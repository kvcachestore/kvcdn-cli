use anyhow::{Context, Result};
use plotters::prelude::*;
use plotters::style::FontStyle;
use std::fs;

use crate::output::resolve_output_path;

#[derive(Debug)]
pub struct BenchRow {
    pub tokens: usize,
    pub prefill_s: f64,
    pub kv_attn_s: f64,
}

fn parse_csv(path: &std::path::Path) -> Result<Vec<BenchRow>> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() != 6 {
            anyhow::bail!("invalid CSV row {}: {}", i + 1, line);
        }
        rows.push(BenchRow {
            tokens: cols[0]
                .parse()
                .with_context(|| format!("tokens in row {}", i + 1))?,
            prefill_s: cols[1]
                .parse()
                .with_context(|| format!("prefill_s in row {}", i + 1))?,
            kv_attn_s: cols[2]
                .parse()
                .with_context(|| format!("kv_attn_s in row {}", i + 1))?,
        });
    }
    Ok(rows)
}

fn load_system_sans_font() -> Result<&'static [u8]> {
    let mut db = fontdb::Database::new();
    db.load_system_fonts();

    let ids: Vec<fontdb::ID> = db.faces().map(|f| f.id).collect();
    let mut candidates: Vec<(fontdb::ID, String)> = Vec::new();
    for id in ids {
        if let Some(face) = db.face(id) {
            let families: Vec<_> = face
                .families
                .iter()
                .map(|(name, _)| name.to_lowercase())
                .collect();
            let is_sans = families
                .iter()
                .any(|n| n.contains("sans") || n.contains("arial") || n.contains("helvetica"));
            if is_sans {
                candidates.push((id, families.join(", ")));
            }
        }
    }

    // Prefer well-known sans fonts, then fall back to any sans candidate.
    candidates.sort_by(|a, b| {
        let score = |s: &str| {
            let s = s.to_lowercase();
            if s.contains("liberation sans") {
                0
            } else if s.contains("noto sans") {
                1
            } else if s.contains("dejavu sans") {
                2
            } else if s.contains("arial") {
                3
            } else {
                4
            }
        };
        score(&a.1).cmp(&score(&b.1))
    });

    let id = candidates
        .first()
        .map(|(id, _)| *id)
        .context("no system sans-serif font found")?;

    let (source, _face_index) = db
        .face_source(id)
        .context("missing source for selected font")?;
    let bytes: Vec<u8> = match source {
        fontdb::Source::Binary(data) => data.as_ref().as_ref().to_vec(),
        fontdb::Source::File(path) => {
            fs::read(&path).with_context(|| format!("reading {path:?}"))?
        }
        fontdb::Source::SharedFile(_path, data) => data.as_ref().as_ref().to_vec(),
    };

    Ok(Box::leak(bytes.into_boxed_slice()))
}

fn break_even(c_prefill: f64, c_load: f64) -> f64 {
    if c_prefill <= c_load {
        f64::INFINITY
    } else {
        c_prefill / (c_prefill - c_load)
    }
}

/// Plot amortized per-call cost vs reuse count N to a PNG.
///
/// Matches the cost model in kvstore/plot.py:
///   from-scratch: flat per-call cost = C_prefill
///   KV-reuse:     per-call cost = C_prefill / N + C_load
pub fn run(
    csv_path: Option<&str>,
    out_path: Option<&str>,
    max_n: usize,
    model_name: &str,
) -> Result<()> {
    register_font_once()?;

    let csv_path = resolve_output_path("plot", model_name, &csv_path.map(String::from), "csv")?;
    let png_path = resolve_output_path("plot", model_name, &out_path.map(String::from), "png")?;
    run_with_paths(&csv_path, &png_path, max_n, model_name)
}

fn register_font_once() -> Result<()> {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    static mut FONT_DATA: Option<&'static [u8]> = None;
    let mut result: Result<()> = Ok(());
    ONCE.call_once(|| {
        let font_data = match load_system_sans_font() {
            Ok(data) => data,
            Err(e) => {
                result = Err(e);
                return;
            }
        };
        // SAFETY: set only inside call_once and read only afterwards.
        unsafe { FONT_DATA = Some(font_data) };
        if plotters::style::register_font("sans-serif", FontStyle::Normal, font_data).is_err() {
            result = Err(anyhow::anyhow!("failed to register system sans-serif font"));
        }
    });
    result
}

pub fn run_with_paths(
    csv_path: &std::path::Path,
    png_path: &std::path::Path,
    max_n: usize,
    _model_name: &str,
) -> Result<()> {
    let rows = parse_csv(csv_path)?;
    if rows.is_empty() {
        println!("No benchmark rows found in {}", csv_path.display());
        return Ok(());
    }

    if let Some(parent) = png_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let min_cost = rows
        .iter()
        .map(|r| r.kv_attn_s)
        .fold(f64::INFINITY, |a, b| a.min(b));
    let max_cost = rows
        .iter()
        .map(|r| r.prefill_s)
        .fold(0.0f64, |a, b| a.max(b));
    if !(min_cost > 0.0 && max_cost > 0.0) {
        anyhow::bail!("non-positive costs; cannot plot on log scale");
    }
    let y_low = min_cost * 0.8;
    let y_high = max_cost * 1.2;

    let root = BitMapBackend::new(&png_path, (840, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let root = root.margin(10, 10, 10, 10);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Per-call cost vs reuse count: prefill-from-scratch vs KV-reuse",
            ("sans-serif", 20),
        )
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d((1usize..max_n).log_scale(), (y_low..y_high).log_scale())?;

    chart
        .configure_mesh()
        .x_desc("reuse count N (number of agents / queries reading the doc)")
        .y_desc("amortized per-call cost (s)")
        .draw()?;

    let palette = [&RED, &BLUE, &GREEN, &MAGENTA, &CYAN, &BLACK, &YELLOW];

    println!(
        "{:>8}  {:>13}  {:>11}  {:>7}  {:>13}",
        "tokens", "C_prefill(s)", "C_load(s)", "N*", "saving@max_n"
    );

    for (idx, row) in rows.iter().enumerate() {
        let color = palette[idx % palette.len()];
        let ns: Vec<usize> = (1..=max_n).collect();
        let points: Vec<(usize, f64)> = ns
            .iter()
            .map(|&n| (n, row.prefill_s / n as f64 + row.kv_attn_s))
            .collect();
        chart
            .draw_series(LineSeries::new(points, *color))?
            .label(format!("KV-reuse ({} tok)", row.tokens))
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], *color));

        let ne = break_even(row.prefill_s, row.kv_attn_s);
        let saving = max_n as f64 * row.prefill_s - (row.prefill_s + max_n as f64 * row.kv_attn_s);
        println!(
            "{:>8}  {:>13.4}  {:>11.4}  {:>7.2}  {:>12.1}s",
            row.tokens, row.prefill_s, row.kv_attn_s, ne, saving
        );
    }

    let cp_long = rows.last().unwrap().prefill_s;
    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(1, cp_long), (max_n, cp_long)],
            BLACK.mix(0.5),
        )))?
        .label(format!(
            "from-scratch ({} tok)",
            rows.last().unwrap().tokens
        ))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK.mix(0.5)));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    println!("\nplot PNG: {}", png_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn plot_produces_png_from_synthetic_csv() -> Result<()> {
        register_font_once()?;

        let dir = tempfile::tempdir()?;
        let csv_path = dir.path().join("bench.csv");
        let png_path = dir.path().join("out.png");

        let mut file = fs::File::create(&csv_path)?;
        writeln!(
            file,
            "tokens,prefill_s,kv_attn_s,kv_mb,prefill_gflops,speedup_compute"
        )?;
        writeln!(file, "128,0.0100,0.0005,1.0,5.0,20.00")?;
        writeln!(file, "256,0.0200,0.0010,2.0,5.0,20.00")?;

        run_with_paths(&csv_path, &png_path, 1000, "test-model")?;

        let meta = fs::metadata(&png_path)?;
        assert!(meta.len() > 0, "plot PNG was empty");
        Ok(())
    }
}
