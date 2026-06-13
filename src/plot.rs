use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

#[derive(Debug)]
pub struct BenchRow {
    pub tokens: usize,
    pub prefill_s: f64,
    pub continuation_s: f64,
    pub speedup: f64,
}

/// ASCII visualization of the amortized cost from reusing a KV cache.
///
/// A full plotting backend (`plotters`) was avoided because it pulls in
/// system-level font/fontconfig dependencies that are not guaranteed to be
/// present in minimal build environments.  This keeps the crate self-contained
/// and still gives an informative summary of the benchmark results.
pub fn run<P: AsRef<Path>>(csv_path: P) -> Result<()> {
    let text = fs::read_to_string(&csv_path)
        .with_context(|| format!("reading {}", csv_path.as_ref().display()))?;

    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() != 4 {
            anyhow::bail!("invalid CSV row {}: {}", i + 1, line);
        }
        rows.push(BenchRow {
            tokens: cols[0].parse().with_context(|| format!("tokens in row {}", i + 1))?,
            prefill_s: cols[1].parse().with_context(|| format!("prefill_s in row {}", i + 1))?,
            continuation_s: cols[2]
                .parse()
                .with_context(|| format!("continuation_s in row {}", i + 1))?,
            speedup: cols[3].parse().with_context(|| format!("speedup in row {}", i + 1))?,
        });
    }

    if rows.is_empty() {
        println!("No benchmark rows found in {}", csv_path.as_ref().display());
        return Ok(());
    }

    println!("\nBenchmark scaling summary\n");
    println!(
        "{:>8}  {:>12}  {:>16}  {:>8}  {:>18}",
        "tokens", "prefill_s", "continuation_s", "speedup", "amortized_cost_s"
    );
    println!("{:-<85}", "");

    for row in &rows {
        let amortized = row.prefill_s / row.speedup;
        println!(
            "{:>8}  {:>12.3}  {:>16.3}  {:>8.2}x  {:>18.3}",
            row.tokens, row.prefill_s, row.continuation_s, row.speedup, amortized
        );
    }

    println!("\nAmortized per-call cost (prefill + n × continuation) / n reuses\n");
    print!("{:>8}  ", "reuses");
    for row in &rows {
        print!("{:>12} ", format!("{} tok", row.tokens));
    }
    println!();
    println!("{:-<60}", "");

    let max_reuses = 10;
    for n in 1..=max_reuses {
        print!("{:>8}  ", n);
        for row in &rows {
            let total = row.prefill_s + (n - 1) as f64 * row.continuation_s;
            let avg = total / n as f64;
            print!("{:>12.3} ", avg);
        }
        println!();
    }

    println!();
    Ok(())
}
