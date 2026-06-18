use std::fs;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Tensor};

use crate::cli::BenchmarkArgs;
use crate::model::{load_model, load_model_on, resolve_revision};
use crate::models::ModelConfig;
use crate::output::{atomic_write, resolve_output_path};
use crate::tokenize::{context_of_length, encode};

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    }
}

fn device_synchronize(device: &candle_core::Device) -> Result<()> {
    use candle_core::backend::BackendDevice;
    match device {
        candle_core::Device::Cpu => Ok(()),
        candle_core::Device::Cuda(dev) => dev.synchronize().map_err(|e| anyhow::anyhow!(e)),
        candle_core::Device::Metal(dev) => dev.synchronize().map_err(|e| anyhow::anyhow!(e)),
    }
}

fn approx_prefill_flops(tokens: usize, cfg: &ModelConfig) -> f64 {
    // Count trainable parameters the same way kvstore does.
    let hidden = cfg.hidden_size as f64;
    let intermediate = cfg.intermediate_size as f64;
    let vocab = cfg.vocab_size as f64;
    let layers = cfg.num_hidden_layers as f64;
    let d_model = cfg.hidden_size as f64;
    let n_heads = cfg.num_attention_heads as f64;
    let _kv_heads = cfg.num_key_value_heads as f64;
    let head_dim = cfg.head_dim as f64;

    let embed = vocab * hidden;
    // Each transformer layer: attention q/k/v/o projections + MLP gate/up/down.
    let attn_proj = 4.0 * hidden * (n_heads * head_dim);
    let mlp = 2.0 * hidden * intermediate + hidden * intermediate;
    let layer_params = attn_proj + mlp;
    let lm_head = embed;
    let norm = hidden; // final layer norm (negligible, but kept for completeness)
    let n_params = embed + layers * layer_params + lm_head + norm;

    let n = tokens as f64;
    let dense = 2.0 * n_params * n;
    let attn = 2.0 * layers * n * n * d_model;
    dense + attn
}

fn time_full_prefill(
    bundle: &mut crate::model::ModelBundle,
    tokens: &[u32],
    reps: usize,
) -> Result<f64> {
    let input = Tensor::new(tokens, &bundle.device)?.unsqueeze(0)?;
    // Warmup.
    bundle.model.clear_kv_cache();
    let _ = bundle.model.forward(&input, 0)?;
    device_synchronize(&bundle.device)?;

    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        bundle.model.clear_kv_cache();
        device_synchronize(&bundle.device)?;
        let start = Instant::now();
        let _ = bundle.model.forward(&input, 0)?;
        device_synchronize(&bundle.device)?;
        times.push(start.elapsed().as_secs_f64());
    }
    Ok(median(times))
}

/// One forward step over a resident KV cache (single continuation token).
fn one_resident_step(bundle: &mut crate::model::ModelBundle, ctx_len: usize) -> Result<()> {
    let input = Tensor::new(&[0u32], &bundle.device)?.unsqueeze(0)?;
    let _ = bundle.model.forward(&input, ctx_len)?;
    Ok(())
}

fn time_kv_continuation(
    bundle: &mut crate::model::ModelBundle,
    ctx_tokens: &[u32],
    reps: usize,
) -> Result<f64> {
    // Build a fresh resident cache for each rep.
    let ctx_input = Tensor::new(ctx_tokens, &bundle.device)?.unsqueeze(0)?;

    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        bundle.model.clear_kv_cache();
        let _ = bundle.model.forward(&ctx_input, 0)?;
        let cache = bundle.model.get_kv_cache()?;

        // Warmup on a copy of the resident cache.
        bundle.model.set_kv_cache(cache.clone())?;
        one_resident_step(bundle, ctx_tokens.len())?;
        device_synchronize(&bundle.device)?;

        bundle.model.set_kv_cache(cache)?;
        device_synchronize(&bundle.device)?;
        let start = Instant::now();
        one_resident_step(bundle, ctx_tokens.len())?;
        device_synchronize(&bundle.device)?;
        times.push(start.elapsed().as_secs_f64());
    }
    Ok(median(times))
}

pub fn run(args: BenchmarkArgs) -> Result<()> {
    let revision = resolve_revision(args.revision.as_deref());
    let mut bundle = if let Some(device) = args.device {
        load_model_on(&args.model, &revision, DType::F16, device)?
    } else {
        load_model(&args.model, &revision, DType::F16)?
    };

    let context_text = if let Some(path) = &args.context_file {
        Some(fs::read_to_string(path).with_context(|| format!("reading context file {path:?}"))?)
    } else {
        None
    };

    let lengths = args.lengths.unwrap_or_else(|| {
        vec![255, 485, 945, 1888, 3774]
            .into_iter()
            .take_while(|&l| l <= 4096)
            .collect()
    });

    let output_path = resolve_output_path("benchmark", &args.model, &args.output, "csv")?;
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let cfg = bundle.model.config().clone();
    let kv_heads = cfg.num_key_value_heads as f64;
    let head_dim = cfg.head_dim as f64;
    let bytes_per_tok = 2.0 * cfg.num_hidden_layers as f64 * kv_heads * head_dim * 2.0; // fp16

    #[derive(Debug)]
    struct Row {
        tokens: usize,
        prefill_s: f64,
        kv_attn_s: f64,
        kv_mb: f64,
        prefill_gflops: f64,
        speedup_compute: f64,
    }

    let mut rows = Vec::with_capacity(lengths.len());
    for &len in &lengths {
        let ctx_tokens = if let Some(text) = &context_text {
            encode(&bundle.tokenizer, text, true)?
                .into_iter()
                .cycle()
                .take(len)
                .collect::<Vec<_>>()
        } else {
            context_of_length(&bundle.tokenizer, len)?
        };
        let prefill_s = time_full_prefill(&mut bundle, &ctx_tokens, args.reps.max(1))?;
        let kv_attn_s = time_kv_continuation(&mut bundle, &ctx_tokens, args.reps.max(1))?;
        let flops = approx_prefill_flops(len, &cfg);
        rows.push(Row {
            tokens: len,
            prefill_s,
            kv_attn_s,
            kv_mb: (len as f64) * bytes_per_tok / 1e6,
            prefill_gflops: flops / 1e9,
            speedup_compute: if kv_attn_s > 0.0 {
                prefill_s / kv_attn_s
            } else {
                0.0
            },
        });
    }

    atomic_write(&output_path, |file| {
        use std::io::Write;
        writeln!(
            file,
            "tokens,prefill_s,kv_attn_s,kv_mb,prefill_gflops,speedup_compute"
        )?;
        for r in &rows {
            writeln!(
                file,
                "{},{:.4},{:.5},{:.2},{:.1},{:.2}",
                r.tokens, r.prefill_s, r.kv_attn_s, r.kv_mb, r.prefill_gflops, r.speedup_compute
            )?;
        }
        Ok(())
    })
    .with_context(|| format!("writing {}", output_path.display()))?;

    println!("device={:?}  model={}", bundle.device, args.model);
    println!("benchmark CSV: {}", output_path.display());
    println!(
        "{:>8}  {:>10}  {:>12}  {:>8}  {:>14}  {:>15}",
        "tokens", "prefill_s", "kv_attn_s", "kv_mb", "prefill_gflops", "speedup_compute"
    );
    for r in &rows {
        println!(
            "{:>8}  {:>10.4}  {:>12.5}  {:>8.2}  {:>14.1}  {:>15.2}x",
            r.tokens, r.prefill_s, r.kv_attn_s, r.kv_mb, r.prefill_gflops, r.speedup_compute
        );
    }

    Ok(())
}
