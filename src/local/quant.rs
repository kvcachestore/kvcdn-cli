use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::DType;

use crate::cli::QuantArgs;
use crate::core::output::{ensure_kv_extension, resolve_output_path};
use crate::local::continuation::generate;
use crate::local::kv_io;
use crate::local::kv_quant::{max_quant_error, quantize_kv};
use crate::local::prefill::prefill;
use crate::local::tokenize::{context_of_length, encode};
use crate::models::engine::{load_model, load_model_on, resolve_revision};

pub fn run(args: QuantArgs) -> Result<()> {
    let revision = resolve_revision(args.revision.as_deref());
    let mut bundle = if let Some(device) = args.device {
        load_model_on(&args.model, &revision, DType::F16, device)?
    } else {
        load_model(&args.model, &revision, DType::F16)?
    };

    let ctx_tokens = if let Some(path) = &args.context_file {
        encode(
            &bundle.tokenizer,
            &fs::read_to_string(path).with_context(|| format!("reading context file {path:?}"))?,
            true,
        )?
    } else {
        context_of_length(&bundle.tokenizer, args.context_tokens)?
    };
    let question = &args.question;
    let q_tokens = encode(&bundle.tokenizer, question, false)?;

    println!("device={:?}  model={}", bundle.device, args.model);

    let target_dtype = kv_io::parse_dtype(&args.target_dtype)?;

    // Prefill context and save original KV artifact.
    bundle.model.clear_kv_cache();
    let cache = prefill(&mut bundle, &ctx_tokens)?;
    let input_path = resolve_output_path("quant", &args.model, &args.input, "kv")?;
    if let Some(parent) = input_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let orig_art = kv_io::save_kv(&cache, &input_path, &args.model)?;
    println!(
        "saved original KV: {} tokens, {} layers, {:.2} MB, dtype={}",
        orig_art.num_tokens,
        orig_art.num_layers,
        orig_art.nbytes as f64 / 1e6,
        orig_art.dtype
    );
    println!("input KV path: {}", input_path.display());

    // Quantize and save.
    let output_path = match &args.output {
        Some(p) => ensure_kv_extension(Path::new(p)),
        None => {
            let base = input_path
                .file_stem()
                .map(|s| s.to_os_string())
                .unwrap_or_else(|| input_path.as_os_str().to_owned());
            input_path.with_file_name(format!("{}.q8.kv", base.to_string_lossy()))
        }
    };
    let quantized = quantize_kv(&cache)?;
    let q_art = kv_io::save_quantized_kv(&quantized, &output_path, &args.model, target_dtype)?;
    println!("output KV path: {}", output_path.display());
    println!(
        "saved quantized KV: {} tokens, {} layers, {:.2} MB, dtype={}",
        q_art.num_tokens,
        q_art.num_layers,
        q_art.nbytes as f64 / 1e6,
        q_art.dtype
    );
    println!(
        "compression ratio: {:.2}x",
        orig_art.nbytes as f64 / q_art.nbytes as f64
    );

    // Sanity-check dequantization error on the raw tensors.
    let err = max_quant_error(&cache, target_dtype)?;
    println!("max dequantization error: {:.6}", err);

    if args.verify {
        // Path A: original KV cache.
        let (orig_cache, _) = kv_io::load_kv(&input_path, &bundle.device)?;
        bundle.model.set_kv_cache(orig_cache)?;
        let a = generate(&mut bundle, &q_tokens, ctx_tokens.len(), args.n)?;

        // Path B: quantized -> dequantized KV cache. Convert to the model's F16 working dtype.
        let (q_cache, _) = kv_io::load_kv(&output_path, &bundle.device)?;
        let q_cache_f16: Vec<_> = q_cache
            .iter()
            .map(|(k, v)| Ok((k.to_dtype(DType::F16)?, v.to_dtype(DType::F16)?)))
            .collect::<Result<Vec<_>>>()?;
        bundle.model.set_kv_cache(q_cache_f16)?;
        let b = generate(&mut bundle, &q_tokens, ctx_tokens.len(), args.n)?;

        let a_text = bundle
            .tokenizer
            .decode(&a, false)
            .map_err(|e| anyhow::anyhow!(e))?;
        let b_text = bundle
            .tokenizer
            .decode(&b, false)
            .map_err(|e| anyhow::anyhow!(e))?;

        println!("\nOriginal KV: {a_text:?}");
        println!("Quantized KV: {b_text:?}");

        let mut mismatches: Vec<usize> = Vec::new();
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            if x != y {
                mismatches.push(i);
            }
        }
        let matched = mismatches.is_empty();
        println!(
            "\nint8 vs fp16 token match: {}/{}",
            args.n - mismatches.len(),
            args.n
        );
        println!("fp16: {a_text:?}");
        println!("int8: {b_text:?}");

        if !matched {
            let idx_list: Vec<String> = mismatches.iter().map(|i| i.to_string()).collect();
            anyhow::bail!(
                "quantized KV continuation did not match original KV (mismatches at indices: {})",
                idx_list.join(", ")
            );
        }
    }

    Ok(())
}
