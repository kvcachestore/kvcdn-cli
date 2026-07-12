use std::fs;

use anyhow::{Context, Result};
use candle_core::DType;

use crate::cli::VerifyArgs;
use crate::core::output::resolve_output_path;
use crate::local::continuation::generate;
use crate::local::kv_io;
use crate::local::prefill::prefill;
use crate::local::tokenize::encode;
use crate::models::engine::{load_model, resolve_revision};

pub fn run(args: VerifyArgs) -> Result<()> {
    let mut bundle = load_model(
        &args.model,
        &resolve_revision(args.revision.as_deref()),
        DType::F16,
    )?;

    let context = if let Some(path) = &args.context_file {
        fs::read_to_string(path).with_context(|| format!("reading context file {path:?}"))?
    } else {
        ("Retrieval-augmented generation reuses documents across many ".to_string()
            + "queries. Prefilling the same long document repeatedly wastes "
            + "compute. ")
            .repeat(20)
    };
    let question = &args.question;

    let ctx_tokens = encode(&bundle.tokenizer, &context, true)?;
    let q_tokens = encode(&bundle.tokenizer, question, false)?;

    println!("device={:?}  model={}", bundle.device, args.model);

    // Path A: prefill everything from scratch, then continue.
    bundle.model.clear_kv_cache();
    let full_tokens = [ctx_tokens.clone(), q_tokens.clone()].concat();
    let a = generate(&mut bundle, &full_tokens, 0, args.n)?;

    // Path B: prefill only context, save KV, load KV, continue.
    bundle.model.clear_kv_cache();
    prefill(&mut bundle, &ctx_tokens)?;
    let cache = bundle.model.get_kv_cache()?;
    let kv_path = resolve_output_path("verify", &args.model, &args.kv_path, "kv")?;
    if let Some(parent) = kv_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let art = kv_io::save_kv_with_meta(
        &cache,
        &kv_path,
        &args.model,
        Some(context.clone()),
        Some(ctx_tokens.clone()),
    )?;
    println!(
        "saved KV: {} tokens, {} layers, {:.1} MB, dtype={}",
        art.num_tokens,
        art.num_layers,
        art.nbytes as f64 / 1e6,
        art.dtype
    );
    println!("KV artifact path: {}", kv_path.display());

    let (cache, _) = kv_io::load_kv(&kv_path, &bundle.device)?;
    bundle.model.set_kv_cache(cache)?;
    let b = generate(&mut bundle, &q_tokens, ctx_tokens.len(), args.n)?;

    let mut mismatches: Vec<usize> = Vec::new();
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x != y {
            mismatches.push(i);
        }
    }
    let matched = mismatches.is_empty();
    let match_count = args.n - mismatches.len();

    let a_text = bundle
        .tokenizer
        .decode(&a, false)
        .map_err(|e| anyhow::anyhow!(e))?;
    let b_text = bundle
        .tokenizer
        .decode(&b, false)
        .map_err(|e| anyhow::anyhow!(e))?;

    println!("\nPath A (scratch): {a_text:?}");
    println!("Path B (kv):      {b_text:?}");
    println!(
        "\ntoken match: {}/{}  ({})",
        match_count,
        args.n,
        if matched { "PASS" } else { "MISMATCH" }
    );

    if !matched {
        let idx_list: Vec<String> = mismatches.iter().map(|i| i.to_string()).collect();
        anyhow::bail!(
            "KV reuse did not match scratch prefill (mismatches at indices: {})",
            idx_list.join(", ")
        );
    }
    Ok(())
}
