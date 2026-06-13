use std::fs;

use anyhow::Result;
use candle_core::DType;

use crate::cli::VerifyArgs;
use crate::continue_mod::generate;
use crate::kv_io;
use crate::model::{DEFAULT_REVISION, load_model};
use crate::prefill::prefill;

fn encode(tokenizer: &tokenizers::Tokenizer, text: &str, add_special_tokens: bool) -> Vec<u32> {
    tokenizer
        .encode(text, add_special_tokens)
        .expect("tokenization failed")
        .get_ids()
        .to_vec()
}

pub fn run(args: VerifyArgs) -> Result<()> {
    let mut bundle = load_model(&args.model, DEFAULT_REVISION, DType::F16)?;

    let context = ("Retrieval-augmented generation reuses documents across many ".to_string()
        + "queries. Prefilling the same long document repeatedly wastes "
        + "compute. ")
        .repeat(20);
    let question = "\n\nQ: Summarize the key claim in one sentence.\nA:";

    let ctx_tokens = encode(&bundle.tokenizer, &context, true);
    let q_tokens = encode(&bundle.tokenizer, question, false);

    println!("device={:?}  model={}", bundle.device, args.model);

    // Path A: prefill everything from scratch, then continue.
    bundle.model.clear_kv_cache();
    let full_tokens = [ctx_tokens.clone(), q_tokens.clone()].concat();
    let a = generate(&mut bundle, &full_tokens, 0, args.n)?;

    // Path B: prefill only context, save KV, load KV, continue.
    bundle.model.clear_kv_cache();
    prefill(&mut bundle, &ctx_tokens)?;
    let cache = bundle.model.get_kv_cache();
    fs::create_dir_all("results")?;
    let art = kv_io::save_kv(&cache, &args.kv_path, &args.model)?;
    println!(
        "saved KV: {} tokens, {} layers, {:.1} MB, dtype={}",
        art.num_tokens,
        art.num_layers,
        art.nbytes as f64 / 1e6,
        art.dtype
    );

    let (cache, _) = kv_io::load_kv(&args.kv_path, &bundle.device)?;
    bundle.model.set_kv_cache(cache);
    let b = generate(&mut bundle, &q_tokens, ctx_tokens.len(), args.n)?;

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

    let matched = a == b;
    println!(
        "\ntoken match: {}/{}  ({})",
        if matched { args.n } else { 0 },
        args.n,
        if matched { "PASS" } else { "MISMATCH" }
    );

    if !matched {
        anyhow::bail!("KV reuse did not match scratch prefill");
    }
    Ok(())
}
