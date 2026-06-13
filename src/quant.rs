use std::fs;

use anyhow::Result;
use candle_core::DType;

use crate::cli::QuantArgs;
use crate::continue_mod::generate;
use crate::kv_io;
use crate::kv_quant::{max_quant_error, quantize_kv};
use crate::model::{DEFAULT_REVISION, load_model};
use crate::prefill::prefill;

fn encode(tokenizer: &tokenizers::Tokenizer, text: &str, add_special_tokens: bool) -> Vec<u32> {
    tokenizer
        .encode(text, add_special_tokens)
        .expect("tokenization failed")
        .get_ids()
        .to_vec()
}

fn context_of_length(tokenizer: &tokenizers::Tokenizer, len: usize) -> Vec<u32> {
    let sentence = "Retrieval-augmented generation reuses documents across many queries. ";
    let mut tokens = encode(tokenizer, sentence, true);
    while tokens.len() < len {
        tokens.extend(encode(tokenizer, sentence, false));
    }
    tokens.truncate(len);
    tokens
}

pub fn run(args: QuantArgs) -> Result<()> {
    let mut bundle = load_model(&args.model, DEFAULT_REVISION, DType::F16)?;

    let ctx_tokens = context_of_length(&bundle.tokenizer, args.context_tokens);
    let question = "\n\nQ: Summarize the key claim in one sentence.\nA:";
    let q_tokens = encode(&bundle.tokenizer, question, false);

    println!("device={:?}  model={}", bundle.device, args.model);

    // Prefill context and save original KV artifact.
    bundle.model.clear_kv_cache();
    let cache = prefill(&mut bundle, &ctx_tokens)?;
    fs::create_dir_all("results")?;
    let orig_art = kv_io::save_kv(&cache, &args.input, &args.model)?;
    println!(
        "saved original KV: {} tokens, {} layers, {:.2} MB, dtype={}",
        orig_art.num_tokens,
        orig_art.num_layers,
        orig_art.nbytes as f64 / 1e6,
        orig_art.dtype
    );

    // Quantize and save.
    let quantized = quantize_kv(&cache)?;
    let q_art = kv_io::save_quantized_kv(&quantized, &args.output, &args.model, DType::F16)?;
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
    let err = max_quant_error(&cache, DType::F16)?;
    println!("max dequantization error: {:.6}", err);

    if args.verify {
        // Path A: original KV cache.
        let (orig_cache, _) = kv_io::load_kv(&args.input, &bundle.device)?;
        bundle.model.set_kv_cache(orig_cache);
        let a = generate(&mut bundle, &q_tokens, ctx_tokens.len(), args.n)?;

        // Path B: quantized -> dequantized KV cache.
        let (q_cache, _) = kv_io::load_kv(&args.output, &bundle.device)?;
        bundle.model.set_kv_cache(q_cache);
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

        let matched = a == b;
        println!(
            "\ntoken match: {}/{}  ({})",
            if matched { args.n } else { 0 },
            args.n,
            if matched { "PASS" } else { "MISMATCH" }
        );

        if !matched {
            anyhow::bail!("quantized KV continuation did not match original KV");
        }
    }

    Ok(())
}
