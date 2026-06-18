use std::fs;

use anyhow::{Context, Result};
use candle_core::{DType, Tensor};

use crate::cli::DiagArgs;
use crate::core::output::resolve_output_path;
use crate::local::kv_io;
use crate::local::tokenize::encode;
use crate::models::engine::{load_model, resolve_revision};

pub fn run(args: DiagArgs) -> Result<()> {
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

    // Path A: full prefill.
    bundle.model.clear_kv_cache();
    let full_tokens = [ctx_tokens.clone(), q_tokens.clone()].concat();
    let input = Tensor::new(full_tokens.as_slice(), &bundle.device)?.unsqueeze(0)?;
    let logits_a = bundle.model.forward(&input, 0)?;

    // Path B: context prefill + KV load.
    bundle.model.clear_kv_cache();
    let input = Tensor::new(ctx_tokens.as_slice(), &bundle.device)?.unsqueeze(0)?;
    let _ = bundle.model.forward(&input, 0)?;
    let cache = bundle.model.get_kv_cache()?;
    let kv_path = resolve_output_path("diag", &args.model, &args.output, "kv")?;
    if let Some(parent) = kv_path.parent() {
        fs::create_dir_all(parent)?;
    }
    kv_io::save_kv(&cache, &kv_path, &args.model)?;
    println!("diag KV path: {}", kv_path.display());

    let (cache, _) = kv_io::load_kv(&kv_path, &bundle.device)?;
    bundle.model.set_kv_cache(cache)?;
    let input = Tensor::new(q_tokens.as_slice(), &bundle.device)?.unsqueeze(0)?;
    let logits_b = bundle.model.forward(&input, ctx_tokens.len())?;

    // Compare logits for the last question token.
    let logits_a = logits_a.squeeze(0)?.squeeze(0)?;
    let logits_b = logits_b.squeeze(0)?.squeeze(0)?;

    let a_argmax = logits_a.argmax(0)?.to_scalar::<u32>()?;
    let b_argmax = logits_b.argmax(0)?.to_scalar::<u32>()?;
    let diff = (&logits_a - &logits_b)?
        .abs()?
        .max_all()?
        .to_dtype(DType::F32)?
        .to_scalar::<f32>()?;

    println!("ctx_len={}  q_len={}", ctx_tokens.len(), q_tokens.len());
    println!(
        "A argmax token: {a_argmax} -> {:?}",
        decode_one(&bundle.tokenizer, a_argmax)
    );
    println!(
        "B argmax token: {b_argmax} -> {:?}",
        decode_one(&bundle.tokenizer, b_argmax)
    );
    println!("max abs logit diff: {diff:.4}");
    println!(
        "{}",
        if a_argmax == b_argmax {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );

    let top_a = top_k_tokens(&logits_a, &bundle.tokenizer, 5)?;
    let top_b = top_k_tokens(&logits_b, &bundle.tokenizer, 5)?;
    print_top_k("A", &top_a, &bundle.tokenizer);
    print_top_k("B", &top_b, &bundle.tokenizer);

    Ok(())
}

fn decode_one(tokenizer: &tokenizers::Tokenizer, id: u32) -> String {
    tokenizer
        .decode(&[id], false)
        .unwrap_or_else(|_| format!("<decode_error:{id}>"))
}

fn top_k_tokens(
    logits: &Tensor,
    _tokenizer: &tokenizers::Tokenizer,
    k: usize,
) -> Result<Vec<(u32, f32)>> {
    let flat = logits.flatten(0, logits.dims().len() - 1)?;
    let flat = flat.to_dtype(DType::F32)?;
    let data = flat.to_vec1::<f32>()?;
    let mut indexed: Vec<(usize, f32)> = data.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(indexed
        .into_iter()
        .take(k)
        .map(|(idx, score)| (idx as u32, score))
        .collect())
}

fn print_top_k(name: &str, tokens: &[(u32, f32)], tokenizer: &tokenizers::Tokenizer) {
    let decoded: Vec<String> = tokens
        .iter()
        .map(|(id, _)| decode_one(tokenizer, *id))
        .collect();
    println!("  top5 {name}: {decoded:?}");
}
