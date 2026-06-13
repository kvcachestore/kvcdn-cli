use std::fs;

use anyhow::Result;
use candle_core::{DType, Tensor};

use crate::cli::DiagArgs;
use crate::kv_io;
use crate::model::{DEFAULT_REVISION, load_model};

fn encode(tokenizer: &tokenizers::Tokenizer, text: &str, add_special_tokens: bool) -> Vec<u32> {
    tokenizer
        .encode(text, add_special_tokens)
        .expect("tokenization failed")
        .get_ids()
        .to_vec()
}

pub fn run(args: DiagArgs) -> Result<()> {
    let mut bundle = load_model(&args.model, DEFAULT_REVISION, DType::F16)?;

    let context = ("Retrieval-augmented generation reuses documents across many ".to_string()
        + "queries. Prefilling the same long document repeatedly wastes "
        + "compute. ")
        .repeat(20);
    let question = "\n\nQ: Summarize the key claim in one sentence.\nA:";

    let ctx_tokens = encode(&bundle.tokenizer, &context, true);
    let q_tokens = encode(&bundle.tokenizer, question, false);

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
    let cache = bundle.model.get_kv_cache();
    fs::create_dir_all("results")?;
    kv_io::save_kv(&cache, "results/diag.kv", &args.model)?;

    let (cache, _) = kv_io::load_kv("results/diag.kv", &bundle.device)?;
    bundle.model.set_kv_cache(cache);
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

    println!("A argmax token: {a_argmax}");
    println!("B argmax token: {b_argmax}");
    println!("argmax match: {}", a_argmax == b_argmax);
    println!("max abs logit diff: {diff:.4}");

    Ok(())
}
