use std::fs;
use std::io::Write;
use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Tensor};

use crate::cli::BenchmarkArgs;
use crate::continue_mod::generate;
use crate::model::{load_model, DEFAULT_REVISION};

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

fn time_full_prefill(bundle: &mut crate::model::ModelBundle, tokens: &[u32]) -> Result<f64> {
    bundle.model.clear_kv_cache();
    let input = Tensor::new(tokens, &bundle.device)?.unsqueeze(0)?;
    let start = Instant::now();
    let _ = bundle.model.forward(&input, 0)?;
    Ok(start.elapsed().as_secs_f64())
}

fn time_continuation(
    bundle: &mut crate::model::ModelBundle,
    ctx_tokens: &[u32],
    q_tokens: &[u32],
) -> Result<f64> {
    bundle.model.clear_kv_cache();
    let input = Tensor::new(ctx_tokens, &bundle.device)?.unsqueeze(0)?;
    let _ = bundle.model.forward(&input, 0)?;
    let start = Instant::now();
    let _ = generate(bundle, q_tokens, ctx_tokens.len(), 1)?;
    Ok(start.elapsed().as_secs_f64())
}

pub fn run(args: BenchmarkArgs) -> Result<()> {
    let mut bundle = load_model(&args.model, DEFAULT_REVISION, DType::F16)?;

    let lengths = args.lengths.unwrap_or_else(|| {
        vec![255, 485, 945, 1888, 3774]
            .into_iter()
            .take_while(|&l| l <= 4096)
            .collect()
    });

    let question = "\n\nQ: Summarize the key claim in one sentence.\nA:";
    let q_tokens = encode(&bundle.tokenizer, question, false);

    fs::create_dir_all("results")?;
    let mut file = fs::File::create("results/bench.csv")?;
    writeln!(file, "tokens,prefill_s,continuation_s,speedup")?;

    println!("device={:?}  model={}", bundle.device, args.model);
    println!("tokens,prefill_s,continuation_s,speedup");

    for &len in &lengths {
        let ctx_tokens = context_of_length(&bundle.tokenizer, len);
        let c_prefill = time_full_prefill(&mut bundle, &ctx_tokens)?;
        let c_continue = time_continuation(&mut bundle, &ctx_tokens, &q_tokens)?;
        let speedup = c_prefill / c_continue;
        writeln!(
            file,
            "{},{:.6},{:.6},{:.2}",
            len, c_prefill, c_continue, speedup
        )?;
        println!(
            "{},{:.6},{:.6},{:.2}",
            len, c_prefill, c_continue, speedup
        );
    }

    Ok(())
}
