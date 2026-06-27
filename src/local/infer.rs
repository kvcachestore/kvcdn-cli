use std::fs;
use std::io::{self, Read};

use anyhow::{Context, Result};
use candle_core::DType;

use crate::cli::InferArgs;
use crate::core::common;
use crate::local::continuation::generate_with_model;
use crate::local::kv_io;
use crate::local::tokenize::encode;
use crate::models::engine::{load_model_on, resolve_revision};

pub fn run(args: InferArgs) -> Result<Vec<u32>> {
    let device = args
        .device
        .clone()
        .unwrap_or_else(|| common::pick_device().unwrap_or(candle_core::Device::Cpu));
    let mut bundle = load_model_on(
        &args.model,
        &resolve_revision(args.revision.as_deref()),
        DType::F16,
        device.clone(),
    )?;

    let mut kv_bytes = Vec::new();
    if args.input == "-" {
        io::stdin()
            .read_to_end(&mut kv_bytes)
            .context("reading KV artifact from stdin")?;
    } else {
        kv_bytes = fs::read(&args.input)
            .with_context(|| format!("reading KV artifact from {}", args.input))?;
    }

    let tmp_path = std::env::temp_dir().join(format!("kvcdn-infer-{}.kv", std::process::id()));
    fs::write(&tmp_path, &kv_bytes).context("writing temporary KV artifact")?;

    let (cache, _) =
        kv_io::load_kv(&tmp_path, &bundle.device).with_context(|| "loading KV artifact")?;
    let num_ctx_tokens = cache[0].0.dim(2).unwrap_or(0);
    bundle
        .model
        .set_kv_cache(cache)
        .with_context(|| "setting KV cache on model")?;

    let q_tokens = encode(&bundle.tokenizer, &args.question, false)
        .with_context(|| "encoding continuation prompt")?;
    let generated = generate_with_model(
        &mut *bundle.model,
        &bundle.device,
        &q_tokens,
        num_ctx_tokens,
        args.n,
    )
    .with_context(|| "running continuation generation")?;

    let _ = fs::remove_file(&tmp_path);

    Ok(generated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_args_parse_and_run_fails_cleanly_on_missing_file() {
        let args = InferArgs {
            model: "Qwen/Qwen3-0.6B".to_string(),
            input: "/nonexistent/artifact.kv".to_string(),
            question: "Q".to_string(),
            n: 1,
            revision: None,
            device: Some(candle_core::Device::Cpu),
        };
        assert!(run(args).is_err());
    }
}
