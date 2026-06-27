use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

use crate::local::continuation::generate_with_model;
use crate::local::kv_io;
use crate::local::tokenize::encode;
use crate::models::{CausalLM, KVCache};

/// Result of one validation phase.
#[derive(Debug, Clone)]
pub struct PhaseResult {
    pub phase: &'static str,
    pub tokens: Vec<u32>,
    pub duration: std::time::Duration,
}

/// Overall validation report for a single adapter.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub model: String,
    pub passed: bool,
    pub max_tokens: usize,
    pub phases: Vec<PhaseResult>,
    pub mismatches: Vec<(String, usize, u32, u32)>,
    pub error: Option<String>,
}

impl ValidationReport {
    fn new(model: String, max_tokens: usize) -> Self {
        Self {
            model,
            passed: false,
            max_tokens,
            phases: Vec::new(),
            mismatches: Vec::new(),
            error: None,
        }
    }
}

/// Validate a model adapter by comparing greedy generation from:
/// 1. A full scratch prefill (reference).
/// 2. A saved and reloaded KV cache.
/// 3. A quantized, saved, and dequantized KV cache.
///
/// `context` is prefilled; `question` is appended and used for continuation.
/// All generation is greedy (argmax).
pub fn validate_adapter(
    model: &mut dyn CausalLM,
    device: &Device,
    tokenizer: &Tokenizer,
    model_name: &str,
    context: &str,
    question: &str,
    max_tokens: usize,
) -> Result<ValidationReport> {
    let mut report = ValidationReport::new(model_name.to_string(), max_tokens);

    let ctx_tokens =
        encode(tokenizer, context, true).with_context(|| "encoding validation context")?;
    let q_tokens =
        encode(tokenizer, question, false).with_context(|| "encoding validation question")?;
    let full_tokens = [ctx_tokens.clone(), q_tokens.clone()].concat();

    // Reference phase: prefill everything from scratch, then continue.
    let start = Instant::now();
    model.clear_kv_cache();
    let reference = generate_from_tokens(model, device, &full_tokens, 0, max_tokens)?;
    report.phases.push(PhaseResult {
        phase: "reference",
        tokens: reference.clone(),
        duration: start.elapsed(),
    });

    // Cache phase: prefill context, save/load KV cache, then continue.
    let start = Instant::now();
    model.clear_kv_cache();
    let cache = prefill_model(model, device, &ctx_tokens)?;
    let kv_path = temp_path(model_name, "kv");
    let _art = kv_io::save_kv(&cache, &kv_path, model_name)
        .with_context(|| format!("saving KV cache to {}", kv_path.display()))?;
    let (loaded_cache, _) = kv_io::load_kv(&kv_path, device)
        .with_context(|| format!("loading KV cache from {}", kv_path.display()))?;
    model
        .set_kv_cache(loaded_cache)
        .with_context(|| "restoring loaded KV cache")?;
    let cache_tokens =
        generate_from_tokens(model, device, &q_tokens, ctx_tokens.len(), max_tokens)?;
    report.phases.push(PhaseResult {
        phase: "cache",
        tokens: cache_tokens.clone(),
        duration: start.elapsed(),
    });
    let cache_ok = compare_tokens(&mut report, "cache", &reference, &cache_tokens);

    // Quantized phase: prefill context, quantize KV cache to int8, dequantize
    // back to the original cache dtype, save/load, then continue. This tests the
    // int8 round-trip while keeping the same float precision as the reference run.
    let start = Instant::now();
    model.clear_kv_cache();
    let cache = prefill_model(model, device, &ctx_tokens)?;
    let original_dtype = cache[0].0.dtype();
    let quantized =
        crate::local::kv_quant::quantize_kv(&cache).with_context(|| "quantizing KV cache")?;
    let dequant_cache: KVCache = quantized
        .into_iter()
        .enumerate()
        .map(|(i, (k_s, k_q, v_s, v_q))| {
            let k = crate::local::kv_quant::dequantize_tensor(&k_s, &k_q, original_dtype)
                .with_context(|| format!("dequantizing key tensor for layer {i}"))?;
            let v = crate::local::kv_quant::dequantize_tensor(&v_s, &v_q, original_dtype)
                .with_context(|| format!("dequantizing value tensor for layer {i}"))?;
            Ok((k, v))
        })
        .collect::<Result<_>>()?;
    let q_path = temp_path(model_name, "q8.kv");
    let _q_art = kv_io::save_kv(&dequant_cache, &q_path, model_name)
        .with_context(|| format!("saving dequantized KV cache to {}", q_path.display()))?;
    let (loaded_dequant_cache, _) = kv_io::load_kv(&q_path, device)
        .with_context(|| format!("loading dequantized KV cache from {}", q_path.display()))?;
    model
        .set_kv_cache(loaded_dequant_cache)
        .with_context(|| "restoring dequantized KV cache")?;
    let quant_tokens =
        generate_from_tokens(model, device, &q_tokens, ctx_tokens.len(), max_tokens)?;
    report.phases.push(PhaseResult {
        phase: "quantized",
        tokens: quant_tokens.clone(),
        duration: start.elapsed(),
    });
    let quant_ok = compare_tokens(&mut report, "quantized", &reference, &quant_tokens);

    report.passed = cache_ok && quant_ok;
    Ok(report)
}

fn temp_path(model_name: &str, suffix: &str) -> std::path::PathBuf {
    let safe = model_name.replace(|c: char| !c.is_alphanumeric() && c != '-', "_");
    std::env::temp_dir().join(format!("kvcdn-validate-{safe}.{suffix}"))
}

fn prefill_model(model: &mut dyn CausalLM, device: &Device, tokens: &[u32]) -> Result<KVCache> {
    let input = Tensor::new(tokens, device)?.unsqueeze(0)?;
    let _ = model.forward(&input, 0)?;
    Ok(model.get_kv_cache()?)
}

fn generate_from_tokens(
    model: &mut dyn CausalLM,
    device: &Device,
    tokens: &[u32],
    offset: usize,
    n: usize,
) -> Result<Vec<u32>> {
    generate_with_model(model, device, tokens, offset, n)
}

fn compare_tokens(
    report: &mut ValidationReport,
    phase: &'static str,
    expected: &[u32],
    actual: &[u32],
) -> bool {
    let mut matched = true;
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        if e != a {
            report.mismatches.push((phase.to_string(), i, *e, *a));
            matched = false;
        }
    }
    if expected.len() != actual.len() {
        report.mismatches.push((
            phase.to_string(),
            expected.len().min(actual.len()),
            u32::MAX,
            u32::MAX,
        ));
        matched = false;
    }
    matched
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_tokens_detects_mismatch() {
        let mut report = ValidationReport::new("test".to_string(), 3);
        let ok = compare_tokens(&mut report, "cache", &[1, 2, 3], &[1, 2, 4]);
        assert!(!ok);
        assert_eq!(report.mismatches.len(), 1);
        assert_eq!(report.mismatches[0], ("cache".to_string(), 2, 3, 4));
    }

    #[test]
    fn compare_tokens_detects_length_difference() {
        let mut report = ValidationReport::new("test".to_string(), 3);
        let ok = compare_tokens(&mut report, "cache", &[1, 2], &[1, 2, 3]);
        assert!(!ok);
        assert_eq!(report.mismatches.len(), 1);
    }

    #[test]
    fn temp_path_is_sanitized() {
        let p = temp_path("org/model-1", "kv");
        let name = p.file_name().unwrap().to_string_lossy();
        assert!(name.starts_with("kvcdn-validate-"));
        assert!(name.ends_with(".kv"));
        assert!(!name.contains('/'));
    }
}
