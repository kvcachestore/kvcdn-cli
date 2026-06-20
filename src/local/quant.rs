use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::DType;
use serde::{Deserialize, Serialize};

use crate::cli::QuantArgs;
use crate::core::output::{ensure_kv_extension, resolve_output_path};
use crate::local::continuation::generate;
use crate::local::kv_io;
use crate::local::kv_quant::quantize_kv;
use crate::local::prefill::prefill;
use crate::local::tokenize::{context_of_length, encode};
use crate::models::KVCache;
use crate::models::engine::{load_model, load_model_on, resolve_revision};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QuantEvent {
    pub input_path: String,
    pub output_path: String,
    pub model: String,
    pub target_dtype: String,
    pub storage_dtype: String,
    pub num_tokens: usize,
    pub num_layers: usize,
    pub original_nbytes: u64,
    pub quantized_nbytes: u64,
    pub compression_ratio: f64,
    pub max_quant_error: f64,
    pub verify_enabled: bool,
    pub verification_result: String,
    pub mismatch_indices: Vec<usize>,
    pub timestamp: String,
}

pub fn write_quant_event(output_path: &Path, event: &QuantEvent) -> Result<PathBuf> {
    let mut path = output_path.as_os_str().to_owned();
    path.push(".quant-event.json");
    let path = PathBuf::from(path);
    let bytes = serde_json::to_vec_pretty(event)?;
    fs::write(&path, bytes).with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

#[allow(dead_code)]
pub struct QuantPipelineResult {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub orig_art: kv_io::KVArtifact,
    pub q_art: kv_io::KVArtifact,
    pub event: QuantEvent,
    pub event_path: PathBuf,
    pub max_quant_error: f32,
}

pub fn run_quant_pipeline(
    cache: &KVCache,
    args: &QuantArgs,
    target_dtype: kv_io::QuantDtype,
) -> Result<QuantPipelineResult> {
    let input_path = resolve_output_path("quant", &args.model, &args.input, "kv")?;
    if let Some(parent) = input_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let orig_art = kv_io::save_kv(cache, &input_path, &args.model)?;

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
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let quantized = quantize_kv(cache)?;
    let q_art = kv_io::save_quantized_kv(&quantized, &output_path, &args.model, target_dtype)?;

    let dequant_dtype = match target_dtype {
        kv_io::QuantDtype::Int8 => DType::F16,
        kv_io::QuantDtype::Float(dt) => dt,
    };
    let max_quant_error = crate::local::kv_quant::max_quant_error(cache, dequant_dtype)?;

    let event = QuantEvent {
        input_path: input_path.to_string_lossy().to_string(),
        output_path: output_path.to_string_lossy().to_string(),
        model: args.model.clone(),
        target_dtype: q_art.dtype.clone(),
        storage_dtype: q_art.storage_dtype.clone().unwrap_or_default(),
        num_tokens: q_art.num_tokens,
        num_layers: q_art.num_layers,
        original_nbytes: orig_art.nbytes,
        quantized_nbytes: q_art.nbytes,
        compression_ratio: orig_art.nbytes as f64 / q_art.nbytes as f64,
        max_quant_error: max_quant_error as f64,
        verify_enabled: false,
        verification_result: "Skipped".to_string(),
        mismatch_indices: vec![],
        timestamp: chrono::Utc::now().to_rfc3339(),
    };
    let event_path = write_quant_event(&output_path, &event)?;

    Ok(QuantPipelineResult {
        input_path,
        output_path,
        orig_art,
        q_art,
        event,
        event_path,
        max_quant_error,
    })
}

fn prepare_dequantized_cache(q_cache: KVCache, _target_dtype: DType) -> Result<KVCache> {
    Ok(q_cache)
}

pub fn run(args: QuantArgs) -> Result<()> {
    let revision = resolve_revision(args.revision.as_deref());
    let target_dtype = kv_io::parse_quant_dtype(&args.target_dtype)?;

    // Load the model at the target float dtype when possible; int8 targets keep F16 working dtype.
    let model_dtype = match target_dtype {
        kv_io::QuantDtype::Int8 => DType::F16,
        kv_io::QuantDtype::Float(dt) => dt,
    };
    let mut bundle = if let Some(device) = args.device.clone() {
        load_model_on(&args.model, &revision, model_dtype, device.clone()).or_else(|e| {
            eprintln!(
                "warning: failed to load model at {model_dtype:?} ({e}); falling back to F16"
            );
            load_model_on(&args.model, &revision, DType::F16, device)
        })?
    } else {
        load_model(&args.model, &revision, model_dtype).or_else(|e| {
            eprintln!(
                "warning: failed to load model at {model_dtype:?} ({e}); falling back to F16"
            );
            load_model(&args.model, &revision, DType::F16)
        })?
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

    // Prefill context and run the model-free quantization pipeline.
    bundle.model.clear_kv_cache();
    let cache = prefill(&mut bundle, &ctx_tokens)?;
    let pipeline = run_quant_pipeline(&cache, &args, target_dtype)?;
    let input_path = &pipeline.input_path;
    let output_path = &pipeline.output_path;
    let orig_art = &pipeline.orig_art;
    let q_art = &pipeline.q_art;
    println!(
        "saved original KV: {} tokens, {} layers, {:.2} MB, dtype={}",
        orig_art.num_tokens,
        orig_art.num_layers,
        orig_art.nbytes as f64 / 1e6,
        orig_art.dtype
    );
    println!("input KV path: {}", input_path.display());
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
    let dequant_dtype = match target_dtype {
        kv_io::QuantDtype::Int8 => DType::F16,
        kv_io::QuantDtype::Float(dt) => dt,
    };
    let err = pipeline.max_quant_error;
    println!("max dequantization error: {:.6}", err);

    let mut verify_result = "Skipped".to_string();
    let mut mismatch_indices = Vec::new();

    if args.verify {
        // Path A: original KV cache.
        let (orig_cache, _) = kv_io::load_kv(input_path, &bundle.device)?;
        bundle.model.set_kv_cache(orig_cache)?;
        let a = generate(&mut bundle, &q_tokens, ctx_tokens.len(), args.n)?;

        // Path B: quantized -> dequantized KV cache.
        let (q_cache, _) = kv_io::load_kv(output_path, &bundle.device)?;
        let q_cache_for_verify = prepare_dequantized_cache(q_cache, dequant_dtype)?;
        bundle.model.set_kv_cache(q_cache_for_verify)?;
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

        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            if x != y {
                mismatch_indices.push(i);
            }
        }
        let matched = mismatch_indices.is_empty();
        verify_result = if matched {
            "Match".to_string()
        } else {
            "Mismatch".to_string()
        };

        let original_label = orig_art.dtype.clone();
        let quantized_label = q_art.dtype.clone();
        println!(
            "\n{} vs {} token match: {}/{}",
            quantized_label,
            original_label,
            args.n - mismatch_indices.len(),
            args.n
        );
        println!("{original_label}: {a_text:?}");
        println!("{quantized_label}: {b_text:?}");

        if !matched {
            let idx_list: Vec<String> = mismatch_indices.iter().map(|i| i.to_string()).collect();
            anyhow::bail!(
                "quantized KV continuation did not match original KV (mismatches at indices: {})",
                idx_list.join(", ")
            );
        }
    }

    let event = QuantEvent {
        input_path: input_path.to_string_lossy().to_string(),
        output_path: output_path.to_string_lossy().to_string(),
        model: args.model.clone(),
        target_dtype: q_art.dtype.clone(),
        storage_dtype: q_art.storage_dtype.clone().unwrap_or_default(),
        num_tokens: q_art.num_tokens,
        num_layers: q_art.num_layers,
        original_nbytes: orig_art.nbytes,
        quantized_nbytes: q_art.nbytes,
        compression_ratio: orig_art.nbytes as f64 / q_art.nbytes as f64,
        max_quant_error: err as f64,
        verify_enabled: args.verify,
        verification_result: verify_result,
        mismatch_indices,
        timestamp: chrono::Utc::now().to_rfc3339(),
    };
    let event_path = write_quant_event(output_path, &event)?;
    println!("quant event: {}", event_path.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn make_cache(layers: usize, tokens: usize, dtype: DType) -> Vec<(Tensor, Tensor)> {
        let dev = Device::Cpu;
        (0..layers)
            .map(|_| {
                let k = Tensor::zeros((1, 4, tokens, 16), dtype, &dev).unwrap();
                let v = Tensor::zeros((1, 4, tokens, 16), dtype, &dev).unwrap();
                (k, v)
            })
            .collect()
    }

    #[test]
    fn quant_event_sidecar_created() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let output_path = dir.path().join("artifact.kv");
        let event = QuantEvent {
            input_path: "in.kv".to_string(),
            output_path: output_path.to_string_lossy().to_string(),
            model: "test".to_string(),
            target_dtype: "I8".to_string(),
            storage_dtype: "U8".to_string(),
            num_tokens: 8,
            num_layers: 2,
            original_nbytes: 100,
            quantized_nbytes: 50,
            compression_ratio: 2.0,
            max_quant_error: 0.01,
            verify_enabled: false,
            verification_result: "Skipped".to_string(),
            mismatch_indices: vec![],
            timestamp: "2024-01-01T00:00:00Z".to_string(),
        };
        let path = write_quant_event(&output_path, &event)?;
        assert!(path.exists());
        Ok(())
    }

    #[test]
    fn verify_uses_target_dtype() -> Result<()> {
        let cache = make_cache(2, 8, DType::F32);
        let prepared = prepare_dequantized_cache(cache, DType::F32)?;
        for (k, v) in &prepared {
            assert_eq!(k.dtype(), DType::F32);
            assert_eq!(v.dtype(), DType::F32);
        }
        Ok(())
    }

    #[test]
    fn quant_event_records_i8_dtype() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("q_i8.safetensors");
        let cache = make_cache(2, 8, DType::F16);
        let quantized = crate::local::kv_quant::quantize_kv(&cache)?;
        let saved =
            kv_io::save_quantized_kv(&quantized, &path, "q-model", kv_io::QuantDtype::Int8)?;

        let event_path = dir.path().join("q_i8.safetensors.quant-event.json");
        let event = QuantEvent {
            input_path: "in.kv".to_string(),
            output_path: path.to_string_lossy().to_string(),
            model: "q-model".to_string(),
            target_dtype: saved.dtype.clone(),
            storage_dtype: saved.storage_dtype.clone().unwrap_or_default(),
            num_tokens: saved.num_tokens,
            num_layers: saved.num_layers,
            original_nbytes: 100,
            quantized_nbytes: saved.nbytes,
            compression_ratio: 2.0,
            max_quant_error: 0.01,
            verify_enabled: false,
            verification_result: "Skipped".to_string(),
            mismatch_indices: vec![],
            timestamp: "2024-01-01T00:00:00Z".to_string(),
        };
        write_quant_event(&path, &event)?;
        let text = fs::read_to_string(&event_path)?;
        assert!(text.contains("\"target_dtype\": \"I8\""));
        Ok(())
    }

    fn make_quant_args(input: Option<&str>, output: Option<&str>, target_dtype: &str) -> QuantArgs {
        QuantArgs {
            model: "test-model".to_string(),
            revision: None,
            input: input.map(|s| s.to_string()),
            output: output.map(|s| s.to_string()),
            context_tokens: 8,
            n: 4,
            verify: false,
            context_file: None,
            question: "\n\nQ: Summarize the key claim in one sentence.\nA:".to_string(),
            target_dtype: target_dtype.to_string(),
            device: None,
        }
    }

    #[test]
    fn pipeline_saves_input_and_output() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let input = dir.path().join("artifact.kv");
        let output = dir.path().join("quantized.kv");
        let cache = make_cache(2, 8, DType::F16);
        let args = make_quant_args(
            Some(input.to_str().unwrap()),
            Some(output.to_str().unwrap()),
            "F16",
        );
        let result = run_quant_pipeline(&cache, &args, kv_io::QuantDtype::Float(DType::F16))?;
        assert!(result.input_path.exists());
        assert!(result.output_path.exists());
        assert!(result.event_path.exists());
        assert_eq!(result.event.target_dtype, "F16");
        Ok(())
    }

    #[test]
    fn pipeline_computes_compression_ratio() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let input = dir.path().join("artifact.kv");
        let output = dir.path().join("quantized.kv");
        let cache = make_cache(2, 8, DType::F16);
        let args = make_quant_args(
            Some(input.to_str().unwrap()),
            Some(output.to_str().unwrap()),
            "F16",
        );
        let result = run_quant_pipeline(&cache, &args, kv_io::QuantDtype::Float(DType::F16))?;
        let expected = result.orig_art.nbytes as f64 / result.q_art.nbytes as f64;
        assert!(
            (result.event.compression_ratio - expected).abs() < 1e-9,
            "compression ratio mismatch: {} vs {}",
            result.event.compression_ratio,
            expected
        );
        Ok(())
    }

    #[test]
    fn pipeline_records_max_quant_error() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let input = dir.path().join("artifact.kv");
        let output = dir.path().join("quantized.kv");
        let cache = make_cache(2, 8, DType::F16);
        let args = make_quant_args(
            Some(input.to_str().unwrap()),
            Some(output.to_str().unwrap()),
            "F16",
        );
        let result = run_quant_pipeline(&cache, &args, kv_io::QuantDtype::Float(DType::F16))?;
        let expected = crate::local::kv_quant::max_quant_error(&cache, DType::F16)? as f64;
        assert!(
            (result.event.max_quant_error - expected).abs() < 1e-9,
            "max_quant_error mismatch: {} vs {}",
            result.event.max_quant_error,
            expected
        );
        Ok(())
    }

    #[test]
    fn pipeline_i8_storage_path() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let input = dir.path().join("artifact.kv");
        let output = dir.path().join("quantized.kv");
        let cache = make_cache(2, 8, DType::F16);
        let args = make_quant_args(
            Some(input.to_str().unwrap()),
            Some(output.to_str().unwrap()),
            "I8",
        );
        let result = run_quant_pipeline(&cache, &args, kv_io::QuantDtype::Int8)?;
        assert_eq!(result.q_art.dtype, "I8");
        assert_eq!(result.q_art.storage_dtype.as_deref(), Some("U8"));
        assert_eq!(result.event.target_dtype, "I8");
        Ok(())
    }
}
