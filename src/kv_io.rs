use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::qwen3_model::KVCache;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVArtifact {
    pub model_name: String,
    pub num_layers: usize,
    pub num_tokens: usize,
    pub dtype: String,
    pub nbytes: u64,
    pub quantized: bool,
}

impl KVArtifact {
    fn meta_path<P: AsRef<Path>>(path: P) -> PathBuf {
        let mut p = path.as_ref().as_os_str().to_owned();
        p.push(".json");
        PathBuf::from(p)
    }
}

fn write_meta<P: AsRef<Path>>(path: P, artifact: &KVArtifact) -> Result<()> {
    let meta_path = KVArtifact::meta_path(&path);
    let text = serde_json::to_string_pretty(artifact)?;
    fs::write(&meta_path, text)
        .with_context(|| format!("writing {}", meta_path.display()))?;
    Ok(())
}

fn read_meta<P: AsRef<Path>>(path: P) -> Result<Option<KVArtifact>> {
    let meta_path = KVArtifact::meta_path(&path);
    if !meta_path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(&meta_path)
        .with_context(|| format!("reading {}", meta_path.display()))?;
    let artifact: KVArtifact = serde_json::from_str(&text)
        .with_context(|| format!("parsing {}", meta_path.display()))?;
    Ok(Some(artifact))
}

pub fn save_kv<P: AsRef<Path>>(cache: &KVCache, path: P, model_name: &str) -> Result<KVArtifact> {
    let num_layers = cache.len();
    let num_tokens = cache[0].0.dim(2)?;
    let dtype = format!("{:?}", cache[0].0.dtype());

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for (i, (k, v)) in cache.iter().enumerate() {
        tensors.insert(format!("k_{i}"), k.clone());
        tensors.insert(format!("v_{i}"), v.clone());
    }

    candle_core::safetensors::save(&tensors, &path)?;

    let nbytes = fs::metadata(&path)?.len();
    let artifact = KVArtifact {
        model_name: model_name.to_string(),
        num_layers,
        num_tokens,
        dtype,
        nbytes,
        quantized: false,
    };
    write_meta(&path, &artifact)?;
    Ok(artifact)
}

/// Save a KV cache that has already been split into per-layer quantized
/// tensors: each layer is `(k_scale, k_quant, v_scale, v_quant)`.
pub fn save_quantized_kv<P: AsRef<Path>>(
    layers: &[(Tensor, Tensor, Tensor, Tensor)],
    path: P,
    model_name: &str,
    target_dtype: DType,
) -> Result<KVArtifact> {
    let num_layers = layers.len();
    let num_tokens = layers[0].1.dim(2)?;
    let dtype = format!("{:?}", target_dtype);

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for (i, (k_s, k_q, v_s, v_q)) in layers.iter().enumerate() {
        tensors.insert(format!("k_{i}_q"), k_q.clone());
        tensors.insert(format!("k_{i}_s"), k_s.clone());
        tensors.insert(format!("v_{i}_q"), v_q.clone());
        tensors.insert(format!("v_{i}_s"), v_s.clone());
    }

    candle_core::safetensors::save(&tensors, &path)?;

    let nbytes = fs::metadata(&path)?.len();
    let artifact = KVArtifact {
        model_name: model_name.to_string(),
        num_layers,
        num_tokens,
        dtype,
        nbytes,
        quantized: true,
    };
    write_meta(&path, &artifact)?;
    Ok(artifact)
}

fn dequantize_tensor(scale: &Tensor, q: &Tensor, target_dtype: DType) -> Result<Tensor> {
    let centered = (q.to_dtype(DType::F32)? - 128.0f64)?;
    let x = (centered.broadcast_mul(scale)? / 127.0f64)?;
    Ok(x.to_dtype(target_dtype)?)
}

pub fn load_kv<P: AsRef<Path>>(path: P, device: &Device) -> Result<(KVCache, KVArtifact)> {
    let path = path.as_ref();
    let data = candle_core::safetensors::load(path, device)?;

    // Prefer metadata sidecar; fall back to inference for legacy artifacts.
    let meta = read_meta(path)?;
    let quantized = meta.as_ref().map(|m| m.quantized).unwrap_or_else(|| {
        data.contains_key("k_0_q") || data.contains_key("k_0_s")
    });

    let mut layers = Vec::new();
    if quantized {
        while let Some(k_q) = data.get(&format!("k_{}_q", layers.len())) {
            let idx = layers.len();
            let k_s = data
                .get(&format!("k_{idx}_s"))
                .ok_or_else(|| anyhow::anyhow!("missing k scale for layer {idx}"))?;
            let v_q = data
                .get(&format!("v_{idx}_q"))
                .ok_or_else(|| anyhow::anyhow!("missing v quant for layer {idx}"))?;
            let v_s = data
                .get(&format!("v_{idx}_s"))
                .ok_or_else(|| anyhow::anyhow!("missing v scale for layer {idx}"))?;

            let target_dtype = meta.as_ref().map(|m| parse_dtype(&m.dtype)).unwrap_or(Ok(DType::F16))?;
            let k = dequantize_tensor(k_s, k_q, target_dtype)?;
            let v = dequantize_tensor(v_s, v_q, target_dtype)?;
            layers.push((k, v));
        }
    } else {
        while let Some(k) = data.get(&format!("k_{}", layers.len())) {
            let idx = layers.len();
            let v = data
                .get(&format!("v_{idx}"))
                .ok_or_else(|| anyhow::anyhow!("missing value tensor for layer {idx}"))?;
            layers.push((k.clone(), v.clone()));
        }
    }

    let num_layers = layers.len();
    if num_layers == 0 {
        anyhow::bail!("no KV tensors found in artifact");
    }
    let num_tokens = layers[0].0.dim(2)?;
    let dtype = format!("{:?}", layers[0].0.dtype());
    let nbytes = fs::metadata(path)?.len();

    let artifact = match meta {
        Some(mut a) => {
            a.nbytes = nbytes;
            a.num_layers = num_layers;
            a.num_tokens = num_tokens;
            a.dtype = dtype;
            a
        }
        None => KVArtifact {
            model_name: "unknown".to_string(),
            num_layers,
            num_tokens,
            dtype,
            nbytes,
            quantized: false,
        },
    };

    Ok((layers, artifact))
}

fn parse_dtype(s: &str) -> Result<DType> {
    match s {
        "F16" => Ok(DType::F16),
        "F32" => Ok(DType::F32),
        "BF16" => Ok(DType::BF16),
        other => anyhow::bail!("unsupported target dtype for quantized artifact: {other}"),
    }
}
