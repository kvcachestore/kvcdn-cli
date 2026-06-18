use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::models::KVCache;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KVArtifact {
    pub model_name: String,
    pub num_layers: usize,
    pub num_tokens: usize,
    pub dtype: String,
    pub storage_dtype: Option<String>,
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
    let bytes = serde_json::to_vec_pretty(artifact)?;
    fs::write(&meta_path, bytes).with_context(|| format!("writing {}", meta_path.display()))?;
    Ok(())
}

fn read_meta<P: AsRef<Path>>(path: P) -> Result<Option<KVArtifact>> {
    let meta_path = KVArtifact::meta_path(&path);
    if !meta_path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(&meta_path)
        .with_context(|| format!("reading {}", meta_path.display()))?;
    let artifact: KVArtifact =
        serde_json::from_str(&text).with_context(|| format!("parsing {}", meta_path.display()))?;
    Ok(Some(artifact))
}

pub fn save_kv<P: AsRef<Path>>(cache: &KVCache, path: P, model_name: &str) -> Result<KVArtifact> {
    if cache.is_empty() {
        anyhow::bail!("cannot save empty KV cache");
    }
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
        dtype: dtype.clone(),
        storage_dtype: Some(dtype),
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
    if layers.is_empty() {
        anyhow::bail!("cannot save empty quantized KV cache");
    }
    let num_layers = layers.len();
    let num_tokens = layers[0].1.dim(2)?;
    let dtype = format!("{:?}", target_dtype);
    let storage_dtype = format!("{:?}", layers[0].1.dtype());

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
        storage_dtype: Some(storage_dtype),
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
    let quantized = meta
        .as_ref()
        .map(|m| m.quantized)
        .unwrap_or_else(|| data.contains_key("k_0_q") || data.contains_key("k_0_s"));

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

            let target_dtype = meta
                .as_ref()
                .map(|m| parse_dtype(&m.dtype))
                .unwrap_or(Ok(DType::F16))?;
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

    if let Some(artifact) = meta {
        return Ok((layers, artifact));
    }

    let num_tokens = layers[0].0.dim(2)?;
    let dtype = format!("{:?}", layers[0].0.dtype());
    let nbytes = fs::metadata(path)?.len();

    Ok((
        layers,
        KVArtifact {
            model_name: "unknown".to_string(),
            num_layers,
            num_tokens,
            dtype: dtype.clone(),
            storage_dtype: Some(dtype),
            nbytes,
            quantized: false,
        },
    ))
}

#[derive(Debug, Deserialize)]
struct HeaderTensorInfo {
    dtype: String,
    shape: Vec<usize>,
}

/// Determine the dtype string to report for a quantized artifact when only the
/// safetensors header is available. The header stores the quantized storage
/// dtype (U8/I8), not the original target dtype, so this is a best-effort
/// fallback used only when no JSON sidecar exists.
fn quant_header_dtype(header_dtype: &str) -> String {
    match header_dtype {
        "U8" => "U8".to_string(),
        "I8" => "I8".to_string(),
        _ => "I8".to_string(),
    }
}

/// Read only the safetensors header to infer artifact metadata without
/// loading any tensor data into memory or onto a device.
pub fn read_kv_metadata<P: AsRef<Path>>(path: P) -> Result<KVArtifact> {
    let path = path.as_ref();

    // Prefer the JSON sidecar if present; it carries the authoritative dtype
    // (especially important for quantized artifacts).
    if let Some(artifact) = read_meta(path)? {
        return Ok(artifact);
    }

    let mut file = File::open(path)?;

    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf) as usize;

    // Defensive bounds: a safetensors header should never be this large, and
    // malformed files should not be allowed to allocate unbounded memory.
    const MAX_HEADER_LEN: usize = 16 * 1024 * 1024;
    let file_len = file.metadata()?.len();
    let available = file_len.saturating_sub(8);
    if header_len > MAX_HEADER_LEN || header_len as u64 > available {
        anyhow::bail!("invalid safetensors header length: {header_len} (file size {file_len})");
    }

    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;
    let header: HashMap<String, serde_json::Value> = serde_json::from_slice(&header_buf)?;

    let mut tensors: HashMap<String, HeaderTensorInfo> = HashMap::new();
    for (key, value) in header {
        if key == "__metadata__" {
            continue;
        }
        let Some(obj) = value.as_object() else {
            continue;
        };
        if !(obj.contains_key("dtype")
            && obj.contains_key("shape")
            && obj.contains_key("data_offsets"))
        {
            continue;
        }
        let dtype = obj
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let shape = obj
            .get("shape")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        tensors.insert(key, HeaderTensorInfo { dtype, shape });
    }

    let quantized = tensors.contains_key("k_0_q") || tensors.contains_key("k_0_s");
    let num_layers = if quantized {
        (0..)
            .take_while(|i| tensors.contains_key(&format!("k_{}_q", i)))
            .count()
    } else {
        (0..)
            .take_while(|i| tensors.contains_key(&format!("k_{}", i)))
            .count()
    };
    if num_layers == 0 {
        anyhow::bail!("no KV tensors found in artifact");
    }

    let first_key = if tensors.contains_key("k_0_q") {
        "k_0_q"
    } else {
        "k_0"
    };
    let first = tensors
        .get(first_key)
        .ok_or_else(|| anyhow::anyhow!("missing first KV tensor {}", first_key))?;

    let num_tokens = first
        .shape
        .get(2)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("unexpected KV tensor rank for {}", first_key))?;
    let dtype = if quantized {
        quant_header_dtype(&first.dtype)
    } else {
        first.dtype.clone()
    };
    let storage_dtype = if quantized {
        Some(first.dtype.clone())
    } else {
        None
    };
    let nbytes = file.metadata()?.len();

    Ok(KVArtifact {
        model_name: "unknown".to_string(),
        num_layers,
        num_tokens,
        dtype,
        storage_dtype,
        nbytes,
        quantized,
    })
}

pub fn parse_dtype(s: &str) -> Result<DType> {
    match s {
        "F16" => Ok(DType::F16),
        "F32" => Ok(DType::F32),
        "BF16" => Ok(DType::BF16),
        "FP8" | "F8" | "F8E4M3" => Ok(DType::F8E4M3),
        "FP4" | "F4" => {
            anyhow::bail!(
                "{s} is not supported: candle-core 0.10 does not implement to_dtype(F4); implement custom 4-bit quantization to enable it"
            )
        }
        "FP1" => anyhow::bail!(
            "{s} is not supported: there is no standard 1-bit floating-point dtype; use a custom 1-bit/sign quantization instead"
        ),
        "I8" | "U8" | "I4" | "U4" => anyhow::bail!(
            "{s} is not supported as a dequantization target; current storage is symmetric int8 (U8) and targets must be a float dtype"
        ),
        other => anyhow::bail!("unsupported target dtype for quantized artifact: {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn make_cache(layers: usize, tokens: usize) -> KVCache {
        let dev = Device::Cpu;
        (0..layers)
            .map(|_| {
                let k = Tensor::zeros((1, 4, tokens, 16), DType::F16, &dev).unwrap();
                let v = Tensor::zeros((1, 4, tokens, 16), DType::F16, &dev).unwrap();
                (k, v)
            })
            .collect()
    }

    #[test]
    fn save_and_load_kv_sidecar() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("artifact.safetensors");
        let cache = make_cache(3, 17);
        let saved = save_kv(&cache, &path, "test-model")?;
        assert_eq!(saved.num_layers, 3);
        assert_eq!(saved.num_tokens, 17);
        assert_eq!(saved.dtype, "F16");
        assert!(!saved.quantized);
        assert!(path.with_extension("safetensors.json").exists());

        let (loaded, artifact) = load_kv(&path, &Device::Cpu)?;
        assert_eq!(loaded.len(), 3);
        assert_eq!(artifact.model_name, "test-model");
        assert_eq!(artifact.num_layers, 3);
        assert_eq!(artifact.num_tokens, 17);
        assert_eq!(artifact.dtype, "F16");
        assert!(!artifact.quantized);

        let header_meta = read_kv_metadata(&path)?;
        assert_eq!(header_meta, artifact);
        Ok(())
    }

    #[test]
    fn save_and_load_quantized_kv_sidecar() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("q_artifact.safetensors");
        let cache = make_cache(2, 11);
        let quantized = crate::kv_quant::quantize_kv(&cache)?;
        let saved = save_quantized_kv(&quantized, &path, "q-model", DType::F16)?;
        assert_eq!(saved.num_layers, 2);
        assert_eq!(saved.num_tokens, 11);
        assert_eq!(saved.dtype, "F16");
        assert!(saved.quantized);

        let (loaded, artifact) = load_kv(&path, &Device::Cpu)?;
        assert_eq!(loaded.len(), 2);
        assert_eq!(artifact.model_name, "q-model");
        assert_eq!(artifact.dtype, "F16");
        assert!(artifact.quantized);

        let header_meta = read_kv_metadata(&path)?;
        assert_eq!(header_meta, artifact);
        Ok(())
    }

    #[test]
    fn read_kv_metadata_header_fallback() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("legacy.safetensors");
        let cache = make_cache(4, 23);
        candle_core::safetensors::save(
            &cache
                .iter()
                .enumerate()
                .flat_map(|(i, (k, v))| {
                    [(format!("k_{i}"), k.clone()), (format!("v_{i}"), v.clone())]
                })
                .collect::<HashMap<String, Tensor>>(),
            &path,
        )?;

        let meta = read_kv_metadata(&path)?;
        assert_eq!(meta.num_layers, 4);
        assert_eq!(meta.num_tokens, 23);
        assert_eq!(meta.dtype, "F16");
        assert!(!meta.quantized);
        Ok(())
    }

    #[test]
    fn read_kv_metadata_quantized_header_fallback() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("legacy_q.safetensors");
        let cache = make_cache(2, 7);
        let quantized = crate::kv_quant::quantize_kv(&cache)?;
        let tensors: HashMap<String, Tensor> = quantized
            .iter()
            .enumerate()
            .flat_map(|(i, (k_s, k_q, v_s, v_q))| {
                [
                    (format!("k_{i}_q"), k_q.clone()),
                    (format!("k_{i}_s"), k_s.clone()),
                    (format!("v_{i}_q"), v_q.clone()),
                    (format!("v_{i}_s"), v_s.clone()),
                ]
            })
            .collect();
        candle_core::safetensors::save(&tensors, &path)?;

        let meta = read_kv_metadata(&path)?;
        assert_eq!(meta.num_layers, 2);
        assert_eq!(meta.num_tokens, 7);
        assert!(meta.quantized);
        // Without a sidecar the header only knows the quantized storage dtype.
        assert_eq!(meta.dtype, "U8");
        Ok(())
    }

    #[test]
    fn parse_dtype_accepts_float_targets() {
        assert_eq!(parse_dtype("F16").unwrap(), DType::F16);
        assert_eq!(parse_dtype("F32").unwrap(), DType::F32);
        assert_eq!(parse_dtype("BF16").unwrap(), DType::BF16);
        assert_eq!(parse_dtype("FP8").unwrap(), DType::F8E4M3);
        assert_eq!(parse_dtype("F8").unwrap(), DType::F8E4M3);
        assert_eq!(parse_dtype("F8E4M3").unwrap(), DType::F8E4M3);
    }

    #[test]
    fn parse_dtype_rejects_unsupported_targets() {
        for unsupported in ["I8", "U8", "I4", "U4", "FP4", "F4", "FP1"] {
            let err = parse_dtype(unsupported).unwrap_err().to_string();
            assert!(
                err.contains("not supported"),
                "expected unsupported error for {unsupported}, got: {err}"
            );
        }
    }
}
