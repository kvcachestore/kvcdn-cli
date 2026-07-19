use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::Repo;
use hf_hub::api::sync::{Api, ApiBuilder};
use tokenizers::Tokenizer;

use crate::core::common;
use crate::models::CausalLM;

pub const DEFAULT_REVISION: &str = "c1899de289a04d12100db370d81485cdf75e47ca";

/// Resolve the model revision from an explicit flag, the `HF_REVISION` environment
/// variable, or the pinned `DEFAULT_REVISION`.
pub fn resolve_revision(flag: Option<&str>) -> String {
    flag.map(String::from)
        .or_else(|| std::env::var("HF_REVISION").ok())
        .unwrap_or_else(|| DEFAULT_REVISION.to_string())
}

pub struct ModelBundle {
    pub tokenizer: Tokenizer,
    pub model: Box<dyn CausalLM>,
    pub device: Device,
}

#[derive(serde::Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

#[derive(serde::Deserialize)]
struct RawConfig {
    architectures: Vec<String>,
}

/// Build the Hugging Face Hub API client, honoring the standard cache
/// location environment variables. `HF_HUB_CACHE` (the exact cache directory)
/// takes precedence; otherwise `HF_HOME` is used with `/hub` appended;
/// otherwise the default `~/.cache/huggingface/hub` is used. `HF_ENDPOINT`
/// is honored in all cases via `ApiBuilder::from_env`.
fn hf_api() -> Result<Api> {
    let builder = ApiBuilder::from_env();
    if let Ok(cache_dir) = std::env::var("HF_HUB_CACHE")
        && !cache_dir.is_empty()
    {
        return Ok(builder.with_cache_dir(cache_dir.into()).build()?);
    }
    Ok(builder.build()?)
}

pub fn load_model(model_name: &str, revision: &str, dtype: DType) -> Result<ModelBundle> {
    load_model_on(model_name, revision, dtype, common::pick_device()?)
}

pub fn load_model_on(
    model_name: &str,
    revision: &str,
    dtype: DType,
    device: Device,
) -> Result<ModelBundle> {
    let api = hf_api()?;
    let repo = api.repo(Repo::with_revision(
        model_name.to_string(),
        hf_hub::RepoType::Model,
        revision.to_string(),
    ));

    let tokenizer_file = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(|e| anyhow::anyhow!(e))?;

    let config_file = repo.get("config.json")?;
    let raw: RawConfig = serde_json::from_slice(&std::fs::read(&config_file)?)?;

    let architecture = raw
        .architectures
        .first()
        .map(|s| s.as_str())
        .unwrap_or("unknown");

    // Discover safetensors files: prefer index, fall back to single file.
    let paths: Vec<PathBuf> = if let Ok(index_file) = repo.get("model.safetensors.index.json") {
        let index: SafetensorsIndex = serde_json::from_slice(&std::fs::read(index_file)?)?;
        let mut shard_names: Vec<String> = index
            .weight_map
            .values()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        shard_names.sort();
        shard_names
            .iter()
            .map(|name| repo.get(name))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        vec![repo.get("model.safetensors")?]
    };

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, dtype, &device)? };

    let model: Box<dyn CausalLM> = match crate::models::registry::dispatch(architecture) {
        Some(loader) => loader(&config_file, vb, &device)?,
        None => anyhow::bail!(crate::models::registry::unsupported_message(architecture)),
    };

    Ok(ModelBundle {
        tokenizer,
        model,
        device,
    })
}

pub fn parse_device(s: &str) -> anyhow::Result<Device> {
    match s.to_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "cuda" | "gpu" => Ok(Device::new_cuda(0)?),
        "metal" => Ok(Device::new_metal(0)?),
        _ => anyhow::bail!("unsupported device '{}' (expected cpu, cuda, or metal)", s),
    }
}
