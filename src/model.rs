use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::Repo;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

use crate::common;
use crate::qwen3_model::{Config, ModelForCausalLM};

pub const DEFAULT_REVISION: &str = "c1899de289a04d12100db370d81485cdf75e47ca";

pub struct ModelBundle {
    pub tokenizer: Tokenizer,
    pub model: ModelForCausalLM,
    pub device: Device,
}

#[derive(serde::Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

pub fn load_model(model_name: &str, revision: &str, dtype: DType) -> Result<ModelBundle> {
    let device = common::pick_device()?;
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_name.to_string(),
        hf_hub::RepoType::Model,
        revision.to_string(),
    ));

    let tokenizer_file = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(|e| anyhow::anyhow!(e))?;

    let config_file = repo.get("config.json")?;
    let config: Config = serde_json::from_slice(&std::fs::read(config_file)?)?;

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
    let model = ModelForCausalLM::new(&config, vb)?;

    Ok(ModelBundle {
        tokenizer,
        model,
        device,
    })
}
