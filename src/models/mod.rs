pub mod engine;
pub(crate) mod registry;

pub(crate) mod bloom;
pub(crate) mod chatglm;
pub(crate) mod deepseek2;
pub(crate) mod falcon;
pub(crate) mod gemma;
pub(crate) mod gemma3;
pub(crate) mod glm4;
pub(crate) mod gpt2;
pub(crate) mod granite;
pub(crate) mod granite_moe;
pub(crate) mod llama;
pub(crate) mod mamba;
pub(crate) mod mamba2;
pub(crate) mod olmo;
pub(crate) mod phi3;
pub(crate) mod qwen2;
pub(crate) mod qwen3;
pub(crate) mod rwkv6;
pub(crate) mod stable_lm;
pub(crate) mod starcoder2;

use candle_core::{Result, Tensor};

/// A layer-wise key/value cache exposed by causal language-model adapters.
pub type KVCache = Vec<(Tensor, Tensor)>;

/// Generic model-configuration subset used by the CLI (benchmarking, sizing,
/// metadata). Each adapter maps its architecture-specific config to this.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub num_key_value_heads: usize,
}

/// Trait implemented by every causal language-model adapter that supports
/// explicit KV-cache capture and restoration. This is what allows the CLI to
/// save, load, quantize, and reuse a precomputed KV cache independently of the
/// underlying architecture.
pub trait CausalLM {
    /// Run a forward pass over `input` (shape `(B, L)`) starting at the given
    /// KV-cache `offset`. Returns logits for the last token.
    fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor>;

    /// Reset the internal KV cache so the next `forward` starts from an empty
    /// prefix.
    fn clear_kv_cache(&mut self);

    /// Capture the current KV cache. Errors if `forward` has not yet been called
    /// on any layer.
    fn get_kv_cache(&self) -> Result<KVCache>;

    /// Replace the current KV cache with the supplied one. The number of layers
    /// must match the model.
    fn set_kv_cache(&mut self, cache: KVCache) -> Result<()>;

    /// Return the generic configuration for this model.
    fn config(&self) -> ModelConfig;
}
