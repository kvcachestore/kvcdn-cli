pub mod engine;

pub mod bloom;
pub mod chatglm;
pub mod deepseek2;
pub mod falcon;
pub mod gemma;
pub mod gemma3;
pub mod glm4;
pub mod gpt2;
pub mod granite;
pub mod granite_moe;
pub mod llama;
pub mod mamba;
pub mod mamba2;
pub mod olmo;
pub mod phi3;
pub mod qwen2;
pub mod qwen3;
pub(crate) mod registry;
pub mod rwkv6;
pub mod stable_lm;
pub mod starcoder2;

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
