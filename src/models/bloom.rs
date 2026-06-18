//! Bloom causal language-model adapter with explicit KV-cache access.
//!
//! Implements the Bloom transformer stack (`BloomForCausalLM`) using LayerNorm
//! with bias, fused QKV self-attention, a GELU MLP and learned token
//! embeddings tied to the language-modeling head.  ALiBi is omitted for this
//! first pass and replaced by a standard causal mask.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Activation, Linear, VarBuilder};

use crate::models::{CausalLM, KVCache, ModelConfig};

fn default_layer_norm_epsilon() -> f64 {
    1e-5
}

fn default_skip_bias_add() -> bool {
    true
}

#[derive(Debug, Clone, serde::Deserialize)]
#[allow(dead_code)]
pub(crate) struct Config {
    pub vocab_size: usize,
    #[serde(alias = "n_embed")]
    pub hidden_size: usize,
    #[serde(alias = "n_layer")]
    pub num_hidden_layers: usize,
    #[serde(alias = "n_head")]
    pub num_attention_heads: usize,
    #[serde(default)]
    pub hidden_dropout: f64,
    #[serde(default)]
    pub attention_dropout: f64,
    #[serde(default = "default_layer_norm_epsilon")]
    pub layer_norm_epsilon: f64,
    #[serde(default)]
    pub initializer_range: f64,
    #[serde(default)]
    pub apply_residual_connection_post_layernorm: bool,
    #[serde(default = "default_skip_bias_add")]
    pub skip_bias_add: bool,
    #[serde(default, alias = "n_inner")]
    pub intermediate_size: Option<usize>,
}

impl Config {
    pub(crate) fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    fn intermediate_size(&self) -> usize {
        self.intermediate_size.unwrap_or(4 * self.hidden_size)
    }
}

#[derive(Debug, Clone)]
struct BloomAttention {
    query_key_value: Linear,
    dense: Linear,
    num_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    k_cache: Option<Tensor>,
    v_cache: Option<Tensor>,
}

impl BloomAttention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim();

        let query_key_value =
            candle_nn::linear(hidden_size, 3 * hidden_size, vb.pp("query_key_value"))?;
        let dense = candle_nn::linear(hidden_size, hidden_size, vb.pp("dense"))?;

        Ok(Self {
            query_key_value,
            dense,
            num_heads,
            head_dim,
            hidden_size,
            k_cache: None,
            v_cache: None,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        _offset: usize,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        // Fused QKV projection: shape (b, l, 3 * hidden_size).
        let qkv = self.query_key_value.forward(x)?;
        let q = qkv.narrow(2, 0, self.hidden_size)?;
        let k = qkv.narrow(2, self.hidden_size, self.hidden_size)?;
        let v = qkv.narrow(2, 2 * self.hidden_size, self.hidden_size)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        self.k_cache = Some(match self.k_cache.take() {
            None => k.clone(),
            Some(prev) => Tensor::cat(&[&prev, &k], 2)?,
        });
        self.v_cache = Some(match self.v_cache.take() {
            None => v.clone(),
            Some(prev) => Tensor::cat(&[&prev, &v], 2)?,
        });
        let k = self.k_cache.as_ref().unwrap();
        let v = self.v_cache.as_ref().unwrap();

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(v)?;

        ctx.transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
            .apply(&self.dense)
    }

    fn clear_kv_cache(&mut self) {
        self.k_cache = None;
        self.v_cache = None;
    }

    fn get_kv_cache(&self) -> Option<(Tensor, Tensor)> {
        self.k_cache
            .as_ref()
            .and_then(|k| self.v_cache.as_ref().map(|v| (k.clone(), v.clone())))
    }

    fn set_kv_cache(&mut self, k: Tensor, v: Tensor) {
        self.k_cache = Some(k);
        self.v_cache = Some(v);
    }
}

#[derive(Debug, Clone)]
struct BloomMlp {
    dense_h_to_4h: Linear,
    dense_4h_to_h: Linear,
    act: Activation,
}

impl BloomMlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let intermediate_size = cfg.intermediate_size();
        Ok(Self {
            dense_h_to_4h: candle_nn::linear(
                cfg.hidden_size,
                intermediate_size,
                vb.pp("dense_h_to_4h"),
            )?,
            dense_4h_to_h: candle_nn::linear(
                intermediate_size,
                cfg.hidden_size,
                vb.pp("dense_4h_to_h"),
            )?,
            act: Activation::NewGelu,
        })
    }
}

impl Module for BloomMlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.dense_h_to_4h)?
            .apply(&self.act)?
            .apply(&self.dense_4h_to_h)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    input_layernorm: candle_nn::LayerNorm,
    self_attention: BloomAttention,
    post_attention_layernorm: candle_nn::LayerNorm,
    mlp: BloomMlp,
    apply_residual_connection_post_layernorm: bool,
}

impl DecoderLayer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let input_layernorm = candle_nn::layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_epsilon,
            vb.pp("input_layernorm"),
        )?;
        let self_attention = BloomAttention::new(cfg, vb.pp("self_attention"))?;
        let post_attention_layernorm = candle_nn::layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_epsilon,
            vb.pp("post_attention_layernorm"),
        )?;
        let mlp = BloomMlp::new(cfg, vb.pp("mlp"))?;
        Ok(Self {
            input_layernorm,
            self_attention,
            post_attention_layernorm,
            mlp,
            apply_residual_connection_post_layernorm: cfg.apply_residual_connection_post_layernorm,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        // First sub-layer: input LayerNorm -> self-attention -> residual.
        let normalized = self.input_layernorm.forward(x)?;
        let residual = if self.apply_residual_connection_post_layernorm {
            normalized.clone()
        } else {
            x.clone()
        };
        let attn_output = self.self_attention.forward(&normalized, mask, offset)?;
        let x = (attn_output + residual)?;

        // Second sub-layer: post-attention LayerNorm -> MLP -> residual.
        let normalized = self.post_attention_layernorm.forward(&x)?;
        let residual = if self.apply_residual_connection_post_layernorm {
            normalized.clone()
        } else {
            x.clone()
        };
        let mlp_output = self.mlp.forward(&normalized)?;
        mlp_output + residual
    }

    fn clear_kv_cache(&mut self) {
        self.self_attention.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    word_embeddings: candle_nn::Embedding,
    word_embeddings_layernorm: candle_nn::LayerNorm,
    layers: Vec<DecoderLayer>,
    ln_f: candle_nn::LayerNorm,
    device: Device,
    dtype: DType,
    config: Config,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_t = vb.pp("transformer");
        let word_embeddings =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_t.pp("word_embeddings"))?;
        let word_embeddings_layernorm = candle_nn::layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_epsilon,
            vb_t.pp("word_embeddings_layernorm"),
        )?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_h = vb_t.pp("h");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, vb_h.pp(i))?);
        }

        Ok(Self {
            word_embeddings,
            word_embeddings_layernorm,
            layers,
            ln_f: candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_epsilon, vb_t.pp("ln_f"))?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            config: cfg.clone(),
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    fn clear_kv_cache(&mut self) {
        for l in &mut self.layers {
            l.clear_kv_cache();
        }
    }

    fn causal_mask(&self, b: usize, tgt: usize, offset: usize) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| (0..(tgt + offset)).map(move |j| if j <= i + offset { 0. } else { minf }))
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut h = self
            .word_embeddings
            .forward(input)?
            .apply(&self.word_embeddings_layernorm)?;

        let causal = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset)?)
        };

        for layer in &mut self.layers {
            h = layer.forward(&h, causal.as_ref(), offset)?;
        }
        self.ln_f.forward(&h)
    }

    pub fn get_kv_cache(&self) -> Result<KVCache> {
        self.layers
            .iter()
            .map(|l| {
                l.self_attention.get_kv_cache().ok_or_else(|| {
                    candle_core::Error::Msg("get_kv_cache called before any forward pass".into())
                })
            })
            .collect::<Result<Vec<_>>>()
    }

    pub fn set_kv_cache(&mut self, cache: KVCache) -> Result<()> {
        if cache.len() != self.layers.len() {
            candle_core::bail!(
                "KV cache layer count mismatch: expected {}, got {}",
                self.layers.len(),
                cache.len()
            );
        }
        for (layer, (k, v)) in self.layers.iter_mut().zip(cache) {
            layer.self_attention.set_kv_cache(k, v);
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ModelForCausalLM {
    base: Model,
    lm_head: Linear,
}

impl ModelForCausalLM {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let base = Model::new(cfg, vb.clone())?;
        // Bloom ties the output projection to the input word embeddings.
        let lm_head = Linear::new(base.word_embeddings.embeddings().clone(), None);
        Ok(Self { base, lm_head })
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, l) = input.dims2()?;
        self.base
            .forward(input, offset)?
            .narrow(1, l - 1, 1)?
            .apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base.clear_kv_cache();
    }

    pub fn get_kv_cache(&self) -> Result<KVCache> {
        self.base.get_kv_cache()
    }

    pub fn set_kv_cache(&mut self, cache: KVCache) -> Result<()> {
        self.base.set_kv_cache(cache)
    }
}

fn to_model_config(cfg: &Config) -> ModelConfig {
    ModelConfig {
        vocab_size: cfg.vocab_size,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size(),
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        head_dim: cfg.head_dim(),
        num_key_value_heads: cfg.num_attention_heads,
    }
}

impl CausalLM for ModelForCausalLM {
    fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        self.forward(input, offset)
    }

    fn clear_kv_cache(&mut self) {
        self.clear_kv_cache()
    }

    fn get_kv_cache(&self) -> Result<KVCache> {
        self.get_kv_cache()
    }

    fn set_kv_cache(&mut self, cache: KVCache) -> Result<()> {
        self.set_kv_cache(cache)
    }

    fn config(&self) -> ModelConfig {
        to_model_config(self.base.config())
    }
}

#[cfg(test)]
mod tests {
    use super::Config;

    #[test]
    fn bloom_config_deserializes_with_aliases_and_defaults() {
        let json = r#"{
            "vocab_size": 250880,
            "n_embed": 1024,
            "n_layer": 24,
            "num_attention_heads": 16,
            "hidden_dropout": 0.0,
            "attention_dropout": 0.0,
            "layer_norm_epsilon": 1e-5,
            "initializer_range": 0.02,
            "apply_residual_connection_post_layernorm": false,
            "skip_bias_add": true,
            "pretraining_tp": 1,
            "slow_but_exact": false,
            "bias_dropout_fusion": true
        }"#;
        let cfg: Config = serde_json::from_str(json).expect("config should deserialize");
        assert_eq!(cfg.vocab_size, 250880);
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.intermediate_size(), 4096);
        assert!(!cfg.apply_residual_connection_post_layernorm);
        assert!(cfg.skip_bias_add);
    }

    #[test]
    fn bloom_config_uses_hidden_size_and_n_head_aliases() {
        let json = r#"{
            "vocab_size": 100,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "n_head": 4,
            "n_inner": 256
        }"#;
        let cfg: Config = serde_json::from_str(json).expect("config should deserialize");
        assert_eq!(cfg.hidden_size, 64);
        assert_eq!(cfg.num_hidden_layers, 2);
        assert_eq!(cfg.num_attention_heads, 4);
        assert_eq!(cfg.intermediate_size(), 256);
    }

    #[test]
    fn bloom_config_defaults() {
        let json = r#"{"vocab_size": 10, "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2}"#;
        let cfg: Config = serde_json::from_str(json).expect("config should deserialize");
        assert_eq!(cfg.layer_norm_epsilon, 1e-5);
        assert!(!cfg.apply_residual_connection_post_layernorm);
        assert!(cfg.skip_bias_add);
        assert_eq!(cfg.intermediate_size(), 128);
    }
}
