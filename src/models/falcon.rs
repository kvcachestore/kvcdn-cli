//! Vendored Falcon adapter with explicit KV-cache access.
//!
//! Adapted from `candle-transformers` 0.10.2. Supports the common Falcon
//! RoPE + multi-query attention layout. Alibi and `new_decoder_architecture`
//! are rejected on the first pass.
use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder, embedding, linear_b};

use crate::models::{CausalLM, KVCache, ModelConfig};

const MAX_SEQ_LEN: usize = 5000;

fn default_true() -> bool {
    true
}

fn default_initializer_range() -> f64 {
    0.02
}

// https://huggingface.co/tiiuae/falcon-7b/blob/main/config.json
#[derive(Debug, Clone, serde::Deserialize)]
#[allow(dead_code)]
pub(crate) struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub layer_norm_epsilon: f64,
    #[serde(default = "default_initializer_range")]
    #[allow(dead_code)]
    pub initializer_range: f64,
    #[serde(default = "default_true")]
    #[allow(dead_code)]
    pub use_cache: bool,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    #[serde(default)]
    #[allow(dead_code)]
    pub hidden_dropout: f64,
    #[serde(default)]
    #[allow(dead_code)]
    pub attention_dropout: f64,
    #[serde(default)]
    pub n_head_kv: Option<usize>,
    #[serde(default)]
    pub alibi: bool,
    #[serde(default)]
    pub new_decoder_architecture: bool,
    #[serde(default = "default_true")]
    pub multi_query: bool,
    #[serde(default = "default_true")]
    pub parallel_attn: bool,
    #[serde(default)]
    pub bias: bool,
}

impl Config {
    pub(crate) fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    fn validate(&self) -> Result<()> {
        if self.alibi {
            candle_core::bail!("alibi is not supported");
        }
        if self.new_decoder_architecture {
            candle_core::bail!("new_decoder_architecture is not supported");
        }
        if !self.multi_query {
            candle_core::bail!("non-multi-query Falcon is not supported");
        }
        if self.n_head_kv.is_some() {
            candle_core::bail!("n_head_kv is not supported");
        }
        Ok(())
    }
}

fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    let (weight, bias) = match (vb.get(size, "weight"), vb.get(size, "bias")) {
        (Ok(weight), Ok(bias)) => (weight, bias),
        (Err(err), _) | (_, Err(err)) => {
            if let (Ok(weight), Ok(bias)) = (vb.get(size, "gamma"), vb.get(size, "beta")) {
                (weight, bias)
            } else {
                return Err(err);
            }
        }
    };
    Ok(LayerNorm::new(weight, bias, eps))
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let l = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, l / 2)?;
    let x2 = x.narrow(D::Minus1, l / 2, l - l / 2)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

#[derive(Debug, Clone)]
struct FalconRotaryEmbedding {
    inv_freq: Tensor,
    cache: Option<(usize, Tensor, Tensor)>,
}

impl FalconRotaryEmbedding {
    fn new(device: &Device, cfg: &Config) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / 10000f32.powf(i as f32 / head_dim as f32))
            .collect();
        Ok(Self {
            inv_freq: Tensor::new(inv_freq.as_slice(), device)?,
            cache: None,
        })
    }

    fn cos_sin(
        &mut self,
        seq_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        if let Some((s, cos, sin)) = &self.cache
            && *s == seq_len
        {
            return Ok((cos.clone(), sin.clone()));
        }
        let t = Tensor::arange(0u32, seq_len as u32, device)?
            .to_dtype(dtype)?
            .reshape((seq_len, 1))?;
        let inv_freq = self.inv_freq.to_dtype(dtype)?;
        let freqs = t.matmul(&inv_freq.reshape((1, inv_freq.dim(0)?))?)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        let cos = emb.cos()?;
        let sin = emb.sin()?;
        self.cache = Some((seq_len, cos.clone(), sin.clone()));
        Ok((cos, sin))
    }

    fn forward(&mut self, query: &Tensor, key: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_batch, seq_len, _head_dim) = query.dims3()?;
        let (cos, sin) = self.cos_sin(MAX_SEQ_LEN, query.device(), query.dtype())?;
        let cos = cos.narrow(0, offset, seq_len)?;
        let sin = sin.narrow(0, offset, seq_len)?;
        let qs = (query.broadcast_mul(&cos)? + rotate_half(query)?.broadcast_mul(&sin)?)?;
        let ks = (key.broadcast_mul(&cos)? + rotate_half(key)?.broadcast_mul(&sin)?)?;
        Ok((qs, ks))
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?
        .to_dtype(on_false.dtype())?
        .broadcast_as(shape.dims())?;
    mask.where_cond(&on_true, on_false)
}

#[derive(Debug, Clone)]
struct FalconAttention {
    query_key_value: Linear,
    dense: Linear,
    maybe_rotary: Option<FalconRotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
    inv_norm_factor: f64,
    num_heads: usize,
    head_dim: usize,
    n_head_kv: usize,
}

impl FalconAttention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        if !cfg.multi_query {
            candle_core::bail!("Falcon attention requires multi_query=true");
        }
        let maybe_rotary = if cfg.alibi {
            None
        } else {
            Some(FalconRotaryEmbedding::new(vb.device(), cfg)?)
        };
        let head_dim = cfg.head_dim();
        let hidden_size = cfg.hidden_size;
        // Multi-query: one shared key/value head, plus one query per head.
        let qkv_out_dim = hidden_size + 2 * head_dim;
        let query_key_value =
            linear_b(hidden_size, qkv_out_dim, cfg.bias, vb.pp("query_key_value"))?;
        let dense = linear_b(hidden_size, hidden_size, cfg.bias, vb.pp("dense"))?;
        Ok(Self {
            query_key_value,
            dense,
            maybe_rotary,
            kv_cache: None,
            inv_norm_factor: 1.0 / (head_dim as f64).sqrt(),
            num_heads: cfg.num_attention_heads,
            head_dim,
            n_head_kv: 1,
        })
    }

    fn split_heads(&self, fused_qkv: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b_sz, seq_len, _) = fused_qkv.dims3()?;
        let fused_qkv = fused_qkv.reshape((b_sz, seq_len, self.num_heads + 2, self.head_dim))?;
        let d = fused_qkv.dim(D::Minus2)?;
        let q = fused_qkv.narrow(D::Minus2, 0, d - 2)?;
        let k = fused_qkv.narrow(D::Minus2, d - 2, 1)?;
        let v = fused_qkv.narrow(D::Minus2, d - 1, 1)?;
        Ok((q, k, v))
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let fused_qkv = self.query_key_value.forward(x)?;
        let (query, key, value) = self.split_heads(&fused_qkv)?;
        let (b_sz, seq_len, _, _) = query.dims4()?;
        let query =
            query
                .transpose(1, 2)?
                .reshape((b_sz * self.num_heads, seq_len, self.head_dim))?;
        let key = key
            .transpose(1, 2)?
            .reshape((b_sz * self.n_head_kv, seq_len, self.head_dim))?;
        let value =
            value
                .transpose(1, 2)?
                .reshape((b_sz * self.n_head_kv, seq_len, self.head_dim))?;

        let (query, key) = if let Some(r) = &mut self.maybe_rotary {
            r.forward(&query, &key, offset)?
        } else {
            (query, key)
        };

        let (mut key, mut value) = (key, value);
        if let Some((cache_k, cache_v)) = &self.kv_cache {
            key = Tensor::cat(&[cache_k, &key], 1)?.contiguous()?;
            value = Tensor::cat(&[cache_v, &value], 1)?.contiguous()?;
        }
        self.kv_cache = Some((key.clone(), value.clone()));

        let all_len = offset + seq_len;
        let query = query.reshape((b_sz * self.num_heads, seq_len, self.head_dim))?;
        let key = key.reshape((b_sz * self.n_head_kv, all_len, self.head_dim))?;
        let value = value.reshape((b_sz * self.n_head_kv, all_len, self.head_dim))?;

        let (key, value) = if self.n_head_kv == 1 {
            (
                key.broadcast_as((b_sz * self.num_heads, all_len, self.head_dim))?,
                value.broadcast_as((b_sz * self.num_heads, all_len, self.head_dim))?,
            )
        } else {
            (key, value)
        };

        let attention_scores = (query.matmul(&key.t()?)? * self.inv_norm_factor)?;
        let attention_scores = match mask {
            None => attention_scores,
            Some(mask) => {
                let mask = masked_fill(&mask.to_dtype(DType::F32)?, mask, -1e9)?
                    .to_dtype(query.dtype())?;
                attention_scores.broadcast_add(&mask.squeeze(1)?)?
            }
        };

        let attention_scores =
            candle_nn::ops::softmax(&attention_scores.to_dtype(DType::F32)?, D::Minus1)?
                .to_dtype(x.dtype())?;
        let attn_output = attention_scores
            .matmul(&value)?
            .reshape((b_sz, self.num_heads, seq_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;
        self.dense.forward(&attn_output)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }

    fn get_kv_cache(&self) -> Option<(Tensor, Tensor)> {
        self.kv_cache.as_ref().map(|(k, v)| (k.clone(), v.clone()))
    }

    fn set_kv_cache(&mut self, k: Tensor, v: Tensor) {
        self.kv_cache = Some((k, v));
    }
}

#[derive(Debug, Clone)]
struct FalconMlp {
    dense_h_to_4h: Linear,
    dense_4h_to_h: Linear,
}

impl FalconMlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let b = cfg.bias;
        Ok(Self {
            dense_h_to_4h: linear_b(h, 4 * h, b, vb.pp("dense_h_to_4h"))?,
            dense_4h_to_h: linear_b(4 * h, h, b, vb.pp("dense_4h_to_h"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.dense_h_to_4h.forward(x)?.gelu()?;
        self.dense_4h_to_h.forward(&x)
    }
}

#[derive(Debug, Clone)]
struct FalconDecoderLayer {
    inp_layernorm: LayerNorm,
    self_attention: FalconAttention,
    post_attention_layernorm: Option<LayerNorm>,
    mlp: FalconMlp,
    parallel_attn: bool,
}

impl FalconDecoderLayer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let mlp = FalconMlp::new(cfg, vb.pp("mlp"))?;
        let inp_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_epsilon,
            vb.pp("input_layernorm"),
        )?;
        let self_attention = FalconAttention::new(cfg, vb.pp("self_attention"))?;
        let post_attention_layernorm = if cfg.parallel_attn {
            None
        } else {
            Some(layer_norm(
                cfg.hidden_size,
                cfg.layer_norm_epsilon,
                vb.pp("post_attention_layernorm"),
            )?)
        };
        Ok(Self {
            inp_layernorm,
            self_attention,
            post_attention_layernorm,
            mlp,
            parallel_attn: cfg.parallel_attn,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let residual = x.clone();
        let ln_attn = self.inp_layernorm.forward(x)?;
        let attn_output = self.self_attention.forward(&ln_attn, mask, offset)?;
        let (residual, ln_mlp) = match &self.post_attention_layernorm {
            None => (residual, ln_attn),
            Some(pal) => {
                let residual = (&attn_output + &residual)?;
                let ln_mlp = pal.forward(&residual)?;
                (residual, ln_mlp)
            }
        };
        let mlp_output = self.mlp.forward(&ln_mlp)?;
        let mlp_output = if self.parallel_attn {
            (mlp_output + attn_output)?
        } else {
            mlp_output
        };
        mlp_output + residual
    }

    fn clear_kv_cache(&mut self) {
        self.self_attention.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Model {
    word_embeddings: Embedding,
    layers: Vec<FalconDecoderLayer>,
    ln_f: LayerNorm,
    device: Device,
    dtype: DType,
    config: Config,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        cfg.validate()?;
        let word_embeddings = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("transformer.word_embeddings"),
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("transformer.h");
        for i in 0..cfg.num_hidden_layers {
            layers.push(FalconDecoderLayer::new(cfg, vb_l.pp(i))?);
        }
        Ok(Self {
            word_embeddings,
            layers,
            ln_f: layer_norm(
                cfg.hidden_size,
                cfg.layer_norm_epsilon,
                vb.pp("transformer.ln_f"),
            )?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            config: cfg.clone(),
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }

    fn causal_mask(b_sz: usize, seq_len: usize, offset: usize) -> Result<Tensor> {
        let mask: Vec<_> = (0..seq_len)
            .flat_map(|i| (0..(seq_len + offset)).map(move |j| u8::from(j > i + offset)))
            .collect();
        let mask = Tensor::from_slice(&mask, (seq_len, seq_len + offset), &Device::Cpu)?;
        mask.broadcast_as((b_sz, 1, seq_len, seq_len + offset))
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut h = self.word_embeddings.forward(input)?;
        let mask = if l <= 1 {
            None
        } else {
            Some(Self::causal_mask(b, l, offset)?.to_device(&self.device)?)
        };
        for layer in &mut self.layers {
            h = layer.forward(&h, mask.as_ref(), offset)?;
        }
        self.ln_f.forward(&h)
    }

    fn get_kv_cache(&self) -> Result<KVCache> {
        self.layers
            .iter()
            .map(|l| {
                l.self_attention.get_kv_cache().ok_or_else(|| {
                    candle_core::Error::Msg("get_kv_cache called before any forward pass".into())
                })
            })
            .collect::<Result<Vec<_>>>()
    }

    fn set_kv_cache(&mut self, cache: KVCache) -> Result<()> {
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
        let lm_head = linear_b(cfg.hidden_size, cfg.vocab_size, false, vb.pp("lm_head"))?;
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
        intermediate_size: 4 * cfg.hidden_size,
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        head_dim: cfg.head_dim(),
        num_key_value_heads: cfg.n_head_kv.unwrap_or(1),
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
