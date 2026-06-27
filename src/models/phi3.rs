//! Vendored Phi-3 / Phi-4 adapter with explicit KV-cache access.
//!
//! Adapted from `candle-transformers` 0.10.2.

use candle_core::{D, DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Activation, Linear, RmsNorm, VarBuilder, linear_no_bias};
use std::sync::Arc;

use crate::models::{CausalLM, KVCache, ModelConfig};

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) enum RopeScalingType {
    #[serde(rename = "longrope")]
    LongRope,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[allow(dead_code)]
pub(crate) struct RopeScaling {
    pub short_factor: Vec<f32>,
    pub long_factor: Vec<f32>,
    #[serde(rename = "type")]
    pub type_: RopeScalingType,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[allow(dead_code)]
pub(crate) struct Config {
    pub vocab_size: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub rope_scaling: Option<RopeScaling>,
    pub max_position_embeddings: usize,
    pub original_max_position_embeddings: Option<usize>,
    pub partial_rotary_factor: Option<f64>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl Config {
    pub(crate) fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// GQA: repeat key/value heads to match query heads.
fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        xs.unsqueeze(2)?
            .expand((b, n_kv_head, n_rep, seq_len, head_dim))?
            .reshape((b, n_kv_head * n_rep, seq_len, head_dim))
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    partial_dim: Option<usize>,
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let partial_dim = cfg
            .partial_rotary_factor
            .map(|v| (v * cfg.head_dim() as f64) as usize);
        let dim = partial_dim.unwrap_or(cfg.head_dim());
        let freqs = match cfg.rope_scaling.as_ref() {
            None => {
                let max_seq_len = cfg.max_position_embeddings;
                let inv_freq: Vec<_> = (0..dim)
                    .step_by(2)
                    .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
                    .collect();
                let inv_freq_len = inv_freq.len();
                let inv_freq =
                    Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
                let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
                    .to_dtype(dtype)?
                    .reshape((max_seq_len, 1))?;
                t.matmul(&inv_freq)?
            }
            Some(rope_scaling) => {
                let inv_freq_s: Vec<_> = (0..dim)
                    .step_by(2)
                    .zip(rope_scaling.short_factor.iter())
                    .map(|(i, &f)| f / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
                    .collect();
                let inv_freq_s_len = inv_freq_s.len();
                let inv_freq_s =
                    Tensor::from_vec(inv_freq_s, (1, inv_freq_s_len), dev)?.to_dtype(dtype)?;
                let max_seq_len = cfg.max_position_embeddings;
                match cfg.original_max_position_embeddings {
                    None => {
                        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
                            .to_dtype(dtype)?
                            .reshape((max_seq_len, 1))?;
                        t.matmul(&inv_freq_s)?
                    }
                    Some(original_max_seq_len) => {
                        let t_s = Tensor::arange(0u32, original_max_seq_len as u32, dev)?
                            .to_dtype(dtype)?
                            .reshape((original_max_seq_len, 1))?;
                        let freq_s = t_s.matmul(&inv_freq_s)?;
                        let inv_freq_l: Vec<_> = (0..dim)
                            .step_by(2)
                            .zip(rope_scaling.long_factor.iter())
                            .map(|(i, &f)| f / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
                            .collect();
                        let inv_freq_l_len = inv_freq_l.len();
                        let inv_freq_l = Tensor::from_vec(inv_freq_l, (1, inv_freq_l_len), dev)?
                            .to_dtype(dtype)?;
                        let t_l =
                            Tensor::arange(original_max_seq_len as u32, max_seq_len as u32, dev)?
                                .to_dtype(dtype)?
                                .reshape((max_seq_len - original_max_seq_len, 1))?;
                        let freq_l = t_l.matmul(&inv_freq_l)?;
                        Tensor::cat(&[&freq_s, &freq_l], 0)?
                    }
                }
            }
        };
        Ok(Self {
            partial_dim,
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn rope(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let x = match self.partial_dim {
            None => candle_nn::rotary_emb::rope(&xs.contiguous()?, cos, sin)?,
            Some(dim) => {
                let xs_rot = xs.i((.., .., .., ..dim))?.contiguous()?;
                let xs_pass = xs.i((.., .., .., dim..))?;
                let xs_rot = candle_nn::rotary_emb::rope(&xs_rot, cos, sin)?;
                Tensor::cat(&[&xs_rot, &xs_pass], D::Minus1)?.contiguous()?
            }
        };
        Ok(x)
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = self.rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = self.rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct Attention {
    qkv_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    k_cache: Option<Tensor>,
    v_cache: Option<Tensor>,
}

impl Attention {
    fn new(cfg: &Config, rotary_emb: Arc<RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        let num_kv_groups = num_heads / num_kv_heads;
        let op_size = num_heads * head_dim + 2 * num_kv_heads * head_dim;
        let qkv_proj = linear_no_bias(cfg.hidden_size, op_size, vb.pp("qkv_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;
        Ok(Self {
            qkv_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: cfg.hidden_size,
            rotary_emb,
            k_cache: None,
            v_cache: None,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let qkv = self.qkv_proj.forward(xs)?;
        let query_pos = self.num_heads * self.head_dim;
        let query_states = qkv.narrow(D::Minus1, 0, query_pos)?;
        let key_states = qkv.narrow(D::Minus1, query_pos, self.num_kv_heads * self.head_dim)?;
        let value_states = qkv.narrow(
            D::Minus1,
            query_pos + self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        )?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        self.k_cache = Some(match self.k_cache.take() {
            None => key_states,
            Some(prev) => Tensor::cat(&[&prev, &key_states], 2)?,
        });
        self.v_cache = Some(match self.v_cache.take() {
            None => value_states,
            Some(prev) => Tensor::cat(&[&prev, &value_states], 2)?,
        });
        let key_states = self.k_cache.as_ref().unwrap();
        let value_states = self.v_cache.as_ref().unwrap();

        let key_states = repeat_kv(key_states.clone(), self.num_kv_groups)?.contiguous()?;
        let value_states = repeat_kv(value_states.clone(), self.num_kv_groups)?.contiguous()?;

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;
        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&value_states)?;

        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?
            .apply(&self.o_proj)
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
struct Mlp {
    gate_up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
    i_size: usize,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let gate_up_proj = linear_no_bias(hidden_size, 2 * i_size, vb.pp("gate_up_proj"))?;
        let down_proj = linear_no_bias(i_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
            i_size,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let up_states = xs.apply(&self.gate_up_proj)?;
        let gate = up_states.narrow(D::Minus1, 0, self.i_size)?;
        let up_states = up_states.narrow(D::Minus1, self.i_size, self.i_size)?;
        let up_states = (up_states * gate.apply(&self.act_fn))?;
        up_states.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &Config, rotary_emb: Arc<RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(cfg, rotary_emb, vb.pp("self_attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    device: Device,
    dtype: DType,
    config: Config,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("model.layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                cfg,
                rotary_emb.clone(),
                vb_l.pp(layer_idx),
            )?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm: candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?,
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
        let mut h = self.embed_tokens.forward(input)?;

        let causal = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset)?)
        };

        for layer in &mut self.layers {
            h = layer.forward(&h, causal.as_ref(), offset)?;
        }
        self.norm.forward(&h)
    }

    pub fn get_kv_cache(&self) -> Result<KVCache> {
        self.layers
            .iter()
            .map(|l| {
                l.self_attn.get_kv_cache().ok_or_else(|| {
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
            layer.self_attn.set_kv_cache(k, v);
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
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(base.embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
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
        intermediate_size: cfg.intermediate_size,
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        head_dim: cfg.head_dim(),
        num_key_value_heads: cfg.num_key_value_heads,
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
