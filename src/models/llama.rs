//! Vendored Llama family adapter with explicit KV-cache access.
//!
//! Adapted from `candle-transformers` 0.10.2. The upstream `Cache` keeps K/V
//! tensors internally; this version owns the cache inside the model and exposes
//! `get_kv_cache` / `set_kv_cache` via the `CausalLM` trait.
use candle_core::{D, DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Embedding, Linear, RmsNorm, VarBuilder, embedding, linear_no_bias};
use std::collections::HashMap;
use std::f32::consts::PI;

use crate::models::{CausalLM, KVCache, ModelConfig};

const DEFAULT_MAX_SEQ_LEN: usize = 4096;

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub(crate) enum Llama3RopeType {
    #[serde(rename = "llama3")]
    Llama3,
    #[default]
    #[serde(rename = "default")]
    Default,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub(crate) struct Llama3RopeConfig {
    pub(crate) factor: f32,
    pub(crate) low_freq_factor: f32,
    pub(crate) high_freq_factor: f32,
    pub(crate) original_max_position_embeddings: usize,
    pub(crate) rope_type: Llama3RopeType,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[allow(dead_code)]
#[serde(untagged)]
pub(crate) enum LlamaEosToks {
    Single(u32),
    Multiple(Vec<u32>),
}

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct Config {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) vocab_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: Option<usize>,
    pub(crate) rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    pub(crate) rope_theta: f32,
    #[allow(dead_code)]
    pub(crate) bos_token_id: Option<u32>,
    #[allow(dead_code)]
    pub(crate) eos_token_id: Option<LlamaEosToks>,
    #[serde(default)]
    pub(crate) rope_scaling: Option<Llama3RopeConfig>,
    #[serde(default = "default_max_position_embeddings")]
    pub(crate) max_position_embeddings: usize,
    pub(crate) tie_word_embeddings: Option<bool>,
}

fn default_rope() -> f32 {
    10_000.0
}

fn default_max_position_embeddings() -> usize {
    DEFAULT_MAX_SEQ_LEN
}

impl Config {
    pub(crate) fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    pub(crate) fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub(crate) fn into_runtime_config(self) -> RuntimeConfig {
        RuntimeConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads(),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            rope_scaling: self.rope_scaling,
            max_position_embeddings: self.max_position_embeddings,
            tie_word_embeddings: self.tie_word_embeddings.unwrap_or(false),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RuntimeConfig {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) vocab_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) rope_theta: f32,
    pub(crate) rope_scaling: Option<Llama3RopeConfig>,
    pub(crate) max_position_embeddings: usize,
    pub(crate) tie_word_embeddings: bool,
}

impl RuntimeConfig {
    pub(crate) fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Debug, Clone)]
struct Cache {
    masks: HashMap<(usize, usize), Tensor>,
    cos: Tensor,
    sin: Tensor,
    kvs: Vec<Option<(Tensor, Tensor)>>,
}

fn calculate_inv_freq(cfg: &RuntimeConfig) -> Vec<f32> {
    let head_dim = cfg.head_dim();
    let base: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect();

    match &cfg.rope_scaling {
        None
        | Some(Llama3RopeConfig {
            rope_type: Llama3RopeType::Default,
            ..
        }) => base,
        Some(rope_scaling) => {
            let low_freq_wavelen =
                rope_scaling.original_max_position_embeddings as f32 / rope_scaling.low_freq_factor;
            let high_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                / rope_scaling.high_freq_factor;
            base.into_iter()
                .map(|freq| {
                    let wavelen = 2. * PI / freq;
                    if wavelen < high_freq_wavelen {
                        freq
                    } else if wavelen > low_freq_wavelen {
                        freq / rope_scaling.factor
                    } else {
                        let smooth = (rope_scaling.original_max_position_embeddings as f32
                            / wavelen
                            - rope_scaling.low_freq_factor)
                            / (rope_scaling.high_freq_factor - rope_scaling.low_freq_factor);
                        (1. - smooth) * freq / rope_scaling.factor + smooth * freq
                    }
                })
                .collect()
        }
    }
}

impl Cache {
    fn new(dtype: DType, cfg: &RuntimeConfig, device: &Device) -> Result<Self> {
        let theta = Tensor::new(calculate_inv_freq(cfg), device)?;
        let idx_theta = Tensor::arange(0, cfg.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;
        Ok(Self {
            masks: HashMap::new(),
            cos,
            sin,
            kvs: vec![None; cfg.num_hidden_layers],
        })
    }

    fn mask(&mut self, seq_len: usize, index_pos: usize, device: &Device) -> Result<Tensor> {
        let kv_len = index_pos + seq_len;
        if let Some(mask) = self.masks.get(&(seq_len, kv_len)) {
            Ok(mask.clone())
        } else {
            let mask = build_causal_mask(seq_len, index_pos, device)?;
            self.masks.insert((seq_len, kv_len), mask.clone());
            Ok(mask)
        }
    }
}

fn build_causal_mask(seq_len: usize, index_pos: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            let end = i + index_pos + 1;
            let start = index_pos;
            (0..seq_len).map(move |j| if j + start < end { 1f32 } else { 0f32 })
        })
        .collect();
    Tensor::from_vec(mask, (seq_len, seq_len + index_pos), device)
}

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
struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
}

impl CausalSelfAttention {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize, cache: &Cache) -> Result<Tensor> {
        let (_b_sz, _, seq_len, _hidden_size) = x.dims4()?;
        let cos = cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = cache.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
        device: &Device,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let mut k = self.apply_rotary_emb(&k, index_pos, cache)?;

        if let Some((cache_k, cache_v)) = &cache.kvs[block_idx] {
            k = Tensor::cat(&[cache_k, &k], 2)?.contiguous()?;
            v = Tensor::cat(&[cache_v, &v], 2)?.contiguous()?;
            let k_seq_len = k.dims()[2];
            if k_seq_len > self.max_position_embeddings {
                k = k
                    .narrow(
                        D::Minus1,
                        k_seq_len - self.max_position_embeddings,
                        self.max_position_embeddings,
                    )?
                    .contiguous()?
            }
            let v_seq_len = v.dims()[2];
            if v_seq_len > self.max_position_embeddings {
                v = v
                    .narrow(
                        D::Minus1,
                        v_seq_len - self.max_position_embeddings,
                        self.max_position_embeddings,
                    )?
                    .contiguous()?
            }
        }
        cache.kvs[block_idx] = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, self.num_attention_heads / self.num_key_value_heads)?;
        let v = repeat_kv(v, self.num_attention_heads / self.num_key_value_heads)?;

        let in_dtype = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;
        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = if seq_len == 1 {
            att
        } else {
            let mask = cache
                .mask(seq_len, index_pos, device)?
                .broadcast_as(att.shape())?;
            masked_fill(&att, &mask, f32::NEG_INFINITY)?
        };

        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att
            .matmul(&v.contiguous()?)?
            .to_dtype(in_dtype)?
            .transpose(1, 2)?
            .reshape(&[b_sz, seq_len, hidden_size])?;
        self.o_proj.forward(&y)
    }

    fn load(vb: VarBuilder, cfg: &RuntimeConfig) -> Result<Self> {
        let size_in = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let size_q = head_dim * cfg.num_attention_heads;
        let size_kv = head_dim * cfg.num_key_value_heads;
        let q_proj = linear_no_bias(size_in, size_q, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(size_in, size_kv, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(size_in, size_kv, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(size_q, size_in, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim,
            max_position_embeddings: cfg.max_position_embeddings,
        })
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    mask.where_cond(&on_true, on_false)
}

#[derive(Debug, Clone)]
struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }

    fn load(vb: VarBuilder, cfg: &RuntimeConfig) -> Result<Self> {
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let c_fc1 = linear_no_bias(h_size, i_size, vb.pp("gate_proj"))?;
        let c_fc2 = linear_no_bias(h_size, i_size, vb.pp("up_proj"))?;
        let c_proj = linear_no_bias(i_size, h_size, vb.pp("down_proj"))?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
        })
    }
}

#[derive(Debug, Clone)]
struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
        device: &Device,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward(&x, index_pos, block_idx, cache, device)? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }

    fn load(vb: VarBuilder, cfg: &RuntimeConfig) -> Result<Self> {
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        let rms_1 =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Llama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
    cache: Cache,
    device: Device,
    cfg: RuntimeConfig,
}

impl Llama {
    pub(crate) fn new(vb: VarBuilder, cfg: Config, device: &Device) -> Result<Self> {
        let cfg = cfg.into_runtime_config();
        let wte = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(wte.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        let ln_f = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(vb.pp(format!("model.layers.{i}")), &cfg))
            .collect::<Result<Vec<_>>>()?;
        let cache = Cache::new(vb.dtype(), &cfg, device)?;
        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            cache,
            device: device.clone(),
            cfg,
        })
    }

    pub(crate) fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mut x = self.wte.forward(x)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, &mut self.cache, &self.device)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    pub(crate) fn clear_kv_cache(&mut self) {
        self.cache.kvs.fill(None);
    }

    pub(crate) fn get_kv_cache(&self) -> Result<KVCache> {
        let mut out = Vec::with_capacity(self.cache.kvs.len());
        for (i, kv) in self.cache.kvs.iter().enumerate() {
            let (k, v) = kv.as_ref().ok_or_else(|| {
                candle_core::Error::Msg(format!("missing KV cache for layer {i}"))
            })?;
            out.push((k.clone(), v.clone()));
        }
        Ok(out)
    }

    pub(crate) fn set_kv_cache(&mut self, cache: KVCache) -> Result<()> {
        if cache.len() != self.cfg.num_hidden_layers {
            candle_core::bail!(
                "KV cache layer count mismatch: got {} expected {}",
                cache.len(),
                self.cfg.num_hidden_layers
            );
        }
        self.cache.kvs = cache.into_iter().map(Some).collect();
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ModelForCausalLM {
    base: Llama,
}

impl ModelForCausalLM {
    pub(crate) fn new(cfg: &Config, vb: VarBuilder, device: &Device) -> Result<Self> {
        Ok(Self {
            base: Llama::new(vb, cfg.clone(), device)?,
        })
    }
}

fn to_model_config(cfg: &RuntimeConfig) -> ModelConfig {
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
        self.base.forward(input, offset)
    }

    fn clear_kv_cache(&mut self) {
        self.base.clear_kv_cache()
    }

    fn get_kv_cache(&self) -> Result<KVCache> {
        self.base.get_kv_cache()
    }

    fn set_kv_cache(&mut self, cache: KVCache) -> Result<()> {
        self.base.set_kv_cache(cache)
    }

    fn config(&self) -> ModelConfig {
        to_model_config(&self.base.cfg)
    }
}
