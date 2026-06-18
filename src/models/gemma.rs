//! Vendored Gemma adapter with explicit KV-cache access.
//!
//! Adapted from `candle-transformers` 0.10.2. The upstream `Attention` keeps
//! K/V tensors internally; this version owns the cache inside the model and
//! exposes `get_kv_cache` / `set_kv_cache` via the `CausalLM` trait.
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{
    Activation, Embedding, Linear, RmsNorm as CandleRmsNorm, VarBuilder, embedding, linear_b,
};
use std::collections::HashMap;
use std::sync::Arc;

use crate::models::{CausalLM, KVCache, ModelConfig};

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct Config {
    pub(crate) attention_bias: bool,
    pub(crate) head_dim: usize,
    pub(crate) hidden_act: Option<Activation>,
    pub(crate) hidden_activation: Option<Activation>,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) rope_theta: f64,
    pub(crate) vocab_size: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub(crate) max_position_embeddings: usize,
}

fn default_max_position_embeddings() -> usize {
    4096
}

impl Config {
    pub(crate) fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    fn hidden_act(&self) -> Result<Activation> {
        match (self.hidden_act, self.hidden_activation) {
            (None, Some(act)) | (Some(act), None) => Ok(act),
            (Some(_), Some(_)) => {
                candle_core::bail!("both hidden_act and hidden_activation are set")
            }
            (None, None) => candle_core::bail!("none of hidden_act and hidden_activation are set"),
        }
    }

    pub(crate) fn into_runtime_config(self) -> Result<RuntimeConfig> {
        Ok(RuntimeConfig {
            attention_bias: self.attention_bias,
            head_dim: self.head_dim,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            max_position_embeddings: self.max_position_embeddings,
            hidden_act: self.hidden_act()?,
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RuntimeConfig {
    pub(crate) attention_bias: bool,
    pub(crate) head_dim: usize,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) vocab_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) rope_theta: f64,
    pub(crate) max_position_embeddings: usize,
    pub(crate) hidden_act: Activation,
}

#[derive(Debug, Clone)]
struct Cache {
    masks: HashMap<(usize, usize), Tensor>,
    kvs: Vec<Option<(Tensor, Tensor)>>,
}

fn calculate_inv_freq(cfg: &RuntimeConfig) -> Vec<f32> {
    let dim = cfg.head_dim;
    (0..dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
        .collect()
}

impl Cache {
    fn new(_dtype: DType, cfg: &RuntimeConfig, _device: &Device) -> Result<Self> {
        Ok(Self {
            masks: HashMap::new(),
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

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    mask.where_cond(&on_true, on_false)
}

#[derive(Debug, Clone)]
struct RmsNorm {
    inner: CandleRmsNorm,
    weight: Tensor,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        let inner = CandleRmsNorm::new(weight.clone(), eps);
        Ok(Self { inner, weight })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.inner.forward(x)?;
        x.broadcast_mul(&(&self.weight + 1.0)?)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(_dtype: DType, cfg: &RuntimeConfig, device: &Device) -> Result<Self> {
        let inv_freq = Tensor::new(calculate_inv_freq(cfg), device)?;
        let t = Tensor::arange(0, cfg.max_position_embeddings as u32, device)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq.reshape((1, inv_freq.elem_count()))?)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
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
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
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
    rotary_emb: Arc<RotaryEmbedding>,
}

impl CausalSelfAttention {
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

        let (q, mut k) = self.rotary_emb.apply_rotary_emb_qkv(&q, &k, index_pos)?;

        if let Some((cache_k, cache_v)) = &cache.kvs[block_idx] {
            k = Tensor::cat(&[cache_k, &k], 2)?.contiguous()?;
            v = Tensor::cat(&[cache_v, &v], 2)?.contiguous()?;
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
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let bias = cfg.attention_bias;
        let q_proj = linear_b(hidden_sz, num_heads * head_dim, bias, vb.pp("q_proj"))?;
        let k_proj = linear_b(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("k_proj"))?;
        let v_proj = linear_b(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("v_proj"))?;
        let o_proj = linear_b(num_heads * head_dim, hidden_sz, bias, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: num_heads,
            num_key_value_heads: num_kv_heads,
            head_dim,
            rotary_emb: Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?),
        })
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act: Activation,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let lhs = self.gate_proj.forward(x)?.apply(&self.act)?;
        let rhs = self.up_proj.forward(x)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }

    fn load(vb: VarBuilder, cfg: &RuntimeConfig) -> Result<Self> {
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let gate_proj = linear_b(h_size, i_size, false, vb.pp("gate_proj"))?;
        let up_proj = linear_b(h_size, i_size, false, vb.pp("up_proj"))?;
        let down_proj = linear_b(i_size, h_size, false, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act: cfg.hidden_act,
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
        let x = self.mlp.forward(&self.rms_2.forward(&x)?)?;
        x + residual
    }

    fn load(vb: VarBuilder, cfg: &RuntimeConfig) -> Result<Self> {
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        let rms_1 = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = RmsNorm::new(
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
pub(crate) struct Gemma {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
    cache: Cache,
    device: Device,
    cfg: RuntimeConfig,
}

impl Gemma {
    pub(crate) fn new(vb: VarBuilder, cfg: Config, device: &Device) -> Result<Self> {
        let cfg = cfg.into_runtime_config()?;
        let wte = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = Linear::new(wte.embeddings().clone(), None);
        let ln_f = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
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
        x = (x * (self.cfg.hidden_size as f64).sqrt())?;
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
    base: Gemma,
}

impl ModelForCausalLM {
    pub(crate) fn new(cfg: &Config, vb: VarBuilder, device: &Device) -> Result<Self> {
        Ok(Self {
            base: Gemma::new(vb, cfg.clone(), device)?,
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
        head_dim: cfg.head_dim,
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
