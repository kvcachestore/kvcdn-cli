//! Vendored DeepSeek-V2 / DeepSeek-V3 adapter with explicit KV-cache access.
//!
//! Adapted from `candle-transformers` 0.10.2. Uses Multi-head Latent Attention
//! (MLA): the per-layer "KV cache" captured here is the compressed latent
//! pair `(compressed_kv, k_pe)` rather than the expanded K/V tensors.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{f32::consts::PI, sync::Arc};

use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{
    Activation, Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_b,
    linear_no_bias, rms_norm,
};

use crate::models::{CausalLM, KVCache, ModelConfig};

// ---------------------------------------------------------------------------
// Serde defaults
// ---------------------------------------------------------------------------

fn routed_scaling_factor() -> f64 {
    1.0
}

fn topk_method() -> TopkMethod {
    TopkMethod::Greedy
}

fn scoring_func() -> ScoringFunc {
    ScoringFunc::Softmax
}

fn moe_layer_freq() -> usize {
    1
}

fn first_k_dense_replace() -> usize {
    0
}

fn norm_topk_prob() -> bool {
    false
}

fn hidden_act() -> Activation {
    Activation::Silu
}

fn tie_word_embeddings() -> bool {
    false
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Deserialize)]
enum TopkMethod {
    #[serde(rename = "greedy")]
    Greedy,
    #[serde(rename = "group_limited_greedy")]
    GroupLimitedGreedy,
}

#[derive(Debug, Clone, serde::Deserialize)]
enum ScoringFunc {
    #[serde(rename = "softmax")]
    Softmax,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ScaledRopeType {
    #[serde(alias = "su")]
    #[serde(alias = "longrope")]
    Su,
    #[serde(alias = "yarn")]
    Yarn,
    #[serde(alias = "dynamic")]
    Dynamic,
    #[serde(alias = "linear")]
    Linear,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[allow(dead_code)]
#[serde(untagged)]
pub(crate) enum RopeScaling {
    Yarn {
        original_max_position_embeddings: usize,
        beta_fast: f32,
        beta_slow: f32,
        mscale: f32,
        mscale_all_dim: f32,
        factor: f32,
        #[serde(rename = "type")]
        scaling_type: ScaledRopeType,
    },
    LinearOrDynamic {
        #[serde(rename = "type")]
        scaling_type: ScaledRopeType,
        factor: f64,
    },
}

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub n_shared_experts: Option<usize>,
    pub n_routed_experts: Option<usize>,
    #[serde(default = "routed_scaling_factor")]
    pub routed_scaling_factor: f64,
    pub num_experts_per_tok: Option<usize>,
    #[serde(default = "moe_layer_freq")]
    pub moe_layer_freq: usize,
    #[serde(default = "first_k_dense_replace")]
    pub first_k_dense_replace: usize,
    #[serde(default = "norm_topk_prob")]
    pub norm_topk_prob: bool,
    #[serde(default = "topk_method")]
    topk_method: TopkMethod,
    #[serde(default = "scoring_func")]
    scoring_func: ScoringFunc,
    #[serde(default = "hidden_act")]
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    #[serde(default = "tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    pub rope_theta: f32,
    pub rope_scaling: Option<RopeScaling>,
    pub attention_bias: bool,
    pub q_lora_rank: Option<usize>,
    pub qk_rope_head_dim: usize,
    pub kv_lora_rank: usize,
    pub v_head_dim: usize,
    pub qk_nope_head_dim: usize,
    pub n_group: usize,
    pub topk_group: usize,
}

impl Config {
    pub(crate) fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    fn q_head_dim(&self) -> usize {
        self.qk_rope_head_dim + self.qk_nope_head_dim
    }

    fn softmax_scale(&self) -> f32 {
        let mut scale = 1.0 / (self.q_head_dim() as f32).sqrt();
        if let Some(RopeScaling::Yarn {
            mscale_all_dim,
            factor,
            ..
        }) = self.rope_scaling
        {
            let mscale = RotaryEmbedding::yarn_get_mscale(factor, mscale_all_dim);
            scale *= mscale * mscale;
        }
        scale
    }
}

// ---------------------------------------------------------------------------
// Small helper ops (inlined instead of pulling in rayon/tracing dependencies)
// ---------------------------------------------------------------------------

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    mask.where_cond(&on_true, on_false)
}

/// Split the last dimension into contiguous chunks.
fn split_last_dim(xs: &Tensor, splits: &[usize]) -> Result<Vec<Tensor>> {
    let dim = xs.dim(D::Minus1)?;
    let mut res = Vec::with_capacity(splits.len());
    let mut index = 0;
    for &split in splits {
        if index + split > dim {
            candle_core::bail!("split sizes exceed last dim");
        }
        res.push(xs.narrow(D::Minus1, index, split)?);
        index += split;
    }
    Ok(res)
}

/// Top-k in the last dimension, returning unsorted values/indices.
fn topk_unsorted(xs: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    let sorted_indices = xs.arg_sort_last_dim(false)?;
    let topk_idx_sorted = sorted_indices.narrow(D::Minus1, 0, k)?.contiguous()?;
    let topk_val_sorted = xs.gather(&topk_idx_sorted, D::Minus1)?;

    let reorder = topk_idx_sorted.arg_sort_last_dim(true)?;
    let topk_idx = topk_idx_sorted.gather(&reorder, D::Minus1)?;
    let topk_val = topk_val_sorted.gather(&reorder, D::Minus1)?;
    Ok((topk_val, topk_idx))
}

fn bincount(values: &[u32], minlength: u32) -> Vec<u32> {
    let max_val = values.iter().copied().max().unwrap_or(0);
    let len = ((max_val + 1).max(minlength)) as usize;
    let mut counts = vec![0u32; len];
    for &v in values {
        counts[v as usize] += 1;
    }
    counts
}

/// Returns a `(result_len, ndim)` tensor of nonzero indices in row-major order.
fn nonzero(t: &Tensor) -> Result<Tensor> {
    let data = t.flatten_all()?.to_vec1::<u8>()?;
    let dims = t.dims().to_vec();
    let ndim = dims.len();
    let mut indices = Vec::new();
    // Compute row-major strides so flat index = sum(row[i] * stride[i]).
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    for (idx, &v) in data.iter().enumerate() {
        if v != 0 {
            let mut rem = idx;
            for stride in &strides {
                indices.push((rem / stride) as u32);
                rem %= stride;
            }
        }
    }
    let n = indices.len() / ndim.max(1);
    Tensor::from_vec(indices, (n, ndim), t.device())
}

// ---------------------------------------------------------------------------
// Rotary embeddings (Yarn + unscaled)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

struct RopeConfig {
    rope_scaling: Option<RopeScaling>,
    max_position_embeddings: usize,
    rope_theta: f32,
    qk_rope_head_dim: usize,
}

impl RotaryEmbedding {
    fn new_unscaled(cfg: &RopeConfig, dtype: DType, dev: &Device) -> Result<Self> {
        let max_seq_len = cfg.max_position_embeddings;
        let dim = cfg.qk_rope_head_dim;

        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;

        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    fn yarn_find_correction_dim(
        num_rot: f32,
        dim: usize,
        base: f32,
        max_position_embeddings: usize,
    ) -> f32 {
        (dim as f32 * (max_position_embeddings as f32 / (num_rot * 2. * PI)).ln())
            / (2. * base.ln())
    }

    fn yarn_find_correction_range(
        low_rot: f32,
        high_rot: f32,
        dim: usize,
        base: f32,
        max_position_embeddings: usize,
    ) -> (f32, f32) {
        let low =
            Self::yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings).floor();
        let high =
            Self::yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings).ceil();
        (low.max(0.), high.min(dim as f32 - 1.))
    }

    fn yarn_linear_ramp_mask(min: f32, mut max: f32, dim: usize, dev: &Device) -> Result<Tensor> {
        if min == max {
            max += 0.001;
        }
        let linear_func =
            ((Tensor::arange(0f32, dim as f32, dev)? - min as f64)? / (max as f64 - min as f64))?;
        linear_func.clamp(0., 1.)
    }

    fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
        if scale <= 1. {
            return 1.;
        }
        0.1 * mscale * scale.ln() + 1.
    }

    #[allow(clippy::too_many_arguments)]
    fn new_yarn(
        cfg: &RopeConfig,
        dtype: DType,
        dev: &Device,
        original_max_position_embeddings: usize,
        beta_fast: f32,
        beta_slow: f32,
        factor: f32,
        mscale: f32,
        mscale_all_dim: f32,
    ) -> Result<Self> {
        let freq_extra: Vec<_> = (0..cfg.qk_rope_head_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / cfg.qk_rope_head_dim as f32))
            .collect();
        let freq_extra_len = freq_extra.len();
        let freq_extra = Tensor::from_vec(freq_extra, freq_extra_len, dev)?;
        let freq_inter: Vec<_> = (0..cfg.qk_rope_head_dim)
            .step_by(2)
            .map(|i| 1f32 / (factor * cfg.rope_theta.powf(i as f32 / cfg.qk_rope_head_dim as f32)))
            .collect();
        let freq_inter_len = freq_inter.len();
        let freq_inter = Tensor::from_vec(freq_inter, (1, freq_inter_len), dev)?;

        let (low, high) = Self::yarn_find_correction_range(
            beta_fast,
            beta_slow,
            cfg.qk_rope_head_dim,
            cfg.rope_theta,
            original_max_position_embeddings,
        );
        let inv_freq_mask =
            (1. - Self::yarn_linear_ramp_mask(low, high, cfg.qk_rope_head_dim / 2, dev)?)?;
        let inv_freq = freq_inter
            .broadcast_mul(&(1. - &inv_freq_mask)?)?
            .broadcast_add(&freq_extra.broadcast_mul(&inv_freq_mask)?)?;

        let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;

        let mscale =
            Self::yarn_get_mscale(factor, mscale) / Self::yarn_get_mscale(factor, mscale_all_dim);
        let sin = (freqs.sin()? * mscale as f64)?.to_dtype(dtype)?;
        let cos = (freqs.cos()? * mscale as f64)?.to_dtype(dtype)?;

        Ok(Self { sin, cos })
    }

    fn new(cfg: &RopeConfig, dtype: DType, dev: &Device) -> Result<Self> {
        match &cfg.rope_scaling {
            Some(RopeScaling::LinearOrDynamic { .. }) => {
                candle_core::bail!("linear and dynamic rope are not implemented yet")
            }
            Some(RopeScaling::Yarn {
                original_max_position_embeddings,
                beta_fast,
                beta_slow,
                factor,
                mscale,
                mscale_all_dim,
                ..
            }) => Self::new_yarn(
                cfg,
                dtype,
                dev,
                *original_max_position_embeddings,
                *beta_fast,
                *beta_slow,
                *factor,
                *mscale,
                *mscale_all_dim,
            ),
            None => Self::new_unscaled(cfg, dtype, dev),
        }
    }

    fn forward(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;

        let q_embed = candle_nn::rotary_emb::rope_i(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope_i(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ---------------------------------------------------------------------------
// Attention (MLA)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum QProj {
    Plain(Linear),
    Lora { a: Linear, norm: RmsNorm, b: Linear },
}

impl QProj {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Lora { a, norm, b } => b.forward(&norm.forward(&a.forward(xs)?)?),
            Self::Plain(lin) => lin.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q: QProj,
    kv_a_proj_with_mqa: Linear,
    kv_a_layernorm: RmsNorm,
    kv_b_proj: Linear,
    o_proj: Linear,
    rotary_emb: Arc<RotaryEmbedding>,
    cfg: Config,
    q_head_dim: usize,
    softmax_scale: f64,
    /// MLA latent cache: `(compressed_kv, k_pe)`.
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let q_head_dim = cfg.q_head_dim();
        let q = match cfg.q_lora_rank {
            Some(lora_rank) => {
                let a = linear_b(
                    cfg.hidden_size,
                    lora_rank,
                    cfg.attention_bias,
                    vb.pp("q_a_proj"),
                )?;
                let norm = rms_norm(lora_rank, cfg.rms_norm_eps, vb.pp("q_a_layernorm"))?;
                let b = linear_no_bias(
                    lora_rank,
                    cfg.num_attention_heads * q_head_dim,
                    vb.pp("q_b_proj"),
                )?;
                QProj::Lora { a, norm, b }
            }
            None => QProj::Plain(linear_no_bias(
                cfg.hidden_size,
                cfg.num_attention_heads * q_head_dim,
                vb.pp("q_proj"),
            )?),
        };

        let kv_a_proj_with_mqa = linear_b(
            cfg.hidden_size,
            cfg.kv_lora_rank + cfg.qk_rope_head_dim,
            cfg.attention_bias,
            vb.pp("kv_a_proj_with_mqa"),
        )?;
        let kv_a_layernorm = rms_norm(cfg.kv_lora_rank, cfg.rms_norm_eps, vb.pp("kv_a_layernorm"))?;
        let kv_b_proj = linear_no_bias(
            cfg.kv_lora_rank,
            cfg.num_attention_heads * (q_head_dim - cfg.qk_rope_head_dim + cfg.v_head_dim),
            vb.pp("kv_b_proj"),
        )?;
        let o_proj = linear_b(
            cfg.num_attention_heads * cfg.v_head_dim,
            cfg.hidden_size,
            cfg.attention_bias,
            vb.pp("o_proj"),
        )?;

        Ok(Self {
            q,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
            rotary_emb,
            cfg: cfg.clone(),
            q_head_dim,
            softmax_scale: cfg.softmax_scale() as f64,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (bs, seq_len, _) = xs.dims3()?;

        let q = {
            let q = self.q.forward(xs)?;
            q.reshape((bs, seq_len, self.cfg.num_attention_heads, self.q_head_dim))?
                .transpose(1, 2)?
        };
        let q_split = split_last_dim(&q, &[self.cfg.qk_nope_head_dim, self.cfg.qk_rope_head_dim])?;
        let q_nope = q_split[0].clone();
        let q_pe = q_split[1].clone();

        let compressed_kv = self.kv_a_proj_with_mqa.forward(xs)?;
        let ckv_split = split_last_dim(
            &compressed_kv,
            &[self.cfg.kv_lora_rank, self.cfg.qk_rope_head_dim],
        )?;
        let mut compressed_kv = ckv_split[0].clone();
        let k_pe = {
            let k_pe = ckv_split[1].clone();
            k_pe.reshape((bs, seq_len, 1, self.cfg.qk_rope_head_dim))?
                .transpose(1, 2)?
        };

        // Apply rotary embeddings to the new query/key-position tensors at the
        // current offset. The cached k_pe already carries rope from prior steps.
        let (q_pe, k_pe) = self.rotary_emb.forward(&q_pe, &k_pe, seqlen_offset)?;

        // Concatenate with the latent cache if it exists.
        let (compressed_kv, k_pe) = if let Some((prev_ckv, prev_k_pe)) = self.kv_cache.take() {
            compressed_kv = Tensor::cat(&[&prev_ckv, &compressed_kv], 1)?;
            let k_pe = Tensor::cat(&[&prev_k_pe, &k_pe], 2)?;
            (compressed_kv, k_pe)
        } else {
            (compressed_kv, k_pe)
        };
        self.kv_cache = Some((compressed_kv.clone(), k_pe.clone()));
        let total_seq_len = compressed_kv.dim(1)?;

        let kv = {
            let kv = self
                .kv_b_proj
                .forward(&self.kv_a_layernorm.forward(&compressed_kv)?)?;
            kv.reshape((
                bs,
                total_seq_len,
                self.cfg.num_attention_heads,
                self.cfg.qk_nope_head_dim + self.cfg.v_head_dim,
            ))?
            .transpose(1, 2)?
        };
        let kv_split = split_last_dim(&kv, &[self.cfg.qk_nope_head_dim, self.cfg.v_head_dim])?;
        let k_nope = kv_split[0].clone();
        let v = kv_split[1].clone();

        let k_pe = self.kv_cache.as_ref().unwrap().1.clone();

        let q = Tensor::cat(&[q_nope, q_pe], D::Minus1)?;
        let k = Tensor::cat(
            &[
                k_nope,
                k_pe.repeat((1, self.cfg.num_attention_heads, 1, 1))?,
            ],
            D::Minus1,
        )?;

        let attn_out = {
            let att = (q.contiguous()?.matmul(&k.t()?.contiguous()?)? * self.softmax_scale)?;
            let att = match attention_mask {
                Some(mask) => att.broadcast_add(mask)?,
                None => att,
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.matmul(&v.contiguous()?)?.transpose(1, 2)?.reshape((
                bs,
                seq_len,
                self.cfg.num_attention_heads * self.cfg.v_head_dim,
            ))?
        };

        self.o_proj.forward(&attn_out)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }

    fn get_kv_cache(&self) -> Option<(Tensor, Tensor)> {
        self.kv_cache.as_ref().map(|(k, v)| (k.clone(), v.clone()))
    }

    fn set_kv_cache(&mut self, compressed_kv: Tensor, k_pe: Tensor) {
        self.kv_cache = Some((compressed_kv, k_pe));
    }
}

// ---------------------------------------------------------------------------
// MLP + MoE
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
    act: Activation,
}

impl Mlp {
    fn new(
        cfg: &Config,
        vb: VarBuilder,
        hidden_size: Option<usize>,
        intermediate_size: Option<usize>,
    ) -> Result<Self> {
        let hidden_size = hidden_size.unwrap_or(cfg.hidden_size);
        let intermediate_size = intermediate_size.unwrap_or(cfg.intermediate_size);

        Ok(Self {
            gate: linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
            up: linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down: linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?,
            act: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.gate.forward(xs)?.apply(&self.act)?;
        let rhs = self.up.forward(xs)?;
        self.down.forward(&(&lhs * &rhs)?)
    }
}

#[derive(Debug, Clone)]
struct MoeGate {
    weight: Tensor,
    cfg: Config,
    top_k: usize,
    n_routed_experts: usize,
}

impl MoeGate {
    fn new(cfg: &Config, vb: VarBuilder, n_routed_experts: usize) -> Result<Self> {
        let weight = vb.get((n_routed_experts, cfg.hidden_size), "weight")?;
        Ok(Self {
            weight,
            cfg: cfg.clone(),
            top_k: cfg.num_experts_per_tok.unwrap_or(1),
            n_routed_experts,
        })
    }

    /// (topk_idx, topk_weight)
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let (bs, seq_len, h) = xs.dims3()?;
        let xs = xs.reshape(((), h))?;
        let logits = xs
            .to_dtype(DType::F32)?
            .broadcast_matmul(&self.weight.t()?.to_dtype(DType::F32)?)?;
        let scores = match self.cfg.scoring_func {
            ScoringFunc::Softmax => candle_nn::ops::softmax_last_dim(&logits)?,
        };

        let (mut topk_weight, topk_idx) = match self.cfg.topk_method {
            TopkMethod::Greedy => topk_unsorted(&scores, self.top_k)?,
            TopkMethod::GroupLimitedGreedy => {
                let group_scores = scores
                    .reshape((bs * seq_len, self.cfg.n_group, ()))?
                    .max(D::Minus1)?;
                let group_idx = topk_unsorted(&group_scores, self.cfg.topk_group)?.1;
                let group_mask = group_scores.zeros_like()?.scatter_add(
                    &group_idx,
                    &group_idx.ones_like()?.to_dtype(group_scores.dtype())?,
                    1,
                )?;
                let score_mask = group_mask
                    .unsqueeze(D::Minus1)?
                    .expand((
                        bs * seq_len,
                        self.cfg.n_group,
                        self.n_routed_experts / self.cfg.n_group,
                    ))?
                    .reshape((bs * seq_len, ()))?;
                let tmp_scores = masked_fill(&score_mask, &(1. - &score_mask.ne(0.)?)?, 0.)?;
                topk_unsorted(&tmp_scores, self.top_k)?
            }
        };

        if self.top_k > 1 && self.cfg.norm_topk_prob {
            let denominator = (topk_weight.sum_keepdim(D::Minus1)? + 1e-20)?;
            topk_weight = (topk_weight / denominator)?;
        } else {
            topk_weight = (topk_weight * self.cfg.routed_scaling_factor)?;
        }
        Ok((topk_idx, topk_weight))
    }
}

#[derive(Debug, Clone)]
struct Moe {
    experts: Vec<Mlp>,
    shared_experts: Option<Mlp>,
    gate: MoeGate,
}

impl Moe {
    fn new(
        cfg: &Config,
        vb: VarBuilder,
        n_shared_experts: Option<usize>,
        n_routed_experts: usize,
    ) -> Result<Self> {
        let mut experts = Vec::with_capacity(n_routed_experts);
        for i in 0..n_routed_experts {
            let vb_e = vb.pp("experts").pp(i);
            experts.push(Mlp::new(cfg, vb_e, None, Some(cfg.moe_intermediate_size))?);
        }
        let shared_experts = if let Some(n_shared_experts) = n_shared_experts {
            let intermediate_size = cfg.moe_intermediate_size * n_shared_experts;
            Some(Mlp::new(
                cfg,
                vb.pp("shared_experts"),
                None,
                Some(intermediate_size),
            )?)
        } else {
            None
        };
        let gate = MoeGate::new(cfg, vb.pp("gate"), n_routed_experts)?;
        Ok(Self {
            experts,
            shared_experts,
            gate,
        })
    }

    fn moe_infer(&self, xs: &Tensor, topk_ids: &Tensor, topk_weight: &Tensor) -> Result<Tensor> {
        let mut y = xs.zeros_like()?;
        let counts = bincount(
            &topk_ids.flatten_all()?.to_vec1::<u32>()?,
            self.experts.len() as u32,
        );
        for (i, expert) in self.experts.iter().enumerate() {
            if counts[i] == 0 {
                continue;
            }
            let idx_top = nonzero(&topk_ids.eq(i as f64)?)?.t()?;
            let idx = &idx_top.i(0)?.contiguous()?;
            let top = &idx_top.i(1)?.contiguous()?;

            y = y.index_add(
                idx,
                &expert.forward(&xs.index_select(idx, 0)?)?.broadcast_mul(
                    &topk_weight
                        .index_select(idx, 0)?
                        .gather(&top.unsqueeze(1)?, 1)?
                        .squeeze(1)?
                        .unsqueeze(D::Minus1)?
                        .to_dtype(xs.dtype())?,
                )?,
                0,
            )?;
        }
        Ok(y)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let identity = xs.clone();
        let orig_shape = xs.shape();
        let (topk_idx, topk_weight) = self.gate.forward(xs)?;
        let xs = xs.reshape(((), xs.dim(D::Minus1)?))?;

        let mut y = self
            .moe_infer(&xs, &topk_idx, &topk_weight)?
            .reshape(orig_shape)?;
        if let Some(ref shared_experts) = self.shared_experts {
            y = (y + shared_experts.forward(&identity)?)?;
        }
        Ok(y)
    }
}

#[derive(Debug, Clone)]
enum MoeOrMlp {
    Moe(Box<Moe>),
    Mlp(Box<Mlp>),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(mlp) => mlp.forward(xs),
            Self::Moe(moe) => moe.forward(xs),
        }
    }
}

// ---------------------------------------------------------------------------
// Decoder layer + Model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct DecoderLayer {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    attn: Attention,
    moe_or_mlp: MoeOrMlp,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        layer_idx: usize,
    ) -> Result<Self> {
        let attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let moe_or_mlp = if let Some(n_routed_experts) = cfg.n_routed_experts {
            if layer_idx >= cfg.first_k_dense_replace
                && layer_idx.is_multiple_of(cfg.moe_layer_freq)
            {
                MoeOrMlp::Moe(
                    Moe::new(cfg, vb.pp("mlp"), cfg.n_shared_experts, n_routed_experts)?.into(),
                )
            } else {
                MoeOrMlp::Mlp(Mlp::new(cfg, vb.pp("mlp"), None, None)?.into())
            }
        } else {
            MoeOrMlp::Mlp(Mlp::new(cfg, vb.pp("mlp"), None, None)?.into())
        };

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            attn,
            moe_or_mlp,
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
        let xs = self.attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .moe_or_mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: Embedding,
    norm: RmsNorm,
    layers: Vec<DecoderLayer>,
    dtype: DType,
    device: Device,
    config: Config,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let rope_cfg = RopeConfig {
            rope_scaling: cfg.rope_scaling.clone(),
            max_position_embeddings: cfg.max_position_embeddings,
            rope_theta: cfg.rope_theta,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
        };
        let rotary_emb = Arc::new(RotaryEmbedding::new(&rope_cfg, vb.dtype(), vb.device())?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx), layer_idx)?;
            layers.push(layer);
        }

        Ok(Self {
            embed_tokens,
            norm,
            layers,
            dtype: vb.dtype(),
            device: vb.device().clone(),
            config: cfg.clone(),
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
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
        let mut xs = self.embed_tokens.forward(input)?;

        let causal = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset)?)
        };

        for layer in &mut self.layers {
            xs = layer.forward(&xs, causal.as_ref(), offset)?;
        }
        self.norm.forward(&xs)
    }

    fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }

    fn get_kv_cache(&self) -> Result<KVCache> {
        self.layers
            .iter()
            .map(|l| {
                l.attn.get_kv_cache().ok_or_else(|| {
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
        for (layer, (compressed_kv, k_pe)) in self.layers.iter_mut().zip(cache) {
            layer.attn.set_kv_cache(compressed_kv, k_pe);
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
        head_dim: cfg.q_head_dim(),
        num_key_value_heads: cfg.num_attention_heads,
    }
}

impl CausalLM for ModelForCausalLM {
    fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        self.forward(input, offset)
    }

    fn clear_kv_cache(&mut self) {
        self.clear_kv_cache();
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
