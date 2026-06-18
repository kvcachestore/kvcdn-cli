//! Vendored IBM Granite MoE adapter with explicit KV-cache access.
//!
//! Adapted from the HuggingFace `transformers` GraniteMoE implementation.
//! Granite MoE is Llama-like with Granite-specific scalar multipliers and a
//! sparse Mixture-of-Experts feed-forward block.
#![allow(dead_code)]

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Activation, Linear, RmsNorm, VarBuilder, embedding, linear_no_bias};
use std::f32::consts::PI;
use std::sync::Arc;

use crate::models::{CausalLM, KVCache, ModelConfig};

const DEFAULT_MAX_SEQ_LEN: usize = 4096;

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub(crate) enum GraniteMoeRopeType {
    #[serde(rename = "granite")]
    Granite,
    #[default]
    #[serde(rename = "default")]
    Default,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub(crate) struct GraniteMoeRopeConfig {
    pub(crate) factor: f32,
    pub(crate) low_freq_factor: f32,
    pub(crate) high_freq_factor: f32,
    pub(crate) original_max_position_embeddings: usize,
    pub(crate) rope_type: GraniteMoeRopeType,
}

fn default_rope() -> f32 {
    10_000.0
}

fn default_max_position_embeddings() -> usize {
    DEFAULT_MAX_SEQ_LEN
}

fn default_one_f64() -> f64 {
    1.0
}

fn default_silu() -> Activation {
    Activation::Silu
}

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct Config {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: Option<usize>,
    pub(crate) rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    pub(crate) rope_theta: f32,
    pub(crate) rope_scaling: Option<GraniteMoeRopeConfig>,
    #[serde(default = "default_max_position_embeddings")]
    pub(crate) max_position_embeddings: usize,
    #[serde(default)]
    pub(crate) tie_word_embeddings: bool,
    #[serde(default = "default_one_f64")]
    pub(crate) attention_multiplier: f64,
    #[serde(default = "default_one_f64")]
    pub(crate) residual_multiplier: f64,
    #[serde(default = "default_one_f64")]
    pub(crate) embedding_multiplier: f64,
    #[serde(default = "default_one_f64")]
    pub(crate) logits_scaling: f64,
    #[serde(default = "default_silu")]
    pub(crate) hidden_act: Activation,
    pub(crate) num_local_experts: usize,
    pub(crate) num_experts_per_tok: usize,
}

impl Config {
    pub(crate) fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    pub(crate) fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub(crate) fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub(crate) fn into_runtime_config(self) -> RuntimeConfig {
        RuntimeConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads(),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            rope_scaling: self.rope_scaling,
            max_position_embeddings: self.max_position_embeddings,
            tie_word_embeddings: self.tie_word_embeddings,
            attention_multiplier: self.attention_multiplier,
            residual_multiplier: self.residual_multiplier,
            embedding_multiplier: self.embedding_multiplier,
            logits_scaling: self.logits_scaling,
            hidden_act: self.hidden_act,
            num_local_experts: self.num_local_experts,
            num_experts_per_tok: self.num_experts_per_tok,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RuntimeConfig {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) rope_theta: f32,
    pub(crate) rope_scaling: Option<GraniteMoeRopeConfig>,
    pub(crate) max_position_embeddings: usize,
    pub(crate) tie_word_embeddings: bool,
    pub(crate) attention_multiplier: f64,
    pub(crate) residual_multiplier: f64,
    pub(crate) embedding_multiplier: f64,
    pub(crate) logits_scaling: f64,
    pub(crate) hidden_act: Activation,
    pub(crate) num_local_experts: usize,
    pub(crate) num_experts_per_tok: usize,
}

impl RuntimeConfig {
    pub(crate) fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

fn calculate_inv_freq(cfg: &RuntimeConfig) -> Vec<f32> {
    let head_dim = cfg.head_dim();
    let base: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect();

    match &cfg.rope_scaling {
        None
        | Some(GraniteMoeRopeConfig {
            rope_type: GraniteMoeRopeType::Default,
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

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &RuntimeConfig, device: &Device) -> Result<Self> {
        let theta = Tensor::new(calculate_inv_freq(cfg), device)?;
        let idx_theta = Tensor::arange(0, cfg.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        Ok(Self {
            cos: idx_theta.cos()?.to_dtype(dtype)?,
            sin: idx_theta.sin()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_b, _h, seq_len, _d) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
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
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    attention_multiplier: f64,
    rotary_emb: Arc<RotaryEmbedding>,
    k_cache: Option<Tensor>,
    v_cache: Option<Tensor>,
}

impl Attention {
    fn new(cfg: &RuntimeConfig, rotary_emb: Arc<RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;

        Ok(Self {
            q_proj: linear_no_bias(cfg.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: cfg.hidden_size,
            attention_multiplier: cfg.attention_multiplier,
            rotary_emb,
            k_cache: None,
            v_cache: None,
        })
    }

    fn forward(&mut self, x: &Tensor, attn_mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        self.k_cache = Some(match self.k_cache.take() {
            None => k,
            Some(prev) => Tensor::cat(&[&prev, &k], 2)?,
        });
        self.v_cache = Some(match self.v_cache.take() {
            None => v,
            Some(prev) => Tensor::cat(&[&prev, &v], 2)?,
        });
        let k = self.k_cache.as_ref().unwrap();
        let v = self.v_cache.as_ref().unwrap();

        let k = repeat_kv(k.clone(), self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v.clone(), self.num_kv_groups)?.contiguous()?;

        let in_dtype = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * self.attention_multiplier)?;
        if let Some(m) = attn_mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?.to_dtype(in_dtype)?;

        ctx.transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
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

/// Parallel expert linear layer: a single weight tensor of shape
/// `[num_experts, output_size, input_size]` that is sliced per expert at
/// runtime. This matches the HF GraniteMoE checkpoint layout.
#[derive(Debug, Clone)]
struct ParallelExperts {
    weight: Tensor,
    num_experts: usize,
}

impl ParallelExperts {
    fn new(
        num_experts: usize,
        input_size: usize,
        output_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let weight = vb.get((num_experts, output_size, input_size), "weight")?;
        Ok(Self {
            weight,
            num_experts,
        })
    }

    /// `inputs` must already be sorted/grouped by expert: the first
    /// `expert_size[0]` rows belong to expert 0, the next `expert_size[1]` to
    /// expert 1, etc.
    fn forward(&self, inputs: &Tensor, expert_size: &[usize]) -> Result<Tensor> {
        let mut outputs = Vec::with_capacity(self.num_experts);
        let mut start = 0;
        for (i, &n) in expert_size.iter().enumerate() {
            if n > 0 {
                let slice = inputs.narrow(0, start, n)?;
                let w = self.weight.narrow(0, i, 1)?.squeeze(0)?;
                outputs.push(slice.matmul(&w.t()?)?);
            }
            start += n;
        }
        Tensor::cat(&outputs, 0)
    }
}

/// Top-k routing for the MoE block. Returns the sorted expert assignments and
/// gating weights needed to gather inputs, run experts, and scatter outputs.
#[derive(Debug, Clone)]
struct TopKGating {
    gate: Linear,
    num_experts: usize,
    top_k: usize,
}

impl TopKGating {
    fn new(input_size: usize, num_experts: usize, top_k: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate: linear_no_bias(input_size, num_experts, vb.pp("layer"))?,
            num_experts,
            top_k,
        })
    }

    /// Returns `(batch_index, batch_gates, expert_size)`.
    /// - `batch_index` maps each `(token, expert)` pair back to its token row.
    /// - `batch_gates` is the corresponding routed gate value.
    /// - `expert_size[e]` is the number of token slots assigned to expert `e`.
    fn forward(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor, Vec<usize>)> {
        let device = hidden_states.device();
        let num_tokens = hidden_states.dims()[0];

        let router_logits = self.gate.forward(hidden_states)?;
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        // Top-k expert selection per token.
        let top_k_indices = routing_weights
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.top_k)?
            .contiguous()?;
        let top_k_gates = routing_weights.gather(&top_k_indices, D::Minus1)?;

        // Move to CPU for the grouping logic; the tensors are small metadata.
        let top_k_indices = top_k_indices.to_vec2::<u32>()?;
        let top_k_gates = top_k_gates.to_dtype(DType::F32)?.to_vec2::<f32>()?;

        // Count assignments per expert and build the flattened (token, expert) list.
        let mut expert_size = vec![0usize; self.num_experts];
        let mut entries: Vec<(u32, u32, f32)> = Vec::with_capacity(num_tokens * self.top_k);
        for (token_idx, (experts, gates)) in
            top_k_indices.iter().zip(top_k_gates.iter()).enumerate()
        {
            for (&expert_idx, &gate) in experts.iter().zip(gates.iter()) {
                expert_size[expert_idx as usize] += 1;
                entries.push((expert_idx, token_idx as u32, gate));
            }
        }

        // Sort by expert index so all tokens for expert 0 come first, etc.
        entries.sort_by_key(|(expert, _, _)| *expert);

        let batch_index: Vec<u32> = entries.iter().map(|(_, token, _)| *token).collect();
        let batch_gates: Vec<f32> = entries.iter().map(|(_, _, gate)| *gate).collect();

        let batch_index = Tensor::new(batch_index, device)?;
        let batch_gates = Tensor::new(batch_gates, device)?
            .reshape(((), 1))?
            .to_dtype(hidden_states.dtype())?;

        Ok((batch_index, batch_gates, expert_size))
    }
}

/// Sparse Mixture-of-Experts feed-forward block used by GraniteMoE.
#[derive(Debug, Clone)]
struct MoeMLP {
    input_linear: ParallelExperts,
    output_linear: ParallelExperts,
    router: TopKGating,
    activation: Activation,
}

impl MoeMLP {
    fn new(cfg: &RuntimeConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            input_linear: ParallelExperts::new(
                cfg.num_local_experts,
                cfg.hidden_size,
                cfg.intermediate_size * 2,
                vb.pp("input_linear"),
            )?,
            output_linear: ParallelExperts::new(
                cfg.num_local_experts,
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("output_linear"),
            )?,
            router: TopKGating::new(
                cfg.hidden_size,
                cfg.num_local_experts,
                cfg.num_experts_per_tok,
                vb.pp("router"),
            )?,
            activation: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;

        let (batch_index, batch_gates, expert_size) = self.router.forward(&xs)?;

        // Gather the token rows selected for each expert, grouped by expert.
        let expert_inputs = xs.index_select(&batch_index, 0)?;

        // Run the input projection, split into gate/up streams, and activate.
        let hidden_states = self.input_linear.forward(&expert_inputs, &expert_size)?;
        let chunks = hidden_states.chunk(2, D::Minus1)?;
        let (left, right) = (&chunks[0], &chunks[1]);
        let hidden_states = (self.activation.forward(left)? * right)?;

        // Run the output projection and weight by the routing gates.
        let expert_outputs = self.output_linear.forward(&hidden_states, &expert_size)?;
        let expert_outputs = expert_outputs.broadcast_mul(&batch_gates)?;

        // Scatter-add the expert outputs back to the original token positions.
        let zeros = xs.zeros_like()?;
        let layer_output = zeros.index_add(&batch_index, &expert_outputs, 0)?;
        layer_output.reshape((b_size, seq_len, hidden_dim))
    }
}

impl Module for MoeMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MoeMLP,
    ln1: RmsNorm,
    ln2: RmsNorm,
    residual_multiplier: f64,
}

impl DecoderLayer {
    fn new(cfg: &RuntimeConfig, rotary_emb: Arc<RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(cfg, rotary_emb, vb.pp("self_attn"))?,
            mlp: MoeMLP::new(cfg, vb.pp("block_sparse_moe"))?,
            ln1: candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            ln2: candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            residual_multiplier: cfg.residual_multiplier,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let residual = x;
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, mask, offset)?;
        let x = (residual + h * self.residual_multiplier)?;

        let residual = &x;
        let h2 = self.ln2.forward(&x)?;
        let h2 = h2.apply(&self.mlp)?;
        residual + h2 * self.residual_multiplier
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
    config: RuntimeConfig,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let cfg = cfg.clone().into_runtime_config();
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let rotary = Arc::new(RotaryEmbedding::new(vb.dtype(), &cfg, vb.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(&cfg, rotary.clone(), vb_l.pp(i))?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm: candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            config: cfg,
        })
    }

    pub fn config(&self) -> &RuntimeConfig {
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
        let mut h = (self.embed_tokens.forward(input)? * self.config.embedding_multiplier)?;

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
        let cfg = base.config();
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(base.embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        Ok(Self { base, lm_head })
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, l) = input.dims2()?;
        let logits = self
            .base
            .forward(input, offset)?
            .narrow(1, l - 1, 1)?
            .apply(&self.lm_head)?;
        (logits / self.base.config.logits_scaling)?.to_dtype(DType::F32)
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
