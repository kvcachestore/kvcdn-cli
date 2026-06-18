//! Vendored RWKV v6 adapter with explicit per-layer state access.
//!
//! Adapted from `candle-transformers` 0.10.2. RWKV is an RNN-style model; its
//! per-layer state is exposed through the `CausalLM` trait as `(state, dummy)`
//! pairs so the layer count matches `KVCache = Vec<(Tensor, Tensor)>`.

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{
    Embedding, GroupNorm, LayerNorm, Linear, VarBuilder, embedding, group_norm, layer_norm,
    linear_no_bias,
};

use crate::models::{CausalLM, KVCache, ModelConfig};

fn default_num_attention_heads() -> usize {
    64
}

// https://huggingface.co/RWKV/HF_v5-Eagle-7B/blob/main/configuration_rwkv5.py
// RWKV v6 reuses the same config shape as v5 in candle-transformers.
#[derive(Debug, Clone, serde::Deserialize)]
#[allow(dead_code)]
pub(crate) struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub attention_hidden_size: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    pub head_size: usize,
    pub intermediate_size: Option<usize>,
    pub layer_norm_epsilon: f64,
    pub rescale_every: usize,
}

impl Config {
    pub(crate) fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    fn n_attn_heads(&self) -> usize {
        self.hidden_size / self.head_size
    }

    fn intermediate_size(&self) -> usize {
        self.intermediate_size
            .unwrap_or(((self.hidden_size as f64 * 3.5) as usize) / 32 * 32)
    }
}

#[derive(Debug, Clone)]
struct StatePerLayer {
    extract_key_value: Tensor,
    linear_attention: Tensor,
    feed_forward: Tensor,
}

impl StatePerLayer {
    fn zeros(batch_size: usize, cfg: &Config, dev: &Device) -> Result<Self> {
        let n_heads = cfg.n_attn_heads();
        let head_size = cfg.head_size;
        Ok(Self {
            extract_key_value: Tensor::zeros((batch_size, cfg.hidden_size), DType::F32, dev)?,
            linear_attention: Tensor::zeros(
                (batch_size, n_heads, head_size, head_size),
                DType::F32,
                dev,
            )?,
            feed_forward: Tensor::zeros((batch_size, cfg.hidden_size), DType::F32, dev)?,
        })
    }

    /// Pack the three per-layer state tensors into a single 2-D tensor of shape
    /// `(batch, hidden_size + hidden_size*head_size + hidden_size)`.
    fn pack(&self) -> Result<Tensor> {
        let ekv = self
            .extract_key_value
            .flatten(1, self.extract_key_value.rank() - 1)?
            .contiguous()?;
        let la = self
            .linear_attention
            .flatten(1, self.linear_attention.rank() - 1)?
            .contiguous()?;
        let ff = self
            .feed_forward
            .flatten(1, self.feed_forward.rank() - 1)?
            .contiguous()?;
        Tensor::cat(&[ekv, la, ff], 1)
    }

    /// Reverse `pack`. The shape is reconstructed from `cfg`.
    fn unpack(state: Tensor, cfg: &Config) -> Result<Self> {
        let batch_size = state.dim(0)?;
        let n_heads = cfg.n_attn_heads();
        let head_size = cfg.head_size;
        let hidden_size = cfg.hidden_size;
        let ekv_size = hidden_size;
        let la_size = n_heads * head_size * head_size;
        let ff_size = hidden_size;
        let total = ekv_size + la_size + ff_size;
        if state.dim(1)? != total {
            candle_core::bail!(
                "packed RWKV state size mismatch: expected {}, got {}",
                total,
                state.dim(1)?
            );
        }
        let ekv = state
            .narrow(1, 0, ekv_size)?
            .reshape((batch_size, hidden_size))?;
        let la = state
            .narrow(1, ekv_size, la_size)?
            .reshape((batch_size, n_heads, head_size, head_size))?;
        let ff = state
            .narrow(1, ekv_size + la_size, ff_size)?
            .reshape((batch_size, hidden_size))?;
        Ok(Self {
            extract_key_value: ekv,
            linear_attention: la,
            feed_forward: ff,
        })
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SelfAttention {
    key: Linear,
    receptance: Linear,
    value: Linear,
    gate: Linear,
    output: Linear,
    ln_x: GroupNorm,
    time_mix_x: Tensor,
    time_mix_w: Tensor,
    time_mix_key: Tensor,
    time_mix_value: Tensor,
    time_mix_receptance: Tensor,
    time_decay: Tensor,
    time_faaaa: Tensor,
    time_mix_gate: Tensor,
    time_decay_w1: Tensor,
    time_decay_w2: Tensor,
    time_mix_w1: Tensor,
    time_mix_w2: Tensor,
    layer_id: usize,
    n_attn_heads: usize,
}

impl SelfAttention {
    fn new(layer_id: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let attn_hidden_size = cfg.attention_hidden_size;
        let key = linear_no_bias(hidden_size, attn_hidden_size, vb.pp("key"))?;
        let receptance = linear_no_bias(hidden_size, attn_hidden_size, vb.pp("receptance"))?;
        let value = linear_no_bias(hidden_size, attn_hidden_size, vb.pp("value"))?;
        let gate = linear_no_bias(hidden_size, attn_hidden_size, vb.pp("gate"))?;
        let output = linear_no_bias(attn_hidden_size, hidden_size, vb.pp("output"))?;
        let ln_x = group_norm(
            hidden_size / cfg.head_size,
            hidden_size,
            1e-5,
            vb.pp("ln_x"),
        )?;

        let time_mix_x = vb.get((1, 1, cfg.hidden_size), "time_mix_x")?;
        let time_mix_w = vb.get((1, 1, cfg.hidden_size), "time_mix_w")?;
        let time_mix_key = vb.get((1, 1, cfg.hidden_size), "time_mix_key")?;
        let time_mix_value = vb.get((1, 1, cfg.hidden_size), "time_mix_value")?;
        let time_mix_receptance = vb.get((1, 1, cfg.hidden_size), "time_mix_receptance")?;
        let n_attn_heads = cfg.n_attn_heads();
        let time_decay = vb.get((1, 1, cfg.hidden_size), "time_decay")?;
        let time_faaaa = vb.get((n_attn_heads, cfg.head_size), "time_faaaa")?;
        let time_mix_gate = vb.get((1, 1, cfg.hidden_size), "time_mix_gate")?;
        let time_decay_w1 = vb.get((cfg.hidden_size, n_attn_heads * 2), "time_decay_w1")?;
        let time_decay_w2 = vb.get((n_attn_heads * 2, cfg.hidden_size), "time_decay_w2")?;
        let time_mix_w1 = vb.get((cfg.hidden_size, n_attn_heads * 5), "time_mix_w1")?;
        let time_mix_w2 = vb.get((5, n_attn_heads, cfg.hidden_size), "time_mix_w2")?;
        Ok(Self {
            key,
            value,
            receptance,
            gate,
            output,
            ln_x,
            time_mix_x,
            time_mix_w,
            time_mix_key,
            time_mix_value,
            time_mix_receptance,
            time_decay,
            time_faaaa,
            time_mix_gate,
            time_decay_w1,
            time_decay_w2,
            time_mix_w1,
            time_mix_w2,
            layer_id,
            n_attn_heads,
        })
    }

    fn forward(&self, xs: &Tensor, state: &mut StatePerLayer) -> Result<Tensor> {
        let h = self.n_attn_heads;
        let (b, t, s) = xs.dims3()?;
        let s = s / h;
        let (receptance, key, value, gate, w) = {
            let shifted = state.extract_key_value.clone();
            let shifted = if shifted.rank() == 2 {
                shifted.unsqueeze(1)?
            } else {
                shifted
            };

            let sx = (&shifted - xs)?;
            let xxx = (xs + &sx * &self.time_mix_x)?;
            let xxx = xxx
                .broadcast_matmul(&self.time_mix_w1)?
                .tanh()?
                .reshape((b * t, 5, ()))?
                .transpose(0, 1)?;

            let xxx = xxx.matmul(&self.time_mix_w2)?.reshape((5, b, t, ()))?;

            let (mw, mk, mv, mr, mg) = (xxx.i(0)?, xxx.i(1)?, xxx.i(2)?, xxx.i(3)?, xxx.i(4)?);

            let xw = (xs + &sx * (&self.time_mix_w + &mw)?)?;
            let xk = (xs + &sx * (&self.time_mix_key + &mk)?)?;
            let xv = (xs + &sx * (&self.time_mix_value + &mv)?)?;
            let xr = (xs + &sx * (&self.time_mix_receptance + &mr)?)?;
            let xg = (xs + &sx * (&self.time_mix_gate + &mg)?)?;

            // Keep the time dimension so we can slice the per-token decay in
            // the loop below. The upstream reshape only broadcasts for t == 1.
            let w = (&self.time_decay
                + xw.broadcast_matmul(&self.time_decay_w1)?
                    .tanh()?
                    .broadcast_matmul(&self.time_decay_w2)?)?
            .reshape((b, t, h, s))?
            .permute((0, 2, 3, 1))?;

            let key = self.key.forward(&xk)?;
            let value = self.value.forward(&xv)?;
            let receptance = self.receptance.forward(&xr)?;
            let gate = candle_nn::ops::silu(&self.gate.forward(&xg)?)?;
            state.extract_key_value = xs.i((.., t - 1))?;
            (receptance, key, value, gate, w)
        };

        // linear attention
        let mut state_ = state.linear_attention.clone();
        let key = key.reshape((b, t, h, s))?.permute((0, 2, 3, 1))?;
        let value = value.reshape((b, t, h, s))?.transpose(1, 2)?;
        let receptance = receptance.reshape((b, t, h, s))?.transpose(1, 2)?;

        let w = w.exp()?.neg()?.exp()?;

        let time_faaaa =
            self.time_faaaa
                .reshape(((), 1, 1))?
                .reshape((self.n_attn_heads, (), 1))?;

        let mut out: Vec<Tensor> = Vec::with_capacity(t);
        for t_ in 0..t {
            let rt = receptance.i((.., .., t_..t_ + 1))?.contiguous()?;
            let kt = key.i((.., .., .., t_..t_ + 1))?.contiguous()?;
            let vt = value.i((.., .., t_..t_ + 1))?.contiguous()?;
            let at = kt.matmul(&vt)?;
            let rhs = (time_faaaa.broadcast_mul(&at)? + &state_)?;
            let out_ = rt.matmul(&rhs)?.squeeze(2)?;
            let wt = w.i((.., .., .., t_..t_ + 1))?.contiguous()?;
            state_ = (&at + wt.broadcast_mul(&state_))?;
            out.push(out_)
        }
        let out = Tensor::cat(&out, 1)?.reshape((b * t, h * s, 1))?;
        let out = out.apply(&self.ln_x)?.reshape((b, t, h * s))?;
        let out = (out * gate)?.apply(&self.output)?;
        state.linear_attention = state_;
        Ok(out)
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct FeedForward {
    time_mix_key: Tensor,
    time_mix_receptance: Tensor,
    key: Linear,
    receptance: Linear,
    value: Linear,
    layer_id: usize,
}

impl FeedForward {
    fn new(layer_id: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let int_size = cfg.intermediate_size();
        let key = linear_no_bias(cfg.hidden_size, int_size, vb.pp("key"))?;
        let receptance = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("receptance"))?;
        let value = linear_no_bias(int_size, cfg.hidden_size, vb.pp("value"))?;
        let time_mix_key = vb.get((1, 1, cfg.hidden_size), "time_mix_key")?;
        let time_mix_receptance = vb.get((1, 1, cfg.hidden_size), "time_mix_receptance")?;
        Ok(Self {
            key,
            receptance,
            value,
            time_mix_key,
            time_mix_receptance,
            layer_id,
        })
    }

    fn forward(&self, xs: &Tensor, state: &mut StatePerLayer) -> Result<Tensor> {
        let shifted = state.feed_forward.broadcast_sub(xs)?;
        let key = (xs + shifted.broadcast_mul(&self.time_mix_key)?)?;
        let receptance = (xs + shifted.broadcast_mul(&self.time_mix_receptance)?)?;
        let key = key.apply(&self.key)?.relu()?.sqr()?;
        let value = key.apply(&self.value)?;
        let receptance = candle_nn::ops::sigmoid(&receptance.apply(&self.receptance)?)?;
        state.feed_forward = xs.i((.., xs.dim(1)? - 1))?;
        let xs = (receptance * value)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct Block {
    pre_ln: Option<LayerNorm>,
    ln1: LayerNorm,
    ln2: LayerNorm,
    attention: SelfAttention,
    feed_forward: FeedForward,
}

impl Block {
    fn new(layer_id: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let ln1 = layer_norm(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("ln1"))?;
        let ln2 = layer_norm(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("ln2"))?;
        let pre_ln = if layer_id == 0 {
            let ln = layer_norm(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("pre_ln"))?;
            Some(ln)
        } else {
            None
        };
        let attention = SelfAttention::new(layer_id, cfg, vb.pp("attention"))?;
        let feed_forward = FeedForward::new(layer_id, cfg, vb.pp("feed_forward"))?;
        Ok(Self {
            pre_ln,
            ln1,
            ln2,
            attention,
            feed_forward,
        })
    }

    fn forward(&self, xs: &Tensor, state: &mut StatePerLayer) -> Result<Tensor> {
        let xs = match self.pre_ln.as_ref() {
            None => xs.clone(),
            Some(pre_ln) => xs.apply(pre_ln)?,
        };
        let attention = self.attention.forward(&xs.apply(&self.ln1)?, state)?;
        let xs = (xs + attention)?;
        let feed_forward = self.feed_forward.forward(&xs.apply(&self.ln2)?, state)?;
        let xs = (xs + feed_forward)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embeddings: Embedding,
    blocks: Vec<Block>,
    ln_out: LayerNorm,
    head: Linear,
    rescale_every: usize,
    layers_are_rescaled: bool,
    states: Vec<StatePerLayer>,
    config: Config,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("rwkv");
        let embeddings = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embeddings"))?;
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_b = vb_m.pp("blocks");
        for block_index in 0..cfg.num_hidden_layers {
            let block = Block::new(block_index, cfg, vb_b.pp(block_index))?;
            blocks.push(block)
        }
        let ln_out = layer_norm(cfg.hidden_size, 1e-5, vb_m.pp("ln_out"))?;
        let head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("head"))?;
        let states = (0..cfg.num_hidden_layers)
            .map(|_| StatePerLayer::zeros(1, cfg, vb.device()))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embeddings,
            blocks,
            ln_out,
            head,
            rescale_every: cfg.rescale_every,
            layers_are_rescaled: false,
            states,
            config: cfg.clone(),
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    fn clear_kv_cache(&mut self) {
        self.states = (0..self.config.num_hidden_layers)
            .map(|_| StatePerLayer::zeros(1, &self.config, &self.device))
            .collect::<Result<Vec<_>>>()
            .unwrap_or_default();
    }

    pub fn forward(&mut self, input: &Tensor, _offset: usize) -> Result<Tensor> {
        let b = input.dim(0)?;
        if self.states[0].extract_key_value.dim(0)? != b {
            self.states = (0..self.config.num_hidden_layers)
                .map(|_| StatePerLayer::zeros(b, &self.config, &self.device))
                .collect::<Result<Vec<_>>>()?;
        }
        let mut xs = input.apply(&self.embeddings)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            xs = block.forward(&xs, &mut self.states[block_idx])?;
            if self.layers_are_rescaled && (block_idx + 1) % self.rescale_every == 0 {
                xs = (xs / 2.)?
            }
        }
        xs.apply(&self.ln_out)
    }

    pub fn get_kv_cache(&self) -> Result<KVCache> {
        let dummy = Tensor::zeros((0,), self.dtype, &self.device)?;
        self.states
            .iter()
            .map(|s| {
                let packed = s.pack()?;
                Ok((packed, dummy.clone()))
            })
            .collect::<Result<Vec<_>>>()
    }

    pub fn set_kv_cache(&mut self, cache: KVCache) -> Result<()> {
        if cache.len() != self.states.len() {
            candle_core::bail!(
                "RWKV state layer count mismatch: expected {}, got {}",
                self.states.len(),
                cache.len()
            );
        }
        self.states = cache
            .into_iter()
            .map(|(state, _dummy)| StatePerLayer::unpack(state, &self.config))
            .collect::<Result<Vec<_>>>()?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ModelForCausalLM {
    base: Model,
}

impl ModelForCausalLM {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let base = Model::new(cfg, vb)?;
        Ok(Self { base })
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, l) = input.dims2()?;
        self.base
            .forward(input, offset)?
            .narrow(1, l - 1, 1)?
            .apply(&self.base.head)
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
    let n_heads = cfg.n_attn_heads();
    ModelConfig {
        vocab_size: cfg.vocab_size,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size(),
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: n_heads,
        head_dim: cfg.head_size,
        num_key_value_heads: n_heads,
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
