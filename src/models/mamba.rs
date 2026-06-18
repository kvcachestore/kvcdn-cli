//! Mamba / Mamba1 causal-LM adapter with explicit state capture.
//!
//! Adapted from `candle-transformers` 0.10.2. This is a state-space model, so the
//! per-layer "cache" is a single state tensor rather than a key/value pair. The
//! `CausalLM` trait is satisfied by pairing the packed layer state with an empty
//! dummy tensor.
use candle_core::{D, DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Linear, RmsNorm, VarBuilder, linear, linear_no_bias};

use crate::models::{CausalLM, KVCache, ModelConfig};

fn default_pad() -> usize {
    16
}

fn default_state_size() -> usize {
    16
}

fn default_conv_kernel() -> usize {
    4
}

fn default_expand() -> usize {
    2
}

fn default_layer_norm_epsilon() -> f64 {
    1e-5
}

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct Config {
    #[serde(alias = "hidden_size")]
    pub d_model: usize,
    #[serde(alias = "num_hidden_layers")]
    pub n_layer: usize,
    pub vocab_size: usize,
    #[serde(default = "default_pad")]
    pub pad_vocab_size_multiple: usize,
    #[serde(default = "default_state_size", alias = "d_state")]
    pub state_size: usize,
    #[serde(default = "default_conv_kernel", alias = "d_conv")]
    pub conv_kernel: usize,
    #[serde(default = "default_expand")]
    pub expand: usize,
    #[serde(alias = "intermediate_size")]
    pub d_inner: Option<usize>,
    #[serde(alias = "time_step_rank", alias = "dt_rank")]
    pub time_step_rank: Option<usize>,
    #[serde(default = "default_layer_norm_epsilon")]
    pub layer_norm_epsilon: f64,
}

impl Config {
    pub(crate) fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    fn vocab_size(&self) -> usize {
        let pad = self.pad_vocab_size_multiple;
        self.vocab_size.div_ceil(pad) * pad
    }

    fn d_inner(&self) -> usize {
        self.d_inner.unwrap_or(self.expand * self.d_model)
    }

    fn dt_rank(&self) -> usize {
        self.time_step_rank
            .unwrap_or_else(|| self.d_model.div_ceil(16))
    }

    fn d_state(&self) -> usize {
        self.state_size
    }

    fn d_conv(&self) -> usize {
        self.conv_kernel
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct State {
    hs: Vec<Tensor>,
    prev_xs: Vec<Vec<Tensor>>,
    pos: usize,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
}

impl State {
    fn new(batch_size: usize, cfg: &Config, dtype: DType, device: &Device) -> Result<Self> {
        let n_layer = cfg.n_layer;
        let d_inner = cfg.d_inner();
        let d_state = cfg.d_state();
        let d_conv = cfg.d_conv();
        let mut hs = Vec::with_capacity(n_layer);
        let mut prev_xs = Vec::with_capacity(n_layer);
        let x = Tensor::zeros((batch_size, d_inner), dtype, device)?;
        for _ in 0..n_layer {
            hs.push(Tensor::zeros(
                (batch_size, d_inner, d_state),
                dtype,
                device,
            )?);
            prev_xs.push(vec![x.clone(); d_conv]);
        }
        Ok(Self {
            hs,
            prev_xs,
            pos: 0,
            d_inner,
            d_state,
            d_conv,
        })
    }

    fn pack_layer(&self, layer_idx: usize) -> Result<Tensor> {
        let mut parts = Vec::with_capacity(1 + self.d_conv);
        parts.push(self.hs[layer_idx].flatten_all()?);
        for px in &self.prev_xs[layer_idx] {
            parts.push(px.flatten_all()?);
        }
        Tensor::cat(&parts, 0)
    }
}

#[derive(Clone, Debug)]
struct MambaBlock {
    in_proj: Linear,
    conv1d_bias: Tensor,
    conv1d_weights: Vec<Tensor>,
    x_proj: Linear,
    dt_proj: Linear,
    a_log: Tensor,
    d: Tensor,
    out_proj: Linear,
    dt_rank: usize,
    layer_index: usize,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
}

impl MambaBlock {
    fn new(layer_index: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let d_inner = cfg.d_inner();
        let dt_rank = cfg.dt_rank();
        let d_state = cfg.d_state();
        let d_conv = cfg.d_conv();

        let in_proj = linear_no_bias(cfg.d_model, d_inner * 2, vb.pp("in_proj"))?;
        let x_proj = linear_no_bias(d_inner, dt_rank + d_state * 2, vb.pp("x_proj"))?;
        let dt_proj = linear(dt_rank, d_inner, vb.pp("dt_proj"))?;
        let a_log = vb.get((d_inner, d_state), "A_log")?;
        let d = vb.get(d_inner, "D")?;
        let out_proj = linear_no_bias(d_inner, cfg.d_model, vb.pp("out_proj"))?;
        let conv1d_bias = vb.get(d_inner, "conv1d.bias")?;
        let conv1d_weight = vb.get((d_inner, 1, d_conv), "conv1d.weight")?;
        let mut conv1d_weights = Vec::with_capacity(d_conv);
        for i in 0..d_conv {
            conv1d_weights.push(conv1d_weight.i((.., 0, i))?);
        }

        Ok(Self {
            in_proj,
            conv1d_bias,
            conv1d_weights,
            x_proj,
            dt_proj,
            a_log,
            d,
            out_proj,
            dt_rank,
            layer_index,
            d_inner,
            d_state,
            d_conv,
        })
    }

    fn forward(&self, xs: &Tensor, state: &mut State) -> Result<Tensor> {
        let (b_sz, _dim) = xs.dims2()?;
        let li = self.layer_index;
        let mut xs = xs.apply(&self.in_proj)?.chunk(2, D::Minus1)?;
        let proj_for_silu = xs.remove(1);
        state.prev_xs[li][state.pos % self.d_conv] = xs.remove(0);

        let mut proj_for_conv = self.conv1d_bias.broadcast_as((b_sz, self.d_inner))?;
        for d_c in 0..self.d_conv {
            proj_for_conv = (proj_for_conv
                + self.conv1d_weights[d_c]
                    .broadcast_mul(&state.prev_xs[li][(d_c + 1 + state.pos) % self.d_conv])?)?;
        }
        let proj_for_conv = candle_nn::ops::silu(&proj_for_conv)?;

        let x_proj = self.x_proj.forward(&proj_for_conv)?;
        let delta = x_proj.narrow(D::Minus1, 0, self.dt_rank)?.contiguous()?;
        let b = x_proj.narrow(D::Minus1, self.dt_rank, self.d_state)?;
        let c = x_proj.narrow(D::Minus1, self.dt_rank + self.d_state, self.d_state)?;

        let delta = delta.apply(&self.dt_proj)?;
        let delta = (delta.exp()? + 1.)?.log()?;
        let a = self.a_log.to_dtype(delta.dtype())?.exp()?.neg()?;
        let d = self.d.to_dtype(delta.dtype())?;

        let delta = delta
            .unsqueeze(D::Minus1)?
            .broadcast_as((b_sz, self.d_inner, self.d_state))?;
        let a = a.broadcast_as((b_sz, self.d_inner, self.d_state))?;
        let b = b.broadcast_as((b_sz, self.d_inner, self.d_state))?;
        let proj_for_conv_b =
            proj_for_conv
                .unsqueeze(D::Minus1)?
                .broadcast_as((b_sz, self.d_inner, self.d_state))?;
        state.hs[li] = ((&state.hs[li] * (&delta * &a)?.exp()?)? + &delta * &b * &proj_for_conv_b)?;
        let ss = (state.hs[li]
            .matmul(&c.unsqueeze(D::Minus1)?)?
            .squeeze(D::Minus1)?
            + proj_for_conv.broadcast_mul(&d)?)?;

        let ys = (ss * candle_nn::ops::silu(&proj_for_silu))?;
        ys.apply(&self.out_proj)
    }
}

#[derive(Clone, Debug)]
struct ResidualBlock {
    mixer: MambaBlock,
    norm: RmsNorm,
}

impl ResidualBlock {
    fn new(layer_index: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let norm = candle_nn::rms_norm(cfg.d_model, cfg.layer_norm_epsilon, vb.pp("norm"))?;
        let mixer = MambaBlock::new(layer_index, cfg, vb.pp("mixer"))?;
        Ok(Self { mixer, norm })
    }

    fn forward(&self, xs: &Tensor, state: &mut State) -> Result<Tensor> {
        self.mixer.forward(&xs.apply(&self.norm)?, state)? + xs
    }
}

#[derive(Clone, Debug)]
pub struct Model {
    embedding: candle_nn::Embedding,
    layers: Vec<ResidualBlock>,
    norm_f: RmsNorm,
    lm_head: Linear,
    dtype: DType,
    device: Device,
    config: Config,
    state: Option<State>,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        // Official Mamba HF checkpoints wrap weights under a `backbone` prefix;
        // fall back to the unprefixed layout used by the reference implementation.
        let vb = if vb.contains_tensor("backbone.embeddings.weight") {
            vb.pp("backbone")
        } else {
            vb.clone()
        };

        let embedding = candle_nn::embedding(cfg.vocab_size(), cfg.d_model, vb.pp("embeddings"))?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        let vb_l = vb.pp("layers");
        for layer_idx in 0..cfg.n_layer {
            let layer = ResidualBlock::new(layer_idx, cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm_f = candle_nn::rms_norm(cfg.d_model, cfg.layer_norm_epsilon, vb.pp("norm_f"))?;
        let lm_head = Linear::new(embedding.embeddings().clone(), None);
        Ok(Self {
            embedding,
            layers,
            norm_f,
            lm_head,
            dtype: vb.dtype(),
            device: vb.device().clone(),
            config: cfg.clone(),
            state: None,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l) = input_ids.dims2()?;
        let xs = self.embedding.forward(input_ids)?;

        if self
            .state
            .as_ref()
            .map(|s| s.hs.first().map(|h| h.dims()[0]).unwrap_or(b) != b)
            .unwrap_or(true)
        {
            self.state = Some(State::new(b, &self.config, self.dtype, &self.device)?);
        }
        let state = self.state.as_mut().unwrap();
        state.pos = offset;

        let mut last_x = None;
        for t in 0..l {
            let mut x = xs.narrow(1, t, 1)?.squeeze(1)?;
            for layer in self.layers.iter() {
                x = layer.forward(&x, state)?;
            }
            state.pos += 1;
            last_x = Some(x);
        }

        last_x.unwrap().apply(&self.norm_f)?.apply(&self.lm_head)
    }

    fn clear_kv_cache(&mut self) {
        self.state = None;
    }

    fn get_kv_cache(&self) -> Result<KVCache> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("no state has been generated yet".into()))?;
        let dummy = Tensor::new(&[] as &[f32], &self.device)?.reshape(())?;
        self.layers
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let s = state.pack_layer(i)?;
                Ok((s, dummy.clone()))
            })
            .collect::<Result<Vec<_>>>()
    }

    fn unpack_state_tensor(&self, t: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let t = t.to_device(&self.device)?.to_dtype(self.dtype)?;
        let total = t.elem_count();
        let denom = self.config.d_inner() * self.config.d_state()
            + self.config.d_conv() * self.config.d_inner();
        if total % denom != 0 {
            candle_core::bail!(
                "packed state size {} is not divisible by expected per-token size {}",
                total,
                denom
            );
        }
        let b = total / denom;
        let hs_len = b * self.config.d_inner() * self.config.d_state();
        let px_len = b * self.config.d_inner();
        let flat = t.reshape((total,))?;
        let hs = flat.narrow(0, 0, hs_len)?.reshape((
            b,
            self.config.d_inner(),
            self.config.d_state(),
        ))?;
        let mut prev_xs = Vec::with_capacity(self.config.d_conv());
        for i in 0..self.config.d_conv() {
            prev_xs.push(
                flat.narrow(0, hs_len + i * px_len, px_len)?
                    .reshape((b, self.config.d_inner()))?,
            );
        }
        Ok((hs, prev_xs))
    }

    fn set_kv_cache(&mut self, cache: KVCache) -> Result<()> {
        if cache.len() != self.layers.len() {
            candle_core::bail!(
                "state cache layer count mismatch: expected {}, got {}",
                self.layers.len(),
                cache.len()
            );
        }
        let mut hs = Vec::with_capacity(self.layers.len());
        let mut prev_xs = Vec::with_capacity(self.layers.len());
        for (state_tensor, _dummy) in cache {
            let (h, px) = self.unpack_state_tensor(&state_tensor)?;
            hs.push(h);
            prev_xs.push(px);
        }
        self.state = Some(State {
            hs,
            prev_xs,
            pos: 0,
            d_inner: self.config.d_inner(),
            d_state: self.config.d_state(),
            d_conv: self.config.d_conv(),
        });
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ModelForCausalLM {
    base: Model,
}

impl ModelForCausalLM {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            base: Model::new(cfg, vb)?,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, offset: usize) -> Result<Tensor> {
        self.base.forward(input_ids, offset)
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
        vocab_size: cfg.vocab_size(),
        hidden_size: cfg.d_model,
        intermediate_size: cfg.d_inner(),
        num_hidden_layers: cfg.n_layer,
        num_attention_heads: 1,
        head_dim: cfg.d_model,
        num_key_value_heads: 1,
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
        to_model_config(&self.base.config)
    }
}
