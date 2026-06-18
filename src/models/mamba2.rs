//! Mamba2 causal-LM adapter with explicit state capture.
//!
//! Adapted from `candle-transformers` 0.10.2. Mamba2 is a state-space model, so
//! the per-layer "cache" is a single state tensor rather than a key/value pair.
//! The `CausalLM` trait is satisfied by pairing the packed layer state with an
//! empty dummy tensor.
use candle_core::{D, DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Linear, RmsNorm, VarBuilder, linear_no_bias};

use crate::models::{CausalLM, KVCache, ModelConfig};

const D_CONV: usize = 4;

fn default_d_state() -> usize {
    64
}

fn default_expand() -> usize {
    2
}

fn default_headdim() -> usize {
    64
}

fn default_ngroups() -> usize {
    1
}

fn default_pad_vocab_size_multiple() -> usize {
    16
}

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct Config {
    #[serde(alias = "hidden_size")]
    pub d_model: usize,
    #[serde(alias = "num_hidden_layers")]
    pub n_layer: usize,
    pub vocab_size: usize,
    #[serde(alias = "state_size", default = "default_d_state")]
    pub d_state: usize,
    #[serde(default = "default_expand")]
    pub expand: usize,
    #[serde(alias = "head_dim", default = "default_headdim")]
    pub headdim: usize,
    #[serde(alias = "n_groups", default = "default_ngroups")]
    pub ngroups: usize,
    #[serde(default = "default_pad_vocab_size_multiple")]
    pub pad_vocab_size_multiple: usize,
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
        self.d_model * self.expand
    }

    fn d_xbc(&self) -> usize {
        self.d_inner() + 2 * self.ngroups * self.d_state
    }

    fn nheads(&self) -> usize {
        self.d_inner() / self.headdim
    }
}

#[derive(Clone, Debug)]
struct State {
    hs: Vec<Tensor>,
    conv_states: Vec<Tensor>,
    pos: usize,
}

impl State {
    fn new(batch_size: usize, cfg: &Config, dtype: DType, device: &Device) -> Result<Self> {
        let nheads = cfg.nheads();
        let d_xbc = cfg.d_xbc();
        let mut hs = Vec::with_capacity(cfg.n_layer);
        let mut conv_states = Vec::with_capacity(cfg.n_layer);
        for _ in 0..cfg.n_layer {
            let h = Tensor::zeros(
                (batch_size, nheads, cfg.headdim, cfg.d_state),
                dtype,
                device,
            )?;
            let conv = Tensor::zeros((batch_size, d_xbc, D_CONV), dtype, device)?;
            hs.push(h);
            conv_states.push(conv);
        }
        Ok(Self {
            hs,
            conv_states,
            pos: 0,
        })
    }

    fn pack_layer(&self, layer_idx: usize) -> Result<Tensor> {
        let parts = [
            self.hs[layer_idx].flatten_all()?,
            self.conv_states[layer_idx].flatten_all()?,
        ];
        Tensor::cat(&parts, 0)
    }
}

#[derive(Clone, Debug)]
struct Mamba2Block {
    in_proj: Linear,
    conv1d_weight: Tensor,
    conv1d_bias: Tensor,
    a_log: Tensor,
    d: Tensor,
    dt_bias: Tensor,
    out_proj: Linear,
    norm: RmsNorm,
    d_inner: usize,
    d_state: usize,
    d_xbc: usize,
    headdim: usize,
    nheads: usize,
    ngroups: usize,
    layer_idx: usize,
}

impl Mamba2Block {
    fn new(layer_idx: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let d_inner = cfg.d_inner();
        let nheads = cfg.nheads();
        let ngroups = cfg.ngroups;
        let d_state = cfg.d_state;
        let d_xbc = cfg.d_xbc();

        let proj_size = d_inner + d_xbc + nheads;
        let in_proj = linear_no_bias(cfg.d_model, proj_size, vb.pp("in_proj"))?;

        let conv1d_weight = vb.get((d_xbc, 1, D_CONV), "conv1d.weight")?;
        let conv1d_bias = vb.get(d_xbc, "conv1d.bias")?;

        let a_log = vb.get(nheads, "A_log")?;
        let d = vb.get(nheads, "D")?;
        let dt_bias = vb.get(nheads, "dt_bias")?;

        let out_proj = linear_no_bias(d_inner, cfg.d_model, vb.pp("out_proj"))?;
        let norm = candle_nn::rms_norm(d_inner, 1e-5, vb.pp("norm"))?;

        Ok(Self {
            in_proj,
            conv1d_weight,
            conv1d_bias,
            a_log,
            d,
            dt_bias,
            out_proj,
            norm,
            d_inner,
            d_state,
            d_xbc,
            headdim: cfg.headdim,
            nheads,
            ngroups,
            layer_idx,
        })
    }

    fn forward(&self, xs: &Tensor, state: &mut State) -> Result<Tensor> {
        let (b_sz, _dim) = xs.dims2()?;

        let proj = self.in_proj.forward(xs)?;

        let z = proj.narrow(D::Minus1, 0, self.d_inner)?;
        let xbc = proj.narrow(D::Minus1, self.d_inner, self.d_xbc)?;
        let dt = proj.narrow(D::Minus1, self.d_inner + self.d_xbc, self.nheads)?;

        let xbc_conv = self.apply_conv1d(&xbc, &mut state.conv_states[self.layer_idx])?;
        let xbc_conv = candle_nn::ops::silu(&xbc_conv)?;

        let x_conv = xbc_conv.narrow(D::Minus1, 0, self.d_inner)?;
        let b = xbc_conv.narrow(D::Minus1, self.d_inner, self.ngroups * self.d_state)?;
        let c = xbc_conv.narrow(
            D::Minus1,
            self.d_inner + self.ngroups * self.d_state,
            self.ngroups * self.d_state,
        )?;

        let dt_bias = self.dt_bias.broadcast_as(dt.shape())?;
        let dt = ((&dt + &dt_bias)?.exp()? + 1.)?.log()?; // softplus

        let a = self.a_log.exp()?.neg()?;

        let y = self.ssm_step(&x_conv, &a, &b, &c, &dt, state)?;

        let d = self.d.broadcast_as((b_sz, self.nheads))?;
        let x_skip = x_conv.reshape((b_sz, self.nheads, self.headdim))?;
        let y = (&y + x_skip.broadcast_mul(&d.unsqueeze(D::Minus1)?)?)?;
        let y = y.reshape((b_sz, self.d_inner))?;

        let y = (y * candle_nn::ops::silu(&z)?)?;
        let y = self.norm.forward(&y)?;

        self.out_proj.forward(&y)
    }

    fn apply_conv1d(&self, xbc: &Tensor, conv_state: &mut Tensor) -> Result<Tensor> {
        let (b_sz, d_xbc) = xbc.dims2()?;

        let shifted = conv_state.narrow(D::Minus1, 1, D_CONV - 1)?;
        let xbc_expanded = xbc.unsqueeze(D::Minus1)?;
        *conv_state = Tensor::cat(&[shifted, xbc_expanded], D::Minus1)?;

        let mut result = self.conv1d_bias.broadcast_as((b_sz, d_xbc))?;
        for i in 0..D_CONV {
            let w = self.conv1d_weight.i((.., 0, i))?;
            let xbc_i = conv_state.i((.., .., i))?;
            result = (result + w.broadcast_mul(&xbc_i)?)?;
        }
        Ok(result)
    }

    fn ssm_step(
        &self,
        x: &Tensor,
        a: &Tensor,
        b: &Tensor,
        c: &Tensor,
        dt: &Tensor,
        state: &mut State,
    ) -> Result<Tensor> {
        let (b_sz, _) = x.dims2()?;
        let h = &mut state.hs[self.layer_idx];

        let x = x.reshape((b_sz, self.nheads, self.headdim))?;

        let b = b.reshape((b_sz, self.ngroups, self.d_state))?;
        let c = c.reshape((b_sz, self.ngroups, self.d_state))?;
        let heads_per_group = self.nheads / self.ngroups;
        let b = b
            .unsqueeze(2)?
            .broadcast_as((b_sz, self.ngroups, heads_per_group, self.d_state))?
            .reshape((b_sz, self.nheads, self.d_state))?;
        let c = c
            .unsqueeze(2)?
            .broadcast_as((b_sz, self.ngroups, heads_per_group, self.d_state))?
            .reshape((b_sz, self.nheads, self.d_state))?;

        let dt_a = dt.broadcast_mul(a)?;
        let decay = dt_a.exp()?;
        let decay = decay.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
        let decay = decay.broadcast_as((b_sz, self.nheads, self.headdim, self.d_state))?;

        let x_unsq = x.unsqueeze(D::Minus1)?;
        let b_unsq = b.unsqueeze(2)?;
        let x_b = x_unsq.broadcast_mul(&b_unsq)?;

        let dt_expanded = dt.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
        let dt_expanded =
            dt_expanded.broadcast_as((b_sz, self.nheads, self.headdim, self.d_state))?;

        *h = ((&*h * &decay)? + (&dt_expanded * &x_b)?)?;

        let c_unsq = c.unsqueeze(2)?;
        let c_broadcast = c_unsq.broadcast_as(h.shape())?;
        let y = (&*h * &c_broadcast)?.sum(D::Minus1)?;

        Ok(y)
    }
}

#[derive(Clone, Debug)]
struct ResidualBlock {
    mixer: Mamba2Block,
    norm: RmsNorm,
}

impl ResidualBlock {
    fn new(layer_idx: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let norm = candle_nn::rms_norm(cfg.d_model, 1e-5, vb.pp("norm"))?;
        let mixer = Mamba2Block::new(layer_idx, cfg, vb.pp("mixer"))?;
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
        // Official Mamba2 HF checkpoints wrap weights under a `backbone` prefix;
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
            layers.push(ResidualBlock::new(layer_idx, cfg, vb_l.pp(layer_idx))?);
        }
        let norm_f = candle_nn::rms_norm(cfg.d_model, 1e-5, vb.pp("norm_f"))?;
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

    fn unpack_state_tensor(&self, t: &Tensor) -> Result<(Tensor, Tensor)> {
        let t = t.to_device(&self.device)?.to_dtype(self.dtype)?;
        let total = t.elem_count();
        let per_batch = self.config.nheads() * self.config.headdim * self.config.d_state
            + self.config.d_xbc() * D_CONV;
        if total % per_batch != 0 {
            candle_core::bail!(
                "packed state size {} is not divisible by expected per-batch size {}",
                total,
                per_batch
            );
        }
        let b = total / per_batch;
        let hs_len = b * self.config.nheads() * self.config.headdim * self.config.d_state;
        let conv_len = b * self.config.d_xbc() * D_CONV;
        let flat = t.reshape((total,))?;
        let hs = flat.narrow(0, 0, hs_len)?.reshape((
            b,
            self.config.nheads(),
            self.config.headdim,
            self.config.d_state,
        ))?;
        let conv = flat
            .narrow(0, hs_len, conv_len)?
            .reshape((b, self.config.d_xbc(), D_CONV))?;
        Ok((hs, conv))
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
        let mut conv_states = Vec::with_capacity(self.layers.len());
        for (state_tensor, _dummy) in cache {
            let (h, conv) = self.unpack_state_tensor(&state_tensor)?;
            hs.push(h);
            conv_states.push(conv);
        }
        self.state = Some(State {
            hs,
            conv_states,
            pos: 0,
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
        num_attention_heads: cfg.nheads(),
        head_dim: cfg.headdim,
        num_key_value_heads: cfg.ngroups,
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
