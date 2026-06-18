//! GPT-2 causal language-model adapter with explicit KV-cache access.
//!
//! Implements the GPT-2 transformer stack (`GPT2LMHeadModel`) using learned
//! position embeddings, standard multi-head self-attention, LayerNorm with
//! bias, and a GELU MLP. Weight names follow the Hugging Face checkpoint
//! layout under the `transformer` prefix.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Activation, Linear, VarBuilder};

use crate::models::{CausalLM, KVCache, ModelConfig};

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub(crate) struct Config {
    pub vocab_size: usize,
    pub n_positions: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    #[serde(default)]
    pub n_inner: Option<usize>,
    #[serde(default = "default_activation_function")]
    pub activation_function: Activation,
    #[serde(default = "default_layer_norm_epsilon")]
    pub layer_norm_epsilon: f64,
    #[serde(default)]
    pub resid_pdrop: f64,
    #[serde(default)]
    pub embd_pdrop: f64,
    #[serde(default)]
    pub attn_pdrop: f64,
    #[serde(default)]
    pub initializer_range: f64,
}

fn default_activation_function() -> Activation {
    Activation::NewGelu
}

fn default_layer_norm_epsilon() -> f64 {
    1e-5
}

impl Config {
    pub(crate) fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    fn intermediate_size(&self) -> usize {
        self.n_inner.unwrap_or(4 * self.n_embd)
    }
}

/// Load a GPT-2 Conv1D-style linear layer.
///
/// GPT-2 stores linear weights with shape `(in_features, out_features)`;
/// `candle_nn::Linear` expects `(out_features, in_features)` and transposes
/// at forward time, so we transpose on load.
fn linear_gpt2(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((in_dim, out_dim), "weight")?.t()?;
    let bias = vb.get(out_dim, "bias")?;
    Ok(Linear::new(weight, Some(bias)))
}

#[derive(Debug, Clone)]
struct Gpt2Attention {
    c_attn: Linear,
    c_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    k_cache: Option<Tensor>,
    v_cache: Option<Tensor>,
}

impl Gpt2Attention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.n_embd;
        let num_heads = cfg.n_head;
        let head_dim = cfg.head_dim();
        let c_attn = linear_gpt2(hidden_size, 3 * hidden_size, vb.pp("c_attn"))?;
        let c_proj = linear_gpt2(hidden_size, hidden_size, vb.pp("c_proj"))?;
        Ok(Self {
            c_attn,
            c_proj,
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
        let qkv = self.c_attn.forward(x)?;
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
            .apply(&self.c_proj)
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
struct Gpt2Mlp {
    c_fc: Linear,
    c_proj: Linear,
    act: Activation,
}

impl Gpt2Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let c_fc = linear_gpt2(cfg.n_embd, cfg.intermediate_size(), vb.pp("c_fc"))?;
        let c_proj = linear_gpt2(cfg.intermediate_size(), cfg.n_embd, vb.pp("c_proj"))?;
        Ok(Self {
            c_fc,
            c_proj,
            act: cfg.activation_function,
        })
    }
}

impl Module for Gpt2Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.c_fc)?.apply(&self.act)?.apply(&self.c_proj)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    ln_1: candle_nn::LayerNorm,
    attn: Gpt2Attention,
    ln_2: candle_nn::LayerNorm,
    mlp: Gpt2Mlp,
}

impl DecoderLayer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let ln_1 = candle_nn::layer_norm(cfg.n_embd, cfg.layer_norm_epsilon, vb.pp("ln_1"))?;
        let attn = Gpt2Attention::new(cfg, vb.pp("attn"))?;
        let ln_2 = candle_nn::layer_norm(cfg.n_embd, cfg.layer_norm_epsilon, vb.pp("ln_2"))?;
        let mlp = Gpt2Mlp::new(cfg, vb.pp("mlp"))?;
        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let h = self.ln_1.forward(x)?;
        let h = self.attn.forward(&h, mask, offset)?;
        let x = (x + h)?;
        let h2 = self.ln_2.forward(&x)?;
        let h2 = h2.apply(&self.mlp)?;
        x + h2
    }

    fn clear_kv_cache(&mut self) {
        self.attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    wte: candle_nn::Embedding,
    wpe: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    ln_f: candle_nn::LayerNorm,
    device: Device,
    dtype: DType,
    config: Config,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_t = vb.pp("transformer");
        let wte = candle_nn::embedding(cfg.vocab_size, cfg.n_embd, vb_t.pp("wte"))?;
        let wpe = candle_nn::embedding(cfg.n_positions, cfg.n_embd, vb_t.pp("wpe"))?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        let vb_h = vb_t.pp("h");
        for i in 0..cfg.n_layer {
            layers.push(DecoderLayer::new(cfg, vb_h.pp(i))?);
        }
        Ok(Self {
            wte,
            wpe,
            layers,
            ln_f: candle_nn::layer_norm(cfg.n_embd, cfg.layer_norm_epsilon, vb_t.pp("ln_f"))?,
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
        let dev = input.device();
        let token_emb = self.wte.forward(input)?;
        let positions = Tensor::arange(offset as u32, (offset + l) as u32, dev)?
            .reshape((1, l))?
            .broadcast_as((b, l))?;
        let pos_emb = self.wpe.forward(&positions)?;
        let mut h = (token_emb + pos_emb)?;

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
                l.attn.get_kv_cache().ok_or_else(|| {
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
            layer.attn.set_kv_cache(k, v);
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
        // GPT-2 ties the output embedding to the token embedding by default.
        // If an explicit `lm_head.weight` is present, load it; otherwise reuse wte.
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            linear_gpt2(cfg.n_embd, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            Linear::new(base.wte.embeddings().clone(), None)
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
        hidden_size: cfg.n_embd,
        intermediate_size: cfg.intermediate_size(),
        num_hidden_layers: cfg.n_layer,
        num_attention_heads: cfg.n_head,
        head_dim: cfg.head_dim(),
        num_key_value_heads: cfg.n_head,
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
