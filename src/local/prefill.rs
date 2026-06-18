use anyhow::Result;
use candle_core::Tensor;

use crate::models::KVCache;
use crate::models::engine::ModelBundle;

/// Prefill `tokens` and return the captured KV cache.
pub fn prefill(bundle: &mut ModelBundle, tokens: &[u32]) -> Result<KVCache> {
    let input = Tensor::new(tokens, &bundle.device)?.unsqueeze(0)?;
    let _ = bundle.model.forward(&input, 0)?;
    Ok(bundle.model.get_kv_cache()?)
}
