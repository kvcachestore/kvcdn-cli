use anyhow::Result;
use candle_core::Tensor;

use crate::models::CausalLM;
use crate::models::engine::ModelBundle;

fn argmax_token(logits: &Tensor) -> Result<u32> {
    let logits = logits.squeeze(0)?.squeeze(0)?;
    let idx = logits.argmax(0)?.to_scalar::<u32>()?;
    Ok(idx)
}

/// Greedy-generate `n` tokens from `model`, continuing a prefix of length `start_offset`.
/// `prompt_tokens` are processed first; the first generated token follows them.
pub fn generate_with_model(
    model: &mut dyn CausalLM,
    device: &candle_core::Device,
    prompt_tokens: &[u32],
    start_offset: usize,
    n: usize,
) -> Result<Vec<u32>> {
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return Ok(out);
    }

    let input = Tensor::new(prompt_tokens, device)?.unsqueeze(0)?;
    let logits = model.forward(&input, start_offset)?;
    let mut next = argmax_token(&logits)?;

    for i in 0..n {
        out.push(next);
        let offset = start_offset + prompt_tokens.len() + i;
        let input = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let logits = model.forward(&input, offset)?;
        next = argmax_token(&logits)?;
    }

    Ok(out)
}

/// Backwards-compatible wrapper that generates from a `ModelBundle`.
pub fn generate(
    bundle: &mut ModelBundle,
    prompt_tokens: &[u32],
    start_offset: usize,
    n: usize,
) -> Result<Vec<u32>> {
    generate_with_model(
        &mut *bundle.model,
        &bundle.device,
        prompt_tokens,
        start_offset,
        n,
    )
}
