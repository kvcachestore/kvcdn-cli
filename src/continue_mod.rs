use anyhow::Result;
use candle_core::Tensor;

use crate::model::ModelBundle;

fn argmax_token(logits: &Tensor) -> Result<u32> {
    let logits = logits.squeeze(0)?.squeeze(0)?;
    let idx = logits.argmax(0)?.to_scalar::<u32>()?;
    Ok(idx)
}

/// Continue generation greedily from an existing KV cache.
///
/// `start_offset` is the number of tokens already cached. The first token in
/// `prompt_tokens` is assumed to follow immediately after the cached prefix.
pub fn generate(
    bundle: &mut ModelBundle,
    prompt_tokens: &[u32],
    start_offset: usize,
    n: usize,
) -> Result<Vec<u32>> {
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return Ok(out);
    }

    // Run prompt tokens through the model once; logits are for the last token.
    let input = Tensor::new(prompt_tokens, &bundle.device)?.unsqueeze(0)?;
    let logits = bundle.model.forward(&input, start_offset)?;
    let mut next = argmax_token(&logits)?;

    for i in 0..n {
        out.push(next);
        let offset = start_offset + prompt_tokens.len() + i;
        let input = Tensor::new(&[next], &bundle.device)?.unsqueeze(0)?;
        let logits = bundle.model.forward(&input, offset)?;
        next = argmax_token(&logits)?;
    }

    Ok(out)
}
