use anyhow::{Context, Result};
use candle_core::Tensor;

use crate::cli::SearchArgs;
use crate::local::kv_io::load_kv;
use crate::local::search::ScoredMatch;
use crate::models::CausalLM;
use crate::models::engine::ModelBundle;

fn argmax_token(logits: &Tensor) -> Result<u32> {
    let logits = logits.squeeze(0)?.squeeze(0)?;
    let idx = logits.argmax(0)?.to_scalar::<u32>()?;
    Ok(idx)
}

fn snapshot_kv_cache(model: &dyn CausalLM) -> Result<crate::models::KVCache> {
    Ok(model.get_kv_cache()?)
}

fn restore_kv_cache(model: &mut dyn CausalLM, cache: crate::models::KVCache) -> Result<()> {
    Ok(model.set_kv_cache(cache)?)
}

/// Forward `tokens` starting at `offset` and return logits for the last token.
fn forward_tokens(
    model: &mut dyn CausalLM,
    device: &candle_core::Device,
    tokens: &[u32],
    offset: usize,
) -> Result<Tensor> {
    let input = Tensor::new(tokens, device)?.unsqueeze(0)?;
    Ok(model.forward(&input, offset)?)
}

/// Greedily generate `n` tokens starting from the current KV-cache position.
/// `first_logits` are the logits for the token immediately preceding generation;
/// `start_offset` is the KV-cache length after that token.
fn generate_from_current(
    model: &mut dyn CausalLM,
    device: &candle_core::Device,
    first_logits: &Tensor,
    start_offset: usize,
    n: usize,
) -> Result<Vec<u32>> {
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return Ok(out);
    }
    let mut next = argmax_token(first_logits)?;
    for i in 0..n {
        out.push(next);
        let offset = start_offset + i;
        let input = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let logits = model.forward(&input, offset)?;
        next = argmax_token(&logits)?;
    }
    Ok(out)
}

pub fn run(
    args: &SearchArgs,
    bundle: &mut ModelBundle,
    query_tokens: &[u32],
    best: Option<&ScoredMatch>,
) -> Result<()> {
    if query_tokens.is_empty() {
        anyhow::bail!("--query produced no tokens; cannot generate tree candidates");
    }

    let (first_logits, start_offset) = if let Some(best) = best {
        if best.prefix_len > 0 {
            println!(
                "matched: {} ({} prefix tokens)",
                best.path.display(),
                best.prefix_len
            );
            let (cache, _) = load_kv(&best.path, &bundle.device)
                .with_context(|| format!("loading KV artifact {}", best.path.display()))?;
            bundle.model.set_kv_cache(cache)?;
            let suffix = &query_tokens[best.prefix_len..];
            if suffix.is_empty() {
                // Query exactly matches the prefix; need logits for the last query token.
                let last = query_tokens.last().copied().unwrap_or(0);
                let logits = forward_tokens(
                    &mut *bundle.model,
                    &bundle.device,
                    &[last],
                    best.prefix_len - 1,
                )?;
                (logits, query_tokens.len())
            } else {
                let logits =
                    forward_tokens(&mut *bundle.model, &bundle.device, suffix, best.prefix_len)?;
                (logits, best.prefix_len + suffix.len())
            }
        } else {
            println!("note: no matching artifact; prefilling query from scratch");
            let logits = forward_tokens(&mut *bundle.model, &bundle.device, query_tokens, 0)?;
            (logits, query_tokens.len())
        }
    } else {
        println!("note: no matching artifact; prefilling query from scratch");
        let logits = forward_tokens(&mut *bundle.model, &bundle.device, query_tokens, 0)?;
        (logits, query_tokens.len())
    };

    let snapshot = snapshot_kv_cache(&*bundle.model)?;

    let n = args.tree.unwrap_or(1);
    for i in 0..n {
        restore_kv_cache(&mut *bundle.model, snapshot.clone())?;
        let tokens = generate_from_current(
            &mut *bundle.model,
            &bundle.device,
            &first_logits,
            start_offset,
            args.tree_tokens,
        )?;
        let text = bundle
            .tokenizer
            .decode(&tokens, false)
            .map_err(|e| anyhow::anyhow!(e))?;
        println!("\ncandidate {}: {}", i, text);
    }

    Ok(())
}
