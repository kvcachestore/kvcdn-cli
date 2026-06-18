use anyhow::Result;

/// Encode `text` into token ids.
pub fn encode(
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
    add_special_tokens: bool,
) -> Result<Vec<u32>> {
    Ok(tokenizer
        .encode(text, add_special_tokens)
        .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?
        .get_ids()
        .to_vec())
}

/// Build a token sequence of exactly `len` tokens by repeating a fixed sentence.
pub fn context_of_length(tokenizer: &tokenizers::Tokenizer, len: usize) -> Result<Vec<u32>> {
    let sentence = "Retrieval-augmented generation reuses documents across many queries. ";
    let mut tokens = encode(tokenizer, sentence, true)?;
    while tokens.len() < len {
        tokens.extend(encode(tokenizer, sentence, false)?);
    }
    tokens.truncate(len);
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_tokenizer() -> tokenizers::Tokenizer {
        // A minimal tokenizer with one token per lowercase letter.
        let mut tokenizer = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
        for c in 'a'..='z' {
            let _ = tokenizer.add_tokens(vec![tokenizers::AddedToken::from(c.to_string(), false)]);
        }
        tokenizer
    }

    #[test]
    fn encode_returns_ids() {
        let tokenizer = dummy_tokenizer();
        let ids = encode(&tokenizer, "hello", false).unwrap();
        assert!(!ids.is_empty());
    }

    #[test]
    fn context_of_length_exact_size() {
        let tokenizer = dummy_tokenizer();
        let ids = context_of_length(&tokenizer, 10).unwrap();
        assert_eq!(ids.len(), 10);
    }
}
