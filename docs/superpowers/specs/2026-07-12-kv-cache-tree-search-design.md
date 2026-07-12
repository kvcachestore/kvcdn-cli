# KV-cache prefix search and tree generation

## Summary

Add a `kvcdn search` command that finds the saved KV artifact whose stored prompt
tokens share the longest prefix with a new query. A `--tree N` flag optionally
loads the best-matching artifact and generates `N` greedy continuation
candidates from the shared prefix, paying for the prefix decode only once.

## Motivation

The CLI already creates, quantizes, and verifies KV artifacts, but there is no
way to discover which saved artifact is the best starting point for a new
prompt. Users currently have to remember which `.kv` file corresponds to which
prefix. Storing the source tokens in the artifact sidecar and searching by
longest common prefix solves that. The `--tree` flag adds a cheap way to
explore multiple candidate continuations from the same prefix, similar to the
fork/backtrack behavior demonstrated by `lo-agent`.

## User-facing behavior

### Default mode: prefix search

```bash
kvcdn search \
  --model Qwen/Qwen3-0.6B \
  --dir ~/.local/share/kvcdn/verify \
  --query "Summarize the key claim" \
  [--format table|json]
```

Output (table):

```text
path                                          prefix_tokens  total_tokens  model
/home/user/.local/share/kvcdn/verify/abc.kv   42             512           Qwen/Qwen3-0.6B
```

- Scans `--dir` recursively for `.kv` files.
- Reads the `.kv.json` sidecar for each artifact.
- Skips artifacts whose `model_name` does not match `--model`.
- Tokenizes `--query` with the model tokenizer (default mode only loads the
  tokenizer; `--tree` loads the full model).
- Computes the longest common token prefix between the query and the stored
  artifact tokens.
- Ranks by prefix length (desc), then by total tokens (desc).
- Prints the best match. If no artifact matches, prints a clear message and
  exits 0.

### Tree generation mode

```bash
kvcdn search \
  --model Qwen/Qwen3-0.6B \
  --dir ~/.local/share/kvcdn/verify \
  --query "Summarize the key claim" \
  --tree 4 \
  [--tree-tokens 32]
```

Behavior:

1. Run the prefix search to find the best artifact.
2. If a match with a non-zero prefix exists:
   - Load the matching KV artifact into the model.
   - Run the unmatched query suffix through the model to advance the KV cache.
   - Snapshot the cache.
   - Generate `N` independent greedy continuations from the snapshot.
3. If no artifact matches:
   - Prefill the full query from scratch.
   - Generate `N` candidates anyway and print a note that no prefix reuse
     occurred.

Output:

```text
matched: /home/user/.local/share/kvcdn/verify/abc.kv (42 prefix tokens)

candidate 0: The key claim is that...
candidate 1: The document argues that...
candidate 2: According to the text,...
candidate 3: It claims that...
```

## Data model changes

Extend `KVArtifact` in `src/local/kv_io.rs` with two optional fields:

```rust
pub struct KVArtifact {
    pub model_name: String,
    pub num_layers: usize,
    pub num_tokens: usize,
    pub dtype: String,
    pub storage_dtype: Option<String>,
    pub nbytes: u64,
    pub quantized: bool,
    /// Original prompt text that produced this KV cache, if available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Token IDs of the original prompt, if available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<Vec<u32>>,
}
```

- Both fields are optional for backward compatibility with existing artifacts.
- Artifacts without `tokens` are skipped during search with a warning.
- `verify` stores the context text and token IDs used to build the cache. The
  stored text is the prefix only (the `context`), not the continuation
  `question`.
- `benchmark` does not currently save `.kv` artifacts (it only writes a CSV),
  so no sidecar update is needed there.
- `quant` stores the context text and token IDs used when the input is generated
  on the fly. If `--input` points to an existing artifact in the future, its
  sidecar metadata should be preserved; for now the command generates the cache
  from `--context-file` or a default context and stores that metadata.

## Architecture

### New modules

- `src/local/search.rs` — `SearchArgs` parsing, directory scan, prefix ranking,
  and command dispatch.
- `src/local/search_tree.rs` — KV-cache snapshot, suffix forward, and candidate
  generation helpers.

### Changes to existing modules

- `src/cli.rs` — add `SearchArgs`.
- `src/main.rs` — wire `Cli::Search` to `local::search::run`.
- `src/local/kv_io.rs` — extend `KVArtifact` and update save helpers to accept
  optional prompt/tokens.
- `src/local/verify.rs` — pass prompt and tokens into `save_kv`.
- `src/local/quant.rs` — store prompt/tokens for the context used to build the
  quantized cache.

## Algorithm

### Prefix search

1. Load tokenizer for `--model`.
2. Encode `--query` once → `query_tokens`.
3. For each `.kv` file under `--dir`:
   - Read sidecar.
   - Skip if `model_name != --model`.
   - Skip if `tokens` is `None`.
   - Compute LCP length by iterating `query_tokens` and artifact `tokens`.
4. Sort matches by `(lcp desc, num_tokens desc)`.
5. Print best match.

LCP computation:

```rust
let lcp = query_tokens
    .iter()
    .zip(artifact_tokens.iter())
    .take_while(|(a, b)| a == b)
    .count();
```

### Tree candidate generation

1. Resolve best match as above.
2. Load model on selected device.
3. If a non-zero-prefix match exists:
   - `load_kv(best_path, device)` and `model.set_kv_cache(cache)`.
   - `suffix = &query_tokens[prefix_len..]`.
   - `model.forward(&suffix_tensor, prefix_len)` to advance the cache.
4. Else:
   - `model.forward(&query_tokens_tensor, 0)` to prefill from scratch.
5. Snapshot the current KV cache:
   - `snapshot = model.get_kv_cache()?.clone()`.
6. For `i` in `0..N`:
   - `model.set_kv_cache(snapshot.clone())`.
   - Greedily generate `--tree-tokens` tokens.
   - Decode and print.

## CLI arguments

```rust
#[derive(Parser)]
pub struct SearchArgs {
    /// Hugging Face model identifier whose artifacts to search.
    #[arg(long)]
    pub model: String,
    /// Directory to scan recursively for `.kv` artifacts.
    #[arg(long)]
    pub dir: String,
    /// Query prompt to match against stored artifact prefixes.
    #[arg(long)]
    pub query: String,
    /// Number of greedy candidate continuations to generate from the matched prefix.
    #[arg(long)]
    pub tree: Option<usize>,
    /// Number of tokens to generate per tree candidate.
    #[arg(long, default_value_t = 32)]
    pub tree_tokens: usize,
    /// Output format for the prefix-search result.
    #[arg(long, default_value = "table")]
    pub format: String,
    /// Device to run tree generation on: cpu, cuda, or metal.
    #[arg(long, value_parser = crate::models::engine::parse_device)]
    pub device: Option<candle_core::Device>,
    /// Hugging Face model revision.
    #[arg(long)]
    pub revision: Option<String>,
}
```

## Error handling

- Missing required flags are caught by clap.
- `--dir` not found or not a directory → clear `anyhow` error.
- No matching artifact in default mode → informational message and exit 0.
- No matching artifact in `--tree` mode → prefill from scratch and continue.
- Sidecar missing `tokens` → skip with a warning printed to stderr.
- Model load or inference failure in `--tree` mode → propagate as error.

## Testing

- `kv_io.rs`: round-trip test for `KVArtifact` with `prompt` and `tokens`.
- `search.rs`: unit test with a temporary directory containing sidecars for two
  fake models and two prompts; assert the correct best match and LCP length.
- `search_tree.rs`: use a stub `CausalLM` and fake tokenizer to verify that
  snapshot/restore produces the expected number of distinct greedy candidates.
- `cli.rs`/`main.rs`: parse tests for `SearchArgs` and `Cli::Search`.
- Integration: run `kvcdn verify` with a tiny context, then `kvcdn search`
  against the resulting directory and assert the artifact is found.

## Future work (out of scope)

- Persistent index file (`.kvidx`) for large artifact libraries.
- Fuzzy prefix/distance matching instead of exact token LCP.
- Verifier model or heuristic to pick the best `--tree` candidate instead of
  printing all.
- Backend API endpoint to search remote artifacts.
