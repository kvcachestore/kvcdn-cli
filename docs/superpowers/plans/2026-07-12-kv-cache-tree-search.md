# KV-cache prefix search and tree generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SELL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `kvcdn search` command that finds saved KV artifacts by longest token-prefix match and, with `--tree N`, generates N greedy continuation candidates from the shared prefix.

**Architecture:** Artifact sidecars (`.kv.json`) are extended with optional `prompt` text and `tokens` IDs. `kvcdn search` scans a directory, reads sidecars, tokenizes the query, ranks artifacts by longest common prefix, and optionally loads the best match into the model to fork N greedy decodes from a KV-cache snapshot.

**Tech Stack:** Rust, candle-core, candle-transformers, clap, serde_json, safetensors, tokenizers, anyhow.

---

## File map

| File | Responsibility |
|------|----------------|
| `src/cli.rs` | New `SearchArgs` struct and doc comments. |
| `src/main.rs` | Add `Cli::Search` variant and dispatch to `local::search::run`. |
| `src/local/mod.rs` | Add `pub mod search; pub mod search_tree;`. |
| `src/local/kv_io.rs` | Extend `KVArtifact`; update `save_kv` and `save_quantized_kv` to accept optional prompt/tokens. |
| `src/local/search.rs` | Directory scan, LCP ranking, table/JSON output, `--tree` dispatch. |
| `src/local/search_tree.rs` | KV-cache snapshot, suffix forward, candidate generation. |
| `src/local/verify.rs` | Pass context text and tokens to `save_kv`. |
| `src/local/benchmark.rs` | Pass context text and tokens to `save_kv`. |
| `src/local/quant.rs` | Copy prompt/tokens from input sidecar; pass to output save. |
| `src/local/continuation.rs` | Optionally expose a lower-level generate function that accepts an explicit KV cache snapshot (or reuse existing `generate_with_model`). |

---

### Task 1: Extend `KVArtifact` and save helpers

**Files:**
- Modify: `src/local/kv_io.rs:84-103`
- Modify: `src/local/kv_io.rs:122-150`
- Modify: `src/local/kv_io.rs:154-190`
- Test: `src/local/kv_io.rs:402-581`

- [ ] **Step 1: Write a failing round-trip test for prompt/tokens**

Add inside the existing `#[cfg(test)] mod tests` block:

```rust
#[test]
fn save_and_load_kv_with_prompt_and_tokens() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("artifact.kv");
    let cache = make_cache(2, 5);
    let prompt = "hello world".to_string();
    let tokens = vec![1u32, 2, 3, 4, 5];
    let saved = save_kv_with_meta(&cache, &path, "test-model", Some(prompt.clone()), Some(tokens.clone()))?;
    assert_eq!(saved.prompt, Some(prompt));
    assert_eq!(saved.tokens, Some(tokens));

    let (_loaded, artifact) = load_kv(&path, &Device::Cpu)?;
    assert_eq!(artifact.prompt, Some("hello world".to_string()));
    assert_eq!(artifact.tokens, Some(vec![1, 2, 3, 4, 5]));
    Ok(())
}
```

Run:

```bash
cargo test --lib save_and_load_kv_with_prompt_and_tokens
```

Expected: FAIL — `save_kv_with_meta` not found.

- [ ] **Step 2: Extend `KVArtifact`**

Change the struct in `src/local/kv_io.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KVArtifact {
    pub model_name: String,
    pub num_layers: usize,
    pub num_tokens: usize,
    pub dtype: String,
    pub storage_dtype: Option<String>,
    pub nbytes: u64,
    pub quantized: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<Vec<u32>>,
}
```

- [ ] **Step 3: Add `save_kv_with_meta` and refactor existing `save_kv`**

Replace `save_kv` with:

```rust
pub fn save_kv<P: AsRef<Path>>(
    cache: &KVCache,
    path: P,
    model_name: &str,
) -> Result<KVArtifact> {
    save_kv_with_meta(cache, path, model_name, None, None)
}

pub fn save_kv_with_meta<P: AsRef<Path>>(
    cache: &KVCache,
    path: P,
    model_name: &str,
    prompt: Option<String>,
    tokens: Option<Vec<u32>>,
) -> Result<KVArtifact> {
    if cache.is_empty() {
        anyhow::bail!("cannot save empty KV cache");
    }
    let num_layers = cache.len();
    let num_tokens = cache[0].0.dim(2)?;
    let dtype = format!("{:?}", cache[0].0.dtype());

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for (i, (k, v)) in cache.iter().enumerate() {
        tensors.insert(format!("k_{i}"), k.clone());
        tensors.insert(format!("v_{i}"), v.clone());
    }

    candle_core::safetensors::save(&tensors, &path)?;

    let nbytes = fs::metadata(&path)?.len();
    let artifact = KVArtifact {
        model_name: model_name.to_string(),
        num_layers,
        num_tokens,
        dtype: dtype.clone(),
        storage_dtype: Some(dtype),
        nbytes,
        quantized: false,
        prompt,
        tokens,
    };
    write_meta(&path, &artifact)?;
    Ok(artifact)
}
```

- [ ] **Step 4: Add `save_quantized_kv_with_meta` and refactor `save_quantized_kv`**

Replace `save_quantized_kv` with:

```rust
pub fn save_quantized_kv<P: AsRef<Path>>(
    layers: &[(Tensor, Tensor, Tensor, Tensor)],
    path: P,
    model_name: &str,
    target_dtype: QuantDtype,
) -> Result<KVArtifact> {
    save_quantized_kv_with_meta(layers, path, model_name, target_dtype, None, None)
}

pub fn save_quantized_kv_with_meta<P: AsRef<Path>>(
    layers: &[(Tensor, Tensor, Tensor, Tensor)],
    path: P,
    model_name: &str,
    target_dtype: QuantDtype,
    prompt: Option<String>,
    tokens: Option<Vec<u32>>,
) -> Result<KVArtifact> {
    if layers.is_empty() {
        anyhow::bail!("cannot save empty quantized KV cache");
    }
    let num_layers = layers.len();
    let num_tokens = layers[0].1.dim(2)?;
    let dtype = target_dtype.to_string();
    let storage_dtype = format!("{:?}", layers[0].1.dtype());

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for (i, (k_s, k_q, v_s, v_q)) in layers.iter().enumerate() {
        tensors.insert(format!("k_{i}_q"), k_q.clone());
        tensors.insert(format!("k_{i}_s"), k_s.clone());
        tensors.insert(format!("v_{i}_q"), v_q.clone());
        tensors.insert(format!("v_{i}_s"), v_s.clone());
    }

    candle_core::safetensors::save(&tensors, &path)?;

    let nbytes = fs::metadata(&path)?.len();
    let artifact = KVArtifact {
        model_name: model_name.to_string(),
        num_layers,
        num_tokens,
        dtype,
        storage_dtype: Some(storage_dtype),
        nbytes,
        quantized: true,
        prompt,
        tokens,
    };
    write_meta(&path, &artifact)?;
    Ok(artifact)
}
```

- [ ] **Step 5: Run tests**

Run:

```bash
cargo test --lib save_and_load_kv_with_prompt_and_tokens
cargo test --lib kv_io
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/local/kv_io.rs
git commit -m "feat(kv): store optional prompt and tokens in KV artifact sidecar"
```

---

### Task 2: Add `SearchArgs` to CLI

**Files:**
- Modify: `src/cli.rs`
- Test: `src/cli.rs:166-182` and `src/main.rs:149-171`

- [ ] **Step 1: Add `SearchArgs` struct after `WhoamiArgs`**

Append to `src/cli.rs`:

```rust
/// Search saved KV artifacts for the longest token-prefix match.
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
    /// Generate N greedy candidate continuations from the matched prefix.
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

- [ ] **Step 2: Add a parse test**

Add inside `src/cli.rs` `mod tests`:

```rust
#[test]
fn search_args_parse() {
    let args = SearchArgs::try_parse_from([
        "kvcdn",
        "--model",
        "Qwen/Qwen3-0.6B",
        "--dir",
        "/tmp/kv",
        "--query",
        "hello",
        "--tree",
        "4",
        "--tree-tokens",
        "16",
    ])
    .unwrap();
    assert_eq!(args.model, "Qwen/Qwen3-0.6B");
    assert_eq!(args.dir, "/tmp/kv");
    assert_eq!(args.query, "hello");
    assert_eq!(args.tree, Some(4));
    assert_eq!(args.tree_tokens, 16);
}
```

Run:

```bash
cargo test --lib search_args_parse
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/cli.rs
git commit -m "feat(cli): add SearchArgs for kv search command"
```

---

### Task 3: Wire `Cli::Search` in `main.rs`

**Files:**
- Modify: `src/main.rs:1-120`
- Test: `src/main.rs:149-171`

- [ ] **Step 1: Import `SearchArgs`**

Change the import block at the top of `src/main.rs`:

```rust
use kvcdn::cli::{
    AdminArgs, ApiKeyArgs, BenchmarkArgs, DeleteArgs, DiagArgs, DownloadArgs, InferArgs, ListArgs,
    LoginArgs, LogoutArgs, PlotArgs, QuantArgs, QuotaArgs, SearchArgs, UploadArgs, VerifyArgs,
    WhoamiArgs,
};
```

- [ ] **Step 2: Add variant and dispatch**

In the `Cli` enum add:

```rust
    /// Search saved KV artifacts by token-prefix match.
    Search(SearchArgs),
```

In `command_name`:

```rust
        Cli::Search(_) => "search",
```

In `run_command`:

```rust
        Cli::Search(args) => local::search::run(args),
```

- [ ] **Step 3: Add CLI parse test**

Add inside `src/main.rs` `mod tests`:

```rust
#[test]
fn cli_search_parse() {
    let cli = Cli::try_parse_from([
        "kvcdn",
        "search",
        "--model",
        "Qwen/Qwen3-0.6B",
        "--dir",
        "/tmp/kv",
        "--query",
        "hello",
    ])
    .unwrap();
    match cli {
        Cli::Search(args) => {
            assert_eq!(args.model, "Qwen/Qwen3-0.6B");
            assert_eq!(args.dir, "/tmp/kv");
            assert_eq!(args.query, "hello");
        }
        _ => panic!("expected Search subcommand"),
    }
}
```

Run:

```bash
cargo test --lib cli_search_parse
cargo check
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/main.rs
git commit -m "feat(cli): wire Search subcommand dispatch"
```

---

### Task 4: Implement prefix search in `src/local/search.rs`

**Files:**
- Create: `src/local/search.rs`
- Modify: `src/local/mod.rs`
- Test: `src/local/search.rs` (self-contained unit tests)

- [ ] **Step 1: Create `src/local/search.rs`**

```rust
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

use crate::cli::SearchArgs;
use crate::local::kv_io::{read_kv_metadata, KVArtifact};
use crate::local::tokenize::encode;
use crate::models::engine::{load_model_on, resolve_revision};
use crate::core::common;

#[derive(Debug, Clone)]
pub struct ScoredMatch {
    pub path: PathBuf,
    pub artifact: KVArtifact,
    pub prefix_len: usize,
}

pub fn longest_common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

fn collect_kv_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                out.extend(collect_kv_files(&path));
            } else if path.extension().and_then(|e| e.to_str()) == Some("kv") {
                out.push(path);
            }
        }
    }
    out
}

fn read_artifact(path: &Path) -> Result<KVArtifact> {
    read_kv_metadata(path).with_context(|| format!("reading metadata for {}", path.display()))
}

fn rank_artifacts(
    files: &[PathBuf],
    model_name: &str,
    query_tokens: &[u32],
) -> Vec<ScoredMatch> {
    let mut matches = Vec::new();
    for path in files {
        let artifact = match read_artifact(path) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("warning: {}", e);
                continue;
            }
        };
        if artifact.model_name != model_name {
            continue;
        }
        let Some(tokens) = artifact.tokens.as_ref() else {
            eprintln!(
                "warning: skipping {} (no tokens in sidecar)",
                path.display()
            );
            continue;
        };
        let prefix_len = longest_common_prefix(query_tokens, tokens);
        matches.push(ScoredMatch {
            path: path.clone(),
            artifact,
            prefix_len,
        });
    }
    matches.sort_by(|a, b| {
        b.prefix_len
            .cmp(&a.prefix_len)
            .then_with(|| b.artifact.num_tokens.cmp(&a.artifact.num_tokens))
    });
    matches
}

fn format_table(matches: &[ScoredMatch]) -> String {
    let mut out = String::new();
    out.push_str("path\tprefix_tokens\ttotal_tokens\tmodel\n");
    for m in matches {
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\n",
            m.path.display(),
            m.prefix_len,
            m.artifact.num_tokens,
            m.artifact.model_name
        ));
    }
    out
}

fn format_json(matches: &[ScoredMatch]) -> Result<String> {
    let items: Vec<HashMap<String, serde_json::Value>> = matches
        .iter()
        .map(|m| {
            let mut map = HashMap::new();
            map.insert("path".to_string(), m.path.display().to_string().into());
            map.insert("prefix_tokens".to_string(), m.prefix_len.into());
            map.insert("total_tokens".to_string(), m.artifact.num_tokens.into());
            map.insert(
                "model".to_string(),
                m.artifact.model_name.clone().into(),
            );
            map
        })
        .collect();
    Ok(serde_json::to_string_pretty(&items)?)
}

pub fn run(args: SearchArgs) -> Result<()> {
    let dir = Path::new(&args.dir);
    if !dir.is_dir() {
        anyhow::bail!("--dir '{}' is not a directory", args.dir);
    }

    let files = collect_kv_files(dir);
    if files.is_empty() {
        anyhow::bail!("no .kv artifacts found in {}", args.dir);
    }

    let device = args
        .device
        .clone()
        .unwrap_or_else(|| common::pick_device().unwrap_or(candle_core::Device::Cpu));
    // NOTE: default search only needs the tokenizer; loading the full model is a
    // temporary simplification. If search latency matters, split out a
    // tokenizer-only loader later.
    let bundle = load_model_on(
        &args.model,
        &resolve_revision(args.revision.as_deref()),
        candle_core::DType::F16,
        device.clone(),
    )?;
    let query_tokens = encode(&bundle.tokenizer, &args.query, false)
        .with_context(|| "encoding query")?;

    let matches = rank_artifacts(&files, &args.model, &query_tokens);

    if matches.is_empty() {
        println!("no matching KV artifact found for model {}", args.model);
        return Ok(());
    }

    // NOTE: --tree support is added in Task 5. Default search prints results.
    match args.format.as_str() {
        "json" => println!("{}", format_json(&matches)?),
        _ => print!("{}", format_table(&matches)),
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_artifact(model_name: &str, tokens: Vec<u32>, num_tokens: usize) -> KVArtifact {
        KVArtifact {
            model_name: model_name.to_string(),
            num_layers: 1,
            num_tokens,
            dtype: "F16".to_string(),
            storage_dtype: Some("F16".to_string()),
            nbytes: 100,
            quantized: false,
            prompt: None,
            tokens: Some(tokens),
        }
    }

    #[test]
    fn longest_common_prefix_counts_matching_prefix() {
        assert_eq!(longest_common_prefix(&[1, 2, 3], &[1, 2, 4]), 2);
        assert_eq!(longest_common_prefix(&[1, 2], &[1, 2, 3]), 2);
        assert_eq!(longest_common_prefix(&[], &[1]), 0);
        assert_eq!(longest_common_prefix(&[1, 2, 3], &[1, 2, 3]), 3);
    }

    #[test]
    fn rank_artifacts_picks_longest_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.kv");
        let b = dir.path().join("b.kv");
        // Write dummy .kv files and sidecars.
        fs::write(&a, b"dummy").unwrap();
        fs::write(&b, b"dummy").unwrap();
        let meta_a = make_artifact("m", vec![1, 2, 3, 4], 4);
        let meta_b = make_artifact("m", vec![1, 2, 5, 6], 4);
        fs::write(a.with_extension("kv.json"), serde_json::to_string_pretty(&meta_a).unwrap()).unwrap();
        fs::write(b.with_extension("kv.json"), serde_json::to_string_pretty(&meta_b).unwrap()).unwrap();

        let query = [1u32, 2, 3, 99];
        let files = vec![a.clone(), b.clone()];
        let ranked = rank_artifacts(&files, "m", &query);
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].path, a);
        assert_eq!(ranked[0].prefix_len, 3);
        assert_eq!(ranked[1].path, b);
        assert_eq!(ranked[1].prefix_len, 2);
    }

    #[test]
    fn rank_artifacts_skips_missing_tokens() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.kv");
        fs::write(&a, b"dummy").unwrap();
        let mut meta = make_artifact("m", vec![1], 1);
        meta.tokens = None;
        fs::write(a.with_extension("kv.json"), serde_json::to_string_pretty(&meta).unwrap()).unwrap();

        let files = vec![a];
        let ranked = rank_artifacts(&files, "m", &[1, 2, 3]);
        assert!(ranked.is_empty());
    }
}
```

- [ ] **Step 2: Register module in `src/local/mod.rs`**

```rust
pub mod benchmark;
pub mod continuation;
pub mod diag;
pub mod infer;
pub mod kv_io;
pub mod kv_quant;
pub mod plot;
pub mod prefill;
pub mod quant;
pub mod search;
pub mod search_tree;
pub mod tokenize;
pub mod verify;
```

- [ ] **Step 3: Run tests**

```bash
cargo test --lib search::
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/local/search.rs src/local/mod.rs
git commit -m "feat(search): implement prefix search over KV artifact sidecars"
```

---

### Task 5: Implement tree generation in `src/local/search_tree.rs`

**Files:**
- Create: `src/local/search_tree.rs`
- Modify: `src/local/search.rs` (wire `--tree`)
- Test: `src/local/search_tree.rs` (compile-only smoke test)

- [ ] **Step 1: Create `src/local/search_tree.rs`**

```rust
use anyhow::{Context, Result};
use candle_core::Tensor;

use crate::cli::SearchArgs;
use crate::local::kv_io::load_kv;
use crate::local::search::ScoredMatch;
use crate::models::engine::ModelBundle;
use crate::models::CausalLM;

fn argmax_token(logits: &Tensor) -> Result<u32> {
    let logits = logits.squeeze(0)?.squeeze(0)?;
    let idx = logits.argmax(0)?.to_scalar::<u32>()?;
    Ok(idx)
}

fn snapshot_kv_cache(model: &dyn CausalLM) -> Result<crate::models::KVCache> {
    model.get_kv_cache()
}

fn restore_kv_cache(model: &mut dyn CausalLM, cache: crate::models::KVCache) -> Result<()> {
    model.set_kv_cache(cache)
}

/// Forward `tokens` starting at `offset` and return logits for the last token.
fn forward_tokens(
    model: &mut dyn CausalLM,
    device: &candle_core::Device,
    tokens: &[u32],
    offset: usize,
) -> Result<Tensor> {
    let input = Tensor::new(tokens, device)?.unsqueeze(0)?;
    model.forward(&input, offset)
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
    bundle: &ModelBundle,
    query_tokens: &[u32],
    best: Option<&ScoredMatch>,
) -> Result<()> {
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
                let logits = forward_tokens(
                    &mut *bundle.model,
                    &bundle.device,
                    suffix,
                    best.prefix_len,
                )?;
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
```

- [ ] **Step 2: Wire `--tree` in `src/local/search.rs`**

Change the tail of `run` in `src/local/search.rs`:

```rust
    if matches.is_empty() {
        println!("no matching KV artifact found for model {}", args.model);
        if args.tree.is_some() {
            // With --tree we still want to generate from scratch.
            return crate::local::search_tree::run(&args, &bundle, &query_tokens, None);
        }
        return Ok(());
    }

    if args.tree.is_some() {
        return crate::local::search_tree::run(&args, &bundle, &query_tokens, matches.first());
    }

    match args.format.as_str() {
        "json" => println!("{}", format_json(&matches)?),
        _ => print!("{}", format_table(&matches)),
    }

    Ok(())
}
```

- [ ] **Step 3: Run compile check**

```bash
cargo check
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add src/local/search_tree.rs src/local/search.rs
git commit -m "feat(search): add --tree candidate generation from shared KV prefix"
```

---

### Task 6: Update `verify.rs` to store prompt and tokens

**Files:**
- Modify: `src/local/verify.rs:43-49`

- [ ] **Step 1: Pass prompt/tokens to save helper**

Change:

```rust
let cache = bundle.model.get_kv_cache()?;
let kv_path = resolve_output_path("verify", &args.model, &args.kv_path, "kv")?;
if let Some(parent) = kv_path.parent() {
    fs::create_dir_all(parent)?;
}
let prompt_text = context.clone();
let ctx_tokens_for_save = ctx_tokens.clone();
let art = kv_io::save_kv_with_meta(&cache, &kv_path, &args.model, Some(prompt_text), Some(ctx_tokens_for_save))?;
```

- [ ] **Step 2: Verify compile and existing tests**

```bash
cargo check
cargo test --lib verify
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/local/verify.rs
git commit -m "feat(verify): store source prompt and tokens in artifact sidecar"
```

---

### Task 7: Update `benchmark.rs` to store prompt and tokens

**Files:**
- Modify: `src/local/benchmark.rs`

- [ ] **Step 1: Locate where KV is saved and extend the call**

Find the call to `kv_io::save_kv` and replace with `kv_io::save_kv_with_meta`, passing the current context text and token IDs as `Some(prompt)` and `Some(tokens)`.

- [ ] **Step 2: Verify compile**

```bash
cargo check
```

- [ ] **Step 3: Commit**

```bash
git add src/local/benchmark.rs
git commit -m "feat(benchmark): store source prompt and tokens in artifact sidecar"
```

---

### Task 8: Update `quant.rs` to copy/preserve prompt and tokens

**Files:**
- Modify: `src/local/quant.rs`

- [ ] **Step 1: Read input sidecar and copy prompt/tokens**

After loading the input artifact (or generating it), read `artifact.prompt` and `artifact.tokens`. Pass them to `save_quantized_kv_with_meta`.

If the input is generated on the fly, use the `context` text and its token IDs.

- [ ] **Step 2: Verify compile**

```bash
cargo check
```

- [ ] **Step 3: Commit**

```bash
git add src/local/quant.rs
git commit -m "feat(quant): preserve prompt and tokens in quantized sidecar"
```

---

### Task 9: Full test and lint pass

- [ ] **Step 1: Run unit tests**

```bash
cargo test --workspace
```

Expected: PASS.

- [ ] **Step 2: Run formatting check**

```bash
cargo fmt -- --check
```

Expected: no changes needed (or run `cargo fmt` if needed).

- [ ] **Step 3: Run clippy**

```bash
cargo clippy --all-targets -- -D warnings
```

Expected: PASS.

- [ ] **Step 4: Build release**

```bash
cargo build --release
```

Expected: binary at `target/release/kvcdn`.

- [ ] **Step 5: Manual smoke test (optional but recommended)**

```bash
./target/release/kvcdn verify --model Qwen/Qwen3-0.6B --context-file /tmp/small.txt --question " Q: What? A:"
./target/release/kvcdn search --model Qwen/Qwen3-0.6B --dir ~/.local/share/kvcdn/verify --query "Q: What? A:"
```

- [ ] **Step 6: Commit final fixes**

```bash
git add .
git commit -m "test(search): verify search and tree generation compile and pass tests"
```

---

## Plan self-review

1. **Spec coverage:**
   - `kvcdn search` default prefix search → Task 4.
   - `--tree N` candidate generation → Task 5.
   - Sidecar `prompt`/`tokens` extension → Task 1.
   - `verify`/`benchmark`/`quant` updates → Tasks 6-8.
   - Error handling (missing dir, no match, missing tokens) → Task 4.
   - Tests → embedded in Tasks 1, 4, 9.

2. **Placeholder scan:** No TBD/TODO; all code snippets are concrete.

3. **Type consistency:** `save_kv_with_meta` and `save_quantized_kv_with_meta` signatures match usage in Tasks 6-8. `KVArtifact` field names (`prompt`, `tokens`) are consistent across the plan.
