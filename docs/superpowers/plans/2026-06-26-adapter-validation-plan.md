# Adapter Validation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable validation harness that proves DeepSeek V2/V3, Qwen2, Phi-3, and Gemma 3 adapters produce token-exact output through reference, cached, and quantized KV-cache paths.

**Architecture:** Add a `src/core/validation.rs` library that performs three greedy generation phases per model, an integration test in `tests/adapter_validation.rs` driven by an environment variable, a shared fixture prompt, and a wrapper script. Documentation updates follow once the harness passes.

**Tech Stack:** Rust, Candle, safetensors, tokenizers, cargo test, bash.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/core/validation.rs` | Library helper: greedy reference/cache/quantized validation for a loaded model. |
| `src/core/mod.rs` | Register the new `validation` submodule. |
| `tests/fixtures/context.txt` | Shared prompt (≥ 64 tokens) used by the integration test. |
| `tests/adapter_validation.rs` | Integration test that loads each model in `KVCDN_VALIDATE_MODELS` and runs the harness. |
| `scripts/validate-adapters.sh` | Maintainer wrapper that exports the four target models and runs the integration test. |
| `README.md` | Update "Currently supported architectures" list. |
| `AGENTS.md` | Add note on running `scripts/validate-adapters.sh` for new adapters. |

---

## Task 1: Register the `validation` submodule

**Files:**
- Modify: `src/core/mod.rs`

- [ ] **Step 1: Add `pub mod validation;`**

```rust
pub mod common;
pub mod crypto;
pub mod output;
pub mod validation;
```

- [ ] **Step 2: Check it compiles**

Run: `cargo check`
Expected: succeeds (module is empty for now).

---

## Task 2: Create `src/core/validation.rs` with data types and reference generation

**Files:**
- Create: `src/core/validation.rs`

- [ ] **Step 1: Write the file header, imports, and `ValidationReport`**

```rust
use std::collections::HashSet;

use anyhow::{Context, Result};
use candle_core::Tensor;
use tokenizers::Tokenizer;

use crate::local::continuation::generate;
use crate::local::kv_io;
use crate::local::kv_quant::{dequantize_tensor, quantize_tensor};
use crate::local::prefill::prefill;
use crate::local::tokenize::encode;
use crate::models::CausalLM;

#[derive(Debug, Clone)]
pub struct PhaseResult {
    pub phase: &'static str,
    pub tokens: Vec<u32>,
    pub duration: std::time::Duration,
}

#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub model: String,
    pub passed: bool,
    pub max_tokens: usize,
    pub phases: Vec<PhaseResult>,
    pub mismatches: Vec<(String, usize, u32, u32)>, // phase, index, expected, actual
    pub error: Option<String>,
}

impl ValidationReport {
    fn new(model: String, max_tokens: usize) -> Self {
        Self {
            model,
            passed: false,
            max_tokens,
            phases: Vec::new(),
            mismatches: Vec::new(),
            error: None,
        }
    }
}
```

- [ ] **Step 2: Add `validate_adapter` skeleton and reference phase**

```rust
/// Run reference, cache, and quantized cache validation for `model`.
///
/// `context` is prefilled; `question` is appended and used for continuation.
/// All generation is greedy (argmax). Returns a report describing any mismatch.
pub fn validate_adapter(
    model: &mut dyn CausalLM,
    tokenizer: &Tokenizer,
    model_name: &str,
    context: &str,
    question: &str,
    max_tokens: usize,
) -> Result<ValidationReport> {
    let mut report = ValidationReport::new(model_name.to_string(), max_tokens);

    let ctx_tokens = encode(tokenizer, context, true)
        .with_context(|| "encoding validation context")?;
    let q_tokens = encode(tokenizer, question, false)
        .with_context(|| "encoding validation question")?;
    let full_tokens = [ctx_tokens.clone(), q_tokens.clone()].concat();

    // Reference phase: prefill everything from scratch, then continue.
    let start = std::time::Instant::now();
    model.clear_kv_cache();
    let reference = generate_from_tokens(model, &full_tokens, 0, max_tokens)?;
    report.phases.push(PhaseResult {
        phase: "reference",
        tokens: reference.clone(),
        duration: start.elapsed(),
    });

    // Cache phase and quantized phase will be added in later tasks.
    report.passed = true;
    Ok(report)
}

fn generate_from_tokens(
    model: &mut dyn CausalLM,
    tokens: &[u32],
    offset: usize,
    n: usize,
) -> Result<Vec<u32>> {
    let bundle_tokens = tokens.to_vec();
    let mut bundle = crate::models::engine::ModelBundle {
        tokenizer: tokenizers::Tokenizer::from_bytes(b"{}")?,
        model: Box::new(StubModel),
        device: candle_core::Device::Cpu,
    };
    // TODO: remove stub; we will refactor to avoid ModelBundle in Task 3.
    generate(&mut bundle, &bundle_tokens, offset, n)
}

struct StubModel;
impl CausalLM for StubModel {
    fn forward(&mut self, _input: &Tensor, _offset: usize) -> anyhow::Result<Tensor> {
        unimplemented!()
    }
    fn clear_kv_cache(&mut self) {}
    fn get_kv_cache(&self) -> anyhow::Result<crate::models::KVCache> {
        Ok(Vec::new())
    }
    fn set_kv_cache(&mut self, _cache: crate::models::KVCache) -> anyhow::Result<()> {
        Ok(())
    }
    fn config(&self) -> crate::models::ModelConfig {
        unimplemented!()
    }
}
```

> Note: The stub is intentionally temporary. Task 3 refactors `continuation::generate` to accept `&mut dyn CausalLM` directly so the harness does not need a `ModelBundle`.

- [ ] **Step 3: Check it compiles**

Run: `cargo check`
Expected: succeeds with stub code.

---

## Task 3: Decouple greedy generation from `ModelBundle`

**Files:**
- Modify: `src/local/continuation.rs`

The harness needs to generate tokens using only the model trait, not the full `ModelBundle`. `verify.rs` and `quant.rs` still use `generate(&mut bundle, ...)`, so we keep a backwards-compatible wrapper.

- [ ] **Step 1: Add `generate_with_model` and update `generate` to delegate**

```rust
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
/// `prompt_tokens` are the tokens to process first; the first generated token follows them.
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
    generate_with_model(&mut *bundle.model, &bundle.device, prompt_tokens, start_offset, n)
}
```

- [ ] **Step 2: Run unit tests for `continuation`**

Run: `cargo test --lib local::continuation`
Expected: all tests pass.

---

## Task 4: Implement the cache phase in `validation.rs`

**Files:**
- Modify: `src/core/validation.rs`

- [ ] **Step 1: Replace the temporary `generate_from_tokens` stub with a real helper**

```rust
fn generate_from_tokens(
    model: &mut dyn CausalLM,
    device: &candle_core::Device,
    tokens: &[u32],
    offset: usize,
    n: usize,
) -> Result<Vec<u32>> {
    crate::local::continuation::generate_with_model(model, device, tokens, offset, n)
}
```

Remove the `StubModel` struct and its `CausalLM` implementation.

- [ ] **Step 2: Add a helper to compare token sequences and record mismatches**

```rust
fn compare_tokens(
    report: &mut ValidationReport,
    phase: &'static str,
    expected: &[u32],
    actual: &[u32],
) -> bool {
    let mut matched = true;
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        if e != a {
            report.mismatches.push((phase.to_string(), i, *e, *a));
            matched = false;
        }
    }
    if expected.len() != actual.len() {
        report.mismatches.push((
            phase.to_string(),
            expected.len().min(actual.len()),
            u32::MAX,
            u32::MAX,
        ));
        matched = false;
    }
    matched
}
```

- [ ] **Step 3: Add the cache validation phase to `validate_adapter`**

After the reference phase, insert:

```rust
    // Cache phase: prefill context, save/load KV cache, then continue.
    let start = std::time::Instant::now();
    model.clear_kv_cache();
    let cache = prefill_model(model, &ctx_tokens)?;
    let kv_path = std::env::temp_dir().join(format!("kvcdn-validate-{}.kv", model_name.replace('/', "-")));
    let _art = kv_io::save_kv(&cache, &kv_path, model_name)
        .with_context(|| format!("saving KV cache to {}", kv_path.display()))?;
    let (loaded_cache, _) = kv_io::load_kv(&kv_path, device)
        .with_context(|| format!("loading KV cache from {}", kv_path.display()))?;
    model.set_kv_cache(loaded_cache)
        .with_context(|| "restoring loaded KV cache")?;
    let cache_tokens = generate_from_tokens(model, device, &q_tokens, ctx_tokens.len(), max_tokens)?;
    report.phases.push(PhaseResult {
        phase: "cache",
        tokens: cache_tokens.clone(),
        duration: start.elapsed(),
    });
    let cache_ok = compare_tokens(&mut report, "cache", &reference, &cache_tokens);
```

- [ ] **Step 4: Add `prefill_model` helper**

```rust
fn prefill_model(model: &mut dyn CausalLM, tokens: &[u32]) -> Result<crate::models::KVCache> {
    let input = Tensor::new(tokens, &candle_core::Device::Cpu)?.unsqueeze(0)?;
    let _ = model.forward(&input, 0)?;
    Ok(model.get_kv_cache()?)
}
```

> Note: `prefill_model` duplicates the small `prefill::prefill` helper because `prefill` takes a `ModelBundle`. This keeps the harness decoupled from `ModelBundle`.

- [ ] **Step 5: Update `validate_adapter` signature to accept a device**

```rust
pub fn validate_adapter(
    model: &mut dyn CausalLM,
    device: &candle_core::Device,
    tokenizer: &Tokenizer,
    model_name: &str,
    context: &str,
    question: &str,
    max_tokens: usize,
) -> Result<ValidationReport> {
```

- [ ] **Step 6: Check it compiles**

Run: `cargo check`
Expected: succeeds.

---

## Task 5: Implement the quantized phase in `validation.rs`

**Files:**
- Modify: `src/core/validation.rs`

- [ ] **Step 1: Add the quantized validation phase after the cache phase**

```rust
    // Quantized phase: prefill context, quantize KV cache, save/load/dequantize, then continue.
    let start = std::time::Instant::now();
    model.clear_kv_cache();
    let cache = prefill_model(model, &ctx_tokens)?;
    let quantized = crate::local::kv_quant::quantize_kv(&cache)
        .with_context(|| "quantizing KV cache")?;
    let q_path = std::env::temp_dir().join(format!("kvcdn-validate-{}-q8.kv", model_name.replace('/', "-")));
    let _q_art = kv_io::save_quantized_kv(&quantized, &q_path, model_name, kv_io::QuantDtype::Int8)
        .with_context(|| format!("saving quantized KV cache to {}", q_path.display()))?;
    let (dequant_cache, _) = kv_io::load_kv(&q_path, device)
        .with_context(|| format!("loading quantized KV cache from {}", q_path.display()))?;
    model.set_kv_cache(dequant_cache)
        .with_context(|| "restoring dequantized KV cache")?;
    let quant_tokens = generate_from_tokens(model, device, &q_tokens, ctx_tokens.len(), max_tokens)?;
    report.phases.push(PhaseResult {
        phase: "quantized",
        tokens: quant_tokens.clone(),
        duration: start.elapsed(),
    });
    let quant_ok = compare_tokens(&mut report, "quantized", &reference, &quant_tokens);
```

- [ ] **Step 2: Compute final `passed` flag**

Replace `report.passed = true;` with:

```rust
    report.passed = cache_ok && quant_ok;
    Ok(report)
```

- [ ] **Step 3: Check it compiles**

Run: `cargo check`
Expected: succeeds.

---

## Task 6: Add a shared fixture prompt

**Files:**
- Create: `tests/fixtures/context.txt`

- [ ] **Step 1: Write a ≥64-token prompt**

```text
Retrieval-augmented generation systems reuse long documents across many queries. Prefilling the same document repeatedly wastes GPU time and increases latency. KV caches capture the result of the expensive prefill phase so subsequent questions can skip it. This fixture validates that a model adapter can save, load, and quantize those caches without changing the generated tokens.

Question: What problem do KV caches solve?
Answer:
```

- [ ] **Step 2: Verify token count with a quick ad-hoc test**

Run: `wc -w tests/fixtures/context.txt`
Expected: word count ≥ 64 (token count will be similar).

---

## Task 7: Create the integration test

**Files:**
- Create: `tests/adapter_validation.rs`

- [ ] **Step 1: Write the integration test**

```rust
use std::fs;

use candle_core::DType;
use kvcdn::core::validation::validate_adapter;
use kvcdn::models::engine::{load_model, resolve_revision};

fn models_to_validate() -> Vec<String> {
    std::env::var("KVCDN_VALIDATE_MODELS")
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn fixture_text() -> String {
    fs::read_to_string("tests/fixtures/context.txt").expect("reading fixture")
}

#[test]
fn validate_configured_adapters() {
    let models = models_to_validate();
    if models.is_empty() {
        eprintln!("KVCDN_VALIDATE_MODELS is unset; skipping adapter validation test");
        return;
    }

    let fixture = fixture_text();
    let parts: Vec<&str> = fixture.splitn(2, "Answer:").collect();
    let (context, question) = if parts.len() == 2 {
        (parts[0].trim(), "Answer:")
    } else {
        (fixture.as_str(), "What is the main point?")
    };

    let max_tokens: usize = std::env::var("KVCDN_VALIDATE_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    let mut failures = Vec::new();
    for model_name in models {
        eprintln!("\n=== validating {model_name} ===");
        let revision = resolve_revision(None);
        let mut bundle = match load_model(&model_name, &revision, DType::F16) {
            Ok(b) => b,
            Err(e) => {
                failures.push(format!("{model_name}: load failed: {e}"));
                continue;
            }
        };

        let report = match validate_adapter(
            &mut *bundle.model,
            &bundle.device,
            &bundle.tokenizer,
            &model_name,
            context,
            question,
            max_tokens,
        ) {
            Ok(r) => r,
            Err(e) => {
                failures.push(format!("{model_name}: validation error: {e}"));
                continue;
            }
        };

        eprintln!(
            "{model_name}: reference={} tokens in {:?}, cache={} in {:?}, quant={} in {:?}",
            report.phases[0].tokens.len(),
            report.phases[0].duration,
            report.phases[1].tokens.len(),
            report.phases[1].duration,
            report.phases[2].tokens.len(),
            report.phases[2].duration,
        );

        if !report.passed {
            failures.push(format!("{model_name}: mismatch at {:?}", report.mismatches));
        }
    }

    assert!(failures.is_empty(), "adapter validation failed:\n{}", failures.join("\n"));
}
```

- [ ] **Step 2: Check it compiles**

Run: `cargo test --test adapter_validation --no-run`
Expected: test binary builds.

---

## Task 8: Create the validation script

**Files:**
- Create: `scripts/validate-adapters.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Validate the adapters targeted for promotion to officially supported.
# Requires HF_TOKEN for gated checkpoints (e.g., some Gemma or Phi variants).
export KVCDN_VALIDATE_MODELS="\
deepseek-ai/DeepSeek-V2-Lite-Chat,Qwen/Qwen2-0.5B,microsoft/Phi-3-mini-4k-instruct,google/gemma-3-1b-it"

export KVCDN_VALIDATE_MAX_TOKENS="${KVCDN_VALIDATE_MAX_TOKENS:-16}"

cargo test --test adapter_validation --release
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x scripts/validate-adapters.sh`

---

## Task 9: Update `README.md`

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Expand the supported-architectures list**

In the "Currently supported architectures" section, change:

```markdown
Currently supported architectures:

- `LlamaForCausalLM` (and Llama-3.1/3.2 fine-tunes)
- `MistralForCausalLM` / `MixtralForCausalLM`
- `YiForCausalLM`
- `Qwen3ForCausalLM`
- `GemmaForCausalLM` (and Gemma 2 fine-tunes)
```

to:

```markdown
Currently supported architectures:

- `LlamaForCausalLM` (and Llama-3.1/3.2 fine-tunes)
- `MistralForCausalLM` / `MixtralForCausalLM`
- `YiForCausalLM`
- `DeepseekV2ForCausalLM` / `DeepseekV3ForCausalLM`
- `Qwen2ForCausalLM`
- `Qwen3ForCausalLM`
- `Phi3ForCausalLM`
- `GemmaForCausalLM` (and Gemma 2 fine-tunes)
- `Gemma3ForCausalLM`
```

- [ ] **Step 2: Remove the four families from "Known unsupported"**

Remove `Phi / Phi-3 / Phi-4` from the known-unsupported list if present. (Gemma3, Qwen2, and DeepSeek are not in that list.)

---

## Task 10: Update `AGENTS.md`

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Add validation note under "Things that are easy to miss"**

Append:

```markdown
- When adding or updating a model adapter, run `scripts/validate-adapters.sh` to prove token-exact verify and quantized verify against the four currently tracked reference checkpoints. Add new architectures to the `KVCDN_VALIDATE_MODELS` list in the script once they pass.
```

---

## Task 11: Run local verification

- [ ] **Step 1: Run unit tests**

Run: `cargo test --workspace`
Expected: all unit tests pass.

- [ ] **Step 2: Run the harness against one small model**

Run:
```bash
KVCDN_VALIDATE_MODELS="Qwen/Qwen2-0.5B" KVCDN_VALIDATE_MAX_TOKENS=8 cargo test --test adapter_validation --release
```
Expected: test passes and prints phase timings.

- [ ] **Step 3: Run formatter and clippy**

Run:
```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```
Expected: both pass.

---

## Task 12: Validate all four target adapters

- [ ] **Step 1: Run the full validation script**

Run: `./scripts/validate-adapters.sh`
Expected: all four adapters pass. If any fail, debug the adapter and iterate; do not update README until all four pass.

- [ ] **Step 2: Capture results**

Save the terminal output or add a comment to the PR noting which models passed.

---

## Self-Review Checklist

- [ ] **Spec coverage:** Every section of `docs/superpowers/specs/2026-06-26-adapter-validation-design.md` maps to a task above.
- [ ] **Placeholder scan:** No "TBD", "TODO", or vague steps remain.
- [ ] **Type consistency:** `validate_adapter` uses `&mut dyn CausalLM` and `&candle_core::Device` consistently across tasks.
- [ ] **No new unsupported models:** Tasks only validate adapters already wired in `src/models/registry.rs`.
