# Adapter Validation Harness Design

## Goal

Promote Qwen2 and Phi-3 from "wired in the registry but undocumented" to officially supported architectures by proving that each adapter passes token-exact `kvcdn verify` and token-exact quantized `kvcdn quant --verify`.

DeepSeek V2/V3 and Gemma 3 were also investigated; they are wired in the registry but could not meet the quantized-verify acceptance criterion in this environment. See "Results" below.

## Background

`src/models/registry.rs` dispatches 26 architectures, but the README only lists five as supported (Llama, Mistral, Yi, Qwen3, Gemma). The remaining adapters are implemented but have not been validated against real checkpoints. This work validates the highest-value subset of those adapters and leaves behind a reusable process for the rest.

## Scope

In scope:
- DeepSeekV2ForCausalLM / DeepseekV3ForCausalLM
- Qwen2ForCausalLM
- Phi3ForCausalLM
- Gemma3ForCausalLM

Out of scope (for this iteration):
- Adding brand-new architectures not already wired in `registry.rs`.
- State-space models (Mamba, RWKV) unless the harness proves trivially reusable for them later.
- Inference-server integration.

## Acceptance Criteria

For each target adapter, running the validation harness must:
1. Produce identical token IDs when generating from a fresh prefill (reference run).
2. Produce identical token IDs when saving and reloading the KV cache (cache run).
3. Produce identical token IDs when quantizing the KV cache to int8, then reloading it (quantized run).

After validation, the README "Currently supported architectures" list is updated to include the four new families, and `AGENTS.md` documents how to run the harness when adding a new adapter.

## Components

### 1. `src/core/validation.rs`

A small library module that performs the three validation phases for a loaded model:

```rust
pub fn validate_adapter(
    model: &mut Box<dyn CausalLM>,
    tokenizer: &Tokenizer,
    context: &str,
    question: &str,
    max_tokens: usize,
) -> Result<ValidationReport>
```

The function:
1. Tokenizes `context + question`.
2. Runs a reference greedy generation of `max_tokens` tokens (argmax over logits).
3. Runs prefill, captures KV cache, serializes/deserializes via `kv_io`, and continues greedy generation.
4. Runs prefill, quantizes the cache via `local::kv_quant::quantize_kv_cache` to symmetric int8, dequantizes/loads via `local::kv_quant::dequantize_kv_cache`, and continues greedy generation.
5. Compares token IDs from the cache and quantized runs against the reference.

`ValidationReport` contains per-phase timing and a `pass: bool` flag.

### 2. `tests/adapter_validation.rs`

An integration test that:
- Reads a comma-separated model list from `KVCDN_VALIDATE_MODELS`.
- Loads each model using `engine::load_model`.
- Calls `validation::validate_adapter` with the shared fixture.
- Fails the test if any model reports `pass: false`.

Example invocation:

```bash
KVCDN_VALIDATE_MODELS="deepseek-ai/DeepSeek-V2-Chat-0625,Qwen/Qwen2-0.5B,microsoft/Phi-3-mini-4k-instruct,google/gemma-3-1b-it" \
  cargo test --test adapter_validation
```

The exact checkpoint identifiers are recorded in `scripts/validate-adapters.sh`, not hard-coded in the test.

### 3. `tests/fixtures/context.txt`

A short, architecture-neutral prompt used for all adapters. It should be long enough to exercise the cache (>= 64 tokens) but short enough to run quickly on CPU.

### 4. `scripts/validate-adapters.sh`

A thin wrapper that exports the four target models and runs the integration test:

```bash
#!/usr/bin/env bash
set -euo pipefail
export KVCDN_VALIDATE_MODELS="\
deepseek-ai/DeepSeek-V2-Lite-Chat,Qwen/Qwen2-0.5B,microsoft/Phi-3-mini-4k-instruct,google/gemma-3-1b-it"
cargo test --test adapter_validation --release
```

The script is the documented way for maintainers to validate adapters locally.

## Data Flow

For each model in the list:

1. `engine::load_model` downloads tokenizer + config + safetensors and returns `ModelBundle`.
2. `validation::validate_adapter` tokenizes the fixture.
3. Reference phase:
   - `model.forward(&tokens, 0)` through the full context.
   - Greedy-generate `max_tokens` tokens (argmax), recording token IDs.
4. Cache phase:
   - `model.clear_kv_cache()`.
   - `model.forward(&tokens, 0)` through the full context.
   - `model.get_kv_cache()`.
   - Serialize to a temp file via `kv_io::write_kv_cache`.
   - Deserialize via `kv_io::read_kv_cache` and `model.set_kv_cache()`.
   - Continue generation for `max_tokens` tokens.
   - Compare token IDs to reference.
5. Quantized phase:
   - `model.clear_kv_cache()`.
   - `model.forward(&tokens, 0)` through the full context.
   - Quantize the captured KV cache to symmetric int8 via `local::kv_quant::quantize_kv_cache`.
   - Dequantize/load via `local::kv_quant::dequantize_kv_cache` and set on the model.
   - Continue generation for `max_tokens` tokens.
   - Compare token IDs to reference.

## Error Handling

- Any `Result::Err` in a phase fails the model and prints the phase name plus the error.
- A token mismatch fails the model and prints the first divergent token index plus the surrounding token IDs.
- `MAX_TOKENS` defaults to 16 and is overridable via env var to keep CPU runs fast.
- If `HF_TOKEN` is required for a gated checkpoint, the test fails with the underlying `hf-hub` error.

## Testing

The integration test is the primary test artifact. It does not run in CI by default because it downloads multi-gigabyte checkpoints. CI can opt in by setting `KVCDN_VALIDATE_MODELS` on a scheduled or manually-triggered runner.

Unit tests added as a byproduct:
- Tests for any small pure helper functions introduced in `src/core/validation.rs` (e.g., token-ID comparison utilities).
- `registry.rs` already has tests for loader wiring; those remain.

## Documentation Changes

- Update README.md "Currently supported architectures" to add DeepSeek V2/V3, Qwen2, Phi-3, and Gemma 3.
- Add a one-paragraph note to `AGENTS.md` under "Daily commands" or "Things that are easy to miss" describing `scripts/validate-adapters.sh`.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Adapter has subtle attention/QKV layout bug that only shows on longer contexts. | Use a fixture that exercises at least 64 tokens; if failures appear, extend fixture length. |
| Quantization noise masks a real cache bug. | Compare quantized run against reference, not against cache run. |
| Gated checkpoints require authentication. | Document required `HF_TOKEN` in script comments and `AGENTS.md`. |
| CPU run is too slow for large models. | Choose the smallest available checkpoint for each family; the harness does not require full-size weights. |

## Open Questions

None remaining after design approval.
