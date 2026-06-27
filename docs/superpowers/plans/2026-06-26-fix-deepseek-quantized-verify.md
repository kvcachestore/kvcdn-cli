# Fix DeepSeek-V2 Quantized Verify Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `deepseek-ai/DeepSeek-V2-Lite-Chat` pass the quantized phase of `scripts/validate-adapters.sh` by improving how the int8 quantizer handles rank-3 MLA compressed-KV tensors.

**Architecture:** DeepSeek-V2 uses Multi-head Latent Attention (MLA). The captured "KV cache" is `(compressed_kv, k_pe)` where `compressed_kv` has shape `(batch, seq_len, kv_lora_rank)` instead of the usual rank-4 GQA `(batch, heads, seq_len, head_dim)`. The current `kv_quant::quantize_tensor` computes a single symmetric int8 scale across the entire tensor. For MLA this wastes precision: a single outlier in one latent dimension collapses resolution for all other dimensions. The fix is to add per-token (or per-seq-position) symmetric int8 quantization for rank-3 KV tensors, while keeping per-tensor quantization for rank-4 caches so existing validated adapters (Qwen2, Phi-3) are unaffected.

**Tech Stack:** Rust, Candle, safetensors.

---

## File map

- `src/local/kv_quant.rs` — quantizer implementation and tests.
- `src/core/validation.rs` — validation harness (will reuse improved quantizer unchanged).
- `tests/adapter_validation.rs` — integration test (already passes the quantizer output through).
- `scripts/validate-adapters.sh` — add DeepSeek back to the tracked model list once it passes.
- `README.md` — move DeepSeek to fully supported once it passes.

---

### Task 1: Add failing test for per-token quantization of rank-3 tensors

**Files:**
- Modify: `src/local/kv_quant.rs`

- [ ] **Step 1: Add a unit test that shows per-tensor quantization is too coarse for a rank-3 tensor with per-token outliers.**

```rust
#[test]
fn per_tensor_quantization_wastes_precision_for_rank3_outliers() -> Result<()> {
    let dev = Device::Cpu;
    // Simulate an MLA compressed_kv: (1, seq, latent). Most tokens have small values,
    // but one token has a large outlier latent vector.
    let mut data = vec![0f32; 4 * 512];
    for i in 0..512 {
        data[2 * 512 + i] = 100.0; // token 2 is an outlier
    }
    let t = Tensor::from_vec(data, (1, 4, 512), &dev)?;
    let (scale, q) = quantize_tensor(&t)?;
    let dq = dequantize_tensor(&scale, &q, DType::F32)?;
    let err = (t.to_dtype(DType::F32)? - dq)?.abs()?.max_all()?.to_scalar::<f32>()?;
    // The max error is currently large because the outlier dominates the scale.
    // We want to demonstrate that a per-token quantizer would do better.
    assert!(err > 1.0);
    Ok(())
}
```

- [ ] **Step 2: Run the test and confirm it passes (it documents current behavior).**

Run:
```bash
cargo test --lib local::kv_quant::tests::per_tensor_quantization_wastes_precision_for_rank3_outliers -- --nocapture
```

Expected: PASS.

- [ ] **Step 3: Commit.**

```bash
git add src/local/kv_quant.rs
git commit -m "test(kv_quant): document rank-3 per-tensor quantization precision loss"
```

---

### Task 2: Implement per-token quantization for rank-3 KV tensors

**Files:**
- Modify: `src/local/kv_quant.rs`
- Modify: `src/local/kv_io.rs` (metadata must record quantization axis)
- Modify: `src/local/quant.rs` (max_quant_error must use matching dequant path)

- [ ] **Step 1: Add `Quantization` enum that records whether a tensor is per-tensor or per-token.**

In `src/local/kv_quant.rs`, replace the free `(scale, q)` tuple with a structured type:

```rust
#[derive(Debug, Clone)]
pub enum Quantization {
    /// Single scale for the whole tensor (existing behavior, used for rank-4 GQA caches).
    PerTensor { scale: Tensor, q: Tensor },
    /// One scale per sequence position (used for rank-3 MLA compressed KV).
    PerToken { scales: Tensor, q: Tensor },
}
```

- [ ] **Step 2: Implement `quantize_tensor_per_token`.**

```rust
fn quantize_tensor_per_token(t: &Tensor) -> Result<Quantization> {
    let f32_t = t.to_dtype(DType::F32)?;
    // t is (batch, seq, latent). Compute max over the latent dim for each token.
    let abs = f32_t.abs()?;
    // Reduce along last axis; keep dims so broadcasting works later.
    let max = abs.max(D::Minus1)?.unsqueeze(D::Minus1)?;
    let scale = max.clamp(1e-12f64, f64::MAX)?;

    let normalized = (f32_t.broadcast_div(&scale)? * 127.0f64)?;
    let q = (normalized.round()?.clamp(-127.0f64, 127.0f64)? + 128.0f64)?;
    let q = q
        .to_dtype(DType::U8)
        .context("converting per-token quantized values to U8")?;

    Ok(Quantization::PerToken { scales: scale, q })
}
```

- [ ] **Step 3: Update `quantize_tensor` to dispatch on rank.**

```rust
pub fn quantize_tensor(t: &Tensor) -> Result<Quantization> {
    match t.rank() {
        3 => quantize_tensor_per_token(t),
        _ => {
            let f32_t = t.to_dtype(DType::F32)?;
            let abs = f32_t.abs()?;
            let max = abs.max_all()?;
            let scale = max.clamp(1e-12f64, f64::MAX)?;

            let normalized = (f32_t.broadcast_div(&scale)? * 127.0f64)?;
            let q = (normalized.round()?.clamp(-127.0f64, 127.0f64)? + 128.0f64)?;
            let q = q
                .to_dtype(DType::U8)
                .context("converting quantized values to U8")?;

            Ok(Quantization::PerTensor { scale, q })
        }
    }
}
```

- [ ] **Step 4: Update `dequantize_tensor` to accept `Quantization`.**

```rust
pub fn dequantize_tensor(q: &Quantization, target_dtype: DType) -> Result<Tensor> {
    match q {
        Quantization::PerTensor { scale, q } => dequantize_per_tensor(scale, q, target_dtype),
        Quantization::PerToken { scales, q } => {
            let centered = (q.to_dtype(DType::F32)? - 128.0f64)?;
            let x = (centered.broadcast_mul(scales)? / 127.0f64)?;
            x.to_dtype(target_dtype)
                .with_context(|| format!("dequantizing per-token tensor to {target_dtype:?}"))
        }
    }
}
```

Extract the existing per-tensor logic into `dequantize_per_tensor` to avoid duplication.

- [ ] **Step 5: Update `quantize_kv` to return `Vec<(Quantization, Quantization)>` and `max_quant_error` to use `dequantize_tensor`.**

```rust
pub fn quantize_kv(cache: &KVCache) -> Result<Vec<(Quantization, Quantization)>> {
    cache
        .iter()
        .enumerate()
        .map(|(i, (k, v))| {
            let k_q = quantize_tensor(k)
                .with_context(|| format!("quantizing key tensor for layer {i}"))?;
            let v_q = quantize_tensor(v)
                .with_context(|| format!("quantizing value tensor for layer {i}"))?;
            Ok((k_q, v_q))
        })
        .collect()
}

pub fn max_quant_error(cache: &KVCache, target_dtype: DType) -> Result<f32> {
    let mut max_err: f32 = 0.0;
    for (k, v) in cache {
        let k_q = quantize_tensor(k)?;
        let k_dq = dequantize_tensor(&k_q, target_dtype)?;
        let err = (k.to_dtype(DType::F32)? - k_dq.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        max_err = max_err.max(err);

        let v_q = quantize_tensor(v)?;
        let v_dq = dequantize_tensor(&v_q, target_dtype)?;
        let err = (v.to_dtype(DType::F32)? - v_dq.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        max_err = max_err.max(err);
    }
    Ok(max_err)
}
```

- [ ] **Step 6: Update `save_quantized_kv` and `load_kv` to store per-token scales.**

In `src/local/kv_io.rs`, change `save_quantized_kv` to save the scales tensor shape in the metadata or encode the quantization kind into tensor names. The simplest approach: if `scales` has more than zero elements, save it as `k_{i}_s` with shape `(seq_len, 1)`. The loader can infer per-token vs per-tensor by checking the scale tensor rank: rank-0 scalar means per-tensor, rank-2 `(seq, 1)` means per-token.

Modify `save_quantized_kv` to accept `&[(Quantization, Quantization)]`. Because `Quantization` is in `kv_quant`, `kv_io` can depend on it. Replace the tuple type in the signature.

```rust
pub fn save_quantized_kv<P: AsRef<Path>>(
    layers: &[(Quantization, Quantization)],
    path: P,
    model_name: &str,
    target_dtype: QuantDtype,
) -> Result<KVArtifact> {
    // ... existing setup ...
    for (i, (k_q, v_q)) in layers.iter().enumerate() {
        match k_q {
            Quantization::PerTensor { scale, q } => {
                tensors.insert(format!("k_{i}_q"), q.clone());
                tensors.insert(format!("k_{i}_s"), scale.clone());
            }
            Quantization::PerToken { scales, q } => {
                tensors.insert(format!("k_{i}_q"), q.clone());
                tensors.insert(format!("k_{i}_s"), scales.clone());
            }
        }
        match v_q {
            Quantization::PerTensor { scale, q } => {
                tensors.insert(format!("v_{i}_q"), q.clone());
                tensors.insert(format!("v_{i}_s"), scale.clone());
            }
            Quantization::PerToken { scales, q } => {
                tensors.insert(format!("v_{i}_q"), q.clone());
                tensors.insert(format!("v_{i}_s"), scales.clone());
            }
        }
    }
    // ... rest unchanged ...
}
```

Update `load_kv` dequantization loop to reconstruct `Quantization` from the stored scale shape:

```rust
let k_q = data.get(&format!("k_{idx}_q")).ok_or_else(...)?;
let k_s = data.get(&format!("k_{idx}_s")).ok_or_else(...)?;
let v_q = data.get(&format!("v_{idx}_q")).ok_or_else(...)?;
let v_s = data.get(&format!("v_{idx}_s")).ok_or_else(...)?;

let k_qt = if k_s.rank() == 0 {
    Quantization::PerTensor { scale: k_s.clone(), q: k_q.clone() }
} else {
    Quantization::PerToken { scales: k_s.clone(), q: k_q.clone() }
};
let v_qt = if v_s.rank() == 0 {
    Quantization::PerTensor { scale: v_s.clone(), q: v_q.clone() }
} else {
    Quantization::PerToken { scales: v_s.clone(), q: v_q.clone() }
};

let k = kv_quant::dequantize_tensor(&k_qt, dequant_dtype)?;
let v = kv_quant::dequantize_tensor(&v_qt, dequant_dtype)?;
```

- [ ] **Step 7: Update `src/local/quant.rs` to use the new `Quantization` type.**

`run_quant_pipeline` calls `quantize_kv` and `save_quantized_kv`. Update those call sites to pass the new type. `max_quant_error` signature is unchanged.

- [ ] **Step 8: Update `src/core/validation.rs` to use the new `Quantization` type.**

The validation harness currently does its own dequantization. Update it to:

```rust
let quantized = crate::local::kv_quant::quantize_kv(&cache)?;
let dequant_cache: KVCache = quantized
    .into_iter()
    .enumerate()
    .map(|(i, (k_q, v_q))| {
        let k = crate::local::kv_quant::dequantize_tensor(&k_q, original_dtype)
            .with_context(|| format!("dequantizing key tensor for layer {i}"))?;
        let v = crate::local::kv_quant::dequantize_tensor(&v_q, original_dtype)
            .with_context(|| format!("dequantizing value tensor for layer {i}"))?;
        Ok((k, v))
    })
    .collect::<Result<_>>()?;
```

- [ ] **Step 9: Run tests.**

Run:
```bash
cargo test --workspace
```

Expected: all tests pass.

- [ ] **Step 10: Commit.**

```bash
git add src/local/kv_quant.rs src/local/kv_io.rs src/local/quant.rs src/core/validation.rs
git commit -m "feat(kv_quant): per-token int8 quantization for rank-3 MLA caches"
```

---

### Task 3: Validate DeepSeek-V2-Lite passes quantized verify

**Files:**
- Modify: `scripts/validate-adapters.sh`
- Modify: `README.md`

- [ ] **Step 1: Add DeepSeek back to the tracked validation models.**

In `scripts/validate-adapters.sh`:

```bash
export KVCDN_VALIDATE_MODELS="\
deepseek-ai/DeepSeek-V2-Lite-Chat,Qwen/Qwen2-0.5B,microsoft/Phi-3-mini-4k-instruct"
```

Remove the DeepSeek note from the comments.

- [ ] **Step 2: Run the validation script.**

Run:
```bash
./scripts/validate-adapters.sh
```

Expected: all three models pass.

- [ ] **Step 3: Update README to move DeepSeek to fully supported.**

Move `- DeepseekV2ForCausalLM / DeepseekV3ForCausalLM` from the "Implemented but not yet validated" section to "Currently supported architectures". Remove the explanatory note.

- [ ] **Step 4: Run full checks.**

Run:
```bash
cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test --workspace && ./scripts/validate-adapters.sh
```

Expected: clean.

- [ ] **Step 5: Commit.**

```bash
git add scripts/validate-adapters.sh README.md
git commit -m "feat(models): promote DeepSeek-V2 to supported after per-token quant fix"
```

---

## Spec coverage check

- DeepSeek quantized verify failure root cause: covered by Task 2 per-token quantizer.
- Existing adapters must keep passing: covered by keeping per-tensor path for rank-4 tensors.
- Storage format backward compatibility: **gap**. The new `Quantization` enum changes the serialized layout. Old `.q8.kv` files with scalar scales still load because rank-0 scales map to `PerTensor`. New files with per-token scales will not be readable by old code, but that is acceptable for a forward-only feature.

## Placeholder scan

No TBD/TODO placeholders.

## Type consistency check

- `Quantization` used consistently across `kv_quant`, `kv_io`, `quant`, and `validation`.
- `save_quantized_kv` signature changes from `(Tensor, Tensor, Tensor, Tensor)` to `(Quantization, Quantization)`; all call sites updated in Tasks 2 and 3.
