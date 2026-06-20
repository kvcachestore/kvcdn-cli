use anyhow::{Context, Result};
use candle_core::{DType, Tensor};

use crate::models::KVCache;

/// Per-tensor symmetric int8 quantization.
///
/// Matches kvstore's `quantize_int8`: `scale = max(|t|) / 127` over the whole
/// tensor, and values are rounded to `[-127, 127]`. Candle 0.10 does not
/// expose `DType::I8`, so the signed values are stored in `U8` with a +128
/// offset; the dequantizer recenters before scaling.
pub fn quantize_tensor(t: &Tensor) -> Result<(Tensor, Tensor)> {
    let shape = t.shape();
    if shape.rank() != 4 {
        anyhow::bail!("expected rank-4 KV tensor, got shape {shape:?}");
    }

    let f32_t = t.to_dtype(DType::F32)?;
    let abs = f32_t.abs()?;
    let max = abs.max_all()?;
    let scale = max.clamp(1e-12f64, f64::MAX)?;

    let normalized = (f32_t.broadcast_div(&scale)? * 127.0f64)?;
    let q = (normalized.round()?.clamp(-127.0f64, 127.0f64)? + 128.0f64)?;
    let q = q
        .to_dtype(DType::U8)
        .context("converting quantized values to U8")?;

    Ok((scale, q))
}

/// Dequantize a U8-stored symmetric int8 tensor back to the target dtype.
pub fn dequantize_tensor(scale: &Tensor, q: &Tensor, target_dtype: DType) -> Result<Tensor> {
    let centered = (q.to_dtype(DType::F32)? - 128.0f64)?;
    let x = (centered.broadcast_mul(scale)? / 127.0f64)?;
    x.to_dtype(target_dtype)
        .with_context(|| format!("dequantizing to {target_dtype:?}"))
}

/// Quantize an entire KV cache, returning per-layer `(k_scale, k_quant, v_scale, v_quant)`.
pub fn quantize_kv(cache: &KVCache) -> Result<Vec<(Tensor, Tensor, Tensor, Tensor)>> {
    cache
        .iter()
        .enumerate()
        .map(|(i, (k, v))| {
            let (k_s, k_q) = quantize_tensor(k)
                .with_context(|| format!("quantizing key tensor for layer {i}"))?;
            let (v_s, v_q) = quantize_tensor(v)
                .with_context(|| format!("quantizing value tensor for layer {i}"))?;
            Ok((k_s, k_q, v_s, v_q))
        })
        .collect()
}

/// Compute the maximum absolute dequantization error for a cache.
pub fn max_quant_error(cache: &KVCache, target_dtype: DType) -> Result<f32> {
    let mut max_err: f32 = 0.0;
    for (k, v) in cache {
        let (k_s, k_q) = quantize_tensor(k)?;
        let k_dq = dequantize_tensor(&k_s, &k_q, target_dtype)?;
        let err = (k.to_dtype(DType::F32)? - k_dq.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        max_err = max_err.max(err);

        let (v_s, v_q) = quantize_tensor(v)?;
        let v_dq = dequantize_tensor(&v_s, &v_q, target_dtype)?;
        let err = (v.to_dtype(DType::F32)? - v_dq.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        max_err = max_err.max(err);
    }
    Ok(max_err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn round_trip_quantization() -> Result<()> {
        let dev = Device::Cpu;
        let len: usize = 8 * 32 * 128;
        let deterministic_values = Tensor::arange(0u32, len as u32, &dev)?
            .to_dtype(DType::F32)?
            .reshape((1, 8, 32, 128))?;
        let t =
            (deterministic_values.broadcast_div(&Tensor::new((len / 2) as f32, &dev)?)? - 1.0f64)?;
        let (scale, q) = quantize_tensor(&t)?;
        let dq = dequantize_tensor(&scale, &q, DType::F32)?;
        let err = (t - dq)?.abs()?.max_all()?.to_scalar::<f32>()?;
        println!("max err {err}");
        assert!(err < 0.02);
        Ok(())
    }

    #[test]
    fn round_trip_quantization_fp8() -> Result<()> {
        let dev = Device::Cpu;
        let len: usize = 8 * 32 * 128;
        let deterministic_values = Tensor::arange(0u32, len as u32, &dev)?
            .to_dtype(DType::F32)?
            .reshape((1, 8, 32, 128))?;
        let t =
            (deterministic_values.broadcast_div(&Tensor::new((len / 2) as f32, &dev)?)? - 1.0f64)?;
        let (scale, q) = quantize_tensor(&t)?;
        let dq = dequantize_tensor(&scale, &q, DType::F8E4M3)?;
        let err = (t.to_dtype(DType::F32)? - dq.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        println!("max err {err}");
        assert!(err < 0.5);
        Ok(())
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use candle_core::{Device, Tensor};
    use proptest::prelude::*;
    use proptest::test_runner::TestCaseError;

    /// Generate an arbitrary rank-4 tensor with small, finite f32 values.
    ///
    /// The ranges are intentionally small so each case is fast, while still
    /// exercising many batch/head/token/head-dim combinations.
    fn arb_rank4_tensor() -> impl Strategy<Value = Tensor> {
        (1usize..=2, 1usize..=4, 1usize..=64, 8usize..=64).prop_flat_map(
            |(batch, heads, tokens, head_dim)| {
                let len = batch * heads * tokens * head_dim;
                prop::collection::vec(-10.0f32..=10.0f32, len).prop_map(move |values| {
                    Tensor::from_vec(values, (batch, heads, tokens, head_dim), &Device::Cpu)
                        .expect("valid tensor")
                })
            },
        )
    }

    fn prop_ok<T, E: std::fmt::Display>(
        result: std::result::Result<T, E>,
    ) -> std::result::Result<T, TestCaseError> {
        result.map_err(|e| TestCaseError::fail(e.to_string()))
    }

    proptest! {
        // Keep CI fast: 64 cases is enough to sample the shape/value space.
        #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

        #[test]
        fn roundtrip_error_bounded(t in arb_rank4_tensor()) {
            let (scale, q) = prop_ok(quantize_tensor(&t))?;
            let dq = prop_ok(dequantize_tensor(&scale, &q, DType::F32))?;
            let scale_scalar = prop_ok(scale.to_scalar::<f32>())?;
            let max_err = prop_ok((t.clone() - dq)?.abs()?.max_all()?.to_scalar::<f32>())?;
            prop_assert!(
                max_err <= 0.5 * scale_scalar,
                "max_err {max_err} exceeds 0.5*scale {scale_scalar}"
            );
            prop_assert!(q.shape() == t.shape());
            prop_assert_eq!(q.rank(), 4);
        }

        #[test]
        fn zero_preservation(t in arb_rank4_tensor()) {
            let zeros = prop_ok(Tensor::zeros(t.shape().dims(), DType::F32, t.device()))?;
            let (scale, q) = prop_ok(quantize_tensor(&zeros))?;
            let dq = prop_ok(dequantize_tensor(&scale, &q, DType::F32))?;
            let max_abs = prop_ok(dq.abs()?.max_all()?.to_scalar::<f32>())?;
            prop_assert_eq!(max_abs, 0.0, "dequantized zeros should remain zero (scale={})", prop_ok(scale.to_scalar::<f32>())?);
        }

        #[test]
        fn scale_positivity_and_shape(t in arb_rank4_tensor()) {
            let (scale, q) = prop_ok(quantize_tensor(&t))?;
            let scale_scalar = prop_ok(scale.to_scalar::<f32>())?;
            prop_assert!(scale_scalar.is_finite());
            prop_assert!(scale_scalar > 0.0, "scale must be positive, got {scale_scalar}");
            prop_assert!(q.shape() == t.shape());
            prop_assert_eq!(q.rank(), 4);
        }

        #[test]
        fn determinism(t in arb_rank4_tensor()) {
            let (scale1, q1) = prop_ok(quantize_tensor(&t))?;
            let (scale2, q2) = prop_ok(quantize_tensor(&t))?;
            let scale_diff = prop_ok((scale1 - scale2)?.abs()?.max_all()?.to_scalar::<f32>())?;
            let q_diff = prop_ok((q1.to_dtype(DType::F32)? - q2.to_dtype(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>())?;
            prop_assert_eq!(scale_diff, 0.0, "scale should be deterministic");
            prop_assert_eq!(q_diff, 0.0, "quantized tensor should be deterministic");
        }
    }

    #[test]
    fn all_zeros_scale_clamped() -> Result<()> {
        let dev = Device::Cpu;
        let t = Tensor::zeros((1, 1, 1, 8), DType::F32, &dev)?;
        let (scale, q) = quantize_tensor(&t)?;
        let scale_scalar = scale.to_scalar::<f32>()?;
        assert!(
            (scale_scalar - 1e-12).abs() < 1e-18,
            "expected clamped scale 1e-12, got {scale_scalar}"
        );
        let dq = dequantize_tensor(&scale, &q, DType::F32)?;
        let max_abs = dq.abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(max_abs, 0.0);
        Ok(())
    }

    #[test]
    fn single_element_tensor() -> Result<()> {
        let dev = Device::Cpu;
        let t = Tensor::new(&[[[[1.0f32]]]], &dev)?;
        let (scale, q) = quantize_tensor(&t)?;
        let dq = dequantize_tensor(&scale, &q, DType::F32)?;
        let err = (t - dq)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(err.is_finite());
        assert!(err < 0.02, "single-element error {err} too large");
        Ok(())
    }

    #[test]
    fn large_tensor_quantization() -> Result<()> {
        let dev = Device::Cpu;
        let shape = (1, 8, 256, 64);
        let len: usize = shape.0 * shape.1 * shape.2 * shape.3;
        let deterministic_values = Tensor::arange(0u32, len as u32, &dev)?
            .to_dtype(DType::F32)?
            .reshape(shape)?;
        let t =
            (deterministic_values.broadcast_div(&Tensor::new((len / 2) as f32, &dev)?)? - 1.0f64)?;
        let (scale, q) = quantize_tensor(&t)?;
        let dq = dequantize_tensor(&scale, &q, DType::F32)?;
        let err = (t - dq)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(err < 0.02, "large tensor error {err} too large");
        Ok(())
    }

    #[test]
    fn max_quant_error_for_bf16_and_fp8() -> Result<()> {
        let dev = Device::Cpu;
        let len: usize = 2 * 4 * 8;
        let cache: KVCache = (0..2)
            .map(|layer| {
                let base = (layer * len) as u32;
                let values = Tensor::arange(base, base + len as u32, &dev)?
                    .to_dtype(DType::F32)?
                    .reshape((1, 2, 4, 8))?;
                let t = (values.broadcast_div(&Tensor::new((len / 2) as f32, &dev)?)? - 1.0f64)?;
                let k = t.clone();
                let v = t;
                Ok((k, v))
            })
            .collect::<Result<Vec<_>>>()?;
        let err_bf16 = max_quant_error(&cache, DType::BF16)?;
        assert!(err_bf16.is_finite(), "BF16 error must be finite");
        let err_fp8 = max_quant_error(&cache, DType::F8E4M3)?;
        assert!(err_fp8.is_finite(), "FP8 error must be finite");
        Ok(())
    }
}
