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
