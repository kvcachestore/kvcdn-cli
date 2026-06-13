use candle_core::Device;

pub fn pick_device() -> anyhow::Result<Device> {
    if candle_core::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else if candle_core::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else {
        Ok(Device::Cpu)
    }
}
