use anyhow::Result;
use candle_core::Device;
use candle_nn::VarBuilder;

use crate::models::CausalLM;

pub(crate) type Loader =
    Box<dyn Fn(&std::path::Path, VarBuilder, &Device) -> Result<Box<dyn CausalLM>>>;

fn llama_loader() -> Loader {
    Box::new(|config_path, vb, device| {
        let cfg = crate::models::llama::Config::load(config_path)?;
        Ok(Box::new(crate::models::llama::ModelForCausalLM::new(
            &cfg, vb, device,
        )?))
    })
}

pub(crate) fn dispatch(architecture: &str) -> Option<Loader> {
    // Architectures that share the Llama weight-map / decoder shape.
    match architecture {
        "LlamaForCausalLM" | "MistralForCausalLM" | "MixtralForCausalLM" | "YiForCausalLM" => {
            Some(llama_loader())
        }
        "MambaForCausalLM" => Some(Box::new(|config_path, vb, device| {
            let _ = device;
            let cfg = crate::models::mamba::Config::load(config_path)?;
            Ok(Box::new(crate::models::mamba::ModelForCausalLM::new(
                &cfg, vb,
            )?))
        })),
        "ChatGLMModel" | "ChatGLMForCausalLM" => Some(Box::new(|config_path, vb, device| {
            let _ = device;
            let cfg = crate::models::chatglm::Config::load(config_path)?;
            Ok(Box::new(crate::models::chatglm::ModelForCausalLM::new(
                &cfg, vb,
            )?))
        })),
        "DeepseekV2ForCausalLM" | "DeepseekV3ForCausalLM" => {
            Some(Box::new(|config_path, vb, device| {
                let _ = device;
                let cfg = crate::models::deepseek2::Config::load(config_path)?;
                Ok(Box::new(crate::models::deepseek2::ModelForCausalLM::new(
                    &cfg, vb,
                )?))
            }))
        }
        "FalconForCausalLM" => Some(Box::new(|config_path, vb, device| {
            let _ = device;
            let cfg = crate::models::falcon::Config::load(config_path)?;
            Ok(Box::new(crate::models::falcon::ModelForCausalLM::new(
                &cfg, vb,
            )?))
        })),
        "Phi3ForCausalLM" => Some(Box::new(|config_path, vb, device| {
            let _ = device;
            let cfg = crate::models::phi3::Config::load(config_path)?;
            Ok(Box::new(crate::models::phi3::ModelForCausalLM::new(
                &cfg, vb,
            )?))
        })),
        "OLMoForCausalLM" | "OlmoForCausalLM" => Some(Box::new(|config_path, vb, device| {
            let _ = device;
            let cfg = crate::models::olmo::Config::load(config_path)?;
            Ok(Box::new(crate::models::olmo::ModelForCausalLM::new(
                &cfg, vb,
            )?))
        })),
        "Qwen2ForCausalLM" => Some(Box::new(|config_path, vb, device| {
            let _ = device;
            let cfg = crate::models::qwen2::Config::load(config_path)?;
            Ok(Box::new(crate::models::qwen2::ModelForCausalLM::new(
                &cfg, vb,
            )?))
        })),
        "Qwen3ForCausalLM" => Some(Box::new(|config_path, vb, device| {
            let _ = device;
            let cfg = crate::models::qwen3::Config::load(config_path)?;
            Ok(Box::new(crate::models::qwen3::ModelForCausalLM::new(
                &cfg, vb,
            )?))
        })),
        "StableLmForCausalLM" => Some(Box::new(|config_path, vb, device| {
            let _ = device;
            let cfg = crate::models::stable_lm::Config::load(config_path)?;
            Ok(Box::new(crate::models::stable_lm::ModelForCausalLM::new(
                &cfg, vb,
            )?))
        })),
        "Starcoder2ForCausalLM" => Some(Box::new(|config_path, vb, device| {
            let _ = device;
            let cfg = crate::models::starcoder2::Config::load(config_path)?;
            Ok(Box::new(crate::models::starcoder2::ModelForCausalLM::new(
                &cfg, vb,
            )?))
        })),
        "GemmaForCausalLM" | "Gemma2ForCausalLM" => Some(Box::new(|config_path, vb, device| {
            let cfg = crate::models::gemma::Config::load(config_path)?;
            Ok(Box::new(crate::models::gemma::ModelForCausalLM::new(
                &cfg, vb, device,
            )?))
        })),
        "GPT2LMHeadModel" => Some(Box::new(|config_path, vb, device| {
            let _ = device;
            let cfg = crate::models::gpt2::Config::load(config_path)?;
            Ok(Box::new(crate::models::gpt2::ModelForCausalLM::new(
                &cfg, vb,
            )?))
        })),
        "Gemma3ForCausalLM" => Some(Box::new(|config_path, vb, device| {
            let _ = device;
            let cfg = crate::models::gemma3::Config::load(config_path)?;
            Ok(Box::new(crate::models::gemma3::ModelForCausalLM::new(
                &cfg, vb,
            )?))
        })),
        "GLM4ForCausalLM" => Some(Box::new(|config_path, vb, device| {
            let _ = device;
            let cfg = crate::models::glm4::Config::load(config_path)?;
            Ok(Box::new(crate::models::glm4::ModelForCausalLM::new(
                &cfg, vb,
            )?))
        })),
        "GraniteForCausalLM" => Some(Box::new(|config_path, vb, device| {
            let _ = device;
            let cfg = crate::models::granite::Config::load(config_path)?;
            Ok(Box::new(crate::models::granite::ModelForCausalLM::new(
                &cfg, vb,
            )?))
        })),
        "GraniteMoeForCausalLM" => Some(Box::new(|config_path, vb, device| {
            let _ = device;
            let cfg = crate::models::granite_moe::Config::load(config_path)?;
            Ok(Box::new(crate::models::granite_moe::ModelForCausalLM::new(
                &cfg, vb,
            )?))
        })),
        "BloomForCausalLM" => Some(Box::new(|config_path, vb, device| {
            let _ = device;
            let cfg = crate::models::bloom::Config::load(config_path)?;
            Ok(Box::new(crate::models::bloom::ModelForCausalLM::new(
                &cfg, vb,
            )?))
        })),
        "Mamba2ForCausalLM" => Some(Box::new(|config_path, vb, device| {
            let _ = device;
            let cfg = crate::models::mamba2::Config::load(config_path)?;
            Ok(Box::new(crate::models::mamba2::ModelForCausalLM::new(
                &cfg, vb,
            )?))
        })),
        "RWKV6ForCausalLM" => Some(Box::new(|config_path, vb, device| {
            let _ = device;
            let cfg = crate::models::rwkv6::Config::load(config_path)?;
            Ok(Box::new(crate::models::rwkv6::ModelForCausalLM::new(
                &cfg, vb,
            )?))
        })),
        _ => None,
    }
}

fn known_unsupported(architecture: &str) -> Option<&'static str> {
    match architecture {
        "GPTBigCodeForCausalLM" | "RWKVForCausalLM" | "RWKV7ForCausalLM" => {
            Some("state-space / non-transformer")
        }

        "GLMForCausalLM" => Some("GLM"),

        "KimiForCausalLM" | "MoonlightForCausalLM" | "MoonshotForCausalLM" => {
            Some("Kimi / Moonshot")
        }
        "NemotronForCausalLM" => Some("Nemotron"),
        "PhiForCausalLM" | "Phi4ForCausalLM" => Some("Phi"),
        "GPTNeoXForCausalLM" | "OpenAIGPTForCausalLM" => Some("GPT-NeoX"),
        _ => None,
    }
}

pub(crate) fn unsupported_message(architecture: &str) -> String {
    let supported = supported_architectures().join(", ");
    if let Some(family) = known_unsupported(architecture) {
        format!(
            "{architecture} ({family}) is not yet supported.\n\
             Currently supported: {supported}.\n\
             To request this model, make an issue on GitHub."
        )
    } else {
        format!(
            "unsupported model architecture '{architecture}'\n\
             currently supported: {supported}.\n\
             To request this model, make an issue on GitHub."
        )
    }
}

pub(crate) fn supported_architectures() -> &'static [&'static str] {
    &[
        "LlamaForCausalLM",
        "MistralForCausalLM",
        "MixtralForCausalLM",
        "YiForCausalLM",
        "ChatGLMModel",
        "ChatGLMForCausalLM",
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "FalconForCausalLM",
        "Phi3ForCausalLM",
        "OLMoForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "StableLmForCausalLM",
        "Starcoder2ForCausalLM",
        "GemmaForCausalLM",
        "Gemma2ForCausalLM",
        "Gemma3ForCausalLM",
        "GPT2LMHeadModel",
        "BloomForCausalLM",
        "GLM4ForCausalLM",
        "GraniteForCausalLM",
        "GraniteMoeForCausalLM",
        "MambaForCausalLM",
        "Mamba2ForCausalLM",
        "RWKV6ForCausalLM",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_architectures_have_loaders() {
        for arch in supported_architectures() {
            assert!(dispatch(arch).is_some(), "expected a loader for {arch}",);
        }
    }

    #[test]
    fn llama_compatible_aliases_use_same_loader() {
        let names = [
            "LlamaForCausalLM",
            "MistralForCausalLM",
            "MixtralForCausalLM",
            "YiForCausalLM",
        ];
        for arch in names {
            assert!(
                dispatch(arch).is_some(),
                "expected Llama-compatible loader for {arch}",
            );
        }
    }

    #[test]
    fn known_unsupported_families_return_none() {
        let unsupported = [
            "GLMForCausalLM",
            "KimiForCausalLM",
            "MoonlightForCausalLM",
            "MoonshotForCausalLM",
            "NemotronForCausalLM",
            "PhiForCausalLM",
            "Phi4ForCausalLM",
            "GPTNeoXForCausalLM",
            "OpenAIGPTForCausalLM",
            "RWKVForCausalLM",
            "RWKV7ForCausalLM",
            "GPTBigCodeForCausalLM",
        ];
        for arch in unsupported {
            assert!(
                dispatch(arch).is_none(),
                "{arch} should not have a loader yet",
            );
            assert!(
                known_unsupported(arch).is_some(),
                "{arch} should be in the known-unsupported list",
            );
        }
    }

    #[test]
    fn unsupported_message_mentions_family_and_github() {
        let msg = unsupported_message("PhiForCausalLM");
        assert!(msg.contains("PhiForCausalLM"), "{msg}");
        assert!(msg.contains("Phi"), "{msg}");
        assert!(msg.contains("make an issue on GitHub"), "{msg}");
        assert!(msg.contains("LlamaForCausalLM"), "{msg}");
    }

    #[test]
    fn unknown_architecture_message_mentions_github() {
        let msg = unsupported_message("SomeRandomModelForCausalLM");
        assert!(msg.contains("SomeRandomModelForCausalLM"), "{msg}",);
        assert!(msg.contains("make an issue on GitHub"), "{msg}");
    }

    #[test]
    fn supported_architectures_list_is_stable() {
        let expected = [
            "LlamaForCausalLM",
            "MistralForCausalLM",
            "MixtralForCausalLM",
            "YiForCausalLM",
            "ChatGLMModel",
            "ChatGLMForCausalLM",
            "DeepseekV2ForCausalLM",
            "DeepseekV3ForCausalLM",
            "FalconForCausalLM",
            "Phi3ForCausalLM",
            "OLMoForCausalLM",
            "Qwen2ForCausalLM",
            "Qwen3ForCausalLM",
            "StableLmForCausalLM",
            "Starcoder2ForCausalLM",
            "GemmaForCausalLM",
            "Gemma2ForCausalLM",
            "Gemma3ForCausalLM",
            "GPT2LMHeadModel",
            "BloomForCausalLM",
            "GLM4ForCausalLM",
            "GraniteForCausalLM",
            "GraniteMoeForCausalLM",
            "MambaForCausalLM",
            "Mamba2ForCausalLM",
            "RWKV6ForCausalLM",
        ];
        assert_eq!(supported_architectures(), &expected[..]);
    }
}
