#!/usr/bin/env bash
set -euo pipefail

# Validate the adapters tracked as officially supported. Extend this list only
# after a model passes reference, cache, and quantized verify end-to-end.
#
# Notes on currently excluded targets:
# - DeepSeek-V2-Lite-Chat passes reference and cache verify, but its MLA
#   compressed-KV cache is too sensitive to the current per-tensor symmetric
#   int8 quantizer to pass quantized verify with this fixture.
# - google/gemma-3-1b-it is gated on Hugging Face and requires HF_TOKEN; it
#   cannot be validated in an environment without an access token.
export KVCDN_VALIDATE_MODELS="\
Qwen/Qwen2-0.5B,microsoft/Phi-3-mini-4k-instruct"

export KVCDN_VALIDATE_MAX_TOKENS="${KVCDN_VALIDATE_MAX_TOKENS:-16}"
# Use the latest revision instead of the CLI's pinned default, which targets Qwen3.
export HF_REVISION="${HF_REVISION:-main}"

cargo test --test adapter_validation --release
