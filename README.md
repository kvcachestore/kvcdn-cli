# kvcdn-cli

Releasable command-line tool for KV-cache generation, verification, quantization, and hosted sharing for LLMs.

## Commands

- `kvcdn verify` — verify that loading a saved KV cache produces token-exact output.
- `kvcdn diag` — logits-level diagnostic comparing scratch vs. KV-cache paths.
- `kvcdn benchmark` — measure prefill vs. continuation speedup.
- `kvcdn plot` — summarize benchmark results.
- `kvcdn quant` — quantize a KV artifact and verify continuation accuracy.
- `kvcdn login` — authenticate with the hosted service via OIDC.
- `kvcdn upload` — upload a KV artifact to your private hosted endpoint.

## Local-only usage

No authentication is required for local generation, verification, quantization, or benchmarking.

## Hosted usage

Run `kvcdn login` to authenticate. Use `kvcdn upload <artifact.kv> --name <name>` to store caches in your private hosted endpoint.

## Build

```bash
cargo build --release
```

## License

See upstream repository for licensing terms.
