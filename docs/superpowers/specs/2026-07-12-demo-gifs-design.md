# Demo GIFs Design

## Goal
Add animated terminal GIFs to the README showing real `kvcdn` usage, similar to the live-run clips on https://github.com/IMJONEZZ/lo-agent.

## Scope
Two short GIFs:

1. **Hosted workflow** (`docs/assets/kvcdn-hosted.gif`)
   - `kvcdn quota`
   - `kvcdn upload <artifact.kv> --name demo --org acme --project demo`
   - `kvcdn list --org acme --project demo`
   - `kvcdn download <id> --org acme --project demo --output demo-download.kv`
   - `kvcdn delete <id> --org acme --project demo --yes`
   - Recorded against the production backend `https://kvcachestore.fly.dev`. The API key is supplied via the `KVCDN_API_KEY` environment variable before recording so the key itself is never typed or visible in the terminal.

2. **Local workflow** (`docs/assets/kvcdn-local.gif`)
   - `kvcdn verify --kv-path demo-context.kv --n 4`
   - `kvcdn quant --input demo-context.kv --output demo-context.q8.kv --n 4`
   - The default small model is pre-downloaded off-screen so the recording focuses on the tool output rather than the download.

## Tooling
- `asciinema` (Python package) to record terminal sessions.
- `agg` (Rust crate) to convert `.cast` files to GIFs.
- Both tools installed into isolated local environments (`.venv-r2` for asciinema, `cargo install --root` for agg).

## Process
1. Pre-warm the model cache by running `kvcdn verify` once; save the resulting KV artifact.
2. Quantize the artifact off-screen to produce the upload input.
3. Record the hosted workflow in one `asciinema` cast and the local workflow in another.
4. Convert each cast to a GIF with `agg`.
5. Embed the GIFs in `README.md` under the **Hosted usage** and **Local-only usage** sections.
6. Commit the new assets and README changes.

## Constraints
- The hosted workflow is recorded against production, but the API key is passed via `KVCDN_API_KEY` so it is not visible in the terminal.
- GIFs should be kept short (< 30 s each) by using small models, short continuation counts (`--n 4`), and pre-cached downloads.
