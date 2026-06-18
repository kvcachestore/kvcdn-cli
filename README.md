# kvcdn-cli

Releasable command-line tool for KV-cache generation, verification, quantization, and hosted sharing for LLMs.

## Why this project exists

Large language models spend most of their inference time on the *prefill* phase: processing every token of a long document before generating the first new token. In retrieval-augmented generation, chat, and agent workflows, the same long context is reused across many queries. Re-computing its KV cache for every request wastes GPU time, increases latency, and raises cost.

kvcdn makes that work reusable. It lets you:

- **Generate** a KV cache from a long document once.
- **Verify** that loading the cache and continuing from it is token-exact compared to a full prefill.
- **Quantize** caches to reduce storage and bandwidth without breaking exact continuation.
- **Share** caches privately or publicly through a hosted KVCDN endpoint, so downstream inference services can skip prefill entirely.

In short, kvcdn turns the long-context prefill problem into a cache-and-serve problem. A practical application is an inference service that lets clients supply a KVCDN artifact reference; the service fetches and loads the prefill cache dynamically, then runs continuation generation against the cached context without repeating the expensive prefill.

## Commands

- `kvcdn verify` — verify that loading a saved KV cache produces token-exact output.
- `kvcdn diag` — logits-level diagnostic comparing scratch vs. KV-cache paths.
- `kvcdn benchmark` — measure prefill vs. continuation speedup.
- `kvcdn plot` — summarize benchmark results.
- `kvcdn quant` — quantize a KV artifact and optionally run token-exact verification.
- `kvcdn login` — authenticate with the hosted service via OIDC.
- `kvcdn logout` — remove stored OIDC tokens and API key.
- `kvcdn api-key` — manage the stored KVCDN API key.
- `kvcdn upload` — upload a KV artifact to your hosted KVCDN endpoint.
- `kvcdn list` — list remote KV artifacts in a project.
- `kvcdn download` — download a remote KV artifact by ID.
- `kvcdn delete` — delete a remote KV artifact by ID.
- `kvcdn whoami` — show the current user and active org/project.

## File extension convention

kvcdn uses `.kv` as the file extension for KV cache artifacts (for example, `context.kv`, `model.kv`, or `context.q8.kv` after quantization). The CLI's default output paths, upload/download examples, and hosted workflows all follow this convention.

If you provide an explicit `--kv-path`, `--output`, or `--input` path that does not end in `.kv`, the CLI automatically appends `.kv` and prints a warning so the convention is enforced.

## Local-only usage

No authentication is required for local generation, verification, quantization, or benchmarking.

## Hosted usage

Run `kvcdn login` to authenticate. Use `kvcdn api-key set <key>` to store an API key for non-interactive uploads. Use `kvcdn upload <artifact.kv> --name <name>` to store caches in your hosted KVCDN endpoint. Artifacts are private by default; pass `--visibility public` to make an artifact fetchable without authentication.

Common hosted workflows:

```bash
# Upload an artifact
kvcdn upload context.q8.kv --name context --project acme

# List artifacts in a project
kvcdn list --project acme
kvcdn list --project acme --format json

# Download an artifact
kvcdn download <artifact-id> --project acme --output context.q8.kv

# Delete an artifact
kvcdn delete <artifact-id> --project acme --yes
```

## First-time setup

1. **Create the object-store bucket.** Any S3-compatible store works (Cloudflare R2, AWS S3, MinIO, etc.). Note the endpoint URL, access key, secret key, and bucket name.
2. **Deploy an OIDC provider.** Use any OpenID Connect provider that supports public/PKCE clients (for example, Pocket ID, Keycloak, or Authelia). Register a public client for the CLI with ID `kvcdn-cli`; no redirect URI is required because the CLI uses the OIDC device-code flow.
3. **Create the Fly.io app.** From `backend/`, run `fly apps create kvcachestore`.
4. **Set the backend secrets.** Use the `fly secrets set` command shown in [Hosted backend configuration](#hosted-backend-configuration).
5. **Deploy the backend.** Run the Dagger deploy pipeline locally with a `FLY_API_TOKEN`:
   ```bash
   dagger call -m ci/dagger deploy-backend --src=. --fly-api-token=env:FLY_API_TOKEN
   ```
6. **Configure the CLI.** Create `~/.config/kvcdn/config.toml`:
   ```toml
   api_url = "https://<your-api-host>"
   issuer_url = "https://<your-issuer>"
   client_id = "kvcdn-cli"
   default_org = "acme"
   default_project = "acme"
   ```

   All hosted commands accept `--org` and `--project` overrides. `default_org` is included for forward compatibility; the current backend routes by `project`, so most users will set both to the same value.
7. **Authenticate and store an API key.** Run `kvcdn login` for interactive use, or mint an org-scoped API key with the admin endpoint and save it with `kvcdn api-key set <key>`. API keys are derived from `KVCDN_API_KEY_SEED` and scoped to one of the orgs in `KVCDN_API_KEY_ORGS`.

## Hosted backend configuration

The `backend/` service is deployed to Fly.io and needs the following secrets set on the app (e.g. `kvcachestore`):

```bash
fly secrets set --app kvcachestore \
  KVCDN_S3_ENDPOINT="https://<account-id>.r2.cloudflarestorage.com" \
  KVCDN_S3_ACCESS_KEY="<r2-access-key-id>" \
  KVCDN_S3_SECRET_KEY="<r2-secret-access-key>" \
  KVCDN_S3_BUCKETS="<bucket-name>" \
  KVCDN_CONTROL_BUCKET="<control-bucket-name>" \
  KVCDN_ISSUER_URL="https://<your-issuer>" \
  KVCDN_CLIENT_ID="kvcdn-cli" \
  KVCDN_API_KEY_SEED="$(openssl rand -hex 32)" \
  KVCDN_API_KEY_ORGS="acme,contoso" \
  KVCDN_ADMIN_SECRET="$(openssl rand -hex 32)"
```

For this deployment the OIDC provider is exposed at `https://<your-issuer>`. If you deploy Pocket ID under your own domain, replace that value in both the backend secrets and the CLI config.

- `KVCDN_S3_BUCKETS` — comma-separated list of S3-compatible bucket names (e.g. `kvcdn-artifacts-1,kvcdn-artifacts-2`). Tenants are deterministically assigned to one bucket. If only one bucket is configured, behavior is identical to the original single-bucket setup.
- `KVCDN_S3_BUCKET` — legacy single-bucket name. Used as a fallback if `KVCDN_S3_BUCKETS` is not set.
- `KVCDN_CONTROL_BUCKET` — bucket that stores tenant-to-bucket assignment records. This bucket should be created in the same S3 account/endpoint and is required when `KVCDN_S3_BUCKETS` has more than one entry.
- `KVCDN_ISSUER_URL` — OIDC issuer URL for the Pocket ID (or other OpenID Connect) provider used to authenticate CLI users.
- `KVCDN_CLIENT_ID` — OIDC client ID the CLI presents to the issuer. This must match a public/PKCE client registered in your OIDC provider.
- `KVCDN_API_KEY_SEED` — master secret used to derive organization-scoped API keys with HKDF-SHA256. Generate once with `openssl rand -hex 32` and keep it secret; changing it invalidates all existing API keys.
- `KVCDN_API_KEY_ORGS` — comma-separated list of allowed organization slugs (e.g. `acme,contoso`). Each slug gets a deterministic API key derived from `KVCDN_API_KEY_SEED`.
- `KVCDN_ADMIN_SECRET` — bearer token required to call the `/v1/admin/api-keys` minting endpoint. Generate with `openssl rand -hex 32` and use it only when provisioning new org keys.

### API key management

Each configured org slug has a deterministic key. To mint a key for an org, call the operator-only admin endpoint:

```bash
curl -s -X POST "https://<your-api-host>/v1/admin/api-keys" \
  -H "authorization: Bearer $KVCDN_ADMIN_SECRET" \
  -H "content-type: application/json" \
  -d '{"org_slug":"acme"}'
```

The response contains the derived `api_key` and stable `customer_id`. Save the key locally with:

```bash
kvcdn api-key set <api_key>
```

Org-scoped keys only work for their org. An artifact uploaded with one key is stored under `artifacts/customers/{customer_id}/` and cannot be listed, downloaded, or deleted by another org's key or OIDC subject.

### Required token permissions

| Service | Token / credential | Required permissions |
| --- | --- | --- |
| Cloudflare R2 (or S3-compatible store) | `KVCDN_S3_ACCESS_KEY` + `KVCDN_S3_SECRET_KEY` | `s3:PutObject`, `s3:GetObject`, `s3:DeleteObject` on the bucket. The backend generates presigned PUT URLs for uploads and reads/deletes objects directly, so listing the bucket is **not** required. |
| OIDC provider (Pocket ID) | none stored in backend; public key discovery via `/.well-known/openid-configuration` | The backend only needs to fetch the provider's discovery document and JWKS endpoint, so no client secret or admin token is required. The CLI OIDC client must be configured as **public**; the CLI uses the OIDC device-code flow and does not need a redirect URI. |
| Fly.io | your `flyctl` auth token | Permission to `fly apps create`, `fly secrets set`, `fly deploy`, and `fly volumes create` in the target organization. The CLI app (`kvcachestore`) only needs the secrets above; the tunnel sidecar uses a separate Cloudflare tunnel token. |
| Cloudflare Tunnel | `TUNNEL_TOKEN` for the `cloudflared` sidecar | Permission to run the tunnel (`Access:Apps:Edit` or the tunnel-specific service token, depending on how the token was created). The tunnel only needs to ingress `pocketid.<yourdomain>` to the private Pocket ID app; it does not terminate TLS for the API or object store. |

The CLI reads the same values from `~/.config/kvcdn/config.toml` (or `KVCDN_*` environment variables) so that `kvcdn login`, `kvcdn upload`, and `kvcdn delete` can reach your deployed backend and issuer.

## License

This project is licensed under the [kvcdn-cli Source-Available License](LICENSE).

The Software is provided for personal, educational, research, and non-commercial
evaluation purposes only. All commercial use and all commercial derivative works
based on this source code are strictly prohibited. You may not sell, offer as a
service, or incorporate this Software or any derivative work into a commercial
product or business operation. See the full license text for details.

## Model support

`--model` accepts any Hugging Face model identifier. The loader inspects the model's `config.json` `architectures` field and dispatches to a matching adapter.

### Example model for first-time users

`Qwen/Qwen3-0.6B` is the recommended starting model. It is small enough to run on CPU,
supports the Qwen3 architecture, and is permissively licensed for research and
non-commercial use. Most examples in this README use it as the default:

```bash
kvcdn verify --model Qwen/Qwen3-0.6B --context-file context.txt
```

Replace the model identifier with any supported architecture once you are ready
to move to larger checkpoints.

Currently supported architectures:

- `LlamaForCausalLM` (and Llama-3.1/3.2 fine-tunes)
- `MistralForCausalLM` / `MixtralForCausalLM`
- `YiForCausalLM`
- `Qwen3ForCausalLM`
- `GemmaForCausalLM` (and Gemma 2 fine-tunes)

Known unsupported families (open an issue on GitHub to request them):

- GLM / ChatGLM
- Kimi / Moonshot / Moonlight
- Mamba / RWKV / state-space models
- Nemotron
- Phi / Phi-3 / Phi-4
- GPT-NeoX / GPT-2 / Bloom

Adding a new family only requires implementing the `CausalLM` trait in `src/models/` and registering it in `src/models/registry.rs`.

## Example usage

Verify that a saved KV cache produces token-exact output:

```bash
kvcdn verify --kv-path context.kv \
             --context-file context.txt \
             --question "What is the main point?"
```

Benchmark full-prefill cost against one resident-KV continuation step and write a CSV:

```bash
kvcdn benchmark --lengths 128,256,512 --reps 10 --output bench.csv
kvcdn plot --csv-path bench.csv --model Qwen/Qwen3-0.6B --out amortized-cost.png --max-n 500
```

`plot` produces an 840x600 PNG with log-log axes showing the break-even reuse count. Use `--out` to set the PNG path and `--max-n` to change the reuse-count range (default 1000). The default filename is timestamped under the data directory.

Quantize a KV artifact with a target dequantization dtype and verify continuation accuracy:

```bash
kvcdn quant --context-file context.txt --question "What is the main point?" \
            --target-dtype F32 --verify
```

Supported target dtypes: `F32`, `F16`, `BF16`, `FP8` (F8E4M3). `I8`, `U8`, `I4`, `U4`, `FP4`, and `FP1` are not supported because the artifact stores symmetric int8 values (U8) and Candle 0.10 does not implement `to_dtype` for FP4/FP1.

## Build

```bash
# Local debug build
cargo build

# Local release build
cargo build --release

# Reproducible containerized release build (exports to ./dist/)
./scripts/build-release.sh
```

The Dagger pipeline runs Rust and backend lint/test checks in parallel, then
builds and strips an x86-64 Linux binary, packages it as
`dist/kvcdn-x86_64-unknown-linux-gnu.tar.gz`, generates an SPDX SBOM
(`dist/kvcdn-x86_64-unknown-linux-gnu.sbom.json`), and scans the tarball with
Trivy. If `COSIGN_PRIVATE_KEY` is provided, it also produces cosign signatures
(`.sig` files).

### Runtime requirements

The released Linux binary is built on `debian:bookworm` (glibc 2.36+) and links
dynamically against a few system libraries. It should run on most recent
Debian, Ubuntu, Fedora, and other glibc-based distributions. Before running it,
you can check that the required shared libraries are present:

```bash
ldd kvcdn-x86_64-unknown-linux-gnu
```

Typical dynamic dependencies include:

- `libc.so.6`
- `libssl.so.3` and `libcrypto.so.3`
- `libz.so.1`
- `libstdc++.so.6`, `libgcc_s.so.1`, `libm.so.6`

If your system ships OpenSSL 1.x instead of OpenSSL 3.x, or an older glibc, the
binary will fail to start. In that case, build from source (`cargo build --release`)
or run the Dagger pipeline on a host whose libraries match your target
environment.

### Release build

Build and export the release binary locally using the Dagger module:

```bash
./scripts/build-release.sh
```

Or invoke Dagger directly:

```bash
# Release without signing
dagger call -m ci/dagger release --src=. export --path=./dist

# Release with cosign signing
dagger call -m ci/dagger release --src=. --cosign-key=env:COSIGN_PRIVATE_KEY export --path=./dist
```

The packaged artifact is written to:
`dist/kvcdn-x86_64-unknown-linux-gnu.tar.gz`

Additional artifacts when signing is enabled:
- `dist/kvcdn-x86_64-unknown-linux-gnu.sbom.json` — SPDX SBOM
- `dist/kvcdn-x86_64-unknown-linux-gnu.tar.gz.sig` — tarball cosign signature
- `dist/kvcdn-x86_64-unknown-linux-gnu.sbom.json.sig` — SBOM cosign signature

### CI/CD deploy workflow

The repository uses Dagger (not GitHub Actions) for deployments. Run the deploy
pipeline locally after setting `FLY_API_TOKEN` in your environment:

```bash
export FLY_API_TOKEN="$(flyctl tokens create deploy -a kvcachestore -x 8760h)"
dagger call -m ci/dagger deploy-backend --src=. --fly-api-token=env:FLY_API_TOKEN
```

Create an app-scoped deploy token for the `kvcachestore` app. This token is
limited to deploying only that app and is valid for one year:

```bash
flyctl tokens create deploy -a kvcachestore -x 8760h
```

The pipeline builds the backend Docker image inside Dagger and pushes it to Fly.io's
private app registry, then deploys it to the `kvcachestore` app. The image is never
pushed to a public registry.

## Limits

The backend is designed around presigned S3 URLs and stateless metadata:

- **Artifact size:** limited by your S3 provider's maximum object size and the
  presigned URL expiration (5 minutes). R2 supports objects up to >5 TiB.
- **Metadata:** stored as a small JSON sidecar (`.meta.json`) next to each artifact
  in the bucket. There is no database to back up.
- **Retention:** objects live in your bucket until you delete them via `kvcdn delete`.
- **Concurrent uploads:** each upload is independent; the backend does not throttle.
- **Authentication:** every API call requires a valid OIDC access token or an
  org-scoped `kv_`-prefixed API key derived from `KVCDN_API_KEY_SEED`.
- **Artifact isolation:** each artifact is stored under a per-customer prefix in
  the bucket. One customer cannot list, download, or delete another customer's
  artifacts, even if they know the artifact ID.

## Hosted KV Cache Store beta limits

When uploading to the hosted KV Cache Store service at [kvcachestore.com](https://kvcachestore.com),
the following per-customer limits apply during the beta and will be lifted when the service exits beta:

- **Organizations:** 32
- **Collections:** 16
- **Artifacts:** 64

If you hit a limit, `kvcdn upload` will fail with the portal's quota-exceeded message.
Delete existing resources to free up capacity, or wait until the beta limits are raised.
