# Agent Notes

## Repo layout

- `src/` — Rust CLI (`kvcdn`). Entry point `src/main.rs`.
- `backend/` — TypeScript/Fastify API. Entry point `backend/src/index.ts`.
- `dagger/` — Dagger module for Rust release builds, SBOM generation, Trivy scans, and optional cosign signing.
- `scripts/` — `build-release.sh` and `build-backend-image.sh`.

## Daily commands

Rust CLI:

```bash
cargo test --workspace
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo build --release  # binary at target/release/kvcdn
```

Backend:

```bash
cd backend
npm test          # NODE_OPTIONS='--experimental-vm-modules' jest
npm run build     # tsc -> dist/
npm run dev       # tsx src/index.ts
```

## Tool paths

- `dagger` is used by the release pipeline; make sure it is on your `PATH`.

## Things that are easy to miss

- The CLI and the backend must agree on `KVCDN_CLIENT_ID` / `client_id`. Default is `kvcdn-cli`. Backend `auth.ts` rejects tokens whose `aud` does not match.
- CLI auth now prefers OIDC device-code flow when the issuer advertises it, falling back to a local callback authorization-code flow. No redirect URI registration is required for device-code clients.
- Backend tests mock the OIDC provider via `KVCDN_MOCK_OIDC=true`; they do not need a live OIDC provider.
- The backend also enters mock mode automatically when `KVCDN_ISSUER_URL` is unset, but `KVCDN_MOCK_OIDC=true` makes the mock path explicit.
- Backend S3 setup: `forcePathStyle: true`, default `region: "us-east-1"`.
- Backend artifact S3 layout: `artifacts/customers/{customer_id}/orgs/{org}/projects/{project}/{artifact_id}/` with `.meta.json` metadata sidecar.
- `GET /v1/quota` returns the authenticated customer's quota limits and current utilization (organizations, projects, artifacts, storage bytes). Limits are configurable via `KVCDN_QUOTA_ORGS`, `KVCDN_QUOTA_PROJECTS`, `KVCDN_QUOTA_ARTIFACTS`, and `KVCDN_QUOTA_STORAGE_BYTES`.
- The CLI exposes the same data through `kvcdn quota` (default table output) and `kvcdn quota --format json`.
- `backend/tsconfig.json` excludes `src/**/*.test.ts`; `jest.config.js` uses `ts-jest` with `useESM: true` and maps `.js` imports back to `.ts` files. Do not run `jest` without `NODE_OPTIONS='--experimental-vm-modules'`.
- CLI config resolution order for hosted commands: CLI flag → `KVCDN_*` env var → `~/.config/kvcdn/config.toml` → default.
- KV cache files use the `.kv` extension. Explicit `--kv-path`/`--output`/`--input` paths that do not end in `.kv` are automatically suffixed with `.kv`.
- `scripts/build-release.sh` runs the full Dagger `Release` pipeline (Rust lint/test, backend tests, release build, SBOM, Trivy scan, optional cosign signing). Set `COSIGN_PRIVATE_KEY` to produce `.sig` files.
- `kvcdn login` opens a browser by default; use `--no-browser` in headless/automated environments.
- API keys are deterministic per org: `kv_<hex>` derived from `KVCDN_API_KEY_SEED` + org slug via HKDF-SHA256. Mint them with `POST /v1/admin/api-keys` using `KVCDN_ADMIN_SECRET`.
