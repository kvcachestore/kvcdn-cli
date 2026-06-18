# Contributing to kvcdn-cli

Thank you for your interest in kvcdn-cli. This document covers how to build the
project, run checks, open issues, and submit changes.

## Development environment

You need:

- [Rust](https://rustup.rs/) (latest stable toolchain)
- [Node.js](https://nodejs.org/) (for backend development and tests)
- [Dagger](https://docs.dagger.io/install/) (for release builds and CI)
- A Linux or macOS host (most local development has been validated on Linux)

Clone the repository and verify the workspace builds:

```bash
git clone https://github.com/kvcachestore/kvcdn-cli.git
cd kvcdn-cli
cargo test --workspace
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

## Project layout

- `src/` — Rust CLI (`kvcdn`). Entry point: `src/main.rs`.
- `backend/` — TypeScript/Fastify API. Entry point: `backend/src/index.ts`.
- `dagger/` — Dagger module for Rust release builds, SBOM generation, Trivy
  scans, and optional cosign signing.
- `scripts/` — `build-release.sh` and `build-backend-image.sh`.

## Local development workflow

### Rust CLI

```bash
cargo build --release        # binary at target/release/kvcdn
cargo test --workspace
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

### Backend

```bash
cd backend
npm install
npm test                     # requires NODE_OPTIONS='--experimental-vm-modules'
npm run build                # tsc -> dist/
npm run dev                  # tsx src/index.ts
```

The backend tests mock the OIDC provider via `KVCDN_MOCK_OIDC=true`; they do not
need a live OIDC provider. Mock mode is also enabled automatically when
`KVCDN_ISSUER_URL` is unset.

## Adding a new model architecture

The CLI loads a model from Hugging Face and dispatches to an adapter based on the
`architectures` field in `config.json`. To add support for a new architecture:

1. Implement the `CausalLM` trait in a new file under `src/models/`.
2. Register the adapter in `src/models/registry.rs`.
3. Add a unit or integration test that loads a small public checkpoint and
   verifies token-exact continuation.

## Continuous integration

CI runs through the Dagger module in this repository. Before opening a pull
request, run the PR checks locally:

```bash
dagger call -m dagger pr --src=.
```

This runs `cargo fmt --check`, `cargo clippy`, `cargo test --workspace`, and the
backend test suite in parallel.

## Release builds

The Dagger pipeline runs PR checks, then builds a stripped x86-64 Linux binary,
generates an SPDX SBOM, and scans the binary with Trivy:

```bash
./scripts/build-release.sh
```

If `COSIGN_PRIVATE_KEY` is set, the pipeline also produces cosign signatures.

## Opening issues

- Use the bug report template for crashes, incorrect behavior, or build
  failures.
- Use the feature request template for new commands, model support, or hosted
  API additions.
- Search existing issues before filing a new one.

## Submitting changes

1. Fork the repository and create a branch.
2. Make focused, minimal changes with clear commit messages.
3. Add or update tests for behavioral changes.
4. Run the full local test suite (`cargo test --workspace` and
   `cd backend && npm test`).
5. Open a pull request and fill out the template.

## Code style

- Rust: follow `cargo fmt` and keep `cargo clippy --all-targets -- -D warnings`
  clean.
- TypeScript: follow the existing `eslint`/`prettier` configuration in
  `backend/`.
- Do not suppress type errors with `as any`, `@ts-ignore`, or
  `@ts-expect-error`.

## License

By contributing, you agree that your contributions will be licensed under the
[kvcdn-cli Source-Available License](LICENSE).
