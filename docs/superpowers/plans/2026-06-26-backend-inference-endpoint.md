# Backend KV Inference Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backend endpoint that accepts a KV artifact reference, the user's continuation prompt, and returns generated tokens — turning the hosted backend from pure storage into a cache-and-serve inference service.

**Architecture:** Add `POST /v1/orgs/:org/projects/:project/artifacts/:artifact_id/infer`. The backend resolves the artifact, fetches the KV cache bytes from S3, calls a local Rust inference worker (via `tokio::process::Command` to the release `kvcdn` binary), and streams back generated tokens. This avoids embedding Candle directly in the Node.js backend and reuses the CLI's existing model loading, KV-cache loading, and generation paths. The first version is synchronous: return the full generated text after inference completes.

**Tech Stack:** TypeScript/Fastify backend, Rust CLI binary, S3 presigned URLs or direct S3 get, child process invocation.

---

## File map

- `backend/src/routes/inference.ts` — new route handler.
- `backend/src/index.ts` — register the new routes.
- `backend/src/stores/artifact-store.ts` — maybe add `stream(key)` helper if needed; existing `get(key)` returns a Buffer and is sufficient for now.
- `backend/src/__tests__/inference.test.ts` — new tests.
- `src/main.rs` / `src/cli.rs` — add a hidden/internal `infer` subcommand that loads a KV artifact from stdin/path and runs continuation generation.
- `src/local/infer.rs` — library implementation of the inference path.
- `scripts/build-release.sh` — ensure the backend can locate the `kvcdn` binary; document `KVCDN_BINARY_PATH`.
- `README.md` — document the new hosted inference capability.

---

### Task 1: Add internal Rust `kvcdn infer` command

**Files:**
- Create: `src/cli.rs` additions for `InferArgs`
- Create: `src/local/infer.rs`
- Modify: `src/main.rs` to dispatch `infer`
- Modify: `src/local/mod.rs` to export `infer`

- [ ] **Step 1: Define `InferArgs` in `src/cli.rs`.**

Add after `VerifyArgs`:

```rust
/// Internal command: load a KV artifact and run continuation generation.
/// Not intended for direct user use; invoked by the backend inference endpoint.
#[derive(Parser)]
pub struct InferArgs {
    /// Hugging Face model identifier for the artifact.
    #[arg(long)]
    pub model: String,
    /// Path to the KV artifact (or `-` to read from stdin).
    #[arg(long, default_value = "-")]
    pub input: String,
    /// Continuation prompt text.
    #[arg(long)]
    pub question: String,
    /// Number of tokens to generate.
    #[arg(long, default_value_t = 32)]
    pub n: usize,
    /// Hugging Face model revision.
    #[arg(long)]
    pub revision: Option<String>,
    /// Device to run on.
    #[arg(long, value_parser = crate::models::engine::parse_device)]
    pub device: Option<candle_core::Device>,
}
```

Add `Infer(InferArgs)` to the top-level `Cli` enum.

- [ ] **Step 2: Implement `src/local/infer.rs`.**

```rust
use std::fs;
use std::io::{self, Read};

use anyhow::{Context, Result};
use candle_core::DType;

use crate::cli::InferArgs;
use crate::local::continuation::generate_with_model;
use crate::local::kv_io;
use crate::local::tokenize::encode;
use crate::models::engine::{load_model_on, resolve_revision};

pub fn run(args: InferArgs) -> Result<Vec<u32>> {
    let device = args
        .device
        .clone()
        .unwrap_or_else(|| crate::core::common::pick_device().unwrap_or(candle_core::Device::Cpu));
    let mut bundle = load_model_on(
        &args.model,
        &resolve_revision(args.revision.as_deref()),
        DType::F16,
        device.clone(),
    )?;

    let mut kv_bytes = Vec::new();
    if args.input == "-" {
        io::stdin()
            .read_to_end(&mut kv_bytes)
            .context("reading KV artifact from stdin")?;
    } else {
        kv_bytes = fs::read(&args.input)
            .with_context(|| format!("reading KV artifact from {}", args.input))?;
    }

    // kv_io::load_kv expects a file path. Write to a temp file.
    let tmp_path = std::env::temp_dir().join(format!("kvcdn-infer-{}.kv", std::process::id()));
    fs::write(&tmp_path, &kv_bytes).with_context(|| "writing temporary KV artifact")?;

    let (cache, _) = kv_io::load_kv(&tmp_path, &bundle.device)
        .with_context(|| "loading KV artifact")?;
    bundle
        .model
        .set_kv_cache(cache)
        .with_context(|| "setting KV cache on model")?;

    let q_tokens = encode(&bundle.tokenizer, &args.question, false)
        .with_context(|| "encoding continuation prompt")?;
    let num_ctx_tokens = cache[0].0.dim(2).unwrap_or(0);
    let generated = generate_with_model(
        &mut *bundle.model,
        &bundle.device,
        &q_tokens,
        num_ctx_tokens,
        args.n,
    )
    .with_context(|| "running continuation generation")?;

    // Clean up temp file; ignore errors.
    let _ = fs::remove_file(&tmp_path);

    Ok(generated)
}
```

- [ ] **Step 3: Wire `infer` into `src/main.rs`.**

Add to the match:

```rust
Cli::Infer(args) => {
    let tokens = local::infer::run(args)?;
    for token in tokens {
        println!("{token}");
    }
    Ok(())
}
```

- [ ] **Step 4: Export `infer` in `src/local/mod.rs`.**

```rust
pub mod infer;
```

- [ ] **Step 5: Add a Rust unit test that `infer` loads an artifact and produces tokens.**

In `src/local/infer.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::InferArgs;

    #[test]
    fn infer_runs_without_crashing() {
        // This is a smoke test: without a real model checkpoint in CI,
        // we can only verify argument parsing and path handling.
        let args = InferArgs {
            model: "Qwen/Qwen3-0.6B".to_string(),
            input: "/nonexistent.kv".to_string(),
            question: "Q".to_string(),
            n: 1,
            revision: None,
            device: None,
        };
        assert!(run(args).is_err());
    }
}
```

- [ ] **Step 6: Run Rust checks.**

Run:
```bash
cargo build --release
```

Expected: builds successfully.

- [ ] **Step 7: Commit.**

```bash
git add src/cli.rs src/local/infer.rs src/local/mod.rs src/main.rs
git commit -m "feat(cli): add internal infer command for backend inference"
```

---

### Task 2: Add Fastify inference route

**Files:**
- Create: `backend/src/routes/inference.ts`
- Modify: `backend/src/index.ts`
- Modify: `backend/src/stores/artifact-store.ts` (optional, no change likely)

- [ ] **Step 1: Create `backend/src/routes/inference.ts`.**

```typescript
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { promises as fs } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { FastifyInstance, FastifyReply, FastifyRequest } from "fastify";
import type { Tenant } from "../stores/tenant-resolver.js";

interface InferenceParams {
  org: string;
  project: string;
  artifact_id: string;
}

interface InferenceBody {
  question: string;
  n?: number;
}

async function requireAuth(request: FastifyRequest, fastify: FastifyInstance): Promise<Tenant> {
  const authHeader = request.headers.authorization;
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    throw new Error("Unauthorized");
  }
  const identity = await fastify.identityVerifier.verify(authHeader.slice("Bearer ".length));
  return fastify.tenantResolver.resolve(identity);
}

async function optionalAuth(request: FastifyRequest, fastify: FastifyInstance): Promise<Tenant | undefined> {
  const authHeader = request.headers.authorization;
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return undefined;
  }
  try {
    const identity = await fastify.identityVerifier.verify(authHeader.slice("Bearer ".length));
    return fastify.tenantResolver.resolve(identity);
  } catch {
    return undefined;
  }
}

function kvcdnBinaryPath(): string {
  return process.env.KVCDN_BINARY_PATH ?? "kvcdn";
}

function isValidArtifactId(id: string): boolean {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id);
}

export async function inferenceRoutes(fastify: FastifyInstance): Promise<void> {
  fastify.post<{
    Params: InferenceParams;
    Body: InferenceBody;
  }>("/v1/orgs/:org/projects/:project/artifacts/:artifact_id/infer", async (request, reply) => {
    const tenant = await optionalAuth(request, fastify);
    const { org, project, artifact_id: artifactId } = request.params;
    if (!isValidArtifactId(artifactId)) {
      return reply.status(400).send({ error: "invalid artifact_id" });
    }

    let meta = tenant
      ? await fastify.metadataStore.get(tenant.customerId, org, project, artifactId)
      : undefined;
    if (!meta) {
      meta = await fastify.metadataStore.getPublic(org, project, artifactId);
    }
    if (!meta || !fastify.accessPolicy.canAccess({ customerId: tenant?.customerId }, meta)) {
      return reply.status(tenant ? 404 : 401).send({ error: tenant ? "Artifact not found" : "Unauthorized" });
    }

    const namespace = `artifacts/customers/${meta.customer_id}/orgs/${meta.org}/projects/${meta.project}`;
    const dataKey = `${namespace}/${artifactId}/${meta.name}`;

    let kvBytes: Buffer;
    try {
      kvBytes = await fastify.artifactStore.get(dataKey);
    } catch (err) {
      request.log.warn({ err }, "failed to fetch artifact bytes");
      return reply.status(502).send({ error: "Failed to fetch artifact" });
    }

    const tmpPath = join(tmpdir(), `kvcdn-infer-${artifactId}.kv`);
    await fs.writeFile(tmpPath, kvBytes);

    const question = request.body.question ?? "";
    const n = Math.min(Math.max(Number(request.body.n) || 32, 1), 256);

    const binary = kvcdnBinaryPath();
    const args = [
      "infer",
      "--model", meta.model_name ?? meta.name,
      "--input", tmpPath,
      "--question", question,
      "--n", String(n),
    ];

    return new Promise<void>((resolve) => {
      const child = spawn(binary, args, { stdio: ["ignore", "pipe", "pipe"] });
      let stdout = "";
      let stderr = "";
      child.stdout.on("data", (chunk) => { stdout += chunk; });
      child.stderr.on("data", (chunk) => { stderr += chunk; });
      child.on("error", (err) => {
        request.log.warn({ err }, "kvcdn infer process failed to start");
        fs.unlink(tmpPath).catch(() => {});
        reply.status(502).send({ error: "Inference engine unavailable" });
        resolve();
      });
      child.on("close", async (code) => {
        await fs.unlink(tmpPath).catch(() => {});
        if (code !== 0) {
          request.log.warn({ code, stderr }, "kvcdn infer exited non-zero");
          reply.status(502).send({ error: "Inference failed", details: stderr });
        } else {
          const tokens = stdout
            .split("\n")
            .map((s) => s.trim())
            .filter((s) => s.length > 0)
            .map((s) => Number.parseInt(s, 10))
            .filter((n) => !Number.isNaN(n));
          reply.status(200).send({
            artifact_id: artifactId,
            tokens,
            token_count: tokens.length,
          });
        }
        resolve();
      });
    });
  });
}
```

- [ ] **Step 2: Register inference routes in `backend/src/index.ts`.**

Add import:

```typescript
import { inferenceRoutes } from "./routes/inference.js";
```

Add registration after `quotaRoutes`:

```typescript
fastify.register(inferenceRoutes);
```

- [ ] **Step 3: Add model_name to ArtifactMetadata so the backend knows which model to load.**

The current metadata has `name` (artifact filename) but not the HF model identifier. Add `model_name: string` to `backend/src/stores/metadata-store.ts` and `backend/src/types.ts`, and ensure the upload route captures it in `UploadMetaSchema`. Update `artifactMeta()` in `backend/src/__tests__/helpers.ts` to include a realistic `model_name`.

In `backend/src/types.ts`, extend `UploadMetaSchema`:

```typescript
export const UploadMetaSchema = z.object({
  // existing fields...
  model_name: z.string().min(1),
});
```

In `backend/src/routes/artifacts.ts`, save `model_name` into `artifactMeta`.

- [ ] **Step 4: Run backend type check and tests.**

Run:
```bash
cd backend
npm run build
npm test
```

Expected: build clean; existing tests pass; new inference tests added next will fail until implemented.

- [ ] **Step 5: Commit.**

```bash
git add backend/src/routes/inference.ts backend/src/index.ts backend/src/stores/metadata-store.ts backend/src/types.ts backend/src/routes/artifacts.ts backend/src/__tests__/helpers.ts
git commit -m "feat(backend): add POST /v1/.../artifacts/:id/infer endpoint"
```

---

### Task 3: Add backend tests for inference endpoint

**Files:**
- Create: `backend/src/__tests__/inference.test.ts`

- [ ] **Step 1: Create `backend/src/__tests__/inference.test.ts`.**

```typescript
import { describe, expect, it, beforeAll, afterAll } from "@jest/globals";
import type { FastifyInstance } from "fastify";
import { artifactMeta, getAccessToken, startTestServer } from "./helpers.js";

process.env.KVCDN_CLIENT_ID = "kvcdn-cli";

describe("POST /v1/.../artifacts/:artifact_id/infer", () => {
  let app: FastifyInstance | undefined;
  let baseUrl: string | undefined;

  beforeAll(async () => {
    ({ app, url: baseUrl } = await startTestServer());
    process.env.KVCDN_ISSUER_URL = `${baseUrl}/oidc`;
  });

  afterAll(async () => {
    await app?.close();
    delete process.env.KVCDN_ISSUER_URL;
  });

  it("returns 401 when artifact is private and no auth is provided", async () => {
    const meta = artifactMeta();
    const token = await getAccessToken(baseUrl!);
    const uploadResponse = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify(meta),
    });
    expect(uploadResponse.status).toBe(201);
    const { artifact_id: artifactId } = (await uploadResponse.json()) as { artifact_id: string };

    const inferResponse = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts/${artifactId}/infer`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question: "hello" }),
    });
    expect(inferResponse.status).toBe(401);
  });

  it("returns 502 when kvcdn binary is not available", async () => {
    process.env.KVCDN_BINARY_PATH = "/nonexistent/kvcdn";
    const meta = artifactMeta();
    const token = await getAccessToken(baseUrl!);
    const uploadResponse = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify(meta),
    });
    expect(uploadResponse.status).toBe(201);
    const { artifact_id: artifactId } = (await uploadResponse.json()) as { artifact_id: string };

    const inferResponse = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts/${artifactId}/infer`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify({ question: "hello" }),
    });
    expect(inferResponse.status).toBe(502);
    delete process.env.KVCDN_BINARY_PATH;
  });

  it("returns 400 for an invalid artifact_id", async () => {
    const token = await getAccessToken(baseUrl!);
    const response = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts/not-a-uuid/infer`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify({ question: "hello" }),
    });
    expect(response.status).toBe(400);
  });
});
```

- [ ] **Step 2: Update `backend/src/__tests__/helpers.ts` to include `model_name`.**

```typescript
export function artifactMeta() {
  return {
    name: "context.kv",
    size_bytes: 1024,
    sha256: "a".repeat(64),
    dtype: "F16",
    num_tokens: 64,
    num_layers: 12,
    quantized: false,
    model_name: "Qwen/Qwen3-0.6B",
  };
}
```

- [ ] **Step 3: Run backend tests.**

Run:
```bash
cd backend
npm test -- inference.test.ts
```

Expected: tests pass.

- [ ] **Step 4: Commit.**

```bash
git add backend/src/__tests__/inference.test.ts backend/src/__tests__/helpers.ts
git commit -m "test(backend): add inference endpoint tests"
```

---

### Task 4: Document the new inference endpoint

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Add inference section to README hosted usage.**

After the upload/download examples, add:

```markdown
## Hosted inference

If a KV artifact has been uploaded, you can ask the backend to continue generation from the cached prefill without downloading the artifact locally:

```bash
curl -X POST "https://$API_HOST/v1/orgs/$ORG/projects/$PROJECT/artifacts/$ARTIFACT_ID/infer" \
  -H "Authorization: Bearer $(kvcdn whoami --format json | jq -r .access_token)" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main point?", "n": 32}'
```

The response contains the generated token IDs:

```json
{"artifact_id": "...", "tokens": [123, 456, ...], "token_count": 32}
```

The backend spawns the local `kvcdn` inference engine; set `KVCDN_BINARY_PATH` if the binary is not on the backend's `PATH`.
```

- [ ] **Step 2: Add `KVCDN_BINARY_PATH` note to AGENTS.md.**

Under "Things that are easy to miss":

```markdown
- The hosted inference endpoint (`POST /v1/orgs/:org/projects/:project/artifacts/:artifact_id/infer`) requires the `kvcdn` release binary to be on the backend's `PATH` or pointed to by `KVCDN_BINARY_PATH`.
```

- [ ] **Step 3: Run full checks.**

Run:
```bash
cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test --workspace
cd backend && npm run build && npm test && cd ..
```

Expected: clean.

- [ ] **Step 4: Commit.**

```bash
git add README.md AGENTS.md
git commit -m "docs: document hosted inference endpoint"
```

---

## Spec coverage check

- Cache-and-serve value proposition: covered by new `/infer` endpoint and docs.
- Auth/access policy reuse: covered by `optionalAuth` and existing `canAccess`.
- Binary spawning vs native embedding: covered; binary path chosen to avoid Node.js Candle bindings.

## Placeholder scan

No TBD/TODO placeholders. All code shown.

## Type consistency check

- `ArtifactMetadata.model_name` added consistently in store interface, schema, route meta, and test helper.
- `InferArgs` matches `Cli::Infer` variant.
