# Storage, tenancy, and transfer seam refactor

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace direct S3 imports and raw HTTP upload/download code with deep seams (`ArtifactStore`, `MetadataStore`, `TenantNamespace`, `Transfer`, `CredentialStore`) so the backend supports multiple object stores and the CLI supports testable, swappable upload/download adapters.

**Architecture:** Each seam is a small interface with one production adapter and one in-memory/fake adapter. The artifact route receives adapters via Fastify decorators; the `ApiClient` and upload/download commands receive adapters via constructor injection. No behavior changes for users.

**Tech Stack:** TypeScript/Fastify/Jest for backend; Rust/reqwest/indicatif for CLI.

---

## Task 1: Introduce `ArtifactStore` seam in the backend

**Files:**
- Create: `backend/src/stores/artifact-store.ts`
- Create: `backend/src/stores/s3-artifact-store.ts`
- Create: `backend/src/stores/memory-artifact-store.ts`
- Modify: `backend/src/s3.ts` (re-export the S3 adapter for compatibility)
- Modify: `backend/src/index.ts`
- Modify: `backend/src/routes/artifacts.ts`
- Modify: `backend/src/__tests__/artifacts.test.ts`
- Delete: `backend/src/__mocks__/s3.js`

**Interface (`ArtifactStore`):**

```typescript
export interface StoredObject {
  key: string;
  lastModified?: Date;
  size?: number;
}

export interface ListPage {
  objects: StoredObject[];
  nextContinuationToken?: string;
}

export interface ArtifactStore {
  presignedUploadUrl(key: string, sizeBytes: number): Promise<string>;
  presignedDownloadUrl(key: string): Promise<string>;
  put(key: string, body: Buffer | string, contentType?: string): Promise<void>;
  get(key: string): Promise<Buffer>;
  delete(key: string): Promise<void>;
  list(prefix: string): Promise<StoredObject[]>;
  listPaginated(prefix: string, maxKeys: number, continuationToken?: string): Promise<ListPage>;
}
```

**Steps:**

- [ ] **Step 1: Define the interface**

  Create `backend/src/stores/artifact-store.ts` with the interface above.

- [ ] **Step 2: Move S3 logic into an adapter**

  Create `backend/src/stores/s3-artifact-store.ts`. Copy the implementation from `backend/src/s3.ts`, wrap it in a class `S3ArtifactStore` implementing `ArtifactStore`, and read `process.env` once in the constructor.

  ```typescript
  export class S3ArtifactStore implements ArtifactStore {
    private client: S3Client;
    private bucket: string;

    constructor() {
      const endpoint = process.env.KVCDN_S3_ENDPOINT;
      const accessKeyId = process.env.KVCDN_S3_ACCESS_KEY;
      const secretAccessKey = process.env.KVCDN_S3_SECRET_KEY;
      const bucket = process.env.KVCDN_S3_BUCKET;
      if (!endpoint || !accessKeyId || !secretAccessKey || !bucket) {
        throw new Error("S3 credentials are not configured");
      }
      this.bucket = bucket;
      this.client = new S3Client({
        region: "us-east-1",
        endpoint,
        forcePathStyle: true,
        credentials: { accessKeyId, secretAccessKey },
      });
    }
    // ... implement interface methods
  }
  ```

- [ ] **Step 3: Add in-memory adapter for tests**

  Create `backend/src/stores/memory-artifact-store.ts`:

  ```typescript
  export class MemoryArtifactStore implements ArtifactStore {
    private objects = new Map<string, Buffer>();
    presignedUploadUrl(key: string): Promise<string> { return Promise.resolve(`https://memory.test/${key}?upload=true`); }
    presignedDownloadUrl(key: string): Promise<string> { return Promise.resolve(`https://memory.test/${key}?download=true`); }
    put(key: string, body: Buffer | string): Promise<void> { this.objects.set(key, Buffer.isBuffer(body) ? body : Buffer.from(body, "utf-8")); return Promise.resolve(); }
    get(key: string): Promise<Buffer> { const v = this.objects.get(key); if (!v) throw new Error(`Object not found: ${key}`); return Promise.resolve(v); }
    delete(key: string): Promise<void> { this.objects.delete(key); return Promise.resolve(); }
    list(prefix: string): Promise<StoredObject[]> { ... }
    listPaginated(prefix: string, maxKeys: number, continuationToken?: string): Promise<ListPage> { ... }
  }
  ```

- [ ] **Step 4: Keep `s3.ts` as a compatibility re-export**

  Replace `backend/src/s3.ts` with:

  ```typescript
  import { S3ArtifactStore } from "./stores/s3-artifact-store.js";

  const singleton = new S3ArtifactStore();
  export const presignedPutUrl = singleton.presignedUploadUrl.bind(singleton);
  export const presignedGetUrl = singleton.presignedDownloadUrl.bind(singleton);
  export const putObject = singleton.put.bind(singleton);
  export const getObject = singleton.get.bind(singleton);
  export const deleteObject = singleton.delete.bind(singleton);
  export const listObjects = singleton.list.bind(singleton);
  export const listObjectsPaginated = singleton.listPaginated.bind(singleton);
  ```

- [ ] **Step 5: Register the store on Fastify**

  Modify `backend/src/index.ts` to declare and register a default `ArtifactStore`:

  ```typescript
  import { S3ArtifactStore } from "./stores/s3-artifact-store.js";
  import type { ArtifactStore } from "./stores/artifact-store.js";

  declare module "fastify" {
    interface FastifyInstance {
      artifactStore: ArtifactStore;
    }
  }

  export async function build(store?: ArtifactStore): Promise<FastifyInstance> {
    const fastify = Fastify({ logger: true });
    fastify.decorate("artifactStore", store ?? new S3ArtifactStore());
    // ... rest
  }
  ```

- [ ] **Step 6: Update artifact route to use the store**

  In `backend/src/routes/artifacts.ts`, replace all direct `s3.js` imports with `fastify.artifactStore`. The route signature becomes:

  ```typescript
  export async function artifactsRoutes(fastify: FastifyInstance): Promise<void> { ... }
  ```

  Inside handlers, use `fastify.artifactStore.presignedUploadUrl(...)`, etc.

- [ ] **Step 7: Update tests to inject the memory store**

  In `backend/src/__tests__/artifacts.test.ts`, remove `jest.unstable_mockModule("../s3.js", ...)` and the `putObject` import. Instead:

  ```typescript
  import { MemoryArtifactStore } from "../stores/memory-artifact-store.js";
  import type { ArtifactStore } from "../stores/artifact-store.js";

  const memoryStore = new MemoryArtifactStore();

  async function startTestServer(port = 0): Promise<{ app: FastifyInstance; url: string }> {
    const { build } = await import("../index.js");
    const app = await build(memoryStore);
    // ...
  }
  ```

  Update the SHA-256 completion test to use `memoryStore.put(...)` instead of `putObject(...)`.

- [ ] **Step 8: Delete the old manual mock**

  Remove `backend/src/__mocks__/s3.js`.

- [ ] **Step 9: Run backend tests**

  ```bash
  cd backend
  NODE_OPTIONS='--experimental-vm-modules' npm test
  ```

  Expected: all existing tests pass.

- [ ] **Step 10: Commit**

  ```bash
  git add backend/src/stores backend/src/s3.ts backend/src/index.ts backend/src/routes/artifacts.ts backend/src/__tests__/artifacts.test.ts
  git rm backend/src/__mocks__/s3.js
  git commit -m "backend: introduce ArtifactStore seam with S3 and in-memory adapters"
  ```

---

## Task 2: Move tenant namespace construction into `VerifiedCustomer`

**Files:**
- Modify: `backend/src/auth.ts`
- Modify: `backend/src/routes/artifacts.ts`
- Modify: `backend/src/__tests__/artifacts.test.ts`

**Steps:**

- [ ] **Step 1: Add namespace methods to `VerifiedCustomer`**

  In `backend/src/auth.ts`:

  ```typescript
  export interface VerifiedCustomer {
    tokenPayload: TokenPayload;
    customerId: string;
    namespace(): string;
    artifactPath(org: string, project: string, artifactId: string, name: string): string;
    metaPath(org: string, project: string, artifactId: string): string;
    prefix(org: string, project: string, artifactId: string): string;
  }
  ```

  Implement `namespace()` as `artifacts/customers/${customerId}`; other methods build under that prefix.

- [ ] **Step 2: Update `verifyApiKey` and `verifyBearer`**

  Return objects with the new methods.

- [ ] **Step 3: Replace key helpers in artifact route**

  Remove `artifactPrefix`, `metaKey`, `dataKey` from `artifacts.ts`. Use `customer.namespace()`, `customer.metaPath(...)`, `customer.artifactPath(...)`.

- [ ] **Step 4: Update tests**

  Replace direct key construction in the completion test with `customerId` from `deriveCustomerId` and `memoryStore.put(customer.artifactPath(...), ...)`. (Test code can still use `deriveCustomerId` directly for the data key.)

- [ ] **Step 5: Add isolation unit test**

  Add a test that proves `customerA.namespace()` is not a prefix of `customerB.namespace()` for two different customers.

- [ ] **Step 6: Run tests and commit**

  ```bash
  cd backend && NODE_OPTIONS='--experimental-vm-modules' npm test
  git add backend/src/auth.ts backend/src/routes/artifacts.ts backend/src/__tests__/artifacts.test.ts
  git commit -m "backend: move tenant namespace construction into VerifiedCustomer"
  ```

---

## Task 3: Introduce `MetadataStore` seam

**Files:**
- Create: `backend/src/stores/metadata-store.ts`
- Create: `backend/src/stores/json-metadata-store.ts`
- Create: `backend/src/stores/memory-metadata-store.ts`
- Modify: `backend/src/routes/artifacts.ts`
- Modify: `backend/src/index.ts`
- Modify: `backend/src/__tests__/artifacts.test.ts`

**Interface (`MetadataStore`):**

```typescript
export interface ArtifactMeta {
  artifact_id: string;
  name: string;
  size_bytes: number;
  sha256: string;
  dtype: string;
  storage_dtype?: string;
  num_tokens: number;
  num_layers: number;
  quantized: boolean;
  visibility: "public" | "private";
  customer_id: string;
  created_at: string;
}

export interface MetadataStore {
  load(key: string): Promise<ArtifactMeta | undefined>;
  save(key: string, meta: ArtifactMeta): Promise<void>;
  delete(key: string): Promise<void>;
}
```

**Steps:**

- [ ] **Step 1: Create the interface**

  Create `backend/src/stores/metadata-store.ts`. Reuse and export `ArtifactMeta` from there.

- [ ] **Step 2: Implement JSON metadata store over ArtifactStore**

  Create `backend/src/stores/json-metadata-store.ts`:

  ```typescript
  export class JsonMetadataStore implements MetadataStore {
    constructor(private store: ArtifactStore) {}
    async load(key: string): Promise<ArtifactMeta | undefined> { ... try getObject + JSON.parse ... }
    async save(key: string, meta: ArtifactMeta): Promise<void> { ... putObject JSON buffer ... }
    async delete(key: string): Promise<void> { ... deleteObject ... }
  }
  ```

- [ ] **Step 3: Add memory metadata store for tests**

  Create `backend/src/stores/memory-metadata-store.ts` backed by a `Map<string, ArtifactMeta>`.

- [ ] **Step 4: Register metadata store on Fastify**

  In `backend/src/index.ts`, add `metadataStore` decorator and build parameter.

- [ ] **Step 5: Update artifact route**

  Use `fastify.metadataStore.load/save/delete` for all `.meta.json` operations. Remove JSON.parse/stringify from the route.

- [ ] **Step 6: Update tests**

  Inject `MemoryMetadataStore` alongside `MemoryArtifactStore` in `startTestServer`.

- [ ] **Step 7: Run tests and commit**

  ```bash
  cd backend && NODE_OPTIONS='--experimental-vm-modules' npm test
  git add backend/src/stores backend/src/index.ts backend/src/routes/artifacts.ts backend/src/__tests__/artifacts.test.ts
  git commit -m "backend: introduce MetadataStore seam"
  ```

---

## Task 4: Replace public artifact pointer with `AccessPolicy`

**Files:**
- Create: `backend/src/access-policy.ts`
- Modify: `backend/src/routes/artifacts.ts`
- Modify: `backend/src/stores/metadata-store.ts` (add public flag lookup)
- Modify: `backend/src/__tests__/artifacts.test.ts`

**Steps:**

- [ ] **Step 1: Define `AccessPolicy`**

  Create `backend/src/access-policy.ts`:

  ```typescript
  export interface AccessPolicy {
    canDownload(artifact: ArtifactMeta, requester?: VerifiedCustomer): boolean;
    canList(artifact: ArtifactMeta, requester: VerifiedCustomer): boolean;
    canDelete(artifact: ArtifactMeta, requester: VerifiedCustomer): boolean;
  }

  export class DefaultAccessPolicy implements AccessPolicy {
    canDownload(meta, requester) {
      if (meta.visibility === "public") return true;
      return !!requester && meta.customer_id === requester.customerId;
    }
    canList(meta, requester) { return meta.customer_id === requester.customerId; }
    canDelete(meta, requester) { return meta.customer_id === requester.customerId; }
  }
  ```

- [ ] **Step 2: Update artifact route**

  On upload, do not write a second public pointer. On download, resolve the owner's metadata via `metadataStore` and use `accessPolicy.canDownload`. On delete, use `canDelete`.

- [ ] **Step 3: Update metadata store for lookup by id**

  Add `findById(artifactId: string, org: string, project: string): Promise<{ meta: ArtifactMeta; key: string } | undefined>` to `MetadataStore`. The default implementation can scan the owning customer's prefix; a future adapter can maintain an index.

- [ ] **Step 4: Update tests**

  Public download test should still pass without the duplicate pointer.

- [ ] **Step 5: Run tests and commit**

  ```bash
  cd backend && NODE_OPTIONS='--experimental-vm-modules' npm test
  git add backend/src/access-policy.ts backend/src/stores backend/src/routes/artifacts.ts backend/src/__tests__/artifacts.test.ts
  git commit -m "backend: replace public artifact pointer with AccessPolicy"
  ```

---

## Task 5: Split `auth.ts` into `IdentityVerifier` and `TenantResolver`

**Files:**
- Create: `backend/src/auth/identity-verifier.ts`
- Create: `backend/src/auth/oidc-identity-verifier.ts`
- Create: `backend/src/auth/api-key-identity-verifier.ts`
- Create: `backend/src/auth/tenant-resolver.ts`
- Create: `backend/src/auth/index.ts`
- Delete: `backend/src/auth.ts` (after re-export compat)
- Modify: `backend/src/index.ts`
- Modify: `backend/src/routes/artifacts.ts`
- Modify: `backend/src/routes/api-keys.ts`
- Modify: `backend/src/__tests__/artifacts.test.ts`

**Interfaces:**

```typescript
export interface Identity {
  subject: string; // stable identifier, e.g. "oidc:alice" or "org:acme"
}

export interface IdentityVerifier {
  verify(bearer: string): Promise<Identity>;
}

export interface TenantResolver {
  resolve(identity: Identity): VerifiedCustomer;
}
```

**Steps:**

- [ ] **Step 1: Move OIDC and API-key verification into separate adapters**

  Create `backend/src/auth/oidc-identity-verifier.ts` and `backend/src/auth/api-key-identity-verifier.ts`.

- [ ] **Step 2: Create composite verifier and tenant resolver**

  Create `backend/src/auth/index.ts` that composes both verifiers (try API key, then OIDC) and maps identity subject to customer id.

- [ ] **Step 3: Keep backward-compatible exports**

  Create a thin `backend/src/auth.ts` shim that re-exports from `backend/src/auth/index.ts`, or update imports. Prefer updating imports.

- [ ] **Step 4: Update routes and tests**

  `artifacts.ts` imports `requireAuth`/`optionalAuth` from the new auth module. Tests inject the same verifier/resolver via `build(...)`.

- [ ] **Step 5: Run tests and commit**

  ```bash
  cd backend && NODE_OPTIONS='--experimental-vm-modules' npm test
  git add backend/src/auth backend/src/auth.ts backend/src/routes backend/src/index.ts backend/src/__tests__/artifacts.test.ts
  git commit -m "backend: split auth into IdentityVerifier and TenantResolver seams"
  ```

---

## Task 6: Introduce `Transfer` seam in the CLI

**Files:**
- Create: `src/transfer.rs` (trait + reqwest adapter + fake adapter)
- Modify: `src/upload.rs`
- Modify: `src/download.rs`
- Modify: `src/main.rs` (wire up)
- Modify: `src/api.rs` (return URLs, no HTTP transport)
- Modify: tests

**Trait:**

```rust
pub trait Transfer: Send + Sync {
    fn upload(&self, url: &str, reader: Box<dyn Read>, size: u64, progress: &ProgressBar) -> Result<()>;
    fn download(&self, url: &str, writer: &mut dyn Write, progress: &ProgressBar) -> Result<u64>;
}
```

**Steps:**

- [ ] **Step 1: Define the trait**

  Create `src/transfer.rs`.

- [ ] **Step 2: Implement `ReqwestTransfer`**

  Move the current `reqwest::blocking` upload/download logic from `upload.rs` and `download.rs` into `ReqwestTransfer`.

- [ ] **Step 3: Implement `FakeTransfer` for tests**

  Record bytes sent/received in memory.

- [ ] **Step 4: Update `upload.rs`**

  Accept a `&dyn Transfer` (or `impl Transfer`) parameter, or store it in `UploadCommand`. Remove direct `reqwest` usage.

- [ ] **Step 5: Update `download.rs`**

  Same pattern.

- [ ] **Step 6: Wire in `main.rs`**

  Use `ReqwestTransfer::new()` as the production default.

- [ ] **Step 7: Add tests**

  Write a test that uses `FakeTransfer` to verify exact upload/download bytes without network.

- [ ] **Step 8: Run Rust tests and commit**

  ```bash
  cargo test
  git add src/transfer.rs src/upload.rs src/download.rs src/main.rs
  git commit -m "cli: introduce Transfer seam for upload/download"
  ```

---

## Task 7: Inject `CredentialStore` into `ApiClient`

**Files:**
- Create: `src/credential_store.rs` (trait + file-based impl)
- Modify: `src/api.rs`
- Modify: `src/main.rs`
- Modify: tests

**Steps:**

- [ ] **Step 1: Define the trait**

  ```rust
  pub trait CredentialStore: Send + Sync {
      fn load_api_key(&self) -> Result<Option<String>>;
      fn load_tokens(&self, key: &Key) -> Result<Option<Tokens>>;
      fn save_tokens(&self, tokens: &Tokens, key: &Key) -> Result<()>;
  }
  ```

- [ ] **Step 2: Implement `FileCredentialStore`**

  Move the current `api_key::load_stored_api_key` and `TokenStore` usage into it.

- [ ] **Step 3: Update `ApiClient::new`**

  Accept `Arc<dyn CredentialStore>` and a `Key`. Remove direct calls to `TokenStore`/`api_key::load_stored_api_key`.

- [ ] **Step 4: Update tests**

  Provide a `MemoryCredentialStore` for tests.

- [ ] **Step 5: Run Rust tests and commit**

  ```bash
  cargo test
  git add src/credential_store.rs src/api.rs src/main.rs
  git commit -m "cli: inject CredentialStore into ApiClient"
  ```

---

## Task 8: Full verification and final notes

- [ ] **Step 1: Run backend test suite**

  ```bash
  cd backend
  NODE_OPTIONS='--experimental-vm-modules' npm test
  npm run build
  ```

- [ ] **Step 2: Run Rust test suite**

  ```bash
  cargo test
  cargo fmt --check
  cargo clippy --all-targets -- -D warnings
  ```

- [ ] **Step 3: Update ADR references if needed**

  Ensure `docs/adr/0001-artifact-data-unencrypted-at-rest.md` still reflects the implemented seams.

- [ ] **Step 5: Final commit / branch completion**

  Use `superpowers:finishing-a-development-branch` to present merge options.
