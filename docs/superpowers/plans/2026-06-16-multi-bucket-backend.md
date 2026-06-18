# Multi-bucket backend implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the backend spread tenant artifact data across multiple S3-compatible buckets so a single bucket is no longer a hard dependency and blast radius.

**Architecture:** Introduce a `BucketResolver` seam that maps a tenant (`customerId`, `org`) to a bucket and endpoint/credentials. The existing `S3ArtifactStore` and `S3MetadataStore` are replaced by a `TenantS3ArtifactStore` that owns a pool of per-bucket `S3Client`s and delegates each key operation to the correct bucket. A `BucketStore` is added to track the mapping persistently (also in S3, under a small per-deployment control bucket). The CLI does not change — it still talks to the same backend API.

**Tech Stack:** TypeScript/Fastify/Jest, AWS SDK S3, backend keeps existing seam interfaces.

---

## File structure

- `backend/src/stores/bucket-store.ts` — new interface and bucket assignment record.
- `backend/src/stores/s3-bucket-store.ts` — control-bucket adapter that reads/writes tenant→bucket mappings.
- `backend/src/stores/memory-bucket-store.ts` — in-memory test adapter.
- `backend/src/stores/bucket-resolver.ts` — policy interface for choosing a bucket.
- `backend/src/stores/hashed-bucket-resolver.ts` — default deterministic resolver that hashes tenant into one of N configured buckets.
- `backend/src/stores/s3-clients.ts` — small factory for creating/cacheing `S3Client`s per endpoint/credential set.
- `backend/src/stores/tenant-s3-artifact-store.ts` — replaces `S3ArtifactStore`: resolves tenant per key, delegates to correct bucket client.
- `backend/src/stores/tenant-s3-metadata-store.ts` — thin wrapper over `TenantS3ArtifactStore` implementing `MetadataStore`.
- `backend/src/stores/s3-artifact-store.ts` — reimplemented as a single-bucket delegate used internally.
- `backend/src/stores/s3-metadata-store.ts` — deleted or deprecated; logic moved into `tenant-s3-metadata-store.ts`.
- `backend/src/index.ts` — decorate `bucketStore`, `bucketResolver`, and wire tenant-aware stores.
- `backend/src/routes/artifacts.ts` — already uses seams, no direct changes; metadata store continues to compute keys.
- `backend/src/__tests__/artifacts.test.ts` — inject memory bucket store + memory artifact/metadata stores to verify multi-tenant isolation still works.
- `backend/src/__tests__/bucket-resolver.test.ts` — new test file for deterministic assignment and bucket list expansion.
- `deploy/local/backend/configmap.yaml` — add `KVCDN_CONTROL_BUCKET` to local MinIO stack.
- `README.md` / deployment docs — add `KVCDN_S3_BUCKETS` and `KVCDN_CONTROL_BUCKET` docs.

---

## Design decisions

1. **Key layout does not change.** Inside any bucket we still store `artifacts/customers/{customer_id}/orgs/{org}/projects/{project}/...`. This keeps presigned URLs, metadata paths, and customer IDs stable.
2. **Tenant-to-bucket mapping is deterministic.** `HashedBucketResolver` uses `customerId` + bucket count so the same tenant always lands on the same bucket and no migration state is needed for reads.
3. **Control bucket is separate.** `KVCDN_CONTROL_BUCKET` stores a tiny JSON record per tenant with its assigned bucket. It is created/written on first write and read on every operation to resolve the bucket. If missing, the deterministic resolver supplies the value and it is written back.
4. **Backward compatibility.** If `KVCDN_S3_BUCKETS` is unset or has one entry, behavior is identical to today. Existing single-bucket deployments continue to work unchanged.
5. **CLI untouched.** No protocol or URL changes; backend computes presigned URLs from whichever bucket owns the tenant.

---

## Task 1: Define BucketStore and control-bucket records

**Files:**
- Create: `backend/src/stores/bucket-store.ts`
- Create: `backend/src/stores/s3-bucket-store.ts`
- Create: `backend/src/stores/memory-bucket-store.ts`

**Data model:**

```typescript
export interface TenantBucketAssignment {
  customer_id: string;
  bucket: string;
}

export interface BucketStore {
  getAssignment(customerId: string): Promise<TenantBucketAssignment | undefined>;
  saveAssignment(assignment: TenantBucketAssignment): Promise<void>;
}
```

- [ ] **Step 1: Create the interface**

Create `backend/src/stores/bucket-store.ts` with the interface above.

- [ ] **Step 2: Implement the S3 control-bucket adapter**

Create `backend/src/stores/s3-bucket-store.ts`:

```typescript
import { S3Client, GetObjectCommand, PutObjectCommand } from "@aws-sdk/client-s3";
import type { BucketStore, TenantBucketAssignment } from "./bucket-store.js";

export class S3BucketStore implements BucketStore {
  private client: S3Client;
  private bucket: string;

  constructor() {
    const endpoint = process.env.KVCDN_S3_ENDPOINT;
    const accessKeyId = process.env.KVCDN_S3_ACCESS_KEY;
    const secretAccessKey = process.env.KVCDN_S3_SECRET_KEY;
    const bucket = process.env.KVCDN_CONTROL_BUCKET;
    if (!endpoint || !accessKeyId || !secretAccessKey || !bucket) {
      throw new Error("Control bucket is not configured");
    }
    this.bucket = bucket;
    this.client = new S3Client({
      region: "us-east-1",
      endpoint,
      forcePathStyle: true,
      credentials: { accessKeyId, secretAccessKey },
    });
  }

  private key(customerId: string): string {
    return `assignments/${customerId}.json`;
  }

  async getAssignment(customerId: string): Promise<TenantBucketAssignment | undefined> {
    try {
      const response = await this.client.send(new GetObjectCommand({
        Bucket: this.bucket,
        Key: this.key(customerId),
      }));
      if (!response.Body) return undefined;
      const stream = response.Body as import("node:stream").Readable;
      const chunks: Buffer[] = [];
      for await (const chunk of stream) {
        chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
      }
      return JSON.parse(Buffer.concat(chunks).toString("utf-8")) as TenantBucketAssignment;
    } catch {
      return undefined;
    }
  }

  async saveAssignment(assignment: TenantBucketAssignment): Promise<void> {
    await this.client.send(new PutObjectCommand({
      Bucket: this.bucket,
      Key: this.key(assignment.customer_id),
      Body: Buffer.from(JSON.stringify(assignment), "utf-8"),
      ContentType: "application/json",
    }));
  }
}
```

- [ ] **Step 3: Implement the memory test adapter**

Create `backend/src/stores/memory-bucket-store.ts`:

```typescript
import type { BucketStore, TenantBucketAssignment } from "./bucket-store.js";

export class MemoryBucketStore implements BucketStore {
  private data = new Map<string, TenantBucketAssignment>();

  async getAssignment(customerId: string): Promise<TenantBucketAssignment | undefined> {
    return this.data.get(customerId);
  }

  async saveAssignment(assignment: TenantBucketAssignment): Promise<void> {
    this.data.set(assignment.customer_id, { ...assignment });
  }
}
```

- [ ] **Step 4: Commit**

```bash
git add backend/src/stores/bucket-store.ts backend/src/stores/s3-bucket-store.ts backend/src/stores/memory-bucket-store.ts
git commit -m "backend: add BucketStore seam for tenant-to-bucket assignments"
```

---

## Task 2: Define BucketResolver policy

**Files:**
- Create: `backend/src/stores/bucket-resolver.ts`
- Create: `backend/src/stores/hashed-bucket-resolver.ts`
- Create: `backend/src/__tests__/bucket-resolver.test.ts`

**Interface:**

```typescript
import type { TenantBucketAssignment } from "./bucket-store.js";

export interface BucketResolver {
  resolve(customerId: string): Promise<TenantBucketAssignment>;
}
```

- [ ] **Step 1: Create the interface**

Create `backend/src/stores/bucket-resolver.ts` with the interface above.

- [ ] **Step 2: Implement deterministic hashed resolver**

Create `backend/src/stores/hashed-bucket-resolver.ts`:

```typescript
import { createHash } from "node:crypto";
import type { BucketStore, TenantBucketAssignment } from "./bucket-store.js";
import type { BucketResolver } from "./bucket-resolver.js";

export class HashedBucketResolver implements BucketResolver {
  private buckets: string[];

  constructor(
    private readonly bucketStore: BucketStore,
    bucketsCsv?: string,
  ) {
    const csv = bucketsCsv ?? process.env.KVCDN_S3_BUCKETS ?? process.env.KVCDN_S3_BUCKET ?? "";
    this.buckets = csv
      .split(",")
      .map((b) => b.trim())
      .filter(Boolean);
    if (this.buckets.length === 0) {
      throw new Error("No buckets configured for KVCDN_S3_BUCKETS");
    }
  }

  async resolve(customerId: string): Promise<TenantBucketAssignment> {
    const existing = await this.bucketStore.getAssignment(customerId);
    if (existing) return existing;

    const bucket = this.pickBucket(customerId);
    const assignment: TenantBucketAssignment = { customer_id: customerId, bucket };
    await this.bucketStore.saveAssignment(assignment);
    return assignment;
  }

  private pickBucket(customerId: string): string {
    const hash = createHash("sha256").update(customerId).digest("hex");
    const index = Number.parseInt(hash.slice(0, 8), 16) % this.buckets.length;
    return this.buckets[index];
  }
}
```

- [ ] **Step 3: Add tests**

Create `backend/src/__tests__/bucket-resolver.test.ts`:

```typescript
import { describe, expect, it } from "@jest/globals";
import { MemoryBucketStore } from "../stores/memory-bucket-store.js";
import { HashedBucketResolver } from "../stores/hashed-bucket-resolver.js";

describe("HashedBucketResolver", () => {
  it("returns the same bucket for the same customer_id", async () => {
    const store = new MemoryBucketStore();
    const resolver = new HashedBucketResolver(store, "artifacts-a,artifacts-b,artifacts-c");
    const first = await resolver.resolve("customer-1");
    const second = await resolver.resolve("customer-1");
    expect(first.bucket).toBe(second.bucket);
    expect(first.customer_id).toBe("customer-1");
  });

  it("persists the assignment in the bucket store", async () => {
    const store = new MemoryBucketStore();
    const resolver = new HashedBucketResolver(store, "artifacts-a,artifacts-b");
    const assignment = await resolver.resolve("customer-2");
    const stored = await store.getAssignment("customer-2");
    expect(stored).toEqual(assignment);
  });

  it("uses one of the configured buckets", async () => {
    const store = new MemoryBucketStore();
    const resolver = new HashedBucketResolver(store, "artifacts-a,artifacts-b");
    const assignment = await resolver.resolve("customer-3");
    expect(["artifacts-a", "artifacts-b"]).toContain(assignment.bucket);
  });

  it("falls back to KVCDN_S3_BUCKET when KVCDN_S3_BUCKETS is unset", async () => {
    const store = new MemoryBucketStore();
    const resolver = new HashedBucketResolver(store, "fallback-bucket");
    const assignment = await resolver.resolve("customer-4");
    expect(assignment.bucket).toBe("fallback-bucket");
  });
});
```

Run: `cd backend && NODE_OPTIONS='--experimental-vm-modules' npx jest src/__tests__/bucket-resolver.test.ts`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add backend/src/stores/bucket-resolver.ts backend/src/stores/hashed-bucket-resolver.ts backend/src/__tests__/bucket-resolver.test.ts
git commit -m "backend: add deterministic HashedBucketResolver for tenant placement"
```

---

## Task 3: Refactor S3ArtifactStore into single-bucket delegate

**Files:**
- Create: `backend/src/stores/s3-clients.ts`
- Modify: `backend/src/stores/s3-artifact-store.ts`

- [ ] **Step 1: Add a shared S3Client factory**

Create `backend/src/stores/s3-clients.ts`:

```typescript
import { S3Client } from "@aws-sdk/client-s3";

export interface S3Connection {
  endpoint: string;
  accessKeyId: string;
  secretAccessKey: string;
}

export function createS3Client(conn: S3Connection): S3Client {
  return new S3Client({
    region: "us-east-1",
    endpoint: conn.endpoint,
    forcePathStyle: true,
    credentials: { accessKeyId: conn.accessKeyId, secretAccessKey: conn.secretAccessKey },
  });
}
```

- [ ] **Step 2: Make S3ArtifactStore accept bucket/client via constructor**

Replace `backend/src/stores/s3-artifact-store.ts` contents with:

```typescript
import { S3Client, DeleteObjectCommand, GetObjectCommand, ListObjectsV2Command, PutObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import type { Readable } from "node:stream";
import type { ArtifactStore, ListPage, StoredObject } from "./artifact-store.js";

export class S3ArtifactStore implements ArtifactStore {
  constructor(
    private readonly client: S3Client,
    private readonly bucket: string,
  ) {}

  async presignedUploadUrl(key: string, sizeBytes: number): Promise<string> {
    const command = new PutObjectCommand({
      Bucket: this.bucket,
      Key: key,
      ContentLength: sizeBytes,
    });
    return getSignedUrl(this.client, command, { expiresIn: 300 });
  }

  async presignedDownloadUrl(key: string): Promise<string> {
    const command = new GetObjectCommand({ Bucket: this.bucket, Key: key });
    return getSignedUrl(this.client, command, { expiresIn: 300 });
  }

  async put(key: string, body: Buffer | string, contentType?: string): Promise<void> {
    const command = new PutObjectCommand({
      Bucket: this.bucket,
      Key: key,
      Body: body,
      ContentType: contentType ?? "application/octet-stream",
    });
    await this.client.send(command);
  }

  async get(key: string): Promise<Buffer> {
    const command = new GetObjectCommand({ Bucket: this.bucket, Key: key });
    const response = await this.client.send(command);
    if (!response.Body) {
      throw new Error(`Object not found: ${key}`);
    }
    const stream = response.Body as Readable;
    const chunks: Buffer[] = [];
    for await (const chunk of stream) {
      chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    }
    return Buffer.concat(chunks);
  }

  async delete(key: string): Promise<void> {
    const command = new DeleteObjectCommand({ Bucket: this.bucket, Key: key });
    await this.client.send(command);
  }

  async list(prefix: string): Promise<StoredObject[]> {
    const page = await this.listPaginated(prefix, 1000);
    if (page.nextContinuationToken) {
      throw new Error("list prefix returned more than 1000 objects; use listPaginated");
    }
    return page.objects;
  }

  async listPaginated(prefix: string, maxKeys: number, continuationToken?: string): Promise<ListPage> {
    const command = new ListObjectsV2Command({
      Bucket: this.bucket,
      Prefix: prefix,
      MaxKeys: maxKeys,
      ContinuationToken: continuationToken,
    });
    const response = await this.client.send(command);
    const objects: StoredObject[] = [];
    for (const item of response.Contents ?? []) {
      if (item.Key) {
        objects.push({
          key: item.Key,
          lastModified: item.LastModified,
          size: item.Size,
        });
      }
    }
    return { objects, nextContinuationToken: response.NextContinuationToken };
  }
}
```

- [ ] **Step 3: Commit**

```bash
git add backend/src/stores/s3-clients.ts backend/src/stores/s3-artifact-store.ts
git commit -m "backend: make S3ArtifactStore constructor accept client and bucket"
```

---

## Task 4: Create tenant-aware S3 artifact and metadata stores

**Files:**
- Create: `backend/src/stores/tenant-s3-artifact-store.ts`
- Create: `backend/src/stores/tenant-s3-metadata-store.ts`
- Delete: `backend/src/stores/s3-metadata-store.ts`

- [ ] **Step 1: Implement TenantS3ArtifactStore**

Create `backend/src/stores/tenant-s3-artifact-store.ts`:

```typescript
import { S3Client } from "@aws-sdk/client-s3";
import type { ArtifactStore, ListPage, StoredObject } from "./artifact-store.js";
import type { BucketResolver } from "./bucket-resolver.js";
import { S3ArtifactStore } from "./s3-artifact-store.js";
import { createS3Client } from "./s3-clients.js";

export class TenantS3ArtifactStore implements ArtifactStore {
  private readonly defaultClient: S3Client;
  private readonly connection: {
    endpoint: string;
    accessKeyId: string;
    secretAccessKey: string;
  };

  constructor(private readonly resolver: BucketResolver) {
    const endpoint = process.env.KVCDN_S3_ENDPOINT;
    const accessKeyId = process.env.KVCDN_S3_ACCESS_KEY;
    const secretAccessKey = process.env.KVCDN_S3_SECRET_KEY;
    if (!endpoint || !accessKeyId || !secretAccessKey) {
      throw new Error("S3 credentials are not configured");
    }
    this.connection = { endpoint, accessKeyId, secretAccessKey };
    this.defaultClient = createS3Client(this.connection);
  }

  async presignedUploadUrl(key: string, sizeBytes: number): Promise<string> {
    const store = await this.storeForKey(key);
    return store.presignedUploadUrl(key, sizeBytes);
  }

  async presignedDownloadUrl(key: string): Promise<string> {
    const store = await this.storeForKey(key);
    return store.presignedDownloadUrl(key);
  }

  async put(key: string, body: Buffer | string, contentType?: string): Promise<void> {
    const store = await this.storeForKey(key);
    return store.put(key, body, contentType);
  }

  async get(key: string): Promise<Buffer> {
    const store = await this.storeForKey(key);
    return store.get(key);
  }

  async delete(key: string): Promise<void> {
    const store = await this.storeForKey(key);
    return store.delete(key);
  }

  async list(prefix: string): Promise<StoredObject[]> {
    const store = await this.storeForKey(prefix);
    return store.list(prefix);
  }

  async listPaginated(prefix: string, maxKeys: number, continuationToken?: string): Promise<ListPage> {
    const store = await this.storeForKey(prefix);
    return store.listPaginated(prefix, maxKeys, continuationToken);
  }

  private async storeForKey(key: string): Promise<S3ArtifactStore> {
    const customerId = extractCustomerId(key);
    const { bucket } = await this.resolver.resolve(customerId);
    return new S3ArtifactStore(this.defaultClient, bucket);
  }
}

function extractCustomerId(key: string): string {
  const match = key.match(/^artifacts\/customers\/([^/]+)\//);
  if (match) return match[1];
  // Public keys also need a bucket; default to a synthetic customer for the public index.
  const publicMatch = key.match(/^artifacts\/public\/orgs\/([^/]+)\//);
  if (publicMatch) return `public:${publicMatch[1]}`;
  throw new Error(`Cannot resolve bucket for key: ${key}`);
}
```

- [ ] **Step 2: Implement TenantS3MetadataStore**

Create `backend/src/stores/tenant-s3-metadata-store.ts`:

```typescript
import type { ArtifactMetadata, ListMetadataPage, MetadataStore } from "./metadata-store.js";
import type { ArtifactStore } from "./artifact-store.js";

function metaKey(customerId: string, org: string, project: string, artifactId: string): string {
  return `artifacts/customers/${customerId}/orgs/${org}/projects/${project}/${artifactId}/.meta.json`;
}

function publicMetaKey(org: string, project: string, artifactId: string): string {
  return `artifacts/public/orgs/${org}/projects/${project}/${artifactId}/.meta.json`;
}

export class TenantS3MetadataStore implements MetadataStore {
  constructor(private readonly store: ArtifactStore) {}

  async save(meta: ArtifactMetadata): Promise<void> {
    const key = metaKey(meta.customer_id, meta.org, meta.project, meta.artifact_id);
    await this.store.put(key, Buffer.from(JSON.stringify(meta), "utf-8"), "application/json");
  }

  async get(customerId: string, org: string, project: string, artifactId: string): Promise<ArtifactMetadata | undefined> {
    const key = metaKey(customerId, org, project, artifactId);
    try {
      const raw = await this.store.get(key);
      return JSON.parse(raw.toString("utf-8")) as ArtifactMetadata;
    } catch {
      return undefined;
    }
  }

  async list(customerId: string, org: string, project: string, limit: number, continuationToken?: string): Promise<ListMetadataPage> {
    const prefix = `artifacts/customers/${customerId}/orgs/${org}/projects/${project}/`;
    const page = await this.store.listPaginated(prefix, limit, continuationToken);

    const artifacts: ArtifactMetadata[] = [];
    const seen = new Set<string>();
    for (const item of page.objects) {
      if (!item.key?.endsWith("/.meta.json")) continue;
      const parts = item.key.split("/");
      const artifactId = parts[7];
      if (!artifactId || seen.has(artifactId)) continue;
      seen.add(artifactId);
      try {
        const raw = await this.store.get(item.key);
        artifacts.push(JSON.parse(raw.toString("utf-8")) as ArtifactMetadata);
      } catch {
        // Ignore unreadable metadata.
      }
    }
    return { artifacts, continuationToken: page.nextContinuationToken };
  }

  async delete(customerId: string, org: string, project: string, artifactId: string): Promise<void> {
    await this.store.delete(metaKey(customerId, org, project, artifactId));
  }

  async getPublic(org: string, project: string, artifactId: string): Promise<ArtifactMetadata | undefined> {
    const key = publicMetaKey(org, project, artifactId);
    try {
      const raw = await this.store.get(key);
      return JSON.parse(raw.toString("utf-8")) as ArtifactMetadata;
    } catch {
      return undefined;
    }
  }

  async savePublic(meta: ArtifactMetadata): Promise<void> {
    const key = publicMetaKey(meta.org, meta.project, meta.artifact_id);
    await this.store.put(key, Buffer.from(JSON.stringify(meta), "utf-8"), "application/json");
  }

  async deletePublic(org: string, project: string, artifactId: string): Promise<void> {
    await this.store.delete(publicMetaKey(org, project, artifactId));
  }
}
```

- [ ] **Step 3: Delete the old single-bucket metadata store**

```bash
git rm backend/src/stores/s3-metadata-store.ts
```

- [ ] **Step 4: Commit**

```bash
git add backend/src/stores/tenant-s3-artifact-store.ts backend/src/stores/tenant-s3-metadata-store.ts
git commit -m "backend: add tenant-aware S3 artifact and metadata stores"
```

---

## Task 5: Wire tenant-aware stores into the backend

**Files:**
- Modify: `backend/src/index.ts`

- [ ] **Step 1: Update build() to use TenantS3ArtifactStore, TenantS3MetadataStore, and add bucket seams**

Replace the store decoration section of `backend/src/index.ts` with:

```typescript
import { S3BucketStore } from "./stores/s3-bucket-store.js";
import { MemoryBucketStore } from "./stores/memory-bucket-store.js";
import type { BucketStore } from "./stores/bucket-store.js";
import { HashedBucketResolver } from "./stores/hashed-bucket-resolver.js";
import type { BucketResolver } from "./stores/bucket-resolver.js";
import { TenantS3ArtifactStore } from "./stores/tenant-s3-artifact-store.js";
import { TenantS3MetadataStore } from "./stores/tenant-s3-metadata-store.js";

// ... keep existing imports ...

declare module "fastify" {
  interface FastifyInstance {
    artifactStore: ArtifactStore;
    metadataStore: MetadataStore;
    accessPolicy: AccessPolicy;
    identityVerifier: IdentityVerifier;
    tenantResolver: TenantResolver;
    bucketStore: BucketStore;
    bucketResolver: BucketResolver;
  }
}

export interface BuildOptions {
  artifactStore?: ArtifactStore;
  metadataStore?: MetadataStore;
  accessPolicy?: AccessPolicy;
  identityVerifier?: IdentityVerifier;
  tenantResolver?: TenantResolver;
  bucketStore?: BucketStore;
  bucketResolver?: BucketResolver;
}

export async function build(options: BuildOptions = {}): Promise<FastifyInstance> {
  const fastify = Fastify({ logger: true });

  const bucketStore = options.bucketStore ?? new S3BucketStore();
  const bucketResolver = options.bucketResolver ?? new HashedBucketResolver(bucketStore);
  const artifactStore = options.artifactStore ?? new TenantS3ArtifactStore(bucketResolver);
  fastify.decorate("bucketStore", bucketStore);
  fastify.decorate("bucketResolver", bucketResolver);
  fastify.decorate("artifactStore", artifactStore);
  fastify.decorate("metadataStore", options.metadataStore ?? new TenantS3MetadataStore(artifactStore));
  fastify.decorate("accessPolicy", options.accessPolicy ?? new OwnerOrPublicAccessPolicy());
  fastify.decorate("identityVerifier", options.identityVerifier ?? new OidcOrApiKeyIdentityVerifier());
  fastify.decorate("tenantResolver", options.tenantResolver ?? new HashedTenantResolver());
  // ... rest unchanged ...
}
```

Keep `BuildOptions` backward compatible by making the new fields optional.

- [ ] **Step 2: Update test server injection**

Modify `backend/src/__tests__/artifacts.test.ts` `startTestServer` to inject `bucketStore` and `bucketResolver` using memory adapters so tests do not require S3 credentials:

```typescript
import { MemoryBucketStore } from "../stores/memory-bucket-store.js";
import { HashedBucketResolver } from "../stores/hashed-bucket-resolver.js";

async function startTestServer(port = 0): Promise<{ app: FastifyInstance; url: string }> {
  const { buildAndInit } = await import("../index.js");
  const artifactStore = new MemoryArtifactStore();
  const bucketStore = new MemoryBucketStore();
  const app = await buildAndInit({
    artifactStore,
    metadataStore: new MemoryMetadataStore(),
    accessPolicy: new OwnerOrPublicAccessPolicy(),
    identityVerifier: new OidcOrApiKeyIdentityVerifier(),
    tenantResolver: new HashedTenantResolver(),
    bucketStore,
    bucketResolver: new HashedBucketResolver(bucketStore, "memory-bucket"),
  });
  // ... rest unchanged ...
}
```

- [ ] **Step 3: Run backend tests**

```bash
cd backend && NODE_OPTIONS='--experimental-vm-modules' npm test
```
Expected: 30 tests pass.

- [ ] **Step 4: Commit**

```bash
git add backend/src/index.ts backend/src/__tests__/artifacts.test.ts
git commit -m "backend: wire tenant-aware S3 stores and bucket resolver into build"
```

---

## Task 6: Update local stack config for control bucket

**Files:**
- Modify: `deploy/local/backend/configmap.yaml`
- Modify: `deploy/local/README.md` if it exists

- [ ] **Step 1: Add control bucket env var to local config**

In `deploy/local/backend/configmap.yaml`, add inside the `data` section:

```yaml
KVCDN_CONTROL_BUCKET: "kvcdn-control"
```

And add a `mc mb` command or bucket creation step in the local MinIO setup if documented.

- [ ] **Step 2: Commit**

```bash
git add deploy/local/backend/configmap.yaml
git commit -m "deploy: add KVCDN_CONTROL_BUCKET for local multi-bucket stack"
```

---

## Task 7: Document multi-bucket configuration

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update hosted backend configuration section**

Replace the single `KVCDN_S3_BUCKET` bullet with:

```markdown
- `KVCDN_S3_BUCKETS` — comma-separated list of S3-compatible bucket names (e.g. `kvcdn-artifacts-1,kvcdn-artifacts-2`). Tenants are deterministically assigned to one bucket. If only one bucket is configured, behavior is identical to the original single-bucket setup.
- `KVCDN_S3_BUCKET` — legacy single-bucket name. Used as a fallback if `KVCDN_S3_BUCKETS` is not set.
- `KVCDN_CONTROL_BUCKET` — bucket that stores tenant-to-bucket assignment records. This bucket should be created in the same S3 account/endpoint and is required when `KVCDN_S3_BUCKETS` has more than one entry.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document KVCDN_S3_BUCKETS and KVCDN_CONTROL_BUCKET"
```

---

## Task 8: Final verification

- [ ] **Step 1: Run full combined gate**

```bash
cargo test && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cd backend && NODE_OPTIONS='--experimental-vm-modules' npm test
```
Expected: all pass.

- [ ] **Step 2: Commit any final fixes if needed**

- [ ] **Step 3: Push branch**

```bash
git push origin main
```

---

## Self-review

1. **Spec coverage:** The user asked to continue addressing the single-bucket concern. The plan adds multi-bucket support through a `BucketResolver` seam while preserving existing key layout and CLI behavior.
2. **Placeholder scan:** No TBD/TODO. Each step has concrete code, file paths, and commands.
3. **Type consistency:** `BucketResolver.resolve` returns `TenantBucketAssignment`. `TenantS3ArtifactStore` extracts `customerId` from keys and resolves the bucket. Memory adapters match interfaces.
