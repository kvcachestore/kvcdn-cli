import { describe, expect, it, beforeAll, afterAll } from "@jest/globals";
import { S3Client, CreateBucketCommand, HeadBucketCommand, DeleteBucketCommand, DeleteObjectCommand, ListObjectsV2Command } from "@aws-sdk/client-s3";
import { S3BucketStore } from "../stores/s3-bucket-store.js";
import { HashedBucketResolver } from "../stores/hashed-bucket-resolver.js";
import { TenantS3ArtifactStore } from "../stores/tenant-s3-artifact-store.js";
import { TenantS3MetadataStore } from "../stores/tenant-s3-metadata-store.js";
import { S3ArtifactStore } from "../stores/s3-artifact-store.js";
import { deriveCustomerId } from "../stores/tenant-resolver.js";
import { createS3Client } from "../stores/s3-clients.js";

function getS3Env(): { endpoint: string; accessKeyId: string; secretAccessKey: string } | undefined {
  const endpoint = process.env.KVCDN_S3_ENDPOINT;
  const accessKeyId = process.env.KVCDN_S3_ACCESS_KEY;
  const secretAccessKey = process.env.KVCDN_S3_SECRET_KEY;
  if (!endpoint || !accessKeyId || !secretAccessKey) {
    return undefined;
  }
  return { endpoint, accessKeyId, secretAccessKey };
}

async function ensureBucket(client: S3Client, bucket: string): Promise<void> {
  try {
    await client.send(new HeadBucketCommand({ Bucket: bucket }));
  } catch {
    await client.send(new CreateBucketCommand({ Bucket: bucket }));
  }
}

async function clearBucket(client: S3Client, bucket: string): Promise<void> {
  const listed = await client.send(new ListObjectsV2Command({ Bucket: bucket }));
  for (const item of listed.Contents ?? []) {
    if (item.Key) {
      await client.send(new DeleteObjectCommand({ Bucket: bucket, Key: item.Key }));
    }
  }
}

describe("multi-bucket S3 integration", () => {
  const env = getS3Env();
  if (!env) {
    it.skip("skips when S3 env vars are not configured", () => {});
    return;
  }

  const artifactBuckets = ["kvcdn-test-artifacts-a", "kvcdn-test-artifacts-b"];
  const controlBucket = "kvcdn-test-control";
  const buckets = [...artifactBuckets, controlBucket];

  const client = createS3Client(env);

  let bucketStore: S3BucketStore;
  let artifactStore: TenantS3ArtifactStore;
  let metadataStore: TenantS3MetadataStore;

  beforeAll(async () => {
    process.env.KVCDN_S3_BUCKETS = artifactBuckets.join(",");
    process.env.KVCDN_CONTROL_BUCKET = controlBucket;

    for (const bucket of buckets) {
      await ensureBucket(client, bucket);
      await clearBucket(client, bucket);
    }

    bucketStore = new S3BucketStore();
    const resolver = new HashedBucketResolver(bucketStore);
    artifactStore = new TenantS3ArtifactStore(resolver);
    metadataStore = new TenantS3MetadataStore(artifactStore);
  });

  afterAll(async () => {
    for (const bucket of buckets) {
      await clearBucket(client, bucket);
      try {
        await client.send(new DeleteBucketCommand({ Bucket: bucket }));
      } catch {
        // Ignore cleanup failures.
      }
    }
    delete process.env.KVCDN_S3_BUCKETS;
    delete process.env.KVCDN_CONTROL_BUCKET;
  });

  it("places tenant data in one configured bucket and records the assignment", async () => {
    const customerId = deriveCustomerId("test:customer-1");
    const org = "acme";
    const project = "acme";
    const artifactId = "test-artifact-1";
    const dataKey = `artifacts/customers/${customerId}/orgs/${org}/projects/${project}/${artifactId}/artifact.bin`;
    const content = Buffer.from("multi-bucket-test-content", "utf-8");

    await artifactStore.put(dataKey, content, "application/octet-stream");

    const assignment = await bucketStore.getAssignment(customerId);
    expect(assignment).toBeDefined();
    expect(artifactBuckets).toContain(assignment!.bucket);

    const assignedClient = new S3ArtifactStore(client, assignment!.bucket);
    const stored = await assignedClient.get(dataKey);
    expect(stored.toString("utf-8")).toBe("multi-bucket-test-content");

    await artifactStore.delete(dataKey);
  });

  it("round-trips metadata through the tenant-aware metadata store", async () => {
    const customerId = deriveCustomerId("test:customer-2");
    const org = "acme";
    const project = "acme";
    const artifactId = "test-artifact-2";

    const meta = {
      artifact_id: artifactId,
      customer_id: customerId,
      org,
      project,
      name: "model.kv",
      size_bytes: 42,
      sha256: "deadbeef".repeat(8),
      dtype: "F16",
      storage_dtype: "F16",
      num_tokens: 128,
      num_layers: 32,
      quantized: false,
      visibility: "private" as const,
      created_at: new Date().toISOString(),
    };

    await metadataStore.save(meta);
    const loaded = await metadataStore.get(customerId, org, project, artifactId);
    expect(loaded).toEqual(meta);

    const assignment = await bucketStore.getAssignment(customerId);
    expect(assignment).toBeDefined();
    expect(artifactBuckets).toContain(assignment!.bucket);

    await metadataStore.delete(customerId, org, project, artifactId);
  });

  it("keeps the same tenant on the same bucket across resolves", async () => {
    const customerId = deriveCustomerId("test:customer-3");
    const resolver = new HashedBucketResolver(bucketStore);
    const first = await resolver.resolve(customerId);
    const second = await resolver.resolve(customerId);
    expect(first.bucket).toBe(second.bucket);
    expect(artifactBuckets).toContain(first.bucket);
  });
});
