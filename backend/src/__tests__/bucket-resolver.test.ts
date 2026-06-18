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

  it("throws when a persisted assignment points to a bucket no longer in the configured list", async () => {
    const store = new MemoryBucketStore();
    await store.saveAssignment({ customer_id: "customer-5", bucket: "old-bucket" });
    const resolver = new HashedBucketResolver(store, "new-bucket-a,new-bucket-b");
    await expect(resolver.resolve("customer-5")).rejects.toThrow(
      'Bucket "old-bucket" assigned to customer "customer-5" is not in KVCDN_S3_BUCKETS',
    );
  });
});
