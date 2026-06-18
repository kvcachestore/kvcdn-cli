import { createHash } from "node:crypto";
import type { BucketStore, BucketStoreLogger, TenantBucketAssignment } from "./bucket-store.js";
import type { BucketResolver } from "./bucket-resolver.js";

export class HashedBucketResolver implements BucketResolver {
  private buckets: string[];
  private logger?: BucketStoreLogger;

  constructor(
    private readonly bucketStore: BucketStore,
    bucketsCsv?: string,
    logger?: BucketStoreLogger,
  ) {
    const csv = bucketsCsv ?? process.env.KVCDN_S3_BUCKETS ?? process.env.KVCDN_S3_BUCKET ?? "";
    this.buckets = csv
      .split(",")
      .map((b) => b.trim())
      .filter(Boolean);
    if (this.buckets.length === 0) {
      throw new Error("No buckets configured for KVCDN_S3_BUCKETS");
    }
    this.logger = logger;
  }

  async resolve(customerId: string): Promise<TenantBucketAssignment> {
    const existing = await this.bucketStore.getAssignment(customerId);
    if (existing) {
      if (!this.buckets.includes(existing.bucket)) {
        throw new Error(`Bucket "${existing.bucket}" assigned to customer "${customerId}" is not in KVCDN_S3_BUCKETS`);
      }
      this.logger?.debug?.("resolved tenant bucket from control bucket", { customerId, bucket: existing.bucket });
      return existing;
    }

    const bucket = this.pickBucket(customerId);
    if (!bucket) {
      throw new Error("No buckets configured for KVCDN_S3_BUCKETS");
    }
    const assignment: TenantBucketAssignment = { customer_id: customerId, bucket };
    this.logger?.debug?.("persisting new tenant bucket assignment", { customerId, bucket });
    await this.bucketStore.saveAssignment(assignment);
    this.logger?.debug?.("resolved tenant bucket", { customerId, bucket });
    return assignment;
  }

  private pickBucket(customerId: string): string {
    const hash = createHash("sha256").update(customerId).digest("hex");
    const index = Number.parseInt(hash.slice(0, 8), 16) % this.buckets.length;
    const bucket = this.buckets[index];
    if (!bucket) {
      throw new Error("Resolved bucket index out of range");
    }
    return bucket;
  }
}
