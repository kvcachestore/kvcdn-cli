import { S3Client, GetObjectCommand, PutObjectCommand } from "@aws-sdk/client-s3";
import type { BucketStore, BucketStoreLogger, TenantBucketAssignment } from "./bucket-store.js";

export class S3BucketStore implements BucketStore {
  private client: S3Client;
  private bucket: string;
  private logger?: BucketStoreLogger;

  constructor(logger?: BucketStoreLogger) {
    const endpoint = process.env.KVCDN_S3_ENDPOINT;
    const accessKeyId = process.env.KVCDN_S3_ACCESS_KEY;
    const secretAccessKey = process.env.KVCDN_S3_SECRET_KEY;
    const bucket = process.env.KVCDN_CONTROL_BUCKET;
    if (!endpoint || !accessKeyId || !secretAccessKey || !bucket) {
      throw new Error("Control bucket is not configured");
    }
    this.bucket = bucket;
    this.logger = logger;
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
    } catch (err) {
      this.logger?.warn?.("control bucket read failed; falling back to deterministic resolution", { customerId, error: String(err) });
      return undefined;
    }
  }

  async saveAssignment(assignment: TenantBucketAssignment): Promise<void> {
    try {
      await this.client.send(new PutObjectCommand({
        Bucket: this.bucket,
        Key: this.key(assignment.customer_id),
        Body: Buffer.from(JSON.stringify(assignment), "utf-8"),
        ContentType: "application/json",
      }));
    } catch (err) {
      this.logger?.error?.("failed to save bucket assignment to control bucket", { customerId: assignment.customer_id, bucket: assignment.bucket, error: String(err) });
      throw err;
    }
  }
}
