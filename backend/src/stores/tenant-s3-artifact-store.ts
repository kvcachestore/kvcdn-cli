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
  if (match && match[1]) return match[1];
  // Public keys also need a bucket; default to a synthetic customer for the public index.
  const publicMatch = key.match(/^artifacts\/public\/orgs\/([^/]+)\//);
  if (publicMatch && publicMatch[1]) return `public:${publicMatch[1]}`;
  throw new Error(`Cannot resolve bucket for key: ${key}`);
}
