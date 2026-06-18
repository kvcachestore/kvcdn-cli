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
