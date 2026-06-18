import type { ArtifactStore, ListPage, StoredObject } from "./artifact-store.js";

export class MemoryArtifactStore implements ArtifactStore {
  private objects = new Map<string, Buffer>();

  async presignedUploadUrl(key: string): Promise<string> {
    return `https://memory.test/${key}?upload=true`;
  }

  async presignedDownloadUrl(key: string): Promise<string> {
    return `https://memory.test/${key}?download=true`;
  }

  async put(key: string, body: Buffer | string, _contentType?: string): Promise<void> {
    this.objects.set(key, Buffer.isBuffer(body) ? body : Buffer.from(body, "utf-8"));
  }

  async get(key: string): Promise<Buffer> {
    const value = this.objects.get(key);
    if (!value) {
      throw new Error(`Object not found: ${key}`);
    }
    return Buffer.from(value);
  }

  async delete(key: string): Promise<void> {
    this.objects.delete(key);
  }

  async list(prefix: string): Promise<StoredObject[]> {
    return Array.from(this.objects.keys())
      .filter((key) => key.startsWith(prefix))
      .map((key) => ({
        key,
        lastModified: new Date(),
        size: this.objects.get(key)?.length ?? 0,
      }));
  }

  async listPaginated(prefix: string, maxKeys: number, continuationToken?: string): Promise<ListPage> {
    const allKeys = Array.from(this.objects.keys()).filter((key) => key.startsWith(prefix));
    const startIndex = continuationToken ? Number(continuationToken) : 0;
    const pageKeys = allKeys.slice(startIndex, startIndex + maxKeys);
    const objects = pageKeys.map((key) => ({
      key,
      lastModified: new Date(),
      size: this.objects.get(key)?.length ?? 0,
    }));
    const nextIndex = startIndex + pageKeys.length;
    const nextContinuationToken = nextIndex < allKeys.length ? String(nextIndex) : undefined;
    return { objects, nextContinuationToken };
  }
}
