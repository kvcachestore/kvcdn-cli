import type { ArtifactMetadata, ListMetadataPage, MetadataStore } from "./metadata-store.js";

function key(customerId: string, org: string, project: string, artifactId: string): string {
  return `${customerId}:${org}:${project}:${artifactId}`;
}

function publicKey(org: string, project: string, artifactId: string): string {
  return `public:${org}:${project}:${artifactId}`;
}

export class MemoryMetadataStore implements MetadataStore {
  private data = new Map<string, ArtifactMetadata>();

  async save(meta: ArtifactMetadata): Promise<void> {
    this.data.set(key(meta.customer_id, meta.org, meta.project, meta.artifact_id), { ...meta });
  }

  async get(customerId: string, org: string, project: string, artifactId: string): Promise<ArtifactMetadata | undefined> {
    const value = this.data.get(key(customerId, org, project, artifactId));
    return value ? { ...value } : undefined;
  }

  async list(customerId: string, org: string, project: string, limit: number, continuationToken?: string): Promise<ListMetadataPage> {
    const prefix = `${customerId}:${org}:${project}:`;
    const allKeys = Array.from(this.data.keys())
      .filter((k) => k.startsWith(prefix))
      .sort();
    const startIndex = continuationToken ? Number(continuationToken) : 0;
    const pageKeys = allKeys.slice(startIndex, startIndex + limit);
    const artifacts = pageKeys.map((k) => ({ ...this.data.get(k)! }));
    const nextIndex = startIndex + pageKeys.length;
    const continuationTokenOut = nextIndex < allKeys.length ? String(nextIndex) : undefined;
    return { artifacts, continuationToken: continuationTokenOut };
  }

  async listByCustomer(customerId: string, limit: number, continuationToken?: string): Promise<ListMetadataPage> {
    const prefix = `${customerId}:`;
    const allKeys = Array.from(this.data.keys())
      .filter((k) => k.startsWith(prefix))
      .sort();
    const startIndex = continuationToken ? Number(continuationToken) : 0;
    const pageKeys = allKeys.slice(startIndex, startIndex + limit);
    const artifacts = pageKeys.map((k) => ({ ...this.data.get(k)! }));
    const nextIndex = startIndex + pageKeys.length;
    const continuationTokenOut = nextIndex < allKeys.length ? String(nextIndex) : undefined;
    return { artifacts, continuationToken: continuationTokenOut };
  }

  async delete(customerId: string, org: string, project: string, artifactId: string): Promise<void> {
    this.data.delete(key(customerId, org, project, artifactId));
  }

  async getPublic(org: string, project: string, artifactId: string): Promise<ArtifactMetadata | undefined> {
    const value = this.data.get(publicKey(org, project, artifactId));
    return value && value.visibility === "public" ? { ...value } : undefined;
  }

  async savePublic(meta: ArtifactMetadata): Promise<void> {
    this.data.set(publicKey(meta.org, meta.project, meta.artifact_id), { ...meta });
  }

  async deletePublic(org: string, project: string, artifactId: string): Promise<void> {
    this.data.delete(publicKey(org, project, artifactId));
  }
}
