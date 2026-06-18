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

  async listByCustomer(customerId: string, limit: number, continuationToken?: string): Promise<ListMetadataPage> {
    const prefix = `artifacts/customers/${customerId}/`;
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
