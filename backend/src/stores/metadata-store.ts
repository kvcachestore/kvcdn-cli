export interface ArtifactMetadata {
  artifact_id: string;
  customer_id: string;
  org: string;
  project: string;
  name: string;
  size_bytes: number;
  sha256: string;
  dtype: string;
  storage_dtype?: string;
  num_tokens: number;
  num_layers: number;
  quantized: boolean;
  visibility: "public" | "private";
  created_at: string;
}

export interface ListMetadataPage {
  artifacts: ArtifactMetadata[];
  continuationToken?: string;
}

export interface MetadataStore {
  save(meta: ArtifactMetadata): Promise<void>;
  get(customerId: string, org: string, project: string, artifactId: string): Promise<ArtifactMetadata | undefined>;
  list(customerId: string, org: string, project: string, limit: number, continuationToken?: string): Promise<ListMetadataPage>;
  listByCustomer(customerId: string, limit: number, continuationToken?: string): Promise<ListMetadataPage>;
  delete(customerId: string, org: string, project: string, artifactId: string): Promise<void>;

  getPublic(org: string, project: string, artifactId: string): Promise<ArtifactMetadata | undefined>;
  savePublic(meta: ArtifactMetadata): Promise<void>;
  deletePublic(org: string, project: string, artifactId: string): Promise<void>;
}
