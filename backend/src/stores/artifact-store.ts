export interface StoredObject {
  key: string;
  lastModified?: Date;
  size?: number;
}

export interface ListPage {
  objects: StoredObject[];
  nextContinuationToken?: string;
}

export interface ArtifactStore {
  presignedUploadUrl(key: string, sizeBytes: number): Promise<string>;
  presignedDownloadUrl(key: string): Promise<string>;
  put(key: string, body: Buffer | string, contentType?: string): Promise<void>;
  get(key: string): Promise<Buffer>;
  delete(key: string): Promise<void>;
  list(prefix: string): Promise<StoredObject[]>;
  listPaginated(prefix: string, maxKeys: number, continuationToken?: string): Promise<ListPage>;
}
