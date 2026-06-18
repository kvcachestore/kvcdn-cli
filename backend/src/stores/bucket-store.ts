export interface TenantBucketAssignment {
  customer_id: string;
  bucket: string;
}

export interface BucketStoreLogger {
  debug?(msg: string, ...args: unknown[]): void;
  warn?(msg: string, ...args: unknown[]): void;
  error?(msg: string, ...args: unknown[]): void;
}

export interface BucketStore {
  getAssignment(customerId: string): Promise<TenantBucketAssignment | undefined>;
  saveAssignment(assignment: TenantBucketAssignment): Promise<void>;
}
