import type { BucketStore, TenantBucketAssignment } from "./bucket-store.js";

export class MemoryBucketStore implements BucketStore {
  private data = new Map<string, TenantBucketAssignment>();

  async getAssignment(customerId: string): Promise<TenantBucketAssignment | undefined> {
    return this.data.get(customerId);
  }

  async saveAssignment(assignment: TenantBucketAssignment): Promise<void> {
    this.data.set(assignment.customer_id, { ...assignment });
  }
}
