import type { TenantBucketAssignment } from "./bucket-store.js";

export interface BucketResolver {
  resolve(customerId: string): Promise<TenantBucketAssignment>;
}
