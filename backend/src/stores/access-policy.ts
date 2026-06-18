import type { ArtifactMetadata } from "./metadata-store.js";

export interface Caller {
  customerId?: string;
}

export interface AccessPolicy {
  canAccess(caller: Caller, meta: ArtifactMetadata): boolean;
}

export class OwnerOrPublicAccessPolicy implements AccessPolicy {
  canAccess(caller: Caller, meta: ArtifactMetadata): boolean {
    if (meta.visibility === "public") return true;
    if (caller.customerId && caller.customerId === meta.customer_id) return true;
    return false;
  }
}
