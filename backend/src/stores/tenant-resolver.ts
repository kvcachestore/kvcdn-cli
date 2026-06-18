import { createHash } from "node:crypto";

export interface Tenant {
  customerId: string;
  namespace(org: string, project: string): string;
}

export interface TenantResolver {
  resolve(identity: { sub: string; source: "oidc" | "api-key" }): Tenant;
}

export class HashedTenantResolver implements TenantResolver {
  resolve(identity: { sub: string; source: "oidc" | "api-key" }): Tenant {
    // Preserve the legacy tenant-key scheme: API keys were originally hashed
    // as `org:<org slug>` while OIDC subjects were hashed as `oidc:<sub>`.
    const sourceKey = identity.source === "api-key" ? "org" : identity.source;
    const customerId = deriveCustomerId(`${sourceKey}:${identity.sub}`);
    return { customerId, namespace: makeNamespace(customerId) };
  }
}

// customer_id must be stable across restarts because it forms the S3 prefix and
// appears in every artifact's metadata.
export function deriveCustomerId(source: string): string {
  return createHash("sha256").update(source).digest("hex").slice(0, 32);
}

function makeNamespace(
  customerId: string
): (org: string, project: string) => string {
  return (org: string, project: string) =>
    `artifacts/customers/${customerId}/orgs/${org}/projects/${project}`;
}
