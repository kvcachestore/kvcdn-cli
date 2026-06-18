import type { FastifyInstance, FastifyReply, FastifyRequest } from "fastify";
import type { ArtifactMetadata, MetadataStore } from "../stores/metadata-store.js";
import type { Tenant } from "../stores/tenant-resolver.js";

interface QuotaLimits {
  organizations: number;
  projects: number;
  artifacts: number;
  storage_bytes: number;
}

interface QuotaUsage {
  organizations: number;
  projects: number;
  artifacts: number;
  storage_bytes: number;
}

interface QuotaResponse {
  customer_id: string;
  quota: QuotaLimits;
  used: QuotaUsage;
}

async function requireAuth(request: FastifyRequest, fastify: FastifyInstance): Promise<Tenant> {
  const authHeader = request.headers.authorization;
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    throw new Error("Unauthorized");
  }
  const identity = await fastify.identityVerifier.verify(authHeader.slice("Bearer ".length));
  return fastify.tenantResolver.resolve(identity);
}

function parseLimit(name: string, defaultValue: number): number {
  const raw = process.env[name];
  if (!raw) return defaultValue;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 0) return defaultValue;
  return value;
}

function getQuotaLimits(): QuotaLimits {
  return {
    organizations: parseLimit("KVCDN_QUOTA_ORGS", 10),
    projects: parseLimit("KVCDN_QUOTA_PROJECTS", 100),
    artifacts: parseLimit("KVCDN_QUOTA_ARTIFACTS", 1000),
    storage_bytes: parseLimit("KVCDN_QUOTA_STORAGE_BYTES", 1_099_511_627_776),
  };
}

async function aggregateUsage(
  metadataStore: MetadataStore,
  customerId: string,
): Promise<QuotaUsage> {
  const orgs = new Set<string>();
  const projects = new Set<string>();
  const artifacts = new Set<string>();
  let storageBytes = 0;

  let continuationToken: string | undefined;
  do {
    const page = await metadataStore.listByCustomer(customerId, 1000, continuationToken);
    for (const meta of page.artifacts) {
      orgs.add(meta.org);
      projects.add(`${meta.org}/${meta.project}`);
      artifacts.add(meta.artifact_id);
      storageBytes += meta.size_bytes;
    }
    continuationToken = page.continuationToken;
  } while (continuationToken);

  return {
    organizations: orgs.size,
    projects: projects.size,
    artifacts: artifacts.size,
    storage_bytes: storageBytes,
  };
}

export async function quotaRoutes(fastify: FastifyInstance): Promise<void> {
  fastify.get("/v1/quota", async (request: FastifyRequest, reply: FastifyReply) => {
    let tenant: Tenant;
    try {
      tenant = await requireAuth(request, fastify);
    } catch {
      return reply.status(401).send({ error: "Unauthorized" });
    }

    const used = await aggregateUsage(fastify.metadataStore, tenant.customerId);
    const body: QuotaResponse = {
      customer_id: tenant.customerId,
      quota: getQuotaLimits(),
      used,
    };

    return reply.status(200).send(body);
  });
}
