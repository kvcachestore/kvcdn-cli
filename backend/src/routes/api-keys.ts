import type { FastifyInstance, FastifyReply, FastifyRequest } from "fastify";
import {
  deriveApiKey,
  getApiKeySeed,
  getConfiguredOrgs,
} from "../auth.js";
import { deriveCustomerId } from "../stores/tenant-resolver.js";

interface GenerateBody {
  org_slug: string;
}

export async function apiKeyRoutes(fastify: FastifyInstance): Promise<void> {
  fastify.post("/api/v1/api-keys/verify", async (request: FastifyRequest, reply: FastifyReply) => {
    const authHeader = request.headers.authorization;
    if (!authHeader || !authHeader.startsWith("Bearer ")) {
      return reply.status(401).send({ error: "Unauthorized" });
    }

    const token = authHeader.slice("Bearer ".length);
    try {
      const identity = await fastify.identityVerifier.verify(token);
      const tenant = fastify.tenantResolver.resolve(identity);
      return reply.status(200).send({ customer_id: tenant.customerId });
    } catch {
      return reply.status(401).send({ error: "Unauthorized" });
    }
  });

  // Operator-only endpoint to mint an API key for an org. In production this
  // should be behind a secret admin token; for now it requires KVCDN_ADMIN_SECRET.
  fastify.post<{
    Body: GenerateBody;
  }>("/api/v1/admin/api-keys", async (request: FastifyRequest<{ Body: GenerateBody }>, reply: FastifyReply) => {
    const adminSecret = process.env.KVCDN_ADMIN_SECRET;
    const authHeader = request.headers.authorization;
    if (!adminSecret || !authHeader || authHeader !== `Bearer ${adminSecret}`) {
      return reply.status(401).send({ error: "Unauthorized" });
    }

    const configuredOrgs = getConfiguredOrgs();
    if (configuredOrgs.length === 0) {
      return reply
        .status(400)
        .send({ error: "API key generation is disabled: KVCDN_API_KEY_ORGS is not configured" });
    }

    const orgSlug = request.body.org_slug?.trim().toLowerCase();
    if (!orgSlug) {
      return reply.status(400).send({ error: "missing org_slug" });
    }
    if (!configuredOrgs.includes(orgSlug)) {
      return reply
        .status(400)
        .send({ error: "org_slug is not in the configured KVCDN_API_KEY_ORGS list" });
    }

    const seed = getApiKeySeed();
    const apiKey = deriveApiKey(seed, orgSlug);
    const customerId = deriveCustomerId(`org:${orgSlug}`);
    return reply
      .status(201)
      .send({ api_key: apiKey, customer_id: customerId, org_slug: orgSlug });
  });
}
