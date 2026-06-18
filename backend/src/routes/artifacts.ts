import { createHash } from "node:crypto";
import type { FastifyInstance, FastifyReply, FastifyRequest } from "fastify";
import { v4 as uuidv4 } from "uuid";
import type { Tenant } from "../stores/tenant-resolver.js";
import { UploadMetaSchema } from "../types.js";

interface ArtifactsParams {
  org: string;
  project: string;
  artifact_id: string;
}

interface ArtifactsBody {
  name: string;
  size_bytes: number;
  sha256: string;
  dtype: string;
  storage_dtype?: string;
  num_tokens: number;
  num_layers: number;
  quantized: boolean;
  visibility?: "public" | "private";
}

function dataKey(namespace: string, artifactId: string, name: string): string {
  return `${namespace}/${artifactId}/${name}`;
}

function isValidArtifactId(id: string): boolean {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id);
}

async function requireAuth(request: FastifyRequest, fastify: FastifyInstance): Promise<Tenant> {
  const authHeader = request.headers.authorization;
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    throw new Error("Unauthorized");
  }
  const identity = await fastify.identityVerifier.verify(authHeader.slice("Bearer ".length));
  return fastify.tenantResolver.resolve(identity);
}

async function optionalAuth(request: FastifyRequest, fastify: FastifyInstance): Promise<Tenant | undefined> {
  const authHeader = request.headers.authorization;
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return undefined;
  }
  try {
    const identity = await fastify.identityVerifier.verify(authHeader.slice("Bearer ".length));
    return fastify.tenantResolver.resolve(identity);
  } catch {
    return undefined;
  }
}

export async function artifactsRoutes(fastify: FastifyInstance): Promise<void> {
  fastify.post<{
    Params: ArtifactsParams;
    Body: ArtifactsBody;
  }>("/v1/orgs/:org/projects/:project/artifacts", async (request: FastifyRequest<{ Params: ArtifactsParams; Body: ArtifactsBody }>, reply: FastifyReply) => {
    let tenant: Tenant;
    try {
      tenant = await requireAuth(request, fastify);
    } catch {
      return reply.status(401).send({ error: "Unauthorized" });
    }

    const parseResult = UploadMetaSchema.safeParse(request.body);
    if (!parseResult.success) {
      return reply.status(400).send({ error: "Bad Request", details: parseResult.error.issues });
    }

    const { org, project } = request.params;
    const ns = tenant.namespace(org, project);
    const meta = parseResult.data;
    const artifactId = uuidv4();
    const dataKeyPath = dataKey(ns, artifactId, meta.name);
    const uploadUrl = await fastify.artifactStore.presignedUploadUrl(dataKeyPath, meta.size_bytes);

    const artifactMeta = {
      ...meta,
      customer_id: tenant.customerId,
      org,
      project,
      artifact_id: artifactId,
      created_at: new Date().toISOString(),
      visibility: meta.visibility ?? "private",
    };
    await fastify.metadataStore.save(artifactMeta);
    if (artifactMeta.visibility === "public") {
      await fastify.metadataStore.savePublic(artifactMeta);
    }

    return reply.status(201).send({
      artifact_id: artifactId,
      upload_url: uploadUrl,
    });
  });

  fastify.get<{
    Params: ArtifactsParams;
    Querystring: { limit?: string; continuation?: string };
  }>("/v1/orgs/:org/projects/:project/artifacts", async (request: FastifyRequest<{ Params: ArtifactsParams; Querystring: { limit?: string; continuation?: string } }>, reply: FastifyReply) => {
    let tenant: Tenant;
    try {
      tenant = await requireAuth(request, fastify);
    } catch {
      return reply.status(401).send({ error: "Unauthorized" });
    }

    const { org, project } = request.params;
    const rawLimit = request.query.limit;
    const limit = Math.min(Math.max(Number(rawLimit) || 100, 1), 1000);
    const continuation = request.query.continuation;

    const page = await fastify.metadataStore.list(tenant.customerId, org, project, limit, continuation);

    return reply.status(200).send({
      artifacts: page.artifacts,
      continuation_token: page.continuationToken,
    });
  });

  fastify.get<{
    Params: ArtifactsParams;
  }>("/v1/orgs/:org/projects/:project/artifacts/:artifact_id/download", async (request: FastifyRequest<{ Params: ArtifactsParams }>, reply: FastifyReply) => {
    const tenant = await optionalAuth(request, fastify);
    const { org, project, artifact_id: artifactId } = request.params;
    if (!isValidArtifactId(artifactId)) {
      return reply.status(400).send({ error: "invalid artifact_id" });
    }

    let meta = tenant
      ? await fastify.metadataStore.get(tenant.customerId, org, project, artifactId)
      : undefined;

    if (!meta) {
      meta = await fastify.metadataStore.getPublic(org, project, artifactId);
    }

    if (!meta || !fastify.accessPolicy.canAccess({ customerId: tenant?.customerId }, meta)) {
      return reply.status(tenant ? 404 : 401).send({ error: tenant ? "Artifact not found" : "Unauthorized" });
    }

    const namespace = `artifacts/customers/${meta.customer_id}/orgs/${meta.org}/projects/${meta.project}`;
    const dataKeyPath = dataKey(namespace, artifactId, meta.name);
    const downloadUrl = await fastify.artifactStore.presignedDownloadUrl(dataKeyPath);
    return reply.status(200).send({ artifact_id: artifactId, download_url: downloadUrl });
  });

  fastify.delete<{
    Params: ArtifactsParams;
  }>("/v1/orgs/:org/projects/:project/artifacts/:artifact_id", async (request: FastifyRequest<{ Params: ArtifactsParams }>, reply: FastifyReply) => {
    let tenant: Tenant;
    try {
      tenant = await requireAuth(request, fastify);
    } catch {
      return reply.status(401).send({ error: "Unauthorized" });
    }

    const { org, project, artifact_id: artifactId } = request.params;
    if (!isValidArtifactId(artifactId)) {
      return reply.status(400).send({ error: "invalid artifact_id" });
    }

    const meta = await fastify.metadataStore.get(tenant.customerId, org, project, artifactId);
    if (!meta || !fastify.accessPolicy.canAccess({ customerId: tenant.customerId }, meta)) {
      return reply.status(404).send({ error: "Artifact not found" });
    }

    const prefix = `${tenant.namespace(org, project)}/${artifactId}`;
    const objects = await fastify.artifactStore.list(prefix + "/");
    for (const item of objects) {
      if (item.key) {
        await fastify.artifactStore.delete(item.key);
      }
    }

    await fastify.metadataStore.delete(tenant.customerId, org, project, artifactId);
    if (meta.visibility === "public") {
      await fastify.metadataStore.deletePublic(org, project, artifactId);
    }

    return reply.status(204).send();
  });

  fastify.post<{
    Params: ArtifactsParams;
  }>("/v1/orgs/:org/projects/:project/artifacts/:artifact_id/complete", async (request: FastifyRequest<{ Params: ArtifactsParams }>, reply: FastifyReply) => {
    let tenant: Tenant;
    try {
      tenant = await requireAuth(request, fastify);
    } catch {
      return reply.status(401).send({ error: "Unauthorized" });
    }

    const { org, project, artifact_id: artifactId } = request.params;
    if (!isValidArtifactId(artifactId)) {
      return reply.status(400).send({ error: "invalid artifact_id" });
    }

    const meta = await fastify.metadataStore.get(tenant.customerId, org, project, artifactId);
    if (!meta || !fastify.accessPolicy.canAccess({ customerId: tenant.customerId }, meta)) {
      return reply.status(404).send({ error: "Artifact not found" });
    }

    const dataKeyPath = dataKey(tenant.namespace(org, project), artifactId, meta.name);
    let data: Buffer;
    try {
      data = await fastify.artifactStore.get(dataKeyPath);
    } catch {
      return reply.status(400).send({ error: "Artifact data not found" });
    }

    if (data.length !== meta.size_bytes) {
      return reply.status(400).send({
        error: "size mismatch",
        expected: meta.size_bytes,
        actual: data.length,
      });
    }

    const actualSha256 = createHash("sha256").update(data).digest("hex");
    if (actualSha256 !== meta.sha256.toLowerCase()) {
      return reply.status(400).send({ error: "sha256 mismatch", expected: meta.sha256, actual: actualSha256 });
    }

    return reply.status(200).send({ artifact_id: artifactId, sha256: actualSha256 });
  });
}
