import { createHash } from "node:crypto";
import type { FastifyInstance, FastifyReply, FastifyRequest } from "fastify";
import { v4 as uuidv4 } from "uuid";
import { z } from "zod";
import type { Tenant } from "../stores/tenant-resolver.js";
import type { ArtifactMetadata } from "../stores/metadata-store.js";
import { UploadMetaSchema } from "../types.js";
import type { UploadMeta } from "../types.js";

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
  }>("/api/v1/orgs/:org/projects/:project/artifacts", async (request: FastifyRequest<{ Params: ArtifactsParams; Body: ArtifactsBody }>, reply: FastifyReply) => {
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
  }>("/api/v1/orgs/:org/projects/:project/artifacts", async (request: FastifyRequest<{ Params: ArtifactsParams; Querystring: { limit?: string; continuation?: string } }>, reply: FastifyReply) => {
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
  }>("/api/v1/orgs/:org/projects/:project/artifacts/:artifact_id/download", async (request: FastifyRequest<{ Params: ArtifactsParams }>, reply: FastifyReply) => {
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
  }>("/api/v1/orgs/:org/projects/:project/artifacts/:artifact_id", async (request: FastifyRequest<{ Params: ArtifactsParams }>, reply: FastifyReply) => {
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
  }>("/api/v1/orgs/:org/projects/:project/artifacts/:artifact_id/complete", async (request: FastifyRequest<{ Params: ArtifactsParams }>, reply: FastifyReply) => {
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

  // -------------------------------------------------------------------------
  // Flat /api/v1/artifacts routes, mirroring the production portal API. The
  // kvcdn CLI (>= 0.2.2) targets these. Org/project scoping is implicit per
  // customer, so artifacts live under the default org/project namespace.
  // -------------------------------------------------------------------------

  fastify.post<{
    Body: unknown;
  }>("/api/v1/artifacts/upload-url", async (request: FastifyRequest<{ Body: unknown }>, reply: FastifyReply) => {
    let tenant: Tenant;
    try {
      tenant = await requireAuth(request, fastify);
    } catch {
      return reply.status(401).send({ error: "Unauthorized" });
    }

    const parseResult = FlatUploadSchema.safeParse(request.body);
    if (!parseResult.success) {
      return reply.status(400).send({ error: "Bad Request", details: parseResult.error.issues });
    }
    const meta = flatBodyToMeta(parseResult.data);
    if (!meta) {
      return reply.status(400).send({
        error: "Bad Request",
        message: "a name (or model_name) and a 64-hex checksum (or sha256) are required",
      });
    }

    const artifactId = uuidv4();
    const ns = tenant.namespace(FLAT_ORG, FLAT_PROJECT);
    const dataKeyPath = dataKey(ns, artifactId, meta.name);
    const uploadUrl = await fastify.artifactStore.presignedUploadUrl(dataKeyPath, meta.size_bytes);

    const artifactMeta: ArtifactMetadata = {
      ...meta,
      customer_id: tenant.customerId,
      org: FLAT_ORG,
      project: FLAT_PROJECT,
      artifact_id: artifactId,
      created_at: new Date().toISOString(),
    };
    await fastify.metadataStore.save(artifactMeta);
    if (artifactMeta.visibility === "public") {
      await fastify.metadataStore.savePublic(artifactMeta);
    }

    return reply.status(200).send({
      artifact_id: artifactId,
      url: uploadUrl,
      method: "PUT",
      expires_at: new Date(Date.now() + 900 * 1000).toISOString(),
    });
  });

  fastify.get<{
    Querystring: { limit?: string; continuation?: string };
  }>("/api/v1/artifacts", async (request: FastifyRequest<{ Querystring: { limit?: string; continuation?: string } }>, reply: FastifyReply) => {
    let tenant: Tenant;
    try {
      tenant = await requireAuth(request, fastify);
    } catch {
      return reply.status(401).send({ error: "Unauthorized" });
    }

    const rawLimit = request.query.limit;
    const limit = Math.min(Math.max(Number(rawLimit) || 100, 1), 1000);
    const page = await fastify.metadataStore.listByCustomer(tenant.customerId, limit, request.query.continuation);

    return reply.status(200).send({
      artifacts: page.artifacts.map(toFlatArtifact),
      total: page.artifacts.length,
      limit,
      offset: 0,
      continuation_token: page.continuationToken,
    });
  });

  fastify.post<{
    Params: FlatArtifactParams;
  }>("/api/v1/artifacts/:artifact_id/confirm-upload", async (request: FastifyRequest<{ Params: FlatArtifactParams }>, reply: FastifyReply) => {
    let tenant: Tenant;
    try {
      tenant = await requireAuth(request, fastify);
    } catch {
      return reply.status(401).send({ error: "Unauthorized" });
    }

    const artifactId = request.params.artifact_id;
    if (!isValidArtifactId(artifactId)) {
      return reply.status(400).send({ error: "invalid artifact_id" });
    }

    const meta = await findCustomerArtifact(fastify, tenant.customerId, artifactId);
    if (!meta || !fastify.accessPolicy.canAccess({ customerId: tenant.customerId }, meta)) {
      return reply.status(404).send({ error: "Artifact not found" });
    }

    const dataKeyPath = dataKey(tenant.namespace(meta.org, meta.project), artifactId, meta.name);
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

  fastify.post<{
    Params: FlatArtifactParams;
  }>("/api/v1/artifacts/:artifact_id/download-url", async (request: FastifyRequest<{ Params: FlatArtifactParams }>, reply: FastifyReply) => {
    let tenant: Tenant;
    try {
      tenant = await requireAuth(request, fastify);
    } catch {
      return reply.status(401).send({ error: "Unauthorized" });
    }

    const artifactId = request.params.artifact_id;
    if (!isValidArtifactId(artifactId)) {
      return reply.status(400).send({ error: "invalid artifact_id" });
    }

    const meta = await findCustomerArtifact(fastify, tenant.customerId, artifactId);
    if (!meta || !fastify.accessPolicy.canAccess({ customerId: tenant.customerId }, meta)) {
      return reply.status(404).send({ error: "Artifact not found" });
    }

    const namespace = `artifacts/customers/${meta.customer_id}/orgs/${meta.org}/projects/${meta.project}`;
    const dataKeyPath = dataKey(namespace, artifactId, meta.name);
    const downloadUrl = await fastify.artifactStore.presignedDownloadUrl(dataKeyPath);
    return reply.status(200).send({ artifact_id: artifactId, download_url: downloadUrl });
  });

  fastify.delete<{
    Params: FlatArtifactParams;
  }>("/api/v1/artifacts/:artifact_id", async (request: FastifyRequest<{ Params: FlatArtifactParams }>, reply: FastifyReply) => {
    let tenant: Tenant;
    try {
      tenant = await requireAuth(request, fastify);
    } catch {
      return reply.status(401).send({ error: "Unauthorized" });
    }

    const artifactId = request.params.artifact_id;
    if (!isValidArtifactId(artifactId)) {
      return reply.status(400).send({ error: "invalid artifact_id" });
    }

    const meta = await findCustomerArtifact(fastify, tenant.customerId, artifactId);
    if (!meta || !fastify.accessPolicy.canAccess({ customerId: tenant.customerId }, meta)) {
      return reply.status(404).send({ error: "Artifact not found" });
    }

    const prefix = `${tenant.namespace(meta.org, meta.project)}/${artifactId}`;
    const objects = await fastify.artifactStore.list(prefix + "/");
    for (const item of objects) {
      if (item.key) {
        await fastify.artifactStore.delete(item.key);
      }
    }

    await fastify.metadataStore.delete(tenant.customerId, meta.org, meta.project, artifactId);
    if (meta.visibility === "public") {
      await fastify.metadataStore.deletePublic(meta.org, meta.project, artifactId);
    }

    return reply.status(204).send();
  });
}

// ---------------------------------------------------------------------------
// Flat route helpers
// ---------------------------------------------------------------------------

const FLAT_ORG = "default";
const FLAT_PROJECT = "default";

interface FlatArtifactParams {
  artifact_id: string;
}

const FlatUploadSchema = z.object({
  model_name: z.string().min(1).optional(),
  name: z.string().min(1).optional(),
  dtype: z.string().min(1),
  num_tokens: z.number().int().nonnegative().default(0),
  visibility: z.enum(["public", "private"]).default("private"),
  size_bytes: z.number().int().nonnegative().default(0),
  checksum: z.string().regex(/^[0-9a-fA-F]{64}$/, "expected 64 hex chars").optional(),
  sha256: z.string().regex(/^[0-9a-fA-F]{64}$/, "expected 64 hex chars").optional(),
  metadata: z.record(z.unknown()).default({}),
});

function metaString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function metaNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? Math.floor(value) : undefined;
}

/** Map the flat upload body (production shape) onto the internal metadata. */
function flatBodyToMeta(body: z.infer<typeof FlatUploadSchema>): UploadMeta | undefined {
  const extra = body.metadata;
  const name = metaString(extra.name) ?? body.name ?? body.model_name;
  const sha256 = (body.checksum ?? body.sha256 ?? metaString(extra.sha256))?.toLowerCase();
  if (!name || !sha256) {
    return undefined;
  }
  return {
    name,
    size_bytes: body.size_bytes,
    sha256,
    dtype: body.dtype,
    storage_dtype: metaString(extra.storage_dtype),
    num_tokens: body.num_tokens,
    num_layers: metaNumber(extra.num_layers) ?? 0,
    quantized: extra.quantized === true,
    model_name: body.model_name ?? name,
    visibility: body.visibility,
  };
}

/** Shape stored metadata like the production flat API response. */
function toFlatArtifact(meta: ArtifactMetadata): Record<string, unknown> {
  return {
    id: meta.artifact_id,
    artifact_id: meta.artifact_id,
    model_name: meta.model_name,
    name: meta.name,
    dtype: meta.dtype,
    num_tokens: meta.num_tokens,
    num_layers: meta.num_layers,
    quantized: meta.quantized,
    storage_dtype: meta.storage_dtype ?? null,
    visibility: meta.visibility,
    size_bytes: meta.size_bytes,
    sha256: meta.sha256,
    created_at: meta.created_at,
    public_url: null,
  };
}

/** Locate an artifact across all of the customer's org/project namespaces. */
async function findCustomerArtifact(
  fastify: FastifyInstance,
  customerId: string,
  artifactId: string
): Promise<ArtifactMetadata | undefined> {
  let continuation: string | undefined;
  do {
    const page = await fastify.metadataStore.listByCustomer(customerId, 1000, continuation);
    const found = page.artifacts.find((a) => a.artifact_id === artifactId);
    if (found) {
      return found;
    }
    continuation = page.continuationToken;
  } while (continuation);
  return undefined;
}
