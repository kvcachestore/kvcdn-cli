import { spawn } from "node:child_process";
import { promises as fs } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { FastifyInstance, FastifyReply, FastifyRequest } from "fastify";
import type { Tenant } from "../stores/tenant-resolver.js";

interface InferenceParams {
  org: string;
  project: string;
  artifact_id: string;
}

interface InferenceBody {
  question: string;
  n?: number;
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

function kvcdnBinaryPath(): string {
  return process.env.KVCDN_BINARY_PATH ?? "kvcdn";
}

function isValidArtifactId(id: string): boolean {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id);
}

export async function inferenceRoutes(fastify: FastifyInstance): Promise<void> {
  fastify.post<{
    Params: InferenceParams;
    Body: InferenceBody;
  }>("/api/v1/orgs/:org/projects/:project/artifacts/:artifact_id/infer", async (request, reply) => {
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
    const dataKey = `${namespace}/${artifactId}/${meta.name}`;

    let kvBytes: Buffer;
    try {
      kvBytes = await fastify.artifactStore.get(dataKey);
    } catch (err) {
      request.log.warn({ err }, "failed to fetch artifact bytes");
      return reply.status(502).send({ error: "Failed to fetch artifact" });
    }

    const tmpPath = join(tmpdir(), `kvcdn-infer-${artifactId}.kv`);
    await fs.writeFile(tmpPath, kvBytes);

    const question = request.body.question ?? "";
    const n = Math.min(Math.max(Number(request.body.n) || 32, 1), 256);

    const binary = kvcdnBinaryPath();
    const args = [
      "infer",
      "--model", meta.model_name,
      "--input", tmpPath,
      "--question", question,
      "--n", String(n),
    ];

    return new Promise<void>((resolve) => {
      const child = spawn(binary, args, { stdio: ["ignore", "pipe", "pipe"] });
      let stdout = "";
      let stderr = "";
      child.stdout.on("data", (chunk: Buffer) => { stdout += chunk.toString(); });
      child.stderr.on("data", (chunk: Buffer) => { stderr += chunk.toString(); });
      child.on("error", (err) => {
        request.log.warn({ err }, "kvcdn infer process failed to start");
        fs.unlink(tmpPath).catch(() => undefined);
        reply.status(502).send({ error: "Inference engine unavailable" });
        resolve();
      });
      child.on("close", async (code) => {
        await fs.unlink(tmpPath).catch(() => undefined);
        if (code !== 0) {
          request.log.warn({ code, stderr }, "kvcdn infer exited non-zero");
          reply.status(502).send({ error: "Inference failed", details: stderr });
        } else {
          const tokens = stdout
            .split("\n")
            .map((s) => s.trim())
            .filter((s) => s.length > 0)
            .map((s) => Number.parseInt(s, 10))
            .filter((v) => !Number.isNaN(v));
          reply.status(200).send({
            artifact_id: artifactId,
            tokens,
            token_count: tokens.length,
          });
        }
        resolve();
      });
    });
  });
}
