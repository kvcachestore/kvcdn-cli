import "dotenv/config";
import Fastify from "fastify";
import type { FastifyInstance } from "fastify";
import { artifactsRoutes } from "./routes/artifacts.js";
import { apiKeyRoutes } from "./routes/api-keys.js";
import { quotaRoutes } from "./routes/quota.js";
import { oidcRoutes, getOidcKeyPair } from "./routes/oidc.js";
import { isMockOidc } from "./auth.js";
import type { ArtifactStore } from "./stores/artifact-store.js";
import { TenantS3ArtifactStore } from "./stores/tenant-s3-artifact-store.js";
import { TenantS3MetadataStore } from "./stores/tenant-s3-metadata-store.js";
import type { MetadataStore } from "./stores/metadata-store.js";
import { OwnerOrPublicAccessPolicy } from "./stores/access-policy.js";
import type { AccessPolicy } from "./stores/access-policy.js";
import { OidcOrApiKeyIdentityVerifier } from "./stores/identity-verifier.js";
import type { IdentityVerifier } from "./stores/identity-verifier.js";
import { HashedTenantResolver } from "./stores/tenant-resolver.js";
import type { TenantResolver } from "./stores/tenant-resolver.js";
import { S3BucketStore } from "./stores/s3-bucket-store.js";
import { MemoryBucketStore } from "./stores/memory-bucket-store.js";
import type { BucketStore } from "./stores/bucket-store.js";
import { HashedBucketResolver } from "./stores/hashed-bucket-resolver.js";
import type { BucketResolver } from "./stores/bucket-resolver.js";

declare module "fastify" {
  interface FastifyInstance {
    artifactStore: ArtifactStore;
    metadataStore: MetadataStore;
    accessPolicy: AccessPolicy;
    identityVerifier: IdentityVerifier;
    tenantResolver: TenantResolver;
    bucketStore: BucketStore;
    bucketResolver: BucketResolver;
  }
}


export interface BuildOptions {
  artifactStore?: ArtifactStore;
  metadataStore?: MetadataStore;
  accessPolicy?: AccessPolicy;
  identityVerifier?: IdentityVerifier;
  tenantResolver?: TenantResolver;
  bucketStore?: BucketStore;
  bucketResolver?: BucketResolver;
}

export async function build(options: BuildOptions = {}): Promise<FastifyInstance> {
  const fastify = Fastify({ logger: true });

  const bucketStore = options.bucketStore ?? (process.env.KVCDN_CONTROL_BUCKET ? new S3BucketStore(fastify.log) : new MemoryBucketStore());
  const bucketResolver = options.bucketResolver ?? new HashedBucketResolver(bucketStore, undefined, fastify.log);
  const artifactStore = options.artifactStore ?? new TenantS3ArtifactStore(bucketResolver);
  fastify.decorate("bucketStore", bucketStore);
  fastify.decorate("bucketResolver", bucketResolver);
  fastify.decorate("artifactStore", artifactStore);
  fastify.decorate("metadataStore", options.metadataStore ?? new TenantS3MetadataStore(artifactStore));
  fastify.decorate("accessPolicy", options.accessPolicy ?? new OwnerOrPublicAccessPolicy());
  fastify.decorate("identityVerifier", options.identityVerifier ?? new OidcOrApiKeyIdentityVerifier());
  fastify.decorate("tenantResolver", options.tenantResolver ?? new HashedTenantResolver());
  fastify.get("/health", async () => ({ status: "ok" }));
  fastify.register(artifactsRoutes);
  fastify.register(apiKeyRoutes);
  fastify.register(quotaRoutes);

  if (isMockOidc()) {
    fastify.register(oidcRoutes, { prefix: "/oidc" });
  }

  return fastify;
}

export async function buildAndInit(options: BuildOptions = {}): Promise<FastifyInstance> {
  const app = await build(options);
  if (isMockOidc()) {
    await getOidcKeyPair();
  }
  return app;
}

async function start(): Promise<void> {
  const fastify = await buildAndInit();
  const rawPort = process.env.KVCDN_API_PORT ?? "3000";
  const port = Number(rawPort);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    console.error(`KVCDN_API_PORT must be an integer between 1 and 65535, got: ${rawPort}`);
    process.exit(1);
  }
  const host = process.env.KVCDN_HOST ?? "0.0.0.0";

  try {
    await fastify.listen({ port, host });
    console.log(`Server listening at http://${host}:${port}`);
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
}

if (import.meta.url === new URL(process.argv[1] ?? "", `file://${process.cwd()}/`).href) {
  void start();
}