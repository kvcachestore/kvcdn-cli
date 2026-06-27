import { expect } from "@jest/globals";
import type { FastifyInstance } from "fastify";
import { MemoryArtifactStore } from "../stores/memory-artifact-store.js";
import { MemoryMetadataStore } from "../stores/memory-metadata-store.js";
import { MemoryBucketStore } from "../stores/memory-bucket-store.js";
import { HashedBucketResolver } from "../stores/hashed-bucket-resolver.js";
import { OwnerOrPublicAccessPolicy } from "../stores/access-policy.js";
import { OidcOrApiKeyIdentityVerifier } from "../stores/identity-verifier.js";
import { HashedTenantResolver } from "../stores/tenant-resolver.js";

export async function startTestServer(port = 0): Promise<{ app: FastifyInstance; url: string }> {
  const { buildAndInit } = await import("../index.js");
  const artifactStore = new MemoryArtifactStore();
  const bucketStore = new MemoryBucketStore();
  const app = await buildAndInit({
    artifactStore,
    metadataStore: new MemoryMetadataStore(),
    accessPolicy: new OwnerOrPublicAccessPolicy(),
    identityVerifier: new OidcOrApiKeyIdentityVerifier(),
    tenantResolver: new HashedTenantResolver(),
    bucketStore,
    bucketResolver: new HashedBucketResolver(bucketStore, "memory-bucket"),
  });
  await app.listen({ port, host: "127.0.0.1" });
  const address = app.server.address();
  if (address && typeof address === "object" && "port" in address) {
    return { app, url: `http://127.0.0.1:${address.port}` };
  }
  throw new Error("Failed to get server address");
}

export async function getAccessToken(baseUrl: string, subject = "kvcdn-user"): Promise<string> {
  const response = await fetch(`${baseUrl}/oidc/token`, {
    method: "POST",
    headers: { "content-type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "authorization_code",
      code: `mock-auth-code-${subject}`,
      client_id: process.env.KVCDN_CLIENT_ID ?? "kvcdn-cli",
      redirect_uri: "http://localhost:8080/oidc/callback",
      code_verifier: "test-verifier",
    }),
  });
  expect(response.status).toBe(200);
  const data = (await response.json()) as { access_token: string };
  return data.access_token;
}

export function artifactMeta(overrides: Partial<Record<string, unknown>> = {}) {
  return {
    name: "artifact.kv",
    size_bytes: 1234,
    sha256: "deadbeef".repeat(8),
    dtype: "F16",
    storage_dtype: "F16",
    num_tokens: 128,
    num_layers: 32,
    quantized: false,
    model_name: "Qwen/Qwen3-0.6B",
    ...overrides,
  };
}
