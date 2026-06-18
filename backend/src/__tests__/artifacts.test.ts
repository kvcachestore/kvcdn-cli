import { describe, expect, it, beforeAll, afterAll } from "@jest/globals";
import { createHash } from "node:crypto";
import type { FastifyInstance } from "fastify";
import { deriveApiKey } from "../auth.js";
import { artifactMeta, getAccessToken, startTestServer } from "./helpers.js";
import { MemoryArtifactStore } from "../stores/memory-artifact-store.js";
import { MemoryBucketStore } from "../stores/memory-bucket-store.js";
import { HashedBucketResolver } from "../stores/hashed-bucket-resolver.js";
import { HashedTenantResolver, deriveCustomerId } from "../stores/tenant-resolver.js";

process.env.KVCDN_CLIENT_ID = "kvcdn-cli";

describe("POST /v1/projects/:project/artifacts", () => {
  let app: FastifyInstance | undefined;
  let baseUrl: string | undefined;
  const meta = artifactMeta();

  beforeAll(async () => {
    ({ app, url: baseUrl } = await startTestServer());
    process.env.KVCDN_ISSUER_URL = `${baseUrl}/oidc`;
  });

  afterAll(async () => {
    await app?.close();
    delete process.env.KVCDN_ISSUER_URL;
  });

  it("returns 201 with artifact_id and upload_url for a valid JWT", async () => {
    const token = await getAccessToken(baseUrl!);

    const response = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify(meta),
    });

    expect(response.status).toBe(201);
    const body = (await response.json()) as { artifact_id: string; upload_url: string };
    expect(body.artifact_id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
    expect(body.upload_url).toContain("artifacts/customers/");
    expect(body.upload_url).toContain("/orgs/acme/projects/acme/");
    expect(body.upload_url).toContain(meta.name);
  });

  it("returns 401 when Authorization header is missing", async () => {
    const response = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(meta),
    });

    expect(response.status).toBe(401);
    expect(await response.json()).toMatchObject({ error: "Unauthorized" });
  });

  it("returns 401 for an invalid JWT", async () => {
    const response = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts`, {
      method: "POST",
      headers: { authorization: "Bearer not-a-real-token", "content-type": "application/json" },
      body: JSON.stringify(meta),
    });

    expect(response.status).toBe(401);
    expect(await response.json()).toMatchObject({ error: "Unauthorized" });
  });

  it("returns 401 when the issuer does not match", async () => {
    const response = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts`, {
      method: "POST",
      headers: { authorization: "Bearer not-a-real-token", "content-type": "application/json" },
      body: JSON.stringify(meta),
    });

    expect(response.status).toBe(401);
  });

  it("returns 401 when the audience does not match", async () => {
    const response = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts`, {
      method: "POST",
      headers: { authorization: "Bearer not-a-real-token", "content-type": "application/json" },
      body: JSON.stringify(meta),
    });

    expect(response.status).toBe(401);
  });

  it("returns 401 when Authorization header is missing for DELETE", async () => {
    const response = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts/${meta.sha256}`, {
      method: "DELETE",
    });
    expect(response.status).toBe(401);
  });

  it("returns 401 for an invalid JWT on DELETE", async () => {
    const response = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts/${meta.sha256}`, {
      method: "DELETE",
      headers: { authorization: "Bearer not-a-real-token" },
    });
    expect(response.status).toBe(401);
  });

  it("lists artifacts for a project", async () => {
    const token = await getAccessToken(baseUrl!);
    const uploadResponse = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify(meta),
    });
    expect(uploadResponse.status).toBe(201);

    const listResponse = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts`, {
      headers: { authorization: `Bearer ${token}` },
    });
    expect(listResponse.status).toBe(200);
    const body = (await listResponse.json()) as { artifacts: Array<{ artifact_id: string; name: string }> };
    expect(body.artifacts.length).toBeGreaterThan(0);
    expect(body.artifacts[body.artifacts.length - 1].name).toBe(meta.name);
  });

  it("returns a presigned download URL for an existing artifact", async () => {
    const token = await getAccessToken(baseUrl!);
    const uploadResponse = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify(meta),
    });
    expect(uploadResponse.status).toBe(201);
    const { artifact_id: artifactId } = (await uploadResponse.json()) as { artifact_id: string };

    const downloadResponse = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts/${artifactId}/download`, {
      headers: { authorization: `Bearer ${token}` },
    });
    expect(downloadResponse.status).toBe(200);
    const body = (await downloadResponse.json()) as { artifact_id: string; download_url: string };
    expect(body.artifact_id).toBe(artifactId);
    expect(body.download_url).toContain("artifacts/customers/");
    expect(body.download_url).toContain(`/orgs/acme/projects/acme/${artifactId}/${meta.name}`);
    expect(body.download_url).toContain("download=true");
  });

  it("returns 404 when downloading a missing artifact", async () => {
    const token = await getAccessToken(baseUrl!);
    const response = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts/00000000-0000-0000-0000-000000000000/download`, {
      headers: { authorization: `Bearer ${token}` },
    });
    expect(response.status).toBe(404);
  });

  it("allows unauthenticated download of public artifacts", async () => {
    const token = await getAccessToken(baseUrl!);
    const publicMeta = { ...meta, visibility: "public" };
    const uploadResponse = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify(publicMeta),
    });
    expect(uploadResponse.status).toBe(201);
    const { artifact_id: artifactId } = (await uploadResponse.json()) as { artifact_id: string };

    const downloadResponse = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts/${artifactId}/download`);
    expect(downloadResponse.status).toBe(200);
    const body = (await downloadResponse.json()) as { artifact_id: string; download_url: string };
    expect(body.artifact_id).toBe(artifactId);
    expect(body.download_url).toContain("download=true");
  });

  it("requires authentication to download private artifacts", async () => {
    const token = await getAccessToken(baseUrl!);
    const uploadResponse = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify(meta),
    });
    expect(uploadResponse.status).toBe(201);
    const { artifact_id: artifactId } = (await uploadResponse.json()) as { artifact_id: string };

    const downloadResponse = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts/${artifactId}/download`);
    expect(downloadResponse.status).toBe(401);
  });

  it("verifies sha256 on upload completion", async () => {
    const putObject = async (key: string, body: Buffer | string): Promise<void> => {
      const store = app?.artifactStore as MemoryArtifactStore | undefined;
      if (!store) throw new Error("ArtifactStore not available");
      await store.put(key, body);
    };
    const token = await getAccessToken(baseUrl!);
    const uploadResponse = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify(meta),
    });
    expect(uploadResponse.status).toBe(201);
    const { artifact_id: artifactId } = (await uploadResponse.json()) as { artifact_id: string };

    const customerId = deriveCustomerId("oidc:kvcdn-user");
    const dataKey = `artifacts/customers/${customerId}/orgs/acme/projects/acme/${artifactId}/${meta.name}`;
    await putObject(dataKey, Buffer.from("e2e-dummy-content", "utf-8"));

    const completeResponse = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts/${artifactId}/complete`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}` },
    });
    expect(completeResponse.status).toBe(400);
    const body = (await completeResponse.json()) as { error: string };
    expect(body.error).toBe("size mismatch");

    const actualData = "expected-content-for-sha256";
    const expectedSha256 = createHash("sha256").update(actualData).digest("hex");
    const metaWithCorrectSha = {
      ...meta,
      sha256: expectedSha256,
      size_bytes: Buffer.byteLength(actualData, "utf-8"),
    };
    const upload2 = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify(metaWithCorrectSha),
    });
    expect(upload2.status).toBe(201);
    const { artifact_id: artifactId2 } = (await upload2.json()) as { artifact_id: string };
    const dataKey2 = `artifacts/customers/${customerId}/orgs/acme/projects/acme/${artifactId2}/${meta.name}`;
    await putObject(dataKey2, Buffer.from(actualData, "utf-8"));

    const complete2 = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts/${artifactId2}/complete`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}` },
    });
    expect(complete2.status).toBe(200);
  });

  it("returns 401 when listing without authorization", async () => {
    const response = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts`);
    expect(response.status).toBe(401);
  });
});

describe("API key auth", () => {
  let app: FastifyInstance | undefined;
  let baseUrl: string | undefined;

  const seed = "test-seed-deadbeef";
  const apiKey = deriveApiKey(seed, "acme");
  const meta = artifactMeta();

  beforeAll(async () => {
    process.env.KVCDN_API_KEY_SEED = seed;
    process.env.KVCDN_API_KEY_ORGS = "acme,contoso";
    ({ app, url: baseUrl } = await startTestServer());
    process.env.KVCDN_ISSUER_URL = `${baseUrl}/oidc`;
  });

  afterAll(async () => {
    await app?.close();
    delete process.env.KVCDN_API_KEY_SEED;
    delete process.env.KVCDN_API_KEY_ORGS;
    delete process.env.KVCDN_ISSUER_URL;
  });

  it("returns 200 for a valid API key on /v1/api-keys/verify", async () => {
    const response = await fetch(`${baseUrl}/v1/api-keys/verify`, {
      method: "POST",
      headers: { authorization: `Bearer ${apiKey}` },
    });
    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({ customer_id: deriveCustomerId("org:acme") });
  });

  it("returns 401 for an invalid API key", async () => {
    const response = await fetch(`${baseUrl}/v1/api-keys/verify`, {
      method: "POST",
      headers: { authorization: "Bearer kv_wrong_key" },
    });
    expect(response.status).toBe(401);
  });

  it("accepts API key for artifact upload", async () => {
    const response = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts`, {
      method: "POST",
      headers: { authorization: `Bearer ${apiKey}`, "content-type": "application/json" },
      body: JSON.stringify(meta),
    });
    expect(response.status).toBe(201);
    const body = (await response.json()) as { artifact_id: string; upload_url: string };
    expect(body.artifact_id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
    expect(body.upload_url).toContain("artifacts/customers/");
    expect(body.upload_url).toContain("/orgs/acme/projects/acme/");
  });

  it("returns 401 for an invalid API key on artifact delete", async () => {
    const response = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts/${meta.sha256}`, {
      method: "DELETE",
      headers: { authorization: "Bearer kv_wrong_key" },
    });
    expect(response.status).toBe(401);
  });
});

describe("artifact isolation between orgs", () => {
  let app: FastifyInstance | undefined;
  let baseUrl: string | undefined;

  const seed = "isolation-test-seed";
  const acmeKey = deriveApiKey(seed, "acme");
  const contosoKey = deriveApiKey(seed, "contoso");
  const meta = artifactMeta();

  beforeAll(async () => {
    process.env.KVCDN_API_KEY_SEED = seed;
    process.env.KVCDN_API_KEY_ORGS = "acme,contoso";
    ({ app, url: baseUrl } = await startTestServer());
    process.env.KVCDN_ISSUER_URL = `${baseUrl}/oidc`;
  });

  afterAll(async () => {
    await app?.close();
    delete process.env.KVCDN_API_KEY_SEED;
    delete process.env.KVCDN_API_KEY_ORGS;
    delete process.env.KVCDN_ISSUER_URL;
  });

  it("acme can upload and list its own artifact", async () => {
    const uploadResponse = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts`, {
      method: "POST",
      headers: { authorization: `Bearer ${acmeKey}`, "content-type": "application/json" },
      body: JSON.stringify(meta),
    });
    expect(uploadResponse.status).toBe(201);
    const { artifact_id: artifactId } = (await uploadResponse.json()) as { artifact_id: string };

    const listResponse = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts`, {
      headers: { authorization: `Bearer ${acmeKey}` },
    });
    expect(listResponse.status).toBe(200);
    const body = (await listResponse.json()) as { artifacts: Array<{ artifact_id: string }> };
    expect(body.artifacts.some((a) => a.artifact_id === artifactId)).toBe(true);
  });

  it("contoso cannot list acme's artifacts", async () => {
    const listResponse = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts`, {
      headers: { authorization: `Bearer ${contosoKey}` },
    });
    expect(listResponse.status).toBe(200);
    const body = (await listResponse.json()) as { artifacts: Array<unknown> };
    expect(body.artifacts).toHaveLength(0);
  });

  it("contoso cannot download an acme artifact by id", async () => {
    const uploadResponse = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts`, {
      method: "POST",
      headers: { authorization: `Bearer ${acmeKey}`, "content-type": "application/json" },
      body: JSON.stringify(meta),
    });
    expect(uploadResponse.status).toBe(201);
    const { artifact_id: artifactId } = (await uploadResponse.json()) as { artifact_id: string };

    const downloadResponse = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts/${artifactId}/download`, {
      headers: { authorization: `Bearer ${contosoKey}` },
    });
    expect(downloadResponse.status).toBe(404);
  });

  it("contoso cannot delete an acme artifact by id", async () => {
    const uploadResponse = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts`, {
      method: "POST",
      headers: { authorization: `Bearer ${acmeKey}`, "content-type": "application/json" },
      body: JSON.stringify(meta),
    });
    expect(uploadResponse.status).toBe(201);
    const { artifact_id: artifactId } = (await uploadResponse.json()) as { artifact_id: string };

    const deleteResponse = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts/${artifactId}`, {
      method: "DELETE",
      headers: { authorization: `Bearer ${contosoKey}` },
    });
    expect(deleteResponse.status).toBe(404);

    const ownerDownloadResponse = await fetch(`${baseUrl}/v1/orgs/acme/projects/acme/artifacts/${artifactId}/download`, {
      headers: { authorization: `Bearer ${acmeKey}` },
    });
    expect(ownerDownloadResponse.status).toBe(200);
  });
});

describe("artifact isolation between OIDC subjects", () => {
  let app: FastifyInstance | undefined;
  let baseUrl: string | undefined;

  const meta = artifactMeta();

  beforeAll(async () => {
    ({ app, url: baseUrl } = await startTestServer());
    process.env.KVCDN_ISSUER_URL = `${baseUrl}/oidc`;
  });

  afterAll(async () => {
    await app?.close();
    delete process.env.KVCDN_ISSUER_URL;
  });

  it("alice can upload and list her own artifact", async () => {
    const token = await getAccessToken(baseUrl!, "alice");
    const uploadResponse = await fetch(`${baseUrl}/v1/orgs/personal/projects/personal/artifacts`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify(meta),
    });
    expect(uploadResponse.status).toBe(201);
    const { artifact_id: artifactId } = (await uploadResponse.json()) as { artifact_id: string };

    const listResponse = await fetch(`${baseUrl}/v1/orgs/personal/projects/personal/artifacts`, {
      headers: { authorization: `Bearer ${token}` },
    });
    expect(listResponse.status).toBe(200);
    const body = (await listResponse.json()) as { artifacts: Array<{ artifact_id: string }> };
    expect(body.artifacts.some((a) => a.artifact_id === artifactId)).toBe(true);
  });

  it("bob cannot list alice's artifacts", async () => {
    const aliceToken = await getAccessToken(baseUrl!, "alice");
    const uploadResponse = await fetch(`${baseUrl}/v1/orgs/personal/projects/personal/artifacts`, {
      method: "POST",
      headers: { authorization: `Bearer ${aliceToken}`, "content-type": "application/json" },
      body: JSON.stringify(meta),
    });
    expect(uploadResponse.status).toBe(201);

    const bobToken = await getAccessToken(baseUrl!, "bob");
    const listResponse = await fetch(`${baseUrl}/v1/orgs/personal/projects/personal/artifacts`, {
      headers: { authorization: `Bearer ${bobToken}` },
    });
    expect(listResponse.status).toBe(200);
    const body = (await listResponse.json()) as { artifacts: Array<unknown> };
    expect(body.artifacts).toHaveLength(0);
  });

  it("bob cannot download an alice artifact by id", async () => {
    const aliceToken = await getAccessToken(baseUrl!, "alice");
    const uploadResponse = await fetch(`${baseUrl}/v1/orgs/personal/projects/personal/artifacts`, {
      method: "POST",
      headers: { authorization: `Bearer ${aliceToken}`, "content-type": "application/json" },
      body: JSON.stringify(meta),
    });
    expect(uploadResponse.status).toBe(201);
    const { artifact_id: artifactId } = (await uploadResponse.json()) as { artifact_id: string };

    const bobToken = await getAccessToken(baseUrl!, "bob");
    const downloadResponse = await fetch(`${baseUrl}/v1/orgs/personal/projects/personal/artifacts/${artifactId}/download`, {
      headers: { authorization: `Bearer ${bobToken}` },
    });
    expect(downloadResponse.status).toBe(404);
  });

  it("bob cannot delete an alice artifact by id", async () => {
    const aliceToken = await getAccessToken(baseUrl!, "alice");
    const uploadResponse = await fetch(`${baseUrl}/v1/orgs/personal/projects/personal/artifacts`, {
      method: "POST",
      headers: { authorization: `Bearer ${aliceToken}`, "content-type": "application/json" },
      body: JSON.stringify(meta),
    });
    expect(uploadResponse.status).toBe(201);
    const { artifact_id: artifactId } = (await uploadResponse.json()) as { artifact_id: string };

    const bobToken = await getAccessToken(baseUrl!, "bob");
    const deleteResponse = await fetch(`${baseUrl}/v1/orgs/personal/projects/personal/artifacts/${artifactId}`, {
      method: "DELETE",
      headers: { authorization: `Bearer ${bobToken}` },
    });
    expect(deleteResponse.status).toBe(404);

    const ownerDownloadResponse = await fetch(`${baseUrl}/v1/orgs/personal/projects/personal/artifacts/${artifactId}/download`, {
      headers: { authorization: `Bearer ${aliceToken}` },
    });
    expect(ownerDownloadResponse.status).toBe(200);
  });
});

describe("mock OIDC routes", () => {
  it("are not registered when KVCDN_ISSUER_URL is configured", async () => {
    process.env.KVCDN_ISSUER_URL = "https://auth.external.example";
    const { buildAndInit } = await import("../index.js");
    let app: FastifyInstance | undefined;
    const bucketStore = new MemoryBucketStore();
    try {
      app = await buildAndInit({
        artifactStore: new MemoryArtifactStore(),
        bucketStore,
        bucketResolver: new HashedBucketResolver(bucketStore, "memory-bucket"),
      });
      await app.listen({ port: 0, host: "127.0.0.1" });
      const address = app.server.address();
      if (!address || typeof address !== "object" || !("port" in address)) {
        throw new Error("Failed to get server address");
      }
      const baseUrl = `http://127.0.0.1:${address.port}`;

      const response = await fetch(`${baseUrl}/oidc/.well-known/openid-configuration`);
      expect(response.status).toBe(404);
    } finally {
      await app?.close();
      delete process.env.KVCDN_ISSUER_URL;
    }
  });
});

describe("mock OIDC device-code flow", () => {
  let app: FastifyInstance | undefined;
  let baseUrl: string | undefined;

  beforeAll(async () => {
    ({ app, url: baseUrl } = await startTestServer());
    process.env.KVCDN_ISSUER_URL = `${baseUrl}/oidc`;
  });

  afterAll(async () => {
    await app?.close();
    delete process.env.KVCDN_ISSUER_URL;
  });

  it("advertises device authorization endpoint in discovery", async () => {
    const response = await fetch(`${baseUrl}/oidc/.well-known/openid-configuration`);
    expect(response.status).toBe(200);
    const data = (await response.json()) as { device_authorization_endpoint?: string };
    expect(data.device_authorization_endpoint).toMatch(/\/device\/auth$/);
  });

  it("returns a usable access token after device activation", async () => {
    const authResponse = await fetch(`${baseUrl}/oidc/device/auth`, {
      method: "POST",
      headers: { "content-type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ client_id: process.env.KVCDN_CLIENT_ID ?? "kvcdn-cli" }),
    });
    expect(authResponse.status).toBe(200);
    const authData = (await authResponse.json()) as {
      device_code: string;
      user_code: string;
      verification_uri_complete: string;
      interval: number;
    };

    const pendingResponse = await fetch(`${baseUrl}/oidc/token`, {
      method: "POST",
      headers: { "content-type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        grant_type: "urn:ietf:params:oauth:grant-type:device_code",
        device_code: authData.device_code,
        client_id: process.env.KVCDN_CLIENT_ID ?? "kvcdn-cli",
      }),
    });
    expect(pendingResponse.status).toBe(400);
    const pendingData = (await pendingResponse.json()) as { error: string };
    expect(pendingData.error).toBe("authorization_pending");

    const activateResponse = await fetch(authData.verification_uri_complete);
    expect(activateResponse.status).toBe(200);

    const tokenResponse = await fetch(`${baseUrl}/oidc/token`, {
      method: "POST",
      headers: { "content-type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        grant_type: "urn:ietf:params:oauth:grant-type:device_code",
        device_code: authData.device_code,
        client_id: process.env.KVCDN_CLIENT_ID ?? "kvcdn-cli",
      }),
    });
    expect(tokenResponse.status).toBe(200);
    const tokenData = (await tokenResponse.json()) as { access_token: string };
    expect(tokenData.access_token).toBeTruthy();

    const uploadResponse = await fetch(`${baseUrl}/v1/orgs/personal/projects/personal/artifacts`, {
      method: "POST",
      headers: { authorization: `Bearer ${tokenData.access_token}`, "content-type": "application/json" },
      body: JSON.stringify(artifactMeta({
        name: "device-artifact.kv",
        size_bytes: 8,
        num_tokens: 4,
        num_layers: 2,
      })),
    });
    expect(uploadResponse.status).toBe(201);
  });
});

describe("GET /health", () => {
  let app: FastifyInstance | undefined;
  let baseUrl: string | undefined;

  beforeAll(async () => {
    ({ app, url: baseUrl } = await startTestServer());
  });

  afterAll(async () => {
    await app?.close();
  });

  it("returns ok", async () => {
    const response = await fetch(`${baseUrl}/health`);
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ status: "ok" });
  });
});
