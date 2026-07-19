import { describe, expect, it, beforeAll, afterAll } from "@jest/globals";
import { createHash } from "node:crypto";
import type { FastifyInstance } from "fastify";
import { getAccessToken, startTestServer } from "./helpers.js";
import { MemoryArtifactStore } from "../stores/memory-artifact-store.js";
import { deriveCustomerId } from "../stores/tenant-resolver.js";

process.env.KVCDN_CLIENT_ID = "kvcdn-cli";

describe("flat /api/v1/artifacts routes", () => {
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

  it("supports the CLI upload/confirm/list/download/delete flow", async () => {
    const token = await getAccessToken(baseUrl!);
    const customerId = deriveCustomerId("oidc:kvcdn-user");

    const data = "flat-route-test-content";
    const sha256 = createHash("sha256").update(data).digest("hex");

    // Upload init with the CLI's flat request shape.
    const uploadResponse = await fetch(`${baseUrl}/api/v1/artifacts/upload-url`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify({
        model_name: "flat-model",
        dtype: "F16",
        num_tokens: 64,
        visibility: "private",
        size_bytes: Buffer.byteLength(data),
        checksum: sha256,
        metadata: { name: "flat.kv", sha256, storage_dtype: "F16", num_layers: 12, quantized: false },
      }),
    });
    expect(uploadResponse.status).toBe(200);
    const init = (await uploadResponse.json()) as {
      artifact_id: string;
      url: string;
      method: string;
      expires_at: string;
    };
    expect(init.artifact_id).toMatch(/^[0-9a-f-]{36}$/);
    expect(init.method).toBe("PUT");
    expect(init.url).toContain("/orgs/default/projects/default/");

    // Simulate the presigned PUT by writing to the store directly.
    const store = app?.artifactStore as MemoryArtifactStore | undefined;
    const key = `artifacts/customers/${customerId}/orgs/default/projects/default/${init.artifact_id}/flat.kv`;
    await store!.put(key, Buffer.from(data));

    // Confirm verifies size and sha256 against the stored object.
    const confirm = await fetch(`${baseUrl}/api/v1/artifacts/${init.artifact_id}/confirm-upload`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify({ digest: sha256, size_bytes: Buffer.byteLength(data) }),
    });
    expect(confirm.status).toBe(200);

    // List returns the artifact with CLI-friendly field aliases.
    const list = await fetch(`${baseUrl}/api/v1/artifacts`, {
      headers: { authorization: `Bearer ${token}` },
    });
    expect(list.status).toBe(200);
    const listBody = (await list.json()) as { artifacts: Array<Record<string, unknown>> };
    const listed = listBody.artifacts.find((a) => a.artifact_id === init.artifact_id);
    expect(listed).toBeDefined();
    expect(listed?.id).toBe(init.artifact_id);
    expect(listed?.name).toBe("flat.kv");
    expect(listed?.model_name).toBe("flat-model");
    expect(listed?.sha256).toBe(sha256);
    expect(listed?.num_layers).toBe(12);
    expect(listed?.visibility).toBe("private");

    // Download URL.
    const download = await fetch(`${baseUrl}/api/v1/artifacts/${init.artifact_id}/download-url`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}` },
    });
    expect(download.status).toBe(200);
    const downloadBody = (await download.json()) as { artifact_id: string; download_url: string };
    expect(downloadBody.artifact_id).toBe(init.artifact_id);
    expect(downloadBody.download_url).toContain(init.artifact_id);

    // Delete removes metadata and stored bytes.
    const del = await fetch(`${baseUrl}/api/v1/artifacts/${init.artifact_id}`, {
      method: "DELETE",
      headers: { authorization: `Bearer ${token}` },
    });
    expect(del.status).toBe(204);

    const listAfter = await fetch(`${baseUrl}/api/v1/artifacts`, {
      headers: { authorization: `Bearer ${token}` },
    });
    const listAfterBody = (await listAfter.json()) as { artifacts: Array<Record<string, unknown>> };
    expect(listAfterBody.artifacts.find((a) => a.artifact_id === init.artifact_id)).toBeUndefined();
    await expect(store!.get(key)).rejects.toThrow();
  });

  it("returns 401 without an Authorization header", async () => {
    expect((await fetch(`${baseUrl}/api/v1/artifacts`)).status).toBe(401);
    expect(
      (
        await fetch(`${baseUrl}/api/v1/artifacts/upload-url`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({}),
        })
      ).status
    ).toBe(401);
  });

  it("returns 404 for unknown artifact ids", async () => {
    const token = await getAccessToken(baseUrl!);
    const id = "123e4567-e89b-42d3-a456-426614174000";

    const confirm = await fetch(`${baseUrl}/api/v1/artifacts/${id}/confirm-upload`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify({}),
    });
    expect(confirm.status).toBe(404);

    const del = await fetch(`${baseUrl}/api/v1/artifacts/${id}`, {
      method: "DELETE",
      headers: { authorization: `Bearer ${token}` },
    });
    expect(del.status).toBe(404);
  });

  it("rejects upload bodies without a name or checksum", async () => {
    const token = await getAccessToken(baseUrl!);
    const response = await fetch(`${baseUrl}/api/v1/artifacts/upload-url`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify({ dtype: "F16" }),
    });
    expect(response.status).toBe(400);
  });
});
