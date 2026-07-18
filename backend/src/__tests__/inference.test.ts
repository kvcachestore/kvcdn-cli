import { describe, expect, it, beforeAll, afterAll } from "@jest/globals";
import type { FastifyInstance } from "fastify";
import { artifactMeta, getAccessToken, startTestServer } from "./helpers.js";
import { MemoryArtifactStore } from "../stores/memory-artifact-store.js";
import { createHash } from "node:crypto";

process.env.KVCDN_CLIENT_ID = "kvcdn-cli";

describe("POST /api/v1/orgs/:org/projects/:project/artifacts/:artifact_id/infer", () => {
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

  it("returns 401 when artifact is private and no auth is provided", async () => {
    const meta = artifactMeta();
    const token = await getAccessToken(baseUrl!);
    const uploadResponse = await fetch(`${baseUrl}/api/v1/orgs/acme/projects/acme/artifacts`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify(meta),
    });
    expect(uploadResponse.status).toBe(201);
    const { artifact_id: artifactId } = (await uploadResponse.json()) as { artifact_id: string };

    const inferResponse = await fetch(`${baseUrl}/api/v1/orgs/acme/projects/acme/artifacts/${artifactId}/infer`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question: "hello" }),
    });
    expect(inferResponse.status).toBe(401);
  });

  it("returns 502 when kvcdn binary is not available", async () => {
    process.env.KVCDN_BINARY_PATH = "/nonexistent/kvcdn";
    const meta = artifactMeta();
    const token = await getAccessToken(baseUrl!);
    const uploadResponse = await fetch(`${baseUrl}/api/v1/orgs/acme/projects/acme/artifacts`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify(meta),
    });
    expect(uploadResponse.status).toBe(201);
    const { artifact_id: artifactId } = (await uploadResponse.json()) as { artifact_id: string };

    const inferResponse = await fetch(`${baseUrl}/api/v1/orgs/acme/projects/acme/artifacts/${artifactId}/infer`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify({ question: "hello" }),
    });
    expect(inferResponse.status).toBe(502);
    delete process.env.KVCDN_BINARY_PATH;
  });

  it("returns 400 for an invalid artifact_id", async () => {
    const token = await getAccessToken(baseUrl!);
    const response = await fetch(`${baseUrl}/api/v1/orgs/acme/projects/acme/artifacts/not-a-uuid/infer`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify({ question: "hello" }),
    });
    expect(response.status).toBe(400);
  });

  it("allows unauthenticated inference on public artifacts", async () => {
    const meta = artifactMeta({ visibility: "public" });
    const token = await getAccessToken(baseUrl!);
    const uploadResponse = await fetch(`${baseUrl}/api/v1/orgs/acme/projects/acme/artifacts`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify(meta),
    });
    expect(uploadResponse.status).toBe(201);
    const { artifact_id: artifactId } = (await uploadResponse.json()) as { artifact_id: string };

    process.env.KVCDN_BINARY_PATH = "/nonexistent/kvcdn";
    const inferResponse = await fetch(`${baseUrl}/api/v1/orgs/acme/projects/acme/artifacts/${artifactId}/infer`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question: "hello" }),
    });
    expect(inferResponse.status).toBe(502);
    delete process.env.KVCDN_BINARY_PATH;
  });

  it("returns 502 when artifact bytes are missing", async () => {
    const meta = artifactMeta();
    const token = await getAccessToken(baseUrl!);
    const uploadResponse = await fetch(`${baseUrl}/api/v1/orgs/acme/projects/acme/artifacts`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify(meta),
    });
    expect(uploadResponse.status).toBe(201);
    const { artifact_id: artifactId } = (await uploadResponse.json()) as { artifact_id: string };

    const inferResponse = await fetch(`${baseUrl}/api/v1/orgs/acme/projects/acme/artifacts/${artifactId}/infer`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify({ question: "hello" }),
    });
    expect(inferResponse.status).toBe(502);
  });
});
