import { describe, expect, it, beforeAll, afterAll } from "@jest/globals";
import type { FastifyInstance } from "fastify";
import { deriveApiKey } from "../auth.js";
import { artifactMeta, startTestServer } from "./helpers.js";

process.env.KVCDN_CLIENT_ID = "kvcdn-cli";

describe("GET /api/v1/quota", () => {
  let app: FastifyInstance | undefined;
  let baseUrl: string | undefined;

  const seed = "quota-test-seed";
  const apiKey = deriveApiKey(seed, "acme");

  beforeAll(async () => {
    process.env.KVCDN_API_KEY_SEED = seed;
    process.env.KVCDN_API_KEY_ORGS = "acme,contoso";
    process.env.KVCDN_QUOTA_ORGS = "5";
    process.env.KVCDN_QUOTA_PROJECTS = "20";
    process.env.KVCDN_QUOTA_ARTIFACTS = "200";
    process.env.KVCDN_QUOTA_STORAGE_BYTES = "1048576";
    ({ app, url: baseUrl } = await startTestServer());
  });

  afterAll(async () => {
    await app?.close();
    delete process.env.KVCDN_API_KEY_SEED;
    delete process.env.KVCDN_API_KEY_ORGS;
    delete process.env.KVCDN_QUOTA_ORGS;
    delete process.env.KVCDN_QUOTA_PROJECTS;
    delete process.env.KVCDN_QUOTA_ARTIFACTS;
    delete process.env.KVCDN_QUOTA_STORAGE_BYTES;
  });

  it("returns 401 without authorization", async () => {
    const response = await fetch(`${baseUrl}/api/v1/quota`);
    expect(response.status).toBe(401);
    expect(await response.json()).toMatchObject({ error: "Unauthorized" });
  });

  it("returns zero usage for a customer with no artifacts", async () => {
    const response = await fetch(`${baseUrl}/api/v1/quota`, {
      headers: { authorization: `Bearer ${apiKey}` },
    });
    expect(response.status).toBe(200);
    const body = (await response.json()) as {
      customer_id: string;
      quota: Record<string, number>;
      used: Record<string, number>;
    };
    expect(body.customer_id).toBeDefined();
    expect(body.quota.organizations).toBe(5);
    expect(body.quota.projects).toBe(20);
    expect(body.quota.artifacts).toBe(200);
    expect(body.quota.storage_bytes).toBe(1048576);
    expect(body.used.organizations).toBe(0);
    expect(body.used.projects).toBe(0);
    expect(body.used.artifacts).toBe(0);
    expect(body.used.storage_bytes).toBe(0);
  });

  it("aggregates utilization across orgs and projects", async () => {
    const meta = artifactMeta({ name: "quota-artifact.kv", size_bytes: 100 });

    const upload = async (org: string, project: string) => {
      const response = await fetch(`${baseUrl}/api/v1/orgs/${org}/projects/${project}/artifacts`, {
        method: "POST",
        headers: { authorization: `Bearer ${apiKey}`, "content-type": "application/json" },
        body: JSON.stringify(meta),
      });
      expect(response.status).toBe(201);
    };

    await upload("acme", "project-a");
    await upload("acme", "project-a");
    await upload("acme", "project-b");

    const response = await fetch(`${baseUrl}/api/v1/quota`, {
      headers: { authorization: `Bearer ${apiKey}` },
    });
    expect(response.status).toBe(200);
    const body = (await response.json()) as {
      used: {
        organizations: number;
        projects: number;
        artifacts: number;
        storage_bytes: number;
      };
    };
    expect(body.used.organizations).toBe(1);
    expect(body.used.projects).toBe(2);
    expect(body.used.artifacts).toBe(3);
    expect(body.used.storage_bytes).toBe(300);
  });
});
