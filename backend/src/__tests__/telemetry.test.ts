import { afterAll, beforeAll, describe, expect, it } from "@jest/globals";
import nock from "nock";
import { startTestServer } from "./helpers.js";
import type { FastifyInstance } from "fastify";

describe("POST /api/v1/telemetry", () => {
  let app: FastifyInstance | undefined;
  let baseUrl: string | undefined;

  beforeAll(async () => {
    ({ app, url: baseUrl } = await startTestServer());
  });

  afterAll(async () => {
    await app?.close();
    nock.cleanAll();
  });

  it("returns 204 when forwarding is not configured", async () => {
    delete process.env.KVCDN_TELEMETRY_URL;
    delete process.env.KVCDN_TELEMETRY_SECRET;

    const response = await fetch(`${baseUrl}/api/v1/telemetry`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        command: "verify",
        version: "0.2.0",
        duration_ms: 100,
        success: true,
      }),
    });

    expect(response.status).toBe(204);
  });

  it("rejects malformed events with 400", async () => {
    const response = await fetch(`${baseUrl}/api/v1/telemetry`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ command: "", version: "0.2.0", duration_ms: -1, success: true }),
    });

    expect(response.status).toBe(400);
    const body = (await response.json()) as { error: string };
    expect(body.error).toBe("Invalid telemetry event");
  });

  it("forwards a valid event to the configured upstream", async () => {
    process.env.KVCDN_TELEMETRY_URL = "https://telemetry.example.com";
    process.env.KVCDN_TELEMETRY_SECRET = "secret-123";

    const scope = nock("https://telemetry.example.com")
      .post("/events", (body) => {
        return (
          body.command === "upload" &&
          body.version === "0.2.0" &&
          body.duration_ms === 250 &&
          body.success === false &&
          body.error_kind === "auth_error"
        );
      })
      .matchHeader("authorization", "Bearer secret-123")
      .matchHeader("content-type", "application/json")
      .reply(204);

    const response = await fetch(`${baseUrl}/api/v1/telemetry`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        command: "upload",
        version: "0.2.0",
        duration_ms: 250,
        success: false,
        error_kind: "auth_error",
      }),
    });

    expect(response.status).toBe(204);
    expect(scope.isDone()).toBe(true);
  });

  it("returns 204 when upstream fails", async () => {
    process.env.KVCDN_TELEMETRY_URL = "https://telemetry.example.com";
    process.env.KVCDN_TELEMETRY_SECRET = "secret-123";

    nock("https://telemetry.example.com").post("/events").reply(503);

    const response = await fetch(`${baseUrl}/api/v1/telemetry`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        command: "list",
        version: "0.2.0",
        duration_ms: 50,
        success: true,
      }),
    });

    expect(response.status).toBe(204);
  });

  it("rejects events with an invalid error_kind", async () => {
    const response = await fetch(`${baseUrl}/api/v1/telemetry`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        command: "upload",
        version: "0.2.0",
        duration_ms: 100,
        success: false,
        error_kind: "unknown_error",
      }),
    });

    expect(response.status).toBe(400);
    const body = (await response.json()) as { error: string };
    expect(body.error).toBe("Invalid telemetry event");
  });
});
