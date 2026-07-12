import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { buildApp } from "./index.js";

describe("telemetry service", () => {
  it("GET /health returns ok", async () => {
    const app = buildApp("test-secret");
    const response = await app.inject({
      method: "GET",
      url: "/health",
    });
    assert.equal(response.statusCode, 200);
    assert.deepEqual(JSON.parse(response.body), { status: "ok" });
  });

  it("POST /events rejects missing authorization", async () => {
    const app = buildApp("test-secret");
    const response = await app.inject({
      method: "POST",
      url: "/events",
      payload: { command: "verify", version: "0.2.0", duration_ms: 100, success: true },
    });
    assert.equal(response.statusCode, 401);
    assert.deepEqual(JSON.parse(response.body), { error: "Unauthorized" });
  });

  it("POST /events rejects wrong bearer token", async () => {
    const app = buildApp("test-secret");
    const response = await app.inject({
      method: "POST",
      url: "/events",
      headers: { authorization: "Bearer wrong-secret" },
      payload: { command: "verify", version: "0.2.0", duration_ms: 100, success: true },
    });
    assert.equal(response.statusCode, 401);
  });

  it("POST /events accepts valid bearer token and returns 204", async () => {
    const app = buildApp("test-secret");
    const response = await app.inject({
      method: "POST",
      url: "/events",
      headers: { authorization: "Bearer test-secret" },
      payload: { command: "verify", version: "0.2.0", duration_ms: 100, success: true },
    });
    assert.equal(response.statusCode, 204);
  });
});
