import type { FastifyInstance, FastifyReply, FastifyRequest } from "fastify";
import { z } from "zod";
import * as http from "node:http";
import * as https from "node:https";
import { URL } from "node:url";

const TelemetryEvent = z.object({
  command: z.string().min(1).max(64),
  version: z.string().min(1).max(32),
  duration_ms: z.number().int().nonnegative(),
  success: z.boolean(),
  error_kind: z
    .enum(["auth_error", "network_error", "config_error", "usage_error", "runtime_error"])
    .optional(),
});

type TelemetryEvent = z.infer<typeof TelemetryEvent>;

function getEnv(name: string): string | undefined {
  const value = process.env[name];
  return value && value.length > 0 ? value : undefined;
}

function postJson(url: string, secret: string, body: unknown): Promise<{ status: number; ok: boolean }> {
  const payload = JSON.stringify(body);
  const parsed = new URL(url);
  const options: http.RequestOptions = {
    method: "POST",
    hostname: parsed.hostname,
    port: parsed.port,
    path: parsed.pathname + parsed.search,
    headers: {
      "content-type": "application/json",
      "content-length": Buffer.byteLength(payload),
      authorization: `Bearer ${secret}`,
    },
    timeout: 5000,
  };

  const client = parsed.protocol === "https:" ? https : http;

  return new Promise((resolve, reject) => {
    const req = client.request(options, (res) => {
      res.resume();
      resolve({ status: res.statusCode ?? 0, ok: res.statusCode !== undefined && res.statusCode >= 200 && res.statusCode < 300 });
    });

    req.on("error", reject);
    req.on("timeout", () => {
      req.destroy(new Error("upstream telemetry request timed out"));
    });
    req.write(payload);
    req.end();
  });
}

export async function telemetryRoutes(fastify: FastifyInstance): Promise<void> {
  fastify.post(
    "/v1/telemetry",
    async (request: FastifyRequest, reply: FastifyReply) => {
      let body: TelemetryEvent;
      try {
        body = TelemetryEvent.parse(request.body);
      } catch (err) {
        fastify.log.warn({ err }, "telemetry event validation failed");
        return reply.status(400).send({ error: "Invalid telemetry event" });
      }

      const telemetryUrl = getEnv("KVCDN_TELEMETRY_URL");
      const telemetrySecret = getEnv("KVCDN_TELEMETRY_SECRET");

      if (!telemetryUrl || !telemetrySecret) {
        return reply.status(204).send();
      }

      try {
        const upstream = await postJson(`${telemetryUrl}/events`, telemetrySecret, body);

        if (!upstream.ok) {
          fastify.log.warn(
            { status: upstream.status, url: telemetryUrl },
            "telemetry upstream returned non-success status",
          );
        }

        return reply.status(204).send();
      } catch (err) {
        fastify.log.warn({ err, url: telemetryUrl }, "failed to forward telemetry event");
        return reply.status(204).send();
      }
    },
  );
}
