import Fastify from "fastify";

export function buildApp(secret: string) {
  const fastify = Fastify({ logger: false });

  fastify.get("/health", async () => ({ status: "ok" }));

  fastify.post("/events", async (request, reply) => {
    const auth = request.headers.authorization;
    if (!auth || auth !== `Bearer ${secret}`) {
      return reply.status(401).send({ error: "Unauthorized" });
    }

    const line = JSON.stringify(request.body);
    console.log(line);
    return reply.status(204).send();
  });

  return fastify;
}

function main(): void {
  const secret = process.env.TELEMETRY_SECRET;
  if (!secret || secret.length === 0) {
    console.error("TELEMETRY_SECRET is required");
    process.exit(1);
  }

  const port = Number(process.env.PORT ?? "3000");
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    console.error(`PORT must be an integer between 1 and 65535, got: ${process.env.PORT}`);
    process.exit(1);
  }

  const fastify = buildApp(secret);
  fastify.listen({ port, host: "0.0.0.0" }, (err) => {
    if (err) {
      fastify.log.error(err);
      process.exit(1);
    }
  });
}

if (import.meta.url === new URL(process.argv[1] ?? "", `file://${process.cwd()}/`).href) {
  main();
}
