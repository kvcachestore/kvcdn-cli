import type { FastifyInstance, FastifyReply, FastifyRequest } from "fastify";
import { createPrivateKey, createPublicKey, generateKeyPairSync, randomBytes } from "node:crypto";
import { exportJWK, SignJWT } from "jose";
import type { JWK } from "jose";

export interface OidcKeyPair {
  privateKey: ReturnType<typeof createPrivateKey>;
  publicJwk: JWK;
  kid: string;
}

let keyPairPromise: Promise<OidcKeyPair> | undefined;

// In-memory pending device authorizations. In a real provider these live in
// a database and are shared across processes; for the local mock they only
// need to last long enough for the e2e test to poll.
interface PendingDeviceAuth {
  deviceCode: string;
  userCode: string;
  subject: string;
  createdAt: number;
  expiresIn: number;
  interval: number;
  approved: boolean;
}
const pendingDeviceAuths = new Map<string, PendingDeviceAuth>();
const pendingAuthCodes = new Map<string, PendingAuthCode>();
const DEVICE_CODE_GRANT = "urn:ietf:params:oauth:grant-type:device_code";
const AUTH_CODE_TTL_MS = 5 * 60 * 1000;

export async function getOidcKeyPair(): Promise<OidcKeyPair> {
  if (!keyPairPromise) {
    keyPairPromise = generateKeyPair();
  }
  return keyPairPromise;
}

function generateUserCode(): string {
  return randomBytes(4).toString("hex").toUpperCase();
}

function generateDeviceCode(): string {
  return "mock-device-code-" + randomBytes(16).toString("hex");
}

async function generateKeyPair(): Promise<OidcKeyPair> {
  const { privateKey: pemPrivate, publicKey: pemPublic } = generateKeyPairSync("rsa", {
    modulusLength: 2048,
    publicKeyEncoding: { type: "spki", format: "pem" },
    privateKeyEncoding: { type: "pkcs8", format: "pem" },
  });

  const privateKey = createPrivateKey(pemPrivate);
  const baseJwk = await exportJWK(createPublicKey(pemPublic));
  const publicJwk: JWK = {
    ...baseJwk,
    kty: "RSA",
    kid: "kvcdn-oidc-key-1",
    use: "sig",
    alg: "RS256",
  };

  return { privateKey, publicJwk, kid: publicJwk.kid! };
}

export async function signAccessToken(issuer: string, clientId: string, subject: string): Promise<string> {
  const { privateKey, kid } = await getOidcKeyPair();
  return new SignJWT({ sub: subject, aud: clientId, iss: issuer })
    .setProtectedHeader({ alg: "RS256", kid, typ: "JWT" })
    .setIssuedAt()
    .setExpirationTime("1h")
    .sign(privateKey);
}

interface TokenBody {
  grant_type?: string;
  code?: string;
  redirect_uri?: string;
  code_verifier?: string;
  client_id?: string;
  device_code?: string;
}

interface DeviceAuthBody {
  client_id?: string;
  scope?: string;
}

interface PendingAuthCode {
  code: string;
  subject: string;
  createdAt: number;
  redirectUri: string;
}

export async function oidcRoutes(fastify: FastifyInstance): Promise<void> {
  fastify.addContentTypeParser(
    "application/x-www-form-urlencoded",
    { parseAs: "string" },
    (_request, payload, done) => {
      const params = new URLSearchParams(payload as string);
      const result: Record<string, string> = {};
      for (const [key, value] of params) {
        result[key] = value;
      }
      done(null, result);
    }
  );

  fastify.get("/.well-known/openid-configuration", async (request, reply) => {
    const issuerUrl = buildIssuerUrl(request);
    return reply.send({
      issuer: issuerUrl,
      authorization_endpoint: `${issuerUrl}/auth`,
      token_endpoint: `${issuerUrl}/token`,
      device_authorization_endpoint: `${issuerUrl}/device/auth`,
      jwks_uri: `${issuerUrl}/keys`,
      response_types_supported: ["code"],
      grant_types_supported: [
        "authorization_code",
        "refresh_token",
        DEVICE_CODE_GRANT,
      ],
      subject_types_supported: ["public"],
      id_token_signing_alg_values_supported: ["RS256"],
      token_endpoint_auth_methods_supported: ["none"],
    });
  });

  fastify.get("/keys", async (_request, reply) => {
    const { publicJwk } = await getOidcKeyPair();
    return reply.send({ keys: [publicJwk] });
  });

  fastify.get("/auth", async (request, reply) => {
    const { state, redirect_uri: redirectUri } = request.query as Record<string, string>;
    if (!redirectUri) {
      return reply.status(400).send({ error: "missing redirect_uri" });
    }
    if (!isAllowedRedirectUri(redirectUri)) {
      return reply.status(400).send({ error: "invalid redirect_uri" });
    }

    pruneAuthCodes();
    const code = "mock-auth-code-" + randomBytes(16).toString("hex");
    pendingAuthCodes.set(code, {
      code,
      subject: "kvcdn-user",
      createdAt: Date.now(),
      redirectUri,
    });
    const separator = redirectUri.includes("?") ? "&" : "?";
    return reply.redirect(
      `${redirectUri}${separator}code=${encodeURIComponent(code)}&state=${encodeURIComponent(state ?? "")}`
    );
  });

  fastify.post<{
    Body: DeviceAuthBody;
  }>("/device/auth", async (request, reply) => {
    pruneDeviceAuths();
    const userCode = generateUserCode();
    const deviceCode = generateDeviceCode();
    const issuerUrl = buildIssuerUrl(request);
    const verificationUri = `${issuerUrl}/device/activate`;
    const verificationUriComplete = `${verificationUri}?user_code=${encodeURIComponent(userCode)}`;
    pendingDeviceAuths.set(deviceCode, {
      deviceCode,
      userCode,
      subject: "kvcdn-user",
      createdAt: Date.now(),
      expiresIn: 300,
      interval: 1,
      approved: false,
    });
    return reply.status(200).send({
      device_code: deviceCode,
      user_code: userCode,
      verification_uri: verificationUri,
      verification_uri_complete: verificationUriComplete,
      expires_in: 300,
      interval: 1,
    });
  });

  fastify.get("/device/activate", async (request, reply) => {
    const adminSecret = process.env.KVCDN_ADMIN_SECRET;
    if (adminSecret) {
      const provided =
        (request.query as Record<string, string>).secret ??
        (request.headers["x-activation-secret"] as string | undefined);
      if (provided !== adminSecret) {
        return reply.status(401).send({ error: "Unauthorized" });
      }
    }

    const { user_code: userCode } = request.query as Record<string, string>;
    for (const auth of pendingDeviceAuths.values()) {
      if (auth.userCode === userCode) {
        auth.approved = true;
      }
    }
    return reply.type("text/html").send(`<html><body><h1>Device login approved</h1><p>You may close this window.</p></body></html>`);
  });

  fastify.post<{
    Body: TokenBody;
  }>("/token", async (request, reply) => {
    const { grant_type: grantType, code, device_code: deviceCode } = request.body;
    const issuerUrl = buildIssuerUrl(request);
    const clientId = process.env.KVCDN_CLIENT_ID ?? "kvcdn-cli";

    if (grantType === DEVICE_CODE_GRANT) {
      if (!deviceCode) {
        return reply.status(400).send({ error: "invalid_request", error_description: "missing device_code" });
      }
      const pending = pendingDeviceAuths.get(deviceCode);
      if (!pending) {
        return reply.status(400).send({ error: "expired_token" });
      }
      if (Date.now() > pending.createdAt + pending.expiresIn * 1000) {
        pendingDeviceAuths.delete(deviceCode);
        return reply.status(400).send({ error: "expired_token" });
      }
      if (!pending.approved) {
        return reply.status(400).send({ error: "authorization_pending" });
      }
      pendingDeviceAuths.delete(deviceCode);
      const accessToken = await signAccessToken(issuerUrl, clientId, pending.subject);
      return reply.send({
        access_token: accessToken,
        token_type: "Bearer",
        expires_in: 3600,
        refresh_token: "mock-refresh-token",
      });
    }

    if (grantType === "authorization_code") {
      if (!code || !code.startsWith("mock-auth-code-")) {
        return reply.status(400).send({ error: "invalid_grant" });
      }

      pruneAuthCodes();
      let subject: string;
      const stored = pendingAuthCodes.get(code);
      if (stored) {
        subject = stored.subject;
        pendingAuthCodes.delete(code);
      } else {
        // Backward compatibility: tests may mint deterministic codes of the
        // form mock-auth-code-<subject> without calling /auth first.
        subject = code.slice("mock-auth-code-".length) || "kvcdn-user";
      }
      const accessToken = await signAccessToken(issuerUrl, clientId, subject);

      return reply.send({
        access_token: accessToken,
        token_type: "Bearer",
        expires_in: 3600,
        refresh_token: "mock-refresh-token",
      });
    }

    return reply.status(400).send({ error: "unsupported_grant_type" });
  });
}

function buildIssuerUrl(request: { protocol: string; hostname: string; port?: number | string | null }): string {
  const configured = process.env.KVCDN_ISSUER_URL;
  if (configured) return configured.replace(/\/$/, "");
  const port = request.port;
  const portSuffix = port && String(port) !== "80" && String(port) !== "443" ? `:${port}` : "";
  return `${request.protocol}://${request.hostname}${portSuffix}/oidc`;
}

// Only allow local callback URLs for the mock authorization-code flow. This
// prevents the mock endpoint from being used as an open redirector.
function isAllowedRedirectUri(uri: string): boolean {
  const allowed = process.env.KVCDN_REDIRECT_URI;
  if (allowed) {
    return uri === allowed;
  }
  try {
    const url = new URL(uri);
    return (
      (url.protocol === "http:" || url.protocol === "https:") &&
      (url.hostname === "localhost" || url.hostname === "127.0.0.1")
    );
  } catch {
    return false;
  }
}

function pruneAuthCodes(): void {
  const cutoff = Date.now() - AUTH_CODE_TTL_MS;
  for (const [code, pending] of pendingAuthCodes) {
    if (pending.createdAt < cutoff) {
      pendingAuthCodes.delete(code);
    }
  }
}

function pruneDeviceAuths(): void {
  const now = Date.now();
  for (const [code, pending] of pendingDeviceAuths) {
    if (now > pending.createdAt + pending.expiresIn * 1000) {
      pendingDeviceAuths.delete(code);
    }
  }
}
