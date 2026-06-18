import { hkdfSync } from "node:crypto";
import { jwtVerify, createRemoteJWKSet } from "jose";
import { getOidcKeyPair } from "./routes/oidc.js";

export interface TokenPayload extends Record<string, unknown> {
  sub?: string;
  iss?: string;
  aud?: string | string[];
  exp?: number;
}

let jwksCache: ReturnType<typeof createRemoteJWKSet> | undefined;
let jwksUrlCache: string | undefined;

export function isMockOidc(): boolean {
  return (
    !process.env.KVCDN_ISSUER_URL ||
    process.env.KVCDN_MOCK_OIDC === "true"
  );
}

async function resolveJwksUrl(discoveryBase: string): Promise<string> {
  if (jwksUrlCache) return jwksUrlCache;

  const discoveryUrl = new URL(".well-known/openid-configuration", discoveryBase);
  const response = await fetch(discoveryUrl.href);
  if (!response.ok) {
    throw new Error(`OIDC discovery failed: ${response.status} ${await response.text()}`);
  }

  const discovery = (await response.json()) as { jwks_uri?: string };
  const jwksUri = discovery.jwks_uri;
  if (!jwksUri || typeof jwksUri !== "string") {
    throw new Error("OIDC discovery document missing jwks_uri");
  }

  jwksUrlCache = jwksUri;
  return jwksUri;
}

async function getJwks(discoveryBase: string): Promise<ReturnType<typeof createRemoteJWKSet>> {
  if (jwksCache) return jwksCache;

  const jwksUrl = await resolveJwksUrl(discoveryBase);
  jwksCache = createRemoteJWKSet(new URL(jwksUrl));
  return jwksCache;
}

async function verifyMockToken(token: string): Promise<TokenPayload> {
  const { privateKey } = await getOidcKeyPair();
  const { payload } = await jwtVerify(token, privateKey, {
    issuer: process.env.KVCDN_ISSUER_URL,
    audience: process.env.KVCDN_CLIENT_ID ?? "kvcdn-cli",
  });
  return payload as TokenPayload;
}

export async function verifyToken(token: string): Promise<TokenPayload> {
  if (isMockOidc()) {
    return verifyMockToken(token);
  }

  const rawIssuerUrl = process.env.KVCDN_ISSUER_URL;
  if (!rawIssuerUrl) {
    throw new Error("KVCDN_ISSUER_URL is not configured");
  }

  const issuerUrl = rawIssuerUrl;
  const discoveryBase = issuerUrl.endsWith("/") ? issuerUrl : `${issuerUrl}/`;
  const audience = process.env.KVCDN_CLIENT_ID ?? "kvcdn-cli";

  const jwks = await getJwks(discoveryBase);
  const { payload } = await jwtVerify(token, jwks, {
    issuer: issuerUrl,
    audience,
  });

  return payload as TokenPayload;
}

export function clearJwksCache(): void {
  jwksCache = undefined;
  jwksUrlCache = undefined;
}

export function looksLikeApiKey(token: string): boolean {
  // API keys are kv_ followed by a hex derived key. This is stricter than a
  // plain prefix so that JWTs (which are extremely unlikely to match) are not
  // misrouted to API-key verification.
  return /^kv_[0-9a-fA-F]{32,}$/.test(token);
}

export function getApiKeySeed(): string {
  const seed = process.env.KVCDN_API_KEY_SEED;
  if (!seed) {
    throw new Error("KVCDN_API_KEY_SEED is not configured");
  }
  return seed;
}

export function getConfiguredOrgs(): string[] {
  const orgs = process.env.KVCDN_API_KEY_ORGS;
  if (!orgs) return [];
  return orgs
    .split(",")
    .map((o) => o.trim().toLowerCase())
    .filter(Boolean);
}

// HKDF-SHA256 is the right primitive here: the secret is the seed; the org
// slug is the only thing that varies between customers. Argon2id would add
// latency without extra security because the inputs are high-entropy.
export function deriveApiKey(seed: string, orgSlug: string): string {
  const normalized = orgSlug.trim().toLowerCase();
  if (!normalized) {
    throw new Error("org slug must be non-empty");
  }
  const info = `kvcdn/api-key/v1/${normalized}`;
  const derived = hkdfSync("sha256", seed, Buffer.alloc(0), info, 32);
  return `kv_${Buffer.from(derived).toString("hex")}`;
}
