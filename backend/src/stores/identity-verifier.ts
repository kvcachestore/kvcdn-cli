import {
  deriveApiKey,
  getApiKeySeed,
  getConfiguredOrgs,
  looksLikeApiKey,
  type TokenPayload,
  verifyToken,
} from "../auth.js";

export interface VerifiedIdentity {
  sub: string;
  source: "oidc" | "api-key";
  tokenPayload?: TokenPayload;
}

export interface IdentityVerifier {
  verify(bearer: string): Promise<VerifiedIdentity>;
}

export class OidcOrApiKeyIdentityVerifier implements IdentityVerifier {
  async verify(bearer: string): Promise<VerifiedIdentity> {
    if (looksLikeApiKey(bearer)) {
      return this.verifyApiKey(bearer);
    }

    const payload = await verifyToken(bearer);
    const sub = payload.sub;
    if (!sub) {
      throw new Error("token missing sub claim");
    }
    return { sub, source: "oidc", tokenPayload: payload };
  }

  private verifyApiKey(bearer: string): VerifiedIdentity {
    const seed = getApiKeySeed();
    const orgs = getConfiguredOrgs();
    if (orgs.length === 0) {
      throw new Error("KVCDN_API_KEY_ORGS is not configured");
    }
    for (const org of orgs) {
      if (bearer === deriveApiKey(seed, org)) {
        return { sub: org, source: "api-key" };
      }
    }
    throw new Error("invalid API key");
  }
}

