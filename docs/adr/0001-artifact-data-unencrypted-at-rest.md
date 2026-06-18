# ADR 0001: Artifact data remains unencrypted at rest

## Status

Accepted

## Context

KVCDN stores large pre-computed KV-cache files in an object store so downstream inference services can skip the expensive prefill phase. The product's value depends on fast access to those caches. Customers are concerned about isolation, multi-tenancy, portability, and regulatory compliance.

We evaluated encrypting artifact data at rest (client-side before upload, server-side envelope encryption, or per-object S3 encryption) and concluded that the costs outweigh the benefits for this workload.

## Decision

Artifact **data blobs remain unencrypted** in the object store. Security and tenant isolation are provided by:

1. **Transport encryption**: all API calls and presigned URLs use HTTPS/TLS.
2. **Time-limited presigned URLs**: upload and download URLs expire after 5 minutes and are minted only for the owning Customer or a public artifact.
3. **Tenant-isolated key prefixes**: each Customer's objects live under `artifacts/customers/{customer_id}/...`, where `customer_id` is derived deterministically from the authenticated identity. Listing operations are scoped to the Customer's prefix.
4. **Provider server-side encryption**: we rely on the object store provider's default encryption at rest (e.g., SSE-S3 / SSE-R2) rather than custom encryption.
5. **Encrypted credentials**: OIDC tokens and API keys are encrypted at rest on the CLI side.

Metadata sidecars (`.meta.json`) also remain unencrypted. They contain only artifact attributes and a derived `customer_id`; they do not contain user PII.

## Consequences

- **Performance**: uploads, downloads, and future partial-load optimizations operate on raw bytes with no encryption overhead.
- **Simplicity**: the CLI and backend do not manage data-encryption keys, key rotation, or per-object envelopes.
- **Compliance posture**: GDPR/CCPA "right to be forgotten" is satisfied by deleting the object and metadata. Data residency is handled by provider region selection and documented retention policy.
- **Risk**: a compromised object-store credential with unrestricted bucket access could read raw artifact bytes. This risk is mitigated by scoped credentials, presigned URLs, and per-Customer prefixes.

## Alternatives considered

- **Client-side encryption**: would protect data from the backend, but would force every CLI and inference consumer to perform decryption and key management, erasing the performance advantage.
- **Server-side envelope encryption**: would let the backend control keys while storing ciphertext, but would add CPU cost on every read/write and complicate future streaming/partial-load features.
- **Per-object S3 SSE-KMS**: adds key management cost and audit complexity without addressing tenant isolation better than prefix scoping.

## Future work

If a future customer segment requires encrypted artifacts, introduce a `CryptoAdapter` seam behind which a new adapter can encrypt metadata and/or data without changing the artifact route or CLI transfer logic. That change would require its own ADR covering key management, rotation, and performance impact.
