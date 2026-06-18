# Multi-bucket backend follow-up work

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development or manual execution. Tasks below are sequential because each may affect shared files and documentation.

**Goal:** Make the multi-bucket backend production-ready by validating real S3 behavior, documenting migration, wiring deploy env vars, adding observability, and hardening bucket-list changes.

---

## Task 1: Integration test for real multi-bucket S3 behavior

**Files:**
- Create: `backend/src/__tests__/s3-multi-bucket.test.ts`
- Modify: `backend/src/stores/s3-bucket-store.ts` if needed for testability

**What to test:**
1. Spin up an actual MinIO/S3-compatible store using existing local stack credentials if `KVCDN_S3_ENDPOINT` is set, or skip the suite otherwise.
2. With `KVCDN_S3_BUCKETS=bucket-a,bucket-b` and `KVCDN_CONTROL_BUCKET=kvcdn-control`, create buckets via the S3 client.
3. Instantiate `S3BucketStore`, `HashedBucketResolver`, `TenantS3ArtifactStore`, and `TenantS3MetadataStore`.
4. Write an artifact metadata + object under `artifacts/customers/customer-1/orgs/acme/projects/acme/...`.
5. Assert the object and `.meta.json` land in exactly one of `bucket-a` or `bucket-b`, and that the control bucket contains the assignment record.
6. Read back metadata and object from the same bucket via the tenant-aware stores.

**Implementation notes:**
- Reuse the S3 client factory from `s3-clients.ts`.
- Create a small helper `ensureBucket(client, bucket)` using `HeadBucket`/`CreateBucket`.
- Clean up written objects in `afterAll` to keep tests idempotent.
- Skip the test suite gracefully if env vars are missing, so CI that lacks MinIO still passes.

---

## Task 2: Multi-bucket migration runbook

**Files:**
- Create: `docs/ops/multi-bucket-runbook.md`
- Modify: `README.md` to link to runbook

**Sections:**
1. Pre-requisites: existing single-bucket setup with `KVCDN_S3_BUCKET`.
2. Create the control bucket in the same S3 account/endpoint.
3. Create additional artifact buckets.
4. Update backend secrets: set `KVCDN_S3_BUCKETS=old-bucket,new-bucket-1,new-bucket-2` and `KVCDN_CONTROL_BUCKET=<control>`.
5. Behavior on deploy: existing tenants remain on `old-bucket` because assignments are persisted; new tenants hash to any configured bucket.
6. Caveat: removing buckets from the list after tenants are assigned will break reads for those tenants. Only grow the list.
7. Rollback: unset `KVCDN_S3_BUCKETS` to fall back to `KVCDN_S3_BUCKET` (only safe if no multi-bucket assignments were made, or if all assignments point back to the single bucket).

---

## Task 3: Pass new env vars through Fly deploy pipeline

**Files:**
- Inspect: `ci/dagger/` Dagger module files
- Modify: `ci/dagger/main.go` comment if needed

**What to verify:**
- The `DeployBackend` pipeline uses `flyctl deploy`, which reads runtime env vars and secrets already set on the Fly app via `fly secrets set`.
- No code change is required in the Dagger module because Fly secrets are managed out-of-band.
- Confirm `backend/fly.toml` does not hard-code a single-bucket env block.
- Add or update a comment in `ci/dagger/main.go` documenting that Fly secrets are managed out-of-band.

---

## Task 4: Observability for tenant bucket resolution

**Files:**
- Modify: `backend/src/stores/hashed-bucket-resolver.ts`
- Modify: `backend/src/stores/s3-bucket-store.ts`
- Modify: `backend/src/index.ts` to inject a logger if needed

**What to add:**
- Log `resolved bucket` at `debug` level when a tenant is resolved, including `customerId` and `bucket`.
- Log `bucket assignment persisted` at `debug` when a new assignment is saved.
- In `S3BucketStore.getAssignment`, log control-bucket read failures at `warn` (not error, because we fall back to deterministic hashing).
- In `S3BucketStore.saveAssignment`, log write failures at `error`.

Use the Fastify logger on `fastify.log` if possible; otherwise `console` is acceptable for now. The resolver currently has no logger reference, so pass `logger` as an optional constructor argument or use `process.env.NODE_ENV` guarded `console` logs.

---

## Task 5: Harden against shrinking bucket list

**Files:**
- Modify: `backend/src/stores/hashed-bucket-resolver.ts`
- Modify: `docs/ops/multi-bucket-runbook.md`
- Modify: `backend/src/__tests__/bucket-resolver.test.ts`

**What to add:**
- When `resolve()` finds an existing assignment whose `bucket` is not in the current configured bucket list, throw a clear error: `Bucket "X" assigned to customer "Y" is not in KVCDN_S3_BUCKETS`.
- Add a test that creates a persisted assignment to an old bucket, then instantiates a resolver with a bucket list that excludes it, and asserts the resolver throws.
- Document the constraint and remediation (re-add the bucket or migrate data and assignment records).

---

## Task 6: Final verification

Run:
```bash
cargo test && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cd backend && NODE_OPTIONS='--experimental-vm-modules' npm test
```

---

## Task 7: Push main

```bash
git push origin main
```
