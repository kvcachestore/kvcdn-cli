# Multi-bucket migration runbook

This guide covers moving a kvcdn backend from a single S3-compatible bucket to multiple buckets for better isolation and blast-radius reduction.

## Overview

The backend supports spreading tenant artifact data across several S3-compatible buckets using deterministic hashing. Each tenant is assigned to one bucket, and the mapping is persisted in a separate **control bucket** (`KVCDN_CONTROL_BUCKET`).

Key properties:

- **No data migration required.** Existing tenants keep their existing assignments; new tenants are distributed across the configured buckets.
- **CLI is unchanged.** End users still call the same API and receive presigned URLs computed by the backend.
- **Bucket list should only grow.** Removing a bucket after tenants have been assigned to it will break reads for those tenants.

## Before you start

You need:

- An existing kvcdn backend deployed with a single bucket configured as `KVCDN_S3_BUCKET`.
- Administrative access to the S3-compatible store (R2, AWS S3, MinIO, etc.) to create new buckets.
- A Fly deploy token or equivalent access to redeploy the backend.

## Step 1: create buckets

Create the control bucket plus any additional artifact buckets in the same S3 account/endpoint. For example, using Cloudflare R2:

```bash
# Control bucket — small JSON records only
# Additional artifact buckets — must use the same endpoint, access key, and secret key as the original bucket
```

The exact commands depend on your provider. With MinIO or `mc`, you can run:

```bash
mc mb mystore/kvcdn-control
mc mb mystore/kvcdn-artifacts-1
mc mb mystore/kvcdn-artifacts-2
```

## Step 2: update backend secrets

Set `KVCDN_S3_BUCKETS` to a comma-separated list that includes the existing bucket plus any new ones. Set `KVCDN_CONTROL_BUCKET` to the control bucket you created.

```bash
fly secrets set --app kvcachestore \
  KVCDN_S3_BUCKETS="kvcdn-artifacts-0,kvcdn-artifacts-1,kvcdn-artifacts-2" \
  KVCDN_CONTROL_BUCKET="kvcdn-control"
```

Keep `KVCDN_S3_ENDPOINT`, `KVCDN_S3_ACCESS_KEY`, and `KVCDN_S3_SECRET_KEY` unchanged so all listed buckets are reachable with the same credentials.

## Step 3: deploy

Deploy the backend. The deployment pipeline must pass the new environment variables through to the Fly app (see the `ci/dagger` deploy logic). If you set secrets manually with `fly secrets set`, a normal `fly deploy` is sufficient.

```bash
dagger call -m ci/dagger deploy-backend --src=. --fly-api-token=env:FLY_API_TOKEN
```

## Step 4: verify

After deploy, create a new artifact:

```bash
kvcdn login
kvcdn upload model.kv --name model --project acme
kvcdn list --project acme
```

Then check the control bucket. You should see one assignment record per customer under `assignments/<customer_id>.json`, and the artifact data plus `.meta.json` sidecar should appear in exactly one of the configured artifact buckets.

## How tenant placement works

1. On the first write for a tenant, the backend hashes the `customer_id` and picks one of the configured buckets.
2. It writes `{ customer_id, bucket }` to the control bucket.
3. On subsequent reads and writes, the backend reads the assignment from the control bucket and uses that bucket.
4. If the control bucket read fails, the backend falls back to the deterministic hash. This makes the backend resilient to transient control-bucket outages but means any bucket-list change must be done carefully.

## Important constraints

### Only grow the bucket list

Removing a bucket from `KVCDN_S3_BUCKETS` after tenants are assigned to it will cause errors like:

```
Bucket "kvcdn-artifacts-2" assigned to customer "..." is not in KVCDN_S3_BUCKETS
```

To shrink the bucket list, you must either:

- re-add the missing bucket to `KVCDN_S3_BUCKETS`, or
- migrate the tenant's objects and update the assignment record in the control bucket.

### Single-bucket fallback

If `KVCDN_S3_BUCKETS` is unset, the backend falls back to `KVCDN_S3_BUCKET`. This preserves the original single-bucket behavior and is useful for rollbacks, but only if no multi-bucket assignments have been written.

### Control bucket access

The backend uses the same S3 credentials for the control bucket as for artifact buckets. The control bucket must allow `s3:GetObject` and `s3:PutObject` on keys under `assignments/`.

## Rollback

To return to the original single-bucket deployment:

1. Ensure all existing assignment records in the control bucket point to the single target bucket, or accept that reads will fail for tenants assigned to other buckets.
2. Unset `KVCDN_S3_BUCKETS` and `KVCDN_CONTROL_BUCKET`.
3. Redeploy.

## Troubleshooting

### New uploads fail with "Control bucket is not configured"

Set `KVCDN_CONTROL_BUCKET` and confirm the bucket exists.

### Presigned URLs point to the wrong bucket

Check `assignments/<customer_id>.json` in the control bucket. The customer ID is deterministic and can be found by inspecting an existing artifact's metadata sidecar (`customer_id` field) or by enabling debug logging on the backend.

### Existing tenants still land in the original bucket

This is expected. Existing assignments are persisted, so existing tenants stay where they were. New tenants will be distributed across all configured buckets.
