# KVCDN CLI Auth & Hosted Upload Design

## Status

Approved design. Ready for implementation planning.

## Goals

- Provide a releasable, local-first CLI for KV-cache generation, verification, quantization, and benchmarking.
- Offer a hosted tier where authenticated users can upload KV artifacts to their own private cloud endpoint.
- Authenticate users via OIDC (Pocket-ID) using authorization-code + PKCE.
- Keep the SaaS onboarding path zero-config while remaining configurable for self-hosters.

## Non-goals

- Client-side encryption of artifacts in this iteration (TLS + server-side storage only).
- Billing or quota enforcement in the CLI.
- A backend service implementation; this spec covers the CLI contract only.

## Decisions from brainstorming

1. **Login flow:** browser auto-open by default; `--no-browser` manual-code fallback.
2. **Token storage:** encrypted file in `~/.config/kvcdn/credentials.enc`.
3. **Encryption key:** OS keyring holds the AES key when available; otherwise a separate `~/.config/kvcdn/.key` file with `0600` permissions and a user-visible warning.
4. **Backend/OIDC config:** baked-in SaaS defaults overridable via CLI flags, environment variables, and `~/.config/kvstore/config.toml`.
5. **Upload transport:** CLI initiates upload via backend API, receives a presigned URL, then PUTs the artifact directly.
6. **Organization:** default per-user namespace plus optional `--project` scoping.

## CLI surface

### New subcommands

| Command | Purpose |
|---------|---------|
| `kvcdn login` | Start OIDC flow and persist tokens. |
| `kvcdn login --no-browser` | Print authorization URL and wait for callback/code. |
| `kvcdn logout` | Delete stored tokens and key material. |
| `kvcdn upload <artifact> --name <name> [--project <p>]` | Upload an artifact to the hosted backend. |
| `kvcdn list [--project <p>]` | List uploaded artifacts (deferrable). |
| `kvcdn download <name> [--project <p>]` | Download an artifact (deferrable). |

### Configuration precedence

CLI flags → environment variables → `~/.config/kvcdn/config.toml` → built-in defaults.

Relevant settings:

- `api_url` / `KVCDN_API_URL`
- `issuer_url` / `KVCDN_ISSUER_URL`
- `client_id` / `KVCDN_CLIENT_ID`
- `default_project` / `KVCDN_DEFAULT_PROJECT`

Example `config.toml`:

```toml
api_url = "https://api.kvcdn.example.com"
issuer_url = "https://id.kvcdn.example.com"
client_id = "kvcdn-cli"
default_project = "default"
```

## Auth flow

1. Discover OIDC endpoints via `$ISSUER/.well-known/openid-configuration`.
2. Generate PKCE code verifier, code challenge, and random `state` nonce.
3. Bind a local TCP socket on `127.0.0.1:0`; use the assigned port as the callback URL.
4. Build authorization URL:
   - `response_type=code`
   - `scope=openid profile email`
   - `code_challenge` / `code_challenge_method=S256`
   - `state`
   - `redirect_uri=http://127.0.0.1:<port>/callback`
5. Auto-open the URL in the default browser, or print it with `--no-browser`.
6. Local HTTP handler receives `GET /callback?code=...&state=...`:
   - Validate `state` matches.
   - Extract authorization code.
7. POST to token endpoint:
   - `grant_type=authorization_code`
   - `code`, `redirect_uri`, `code_verifier`, `client_id`
8. Store received `access_token`, `refresh_token`, and `expires_at`.

## Token storage

- Encrypted file: `~/.config/kvcdn/credentials.enc`
- Metadata sidecar: `~/.config/kvcdn/credentials.json` (key ID, expiry, non-secret metadata)
- Encryption: AES-256-GCM or ChaCha20-Poly1305 via a vetted crate (e.g., `ring` or `aes-gcm`).
- Key handling:
  - Preferred: store the encryption key in the OS keyring/keychain (`keyring` crate).
  - Fallback: generate a random key and write it to `~/.config/kvcdn/.key` with `0600` permissions, printing a warning that tokens are only obfuscated.

## Upload flow

1. Read encrypted token store; if access token is near expiry, refresh it first.
2. Compute SHA-256 and file size of the artifact.
3. `POST /v1/projects/{project}/artifacts` with JSON body:
   ```json
   {
     "name": "docs-v1",
     "size_bytes": 12345678,
     "sha256": "abc...",
     "dtype": "F16",
     "num_tokens": 128,
     "num_layers": 28,
     "quantized": false
   }
   ```
4. Backend returns:
   ```json
   {
     "artifact_id": "uuid",
     "upload_url": "https://s3.../presigned",
     "expires_at": "..."
   }
   ```
5. CLI PUTs the artifact to `upload_url` with streaming/chunked body and progress reporting.
6. On success, CLI prints `Uploaded <name> to <project>/<artifact_id>`.

## Error handling

- **Network failures:** retry idempotent calls (discovery, token refresh, upload initiation) with exponential backoff.
- **401 / invalid token:** attempt one refresh; if refresh fails, instruct user to run `kvcdn login`.
- **Browser-open failure:** automatically fall back to `--no-browser` behavior.
- **Large uploads:** stream file in chunks; set long read/write timeouts.
- **State mismatch / CSRF:** abort login and report error.

## Testing strategy

- Unit tests for PKCE verifier/challenge generation and state validation.
- Mock OIDC discovery + token endpoints and mock upload-init endpoint.
- End-to-end test of `--no-browser` login by driving the local callback URL.
- Upload test with a temporary artifact and mocked presigned URL target.

## Deferred work

- `list` and `download` subcommands.
- Client-side encryption of artifacts.
- Resumable multipart uploads.
- Billing/quota display in the CLI.
