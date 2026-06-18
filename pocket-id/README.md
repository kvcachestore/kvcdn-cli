# Pocket ID on Fly.io

Private, branded Pocket ID deployment reachable only through a Cloudflare tunnel.

## Structure

- `fly.toml` — the Pocket ID app. It has no public service and listens on an internal-only address for the Cloudflare tunnel.
- `cloudflared/fly.toml` — the tunnel sidecar that exposes `pocketid.kvcachestore.com` to the internet.
- `dagger/main.go` — Dagger module that deploys both apps to Fly.io.

## Required secrets

Set on the `kvcachestore-pocket-id` Fly app:

```bash
flyctl secrets set --app kvcachestore-pocket-id \
  APP_ENV=production \
  APP_URL=https://<your-issuer> \
  PORT=1411 \
  TRUST_PROXY=true \
  ENCRYPTION_KEY="$(openssl rand -hex 32)" \
  DB_CONNECTION_STRING=/app/backend/data/pocket-id.db
```

Set on the `kvcachestore-pocket-id-cloudflared` Fly app:

```bash
flyctl secrets set --app kvcachestore-pocket-id-cloudflared TUNNEL_TOKEN=<token>
```

## Deploy

```bash
# Create apps and volume (one-time setup)
flyctl apps create kvcachestore-pocket-id
flyctl apps create kvcachestore-pocket-id-cloudflared
flyctl volumes create pocket_id_data --app kvcachestore-pocket-id --size 1 --region iad

# Deploy both apps using Dagger. Tokens expire after 1 hour by default;
# generate a fresh one before each deploy session.
export FLY_API_TOKEN="$(flyctl tokens create deploy -a kvcachestore-pocket-id -x 1h)"
dagger call -m pocket-id/dagger deploy-all --src=. --fly-api-token=env:FLY_API_TOKEN
```

You can also deploy each app independently:

```bash
# Deploy only the Pocket ID app
dagger call -m pocket-id/dagger deploy-pocket-id --src=. --fly-api-token=env:FLY_API_TOKEN

# Deploy only the Cloudflare tunnel sidecar
dagger call -m pocket-id/dagger deploy-tunnel --src=. --fly-api-token=env:FLY_API_TOKEN
```

## Branding

The login page and authorize page branding is controlled in the Pocket ID admin UI. After creating the first admin account:

1. Go to `https://<your-issuer>/admin` and sign in as the admin user.
2. Update the **global login tagline** under Settings → General so it describes KV Cache Store accurately, e.g.:  
   _"Upload and share quantized LLM KV caches across machines."_
3. Update the **OIDC client authorize-page statement** for the `KV Cache Store` client under OIDC Clients → `KV Cache Store` → edit, e.g.:  
   _"KV Cache Store lets you upload and share quantized LLM KV caches between machines."_

## Back up the SQLite database

The live SQLite database is at `/app/backend/data/pocket-id.db` inside the container (mounted from the `pocket_id_data` Fly volume).

```bash
flyctl ssh console --app kvcachestore-pocket-id
# Inside the container:
cp /app/backend/data/pocket-id.db /tmp/pocket-id-$(date +%F).db
```
