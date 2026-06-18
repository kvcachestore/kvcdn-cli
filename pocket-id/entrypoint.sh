#!/bin/sh
set -e

DB_PATH="/app/backend/data/pocket-id.db"

# Ensure the DB directory exists (the Fly volume is mounted here).
mkdir -p /app/backend/data

# The Pocket ID process runs as UID/GID 1000, so the data volume must be
# writable by that user. Root owns the mount point by default.
chown -R 1000:1000 /app/backend/data

# Force the app to use the persistent volume-backed database.
# Pocket ID reads DB_CONNECTION_STRING, not DB_CONNECTION_URL.
export DB_CONNECTION_STRING="$DB_PATH"

# Start the original Pocket ID entrypoint in the background so migrations run
# and create the schema before we try to seed rows.
/app/docker/entrypoint.sh "$@" &
POCKET_PID=$!

# Wait for the database file and schema tables to exist, then seed the
# kvcdn-cli public OIDC client and the verified Resend sender domain.
for i in $(seq 1 60); do
    if [ -f "$DB_PATH" ]; then
        TABLE_COUNT=$(sqlite3 "$DB_PATH" "SELECT count(*) FROM sqlite_master WHERE type='table' AND name IN ('app_config_variables', 'oidc_clients');" 2>/dev/null || echo 0)
        if [ "$TABLE_COUNT" = "2" ]; then
            sqlite3 "$DB_PATH" <<EOF
INSERT OR REPLACE INTO app_config_variables (key, value) VALUES ('appName', 'KV Cache Store');
INSERT OR REPLACE INTO app_config_variables (key, value) VALUES ('smtpHost', 'smtp.resend.com');
INSERT OR REPLACE INTO app_config_variables (key, value) VALUES ('smtpPort', '465');
INSERT OR REPLACE INTO app_config_variables (key, value) VALUES ('smtpUser', 'resend');
INSERT OR REPLACE INTO app_config_variables (key, value) VALUES ('smtpPassword', '$RESEND_API_KEY');
INSERT OR REPLACE INTO app_config_variables (key, value) VALUES ('smtpTls', 'tls');
INSERT OR REPLACE INTO app_config_variables (key, value) VALUES ('smtpSkipCertVerify', 'false');
INSERT OR REPLACE INTO app_config_variables (key, value) VALUES ('smtpFrom', 'noreply@kvcachestore.com');
INSERT OR REPLACE INTO app_config_variables (key, value) VALUES ('emailOneTimeAccessAsUnauthenticatedEnabled', 'true');
INSERT OR REPLACE INTO app_config_variables (key, value) VALUES ('emailVerificationEnabled', 'false');
INSERT OR REPLACE INTO app_config_variables (key, value) VALUES ('homePageUrl', '/settings/account');
INSERT OR REPLACE INTO app_config_variables (key, value) VALUES ('emailsVerified', 'true');
INSERT OR IGNORE INTO oidc_clients (
    id, created_at, name, secret,
    callback_urls, image_type, created_by_id,
    is_public, pkce_enabled, logout_callback_urls,
    credentials, launch_url, requires_reauthentication,
    dark_image_type, is_group_restricted
) VALUES (
    'kvcdn-cli',
    datetime('now'),
    'kvcdn-cli',
    '',
    '["http://127.0.0.1:*", "http://localhost:*"]', NULL, NULL,
    1, 1, '[]',
    '{}', NULL, 0,
    NULL, 0
);
EOF
            break
        fi
    fi
    sleep 1
done

# Keep the container running as long as Pocket ID is alive.
wait "$POCKET_PID"
