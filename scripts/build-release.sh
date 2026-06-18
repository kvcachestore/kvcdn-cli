#!/usr/bin/env bash
set -euo pipefail

# Build the kvcdn release binary using Dagger and export it to ./dist/.
#
# Requires the Dagger CLI: https://docs.dagger.io/install
#
# Usage:
#   ./scripts/build-release.sh
#
# Optional: set COSIGN_PRIVATE_KEY to sign the release tarball and SBOM.
# The value is passed as a Dagger secret; it can be a file path:
#   COSIGN_PRIVATE_KEY=file:/path/to/cosign.key ./scripts/build-release.sh

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if ! command -v dagger >/dev/null 2>&1; then
    echo "dagger CLI not found. Install from https://docs.dagger.io/install" >&2
    exit 1
fi

mkdir -p "$REPO_ROOT/dist"

cd "$REPO_ROOT/ci/dagger"

COSIGN_ARG=""
if [ -n "${COSIGN_PRIVATE_KEY:-}" ]; then
    # Support both env-var contents and file paths.
    if [[ "$COSIGN_PRIVATE_KEY" == /* || "$COSIGN_PRIVATE_KEY" == file:* || "$COSIGN_PRIVATE_KEY" == ./* ]]; then
        COSIGN_ARG="--cosign-key=file:${COSIGN_PRIVATE_KEY#file:}"
    else
        COSIGN_ARG="--cosign-key=env:COSIGN_PRIVATE_KEY"
    fi
fi

if [ -n "$COSIGN_ARG" ]; then
    dagger call release --src="$REPO_ROOT" "$COSIGN_ARG" export --path="$REPO_ROOT/dist"
else
    dagger call release --src="$REPO_ROOT" export --path="$REPO_ROOT/dist"
fi

ls -lh "$REPO_ROOT/dist"
