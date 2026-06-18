#!/usr/bin/env bash
set -euo pipefail

# Build the kvcdn backend Docker image locally.
#
# Usage:
#   ./scripts/build-backend-image.sh [tag]
#
# The default tag is "kvcdn-api:local".

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${1:-kvcdn-api:local}"

cd "$REPO_ROOT"
docker build -t "$TAG" backend/

echo "Built backend image: $TAG"
