#!/usr/bin/env bash
# Start a kind cluster named "kvcdn-local" with a local container registry at
# localhost:5001. Images built on the host can be pushed to the registry and
# pulled by pods in the cluster.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CLUSTER_NAME="${CLUSTER_NAME:-kvcdn-local}"
REG_NAME="${REG_NAME:-kvcdn-local-registry}"
REG_PORT="${REG_PORT:-5001}"

if command -v docker >/dev/null 2>&1; then
  CONTAINER_CLI=docker
elif command -v podman >/dev/null 2>&1; then
  CONTAINER_CLI=podman
else
  echo "ERROR: docker or podman is required" >&2
  exit 1
fi

if ! command -v kind >/dev/null 2>&1; then
  echo "ERROR: kind is required (https://kind.sigs.k8s.io/docs/user/quick-start/#installation)" >&2
  exit 1
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "ERROR: kubectl is required" >&2
  exit 1
fi

teardown() {
  echo "Tearing down kind cluster ${CLUSTER_NAME} and registry ${REG_NAME}..."
  kind delete cluster --name "${CLUSTER_NAME}" || true
  $CONTAINER_CLI rm -f "${REG_NAME}" || true
}

if [ "${1:-}" = "--teardown" ] || [ "${1:-}" = "teardown" ]; then
  teardown
  exit 0
fi

# Start local registry if not already running.
if ! $CONTAINER_CLI inspect -f '{{.State.Running}}' "${REG_NAME}" >/dev/null 2>&1; then
  echo "Starting local registry ${REG_NAME} on port ${REG_PORT}..."
  $CONTAINER_CLI run -d --restart=always -p "127.0.0.1:${REG_PORT}:5000" --name "${REG_NAME}" registry:3
else
  echo "Local registry ${REG_NAME} already running."
fi

# Create kind config that points each node at the local registry.
KIND_CONFIG_DIR="${ROOT_DIR}/.kind"
mkdir -p "${KIND_CONFIG_DIR}"
KIND_CONFIG_FILE="${KIND_CONFIG_DIR}/kind-config.yaml"

cat >"${KIND_CONFIG_FILE}" <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
containerdConfigPatches:
  - |
    [plugins."io.containerd.grpc.v1.cri".registry]
      config_path = "/etc/containerd/certs.d"
EOF

run_kind_create() {
  if command -v systemd-run > /dev/null 2>&1 && systemctl --user status > /dev/null 2>&1; then
    systemd-run --user --scope --property=Delegate=yes -- \
      kind create cluster --name "${CLUSTER_NAME}" --config "${KIND_CONFIG_FILE}"
  else
    kind create cluster --name "${CLUSTER_NAME}" --config "${KIND_CONFIG_FILE}"
  fi
}

if kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
  echo "kind cluster ${CLUSTER_NAME} already exists."
else
  echo "Creating kind cluster ${CLUSTER_NAME}..."
  run_kind_create
fi

# Connect the registry to the kind network so the cluster can reach it.
if [ "$CONTAINER_CLI" = "docker" ]; then
  if [ "$($CONTAINER_CLI inspect -f '{{.HostConfig.NetworkMode}}' "${REG_NAME}")" != "kind" ]; then
    $CONTAINER_CLI network connect kind "${REG_NAME}" || true
  fi
fi

# Configure every node to trust the registry by host alias.
REGISTRY_DIR="/etc/containerd/certs.d/localhost:${REG_PORT}"
for node in $(kind get nodes --name "${CLUSTER_NAME}"); do
  $CONTAINER_CLI exec "${node}" mkdir -p "${REGISTRY_DIR}"
  $CONTAINER_CLI exec "${node}" sh -c "cat > ${REGISTRY_DIR}/hosts.toml <<'INNER'
[host.'http://${REG_NAME}:5000']
  capabilities = ['pull', 'resolve']
INNER"
done

# Document the registry endpoint inside the cluster.
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: local-registry-hosting
  namespace: kube-public
data:
  localRegistryHosting.v1: |
    host: "localhost:${REG_PORT}"
    help: "https://kind.sigs.k8s.io/docs/user/local-registry/"
EOF

# Add helpful aliases to current kubeconfig context.
kubectl config set-context "kind-${CLUSTER_NAME}" --cluster "kind-${CLUSTER_NAME}" --user "kind-${CLUSTER_NAME}" --namespace=kvcdn >/dev/null

echo ""
echo "Cluster '${CLUSTER_NAME}' is ready."
echo "  kubectl context: kind-${CLUSTER_NAME}"
echo "  registry:        localhost:${REG_PORT}"
echo "  example:         docker build -t localhost:${REG_PORT}/kvcdn-api:local backend/"
echo "                   docker push localhost:${REG_PORT}/kvcdn-api:local"
