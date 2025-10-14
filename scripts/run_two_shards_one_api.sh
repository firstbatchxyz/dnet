#!/usr/bin/env bash
set -euo pipefail

# Runs two shard servers and one API server on the same device,
# then loads Qwen/Qwen3-4B-MLX-8bit with a manual split:
#   - Shard A layers: [0..17]
#   - Shard B layers: [18..35]
# Requires: uv, curl, python3

# Configurable ports (override via env before running)
API_HTTP_PORT=${API_HTTP_PORT:-8080}
API_GRPC_PORT=${API_GRPC_PORT:-58080}

SHARD1_HTTP_PORT=${SHARD1_HTTP_PORT:-8081}
SHARD1_GRPC_PORT=${SHARD1_GRPC_PORT:-58081}

SHARD2_HTTP_PORT=${SHARD2_HTTP_PORT:-8082}
SHARD2_GRPC_PORT=${SHARD2_GRPC_PORT:-58082}

MODEL=${MODEL:-"Qwen/Qwen3-4B-MLX-8bit"}

echo "Using ports: API http=${API_HTTP_PORT} grpc=${API_GRPC_PORT}; shard1 http=${SHARD1_HTTP_PORT} grpc=${SHARD1_GRPC_PORT}; shard2 http=${SHARD2_HTTP_PORT} grpc=${SHARD2_GRPC_PORT}"

command -v uv >/dev/null 2>&1 || { echo "uv is required but not found in PATH" >&2; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is required but not found in PATH" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 is required but not found in PATH" >&2; exit 1; }

mkdir -p logs

cleanup() {
  echo "Stopping spawned processes..."
  set +e
  [[ -n "${API_PID:-}" ]] && kill ${API_PID} 2>/dev/null || true
  [[ -n "${SHARD1_PID:-}" ]] && kill ${SHARD1_PID} 2>/dev/null || true
  [[ -n "${SHARD2_PID:-}" ]] && kill ${SHARD2_PID} 2>/dev/null || true
  wait ${API_PID:-} ${SHARD1_PID:-} ${SHARD2_PID:-} 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for() {
  local url=$1
  local retries=${2:-60}
  local delay=${3:-1}
  for ((i=0; i<retries; i++)); do
    if curl "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep "${delay}"
  done
  echo "Timeout waiting for ${url}" >&2
  return 1
}

echo "Starting shard 1..."
uv run dnet-shard --http-port "${SHARD1_HTTP_PORT}" --grpc-port "${SHARD1_GRPC_PORT}" >"logs/shard1.log" 2>&1 &
SHARD1_PID=$!

echo "Starting shard 2..."
uv run dnet-shard --http-port "${SHARD2_HTTP_PORT}" --grpc-port "${SHARD2_GRPC_PORT}" >"logs/shard2.log" 2>&1 &
SHARD2_PID=$!

echo "Starting API..."
uv run dnet-api --http-port "${API_HTTP_PORT}" --grpc-port "${API_GRPC_PORT}" >"logs/api.log" 2>&1 &
API_PID=$!

echo "Waiting for servers to become ready..."
if ! wait_for "http://localhost:${SHARD1_HTTP_PORT}/health" 120 1; then
  echo "Shard1 failed to start; last 80 lines of logs/shard1.log:" >&2
  tail -n 80 "logs/shard1.log" >&2 || true
  exit 1
fi
if ! wait_for "http://localhost:${SHARD2_HTTP_PORT}/health" 120 1; then
  echo "Shard2 failed to start; last 80 lines of logs/shard2.log:" >&2
  tail -n 80 "logs/shard2.log" >&2 || true
  exit 1
fi
if ! wait_for "http://localhost:${API_HTTP_PORT}/v1/health" 120 1; then
  echo "API failed to start; last 80 lines of logs/api.log:" >&2
  tail -n 80 "logs/api.log" >&2 || true
  exit 1
fi

echo "Discovering shard service names from API /v1/devices..."
DEVICES_JSON=$(curl -X GET "http://localhost:${API_HTTP_PORT}/v1/devices")


#echo "Loading model on shards via API /v1/load_model..."
#curl -fsS -X POST "http://localhost:${API_HTTP_PORT}/v1/load_model" \
#  -H 'Content-Type: application/json' \
#  -d "${ASSIGN_JSON}" | tee logs/load_model_response.json

echo "\nAll services are running. Logs: logs/api.log, logs/shard1.log, logs/shard2.log"
echo "Press Ctrl+C to stop."

# Keep the script alive to preserve the trap and child processes.
wait
