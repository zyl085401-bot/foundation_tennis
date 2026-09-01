#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-foundationpose-jetson}"
FULL_INTERVAL_MS="${FULL_INTERVAL_MS:-200}"
FAST_INTERVAL_MS="${FAST_INTERVAL_MS:-50}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/realtime_foundation/outputs}"
RUN_DIR="${OUTPUT_ROOT}/live_jetson_telemetry_$(date +%Y%m%d_%H%M%S)"
RELATIVE_RUN_DIR="${RUN_DIR#${REPO_ROOT}/}"
CONTAINER_RUN_DIR="/workspace/yolo_foundationpose/${RELATIVE_RUN_DIR}"

if [[ "$(uname -m)" != "aarch64" && "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: This telemetry launcher must run natively on the Jetson." >&2
  exit 1
fi
if ! [[ "${FULL_INTERVAL_MS}" =~ ^[0-9]+$ && "${FAST_INTERVAL_MS}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: FULL_INTERVAL_MS and FAST_INTERVAL_MS must be integer milliseconds." >&2
  exit 2
fi
if (( FULL_INTERVAL_MS < 50 || FAST_INTERVAL_MS < 10 )); then
  echo "ERROR: FULL_INTERVAL_MS must be >= 50 and FAST_INTERVAL_MS must be >= 10." >&2
  exit 2
fi
if ! /usr/bin/python3 -c 'from jtop import jtop' >/dev/null 2>&1; then
  echo "ERROR: The Jetson host /usr/bin/python3 cannot import jtop." >&2
  exit 1
fi
if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
elif sudo docker info >/dev/null 2>&1; then
  DOCKER=(sudo docker)
else
  echo "ERROR: Docker daemon is unavailable." >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"
{
  echo "run_dir=${RELATIVE_RUN_DIR}"
  echo "host=$(hostname)"
  echo "started_local=$(date --iso-8601=ns)"
  echo "started_utc=$(date -u --iso-8601=ns)"
  echo "full_interval_ms=${FULL_INTERVAL_MS}"
  echo "fast_interval_ms=${FAST_INTERVAL_MS}"
  echo "event_clock=time.time_ns"
  echo "gpu_fast_load_conversion=raw_divided_by_10"
  sha256sum \
    "${REPO_ROOT}/realtime_foundation/run_realtime.py" \
    "${REPO_ROOT}/realtime_foundation/config.yaml" \
    "${REPO_ROOT}/realtime_foundation/tools/jetson_telemetry.py" \
    "${REPO_ROOT}/realtime_foundation/tools/analyze_jetson_telemetry.py"
  echo
  nvpmodel -q 2>&1 || true
  echo
  jetson_clocks --show 2>&1 || true
} > "${RUN_DIR}/metadata.txt"

setsid /usr/bin/python3 -u "${REPO_ROOT}/realtime_foundation/tools/jetson_telemetry.py" \
  --output-dir "${RUN_DIR}" \
  --full-interval-ms "${FULL_INTERVAL_MS}" \
  --fast-interval-ms "${FAST_INTERVAL_MS}" \
  > "${RUN_DIR}/telemetry_stdout.log" \
  2> "${RUN_DIR}/telemetry_stderr.log" &
telemetry_pid=$!

setsid bash -c 'set -o pipefail; stdbuf -oL /usr/bin/tegrastats --interval "$1" | while IFS= read -r line; do printf "%s %s\n" "$(date --iso-8601=ns)" "$line"; done' \
  _ "${FULL_INTERVAL_MS}" > "${RUN_DIR}/tegrastats.log" 2>&1 &
tegrastats_pid=$!

echo "${telemetry_pid}" > "${RUN_DIR}/telemetry.pid"
echo "${tegrastats_pid}" > "${RUN_DIR}/tegrastats.pid"

cleanup_done=0
cleanup() {
  local status=$?
  if (( cleanup_done )); then
    return
  fi
  cleanup_done=1
  trap - EXIT INT TERM

  if [[ -s "${RUN_DIR}/runtime.container.pid" ]] && "${DOCKER[@]}" inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    container_pid="$(cat "${RUN_DIR}/runtime.container.pid")"
    if "${DOCKER[@]}" exec "${CONTAINER_NAME}" kill -0 "${container_pid}" >/dev/null 2>&1; then
      "${DOCKER[@]}" exec "${CONTAINER_NAME}" kill -INT "${container_pid}" >/dev/null 2>&1 || true
      for _attempt in {1..50}; do
        if ! "${DOCKER[@]}" exec "${CONTAINER_NAME}" kill -0 "${container_pid}" >/dev/null 2>&1; then
          break
        fi
        read -r -t 0.05 _unused || true
      done
      "${DOCKER[@]}" exec "${CONTAINER_NAME}" kill -TERM "${container_pid}" >/dev/null 2>&1 || true
    fi
  fi
  kill -- -"${telemetry_pid}" >/dev/null 2>&1 || true
  kill -- -"${tegrastats_pid}" >/dev/null 2>&1 || true
  wait "${telemetry_pid}" >/dev/null 2>&1 || true
  wait "${tegrastats_pid}" >/dev/null 2>&1 || true

  {
    echo "stopped_local=$(date --iso-8601=ns)"
    echo "stopped_utc=$(date -u --iso-8601=ns)"
    echo "runtime_exit_code=${status}"
  } >> "${RUN_DIR}/metadata.txt"

  if [[ -s "${RUN_DIR}/runtime.log" ]]; then
    /usr/bin/python3 "${REPO_ROOT}/realtime_foundation/tools/analyze_jetson_telemetry.py" "${RUN_DIR}" \
      > "${RUN_DIR}/analysis_stdout.log" \
      2> "${RUN_DIR}/analysis_stderr.log" || true
  fi
  echo "[TELEMETRY] stopped; artifacts=${RELATIVE_RUN_DIR}; exit=${status}"
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

container_running=false
if "${DOCKER[@]}" inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null | grep -qx true; then
  container_running=true
  if "${DOCKER[@]}" exec "${CONTAINER_NAME}" pgrep -f 'realtime_foundation/run_realtime.py' >/dev/null 2>&1; then
    echo "ERROR: run_realtime.py is already active in ${CONTAINER_NAME}." >&2
    exit 1
  fi
fi

runtime_shell="cd /workspace/yolo_foundationpose && echo \$\$ > '${CONTAINER_RUN_DIR}/runtime.container.pid' && exec python3 realtime_foundation/run_realtime.py --config realtime_foundation/config.yaml"

echo "[TELEMETRY] artifacts=${RELATIVE_RUN_DIR}"
echo "[TELEMETRY] full=${FULL_INTERVAL_MS}ms fast=${FAST_INTERVAL_MS}ms"
echo "[TELEMETRY] start the mROS camera publisher before collecting runtime data"

set +e
if [[ "${container_running}" == true ]]; then
  "${DOCKER[@]}" exec -e PYTHONUNBUFFERED=1 "${CONTAINER_NAME}" \
    bash -lc "${runtime_shell}" > >(tee "${RUN_DIR}/runtime.log") 2>&1
  runtime_status=$?
else
  bash "${SCRIPT_DIR}/run_container_jetson.sh" \
    bash -lc "${runtime_shell}" > >(tee "${RUN_DIR}/runtime.log") 2>&1
  runtime_status=$?
fi
set -e
exit "${runtime_status}"
