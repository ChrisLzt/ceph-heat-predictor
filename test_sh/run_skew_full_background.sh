#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_skew_full}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/background}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_ID}.nohup.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/${RUN_ID}.pid}"

run_background() {
  mkdir -p "${LOG_DIR}"

  env \
    RUN_ID="${RUN_ID}" \
    LOG_DIR="${LOG_DIR}" \
    LOG_FILE="${LOG_FILE}" \
    PID_FILE="${PID_FILE}" \
    nohup "$0" --foreground "$@" >"${LOG_FILE}" 2>&1 &
  local pid="$!"
  echo "${pid}" >"${PID_FILE}"

  echo "Started skew data+test run in background"
  echo "PID:      ${pid}"
  echo "Run ID:   ${RUN_ID}"
  echo "Log:      ${LOG_FILE}"
  echo "PID file: ${PID_FILE}"
  echo
  echo "Watch:"
  echo "  tail -f ${LOG_FILE}"
  echo
  echo "Stop:"
  echo "  kill ${pid}"
}

run_foreground() {
  local data_run_id="${DATA_RUN_ID:-${RUN_ID}_data}"
  local test_run_id="${TEST_RUN_ID:-${RUN_ID}_run}"

  echo "Skew full run started: $(date '+%F %T')"
  echo "Run ID: ${RUN_ID}"
  echo "Data run ID: ${data_run_id}"
  echo "Test run ID: ${test_run_id}"
  echo "Arguments: ${*:-<all configs>}"
  echo

  echo "===== PREPARE DATA $(date '+%F %T') ====="
  RUN_ID="${data_run_id}" HP_SAMPLE=0 "${SCRIPT_DIR}/prepare_skew_data.sh" "$@"

  echo
  echo "===== RUN TESTS WITH HP_STATUS $(date '+%F %T') ====="
  RUN_ID="${test_run_id}" HP_SAMPLE=1 "${SCRIPT_DIR}/run_skew_tests_reset_hp.sh" "$@"

  echo
  echo "Skew full run finished: $(date '+%F %T')"
}

if [[ "${1:-}" == "--foreground" ]]; then
  shift
  run_foreground "$@"
else
  run_background "$@"
fi
