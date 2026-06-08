#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_run}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/background}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_ID}.nohup.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/${RUN_ID}.pid}"

mkdir -p "${LOG_DIR}"

env \
  RUN_ID="${RUN_ID}" \
  CONFIG_DIR="${CONFIG_DIR:-${SCRIPT_DIR}/config_skew_run}" \
  HP_SAMPLE="${HP_SAMPLE:-1}" \
  OSD_ID="${OSD_ID:-0}" \
  nohup "${SCRIPT_DIR}/run_skew_tests_reset_hp.sh" "$@" >"${LOG_FILE}" 2>&1 &

pid="$!"
echo "${pid}" >"${PID_FILE}"

echo "Started vdbench skew tests in background"
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
