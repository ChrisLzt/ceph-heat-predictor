#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CONFIG_DIR="${CONFIG_DIR:-${SCRIPT_DIR}/config_skew_run}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_run}"
OSD_ID="${OSD_ID:-0}"
OSD_SERVICE="${OSD_SERVICE:-ceph-osd@${OSD_ID}}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-180}"
WAIT_INTERVAL="${WAIT_INTERVAL:-2}"
VD_ARGS="${VD_ARGS:-}"

export CONFIG_DIR RUN_ID OSD_ID VD_ARGS

resolve_config() {
  local item="$1"

  if [[ -f "${item}" ]]; then
    realpath "${item}"
  elif [[ -f "${CONFIG_DIR}/${item}" ]]; then
    realpath "${CONFIG_DIR}/${item}"
  elif [[ -f "${CONFIG_DIR}/${item}.txt" ]]; then
    realpath "${CONFIG_DIR}/${item}.txt"
  else
    echo "config not found: ${item}" >&2
    return 1
  fi
}

collect_configs() {
  if [[ "$#" -gt 0 ]]; then
    for item in "$@"; do
      resolve_config "${item}"
    done
  else
    find "${CONFIG_DIR}" -maxdepth 1 -type f -name '*.txt' | sort
  fi
}

restart_osd_to_reset_hp_status() {
  echo "Reset heat predictor status by restarting ${OSD_SERVICE}: $(date '+%F %T')"
  sudo systemctl reset-failed "${OSD_SERVICE}" || true
  sudo systemctl stop "${OSD_SERVICE}" || true
  sleep 10
  sudo systemctl start "${OSD_SERVICE}"
}

osd_is_up_in() {
  ceph osd dump 2>/dev/null | awk -v osd="osd.${OSD_ID}" '
    $1 == osd && $2 == "up" && $3 == "in" { found = 1 }
    END { exit found ? 0 : 1 }
  '
}

pgs_are_clean() {
  local pg_stat
  pg_stat="$(ceph pg stat 2>/dev/null || true)"

  [[ -n "${pg_stat}" ]] || return 1
  [[ "${pg_stat}" == *"active+clean"* ]] || return 1
  ! grep -Eq 'unknown|stale|inactive|down|peering|creating|undersized|degraded|remapped|backfill|recovering|recovery_wait|backfill_wait' <<<"${pg_stat}"
}

wait_for_ceph_ready() {
  local elapsed=0

  while (( elapsed <= WAIT_TIMEOUT )); do
    if osd_is_up_in && pgs_are_clean; then
      echo "OSD ${OSD_ID} and PGs are ready after ${elapsed}s"
      ceph pg stat
      return 0
    fi

    if (( elapsed % 10 == 0 )); then
      echo "Waiting for OSD ${OSD_ID} and PGs... ${elapsed}/${WAIT_TIMEOUT}s"
      ceph -s 2>/dev/null | sed -n '1,22p' || true
    fi

    sleep "${WAIT_INTERVAL}"
    elapsed=$((elapsed + WAIT_INTERVAL))
  done

  echo "Ceph did not become ready within ${WAIT_TIMEOUT}s" >&2
  ceph -s >&2 || true
  ceph health detail >&2 || true
  sudo journalctl -u "${OSD_SERVICE}" -n 80 --no-pager >&2 || true
  return 1
}

mapfile -t CONFIGS < <(collect_configs "$@")

if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
  echo "no vdbench config found in ${CONFIG_DIR}" >&2
  exit 1
fi

echo "Run skew tests with heat predictor status reset before each test"
echo "Config dir: ${CONFIG_DIR}"
echo "Run ID: ${RUN_ID}"
echo "OSD service: ${OSD_SERVICE}"
echo "VD_ARGS: ${VD_ARGS}"
echo "Wait timeout: ${WAIT_TIMEOUT}s"

for cfg in "${CONFIGS[@]}"; do
  name="$(basename "${cfg}" .txt)"

  echo
  echo "===== RESET_AND_RUN ${name} $(date '+%F %T') ====="
  restart_osd_to_reset_hp_status
  wait_for_ceph_ready
  "${SCRIPT_DIR}/mytest.sh" "${cfg}"
done

echo
echo "All skew tests finished: $(date '+%F %T')"
