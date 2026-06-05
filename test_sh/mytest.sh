#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="${CONFIG_DIR:-${SCRIPT_DIR}/config}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

VD_DIR="${VD_DIR:-/home/hust/vdbench50406}"
VD_BIN="${VD_DIR}/vdbench"
OUT_BASE="${OUT_BASE:-${SCRIPT_DIR}/out}"
LOG_BASE="${LOG_BASE:-${SCRIPT_DIR}/logs}"
OSD_ID="${OSD_ID:-0}"
CEPH_DAEMON="${CEPH_DAEMON:-osd.${OSD_ID}}"
CEPH_USE_SUDO="${CEPH_USE_SUDO:-1}"
HP_INTERVAL="${HP_INTERVAL:-5}"
HP_SAMPLE="${HP_SAMPLE:-1}"
VD_ARGS="${VD_ARGS:-}"
ACTIVE_SAMPLER_PID=""

RUN_OUT="${OUT_BASE}/${RUN_ID}"
RUN_LOG="${LOG_BASE}/${RUN_ID}"

mkdir -p "${RUN_OUT}" "${RUN_LOG}"

if [[ ! -x "${VD_BIN}" ]]; then
  echo "vdbench not found or not executable: ${VD_BIN}" >&2
  exit 1
fi

dump_hp_status() {
  local label="$1"
  local log_file="$2"

  {
    echo "===== ${label} $(date '+%F %T') ====="
    if [[ "${CEPH_USE_SUDO}" == "1" ]]; then
      if ! sudo -n ceph daemon "${CEPH_DAEMON}" perf dump hp_status; then
        echo "failed to dump hp_status from ${CEPH_DAEMON}; run 'sudo -v' first or set CEPH_USE_SUDO=0"
      fi
    elif ! ceph daemon "${CEPH_DAEMON}" perf dump hp_status; then
      echo "failed to dump hp_status from ${CEPH_DAEMON}"
    fi
    echo
  } >>"${log_file}" 2>&1
}

start_hp_sampler() {
  local log_file="$1"

  while true; do
    dump_hp_status "SAMPLE" "${log_file}"
    sleep "${HP_INTERVAL}"
  done &
  HP_SAMPLER_PID="$!"
}

stop_hp_sampler() {
  local pid="$1"

  if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" 2>/dev/null || true
  fi
}

cleanup() {
  stop_hp_sampler "${ACTIVE_SAMPLER_PID}"
}

trap cleanup EXIT INT TERM

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

configs=()
if [[ "$#" -gt 0 ]]; then
  for item in "$@"; do
    configs+=("$(resolve_config "${item}")")
  done
else
  while IFS= read -r cfg; do
    configs+=("${cfg}")
  done < <(find "${CONFIG_DIR}" -maxdepth 1 -type f -name '*.txt' | sort)
fi

if [[ "${#configs[@]}" -eq 0 ]]; then
  echo "no vdbench config found in ${CONFIG_DIR}" >&2
  exit 1
fi

echo "Run ID: ${RUN_ID}"
echo "Configs: ${CONFIG_DIR}"
echo "Output: ${RUN_OUT}"
echo "Logs:   ${RUN_LOG}"
echo "Ceph daemon: ${CEPH_DAEMON}"
echo "Ceph sudo: ${CEPH_USE_SUDO}"
echo "hp_status interval: ${HP_INTERVAL}s"
echo "hp_status sampling: ${HP_SAMPLE}"
echo "Extra vdbench args: ${VD_ARGS:-<none>}"
echo "Start:  $(date '+%F %T')"

for cfg in "${configs[@]}"; do
  name="$(basename "${cfg}" .txt)"
  out_dir="${RUN_OUT}/${name}"
  vd_log="${RUN_LOG}/${name}.vdbench.log"
  hp_log="${RUN_LOG}/${name}.hp_status.log"
  sampler_pid=""

  mkdir -p "${out_dir}"

  echo
  echo "===== START ${name} $(date '+%F %T') ====="
  echo "Config: ${cfg}"
  echo "Output: ${out_dir}"
  echo "Vdbench log: ${vd_log}"
  echo "hp_status log: ${hp_log}"

  if [[ "${HP_SAMPLE}" == "1" ]]; then
    dump_hp_status "BEFORE ${name}" "${hp_log}"
    start_hp_sampler "${hp_log}"
    sampler_pid="${HP_SAMPLER_PID}"
    ACTIVE_SAMPLER_PID="${sampler_pid}"
  fi

  read -r -a vd_extra_args <<<"${VD_ARGS}"

  if "${VD_BIN}" "${vd_extra_args[@]}" -f "${cfg}" -o "${out_dir}" >"${vd_log}" 2>&1; then
    if [[ "${HP_SAMPLE}" == "1" ]]; then
      stop_hp_sampler "${sampler_pid}"
      ACTIVE_SAMPLER_PID=""
      dump_hp_status "AFTER ${name}" "${hp_log}"
    fi
    echo "===== DONE  ${name} $(date '+%F %T') ====="
  else
    rc="$?"
    if [[ "${HP_SAMPLE}" == "1" ]]; then
      stop_hp_sampler "${sampler_pid}"
      ACTIVE_SAMPLER_PID=""
      dump_hp_status "AFTER_FAILED ${name}" "${hp_log}"
    fi
    echo "===== FAIL  ${name} $(date '+%F %T') exit=${rc} =====" >&2
    exit "${rc}"
  fi
done

echo
echo "All tests finished: $(date '+%F %T')"
