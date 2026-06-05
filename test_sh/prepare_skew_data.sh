#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CONFIG_DIR="${CONFIG_DIR:-${SCRIPT_DIR}/config_skew_data}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_data}"
VD_ARGS="${VD_ARGS:--c}"
HP_SAMPLE="${HP_SAMPLE:-0}"

export CONFIG_DIR RUN_ID VD_ARGS HP_SAMPLE

resolve_config() {
  local item="$1"
  local base

  if [[ -f "${item}" ]]; then
    realpath "${item}"
  elif [[ -f "${CONFIG_DIR}/${item}" ]]; then
    realpath "${CONFIG_DIR}/${item}"
  elif [[ -f "${CONFIG_DIR}/${item}.txt" ]]; then
    realpath "${CONFIG_DIR}/${item}.txt"
  else
    base="$(basename "${item}" .txt)"
    case "${base}" in
      1G文件4K块随机读|1G文件4K块随机写|1G文件4M块顺序读|1G文件4M块顺序写)
        realpath "${CONFIG_DIR}/1G文件数据构造.txt"
        ;;
      4K文件4K块随机读|4K文件4K块随机写)
        realpath "${CONFIG_DIR}/4K文件数据构造.txt"
        ;;
      *)
        echo "config not found: ${item}" >&2
        return 1
        ;;
    esac
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

prepare_lun_dirs() {
  local cfg="$1"

  awk '
    /^sd=/ {
      n = split($0, fields, ",")
      for (i = 1; i <= n; i++) {
        if (fields[i] ~ /^lun=/) {
          path = fields[i]
          sub(/^lun=/, "", path)
          sub(/\/[^\/]+$/, "", path)
          if (path != "") {
            print path
          }
        }
      }
    }
  ' "${cfg}"
}

echo "Prepare skew test data"
echo "Config dir: ${CONFIG_DIR}"
echo "Run ID: ${RUN_ID}"
echo "VD_ARGS: ${VD_ARGS}"
echo "hp_status sampling: ${HP_SAMPLE}"

mapfile -t configs < <(collect_configs "$@")
if [[ "${#configs[@]}" -eq 0 ]]; then
  echo "no vdbench config found in ${CONFIG_DIR}" >&2
  exit 1
fi

while IFS= read -r dir; do
  [[ -n "${dir}" ]] || continue
  mkdir -p "${dir}"
done < <(
  for cfg in "${configs[@]}"; do
    prepare_lun_dirs "${cfg}"
  done | sort -u
)

"${SCRIPT_DIR}/mytest.sh" "${configs[@]}"
