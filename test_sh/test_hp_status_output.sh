#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=${CEPH_BUILD_DIR:-$repo/build}
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

"${CXX:-g++}" -std=c++17 -I"$repo/src" -I"$build/include" \
  "$repo/test_sh/hp_status_output_probe.cc" \
  "$repo/src/mgr/ObjectHeatPredictorStatus.cc" \
  "$repo/src/mgr/ObjectHeatPredictorStatusFormatter.cc" \
  -L"$build/lib" -Wl,-rpath,"$build/lib" -lceph-common \
  -o "$scratch/hp_status_output_probe"

HP_STATUS_OUTPUT_PROBE="$scratch/hp_status_output_probe" \
  python3 "$repo/test_sh/test_hp_status_output.py"

"${CXX:-g++}" -std=c++17 -I"$repo/src" \
  "$repo/test_sh/hp_status_contract_probe.cc" \
  "$repo/src/mgr/ObjectHeatPredictorStatus.cc" \
  -o "$scratch/hp_status_contract_probe"
"$scratch/hp_status_contract_probe"
