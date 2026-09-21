#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=${CEPH_BUILD_DIR:-$repo/build}
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
flags=(-std=c++17 -O1 -g -pthread -I"$repo/src" -I"$build/include" -I"$build/boost/include")
if [[ -n ${HP_SANITIZERS:-} ]]; then
  flags+=(-fno-omit-frame-pointer "-fsanitize=$HP_SANITIZERS")
fi
"${CXX:-g++}" "${flags[@]}" "$repo/test_sh/hp_osd_module_probe.cc" \
  "$repo/src/osd/ObjectHeatPredictor.cc" \
  -L"$build/lib" -Wl,-rpath,"$build/lib" -lceph-common -Wl,--wrap=pthread_rwlock_rdlock \
  -o "$scratch/hp_osd_module_probe"
HP_OSD_MODULE_PROBE="$scratch/hp_osd_module_probe" python3 "$repo/test_sh/test_hp_osd_module.py"
