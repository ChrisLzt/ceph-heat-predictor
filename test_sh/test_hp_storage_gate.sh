#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=${CEPH_BUILD_DIR:-$repo/build}
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
flags=(-std=c++17 -O1 -g -pthread -I"$repo/src" -I"$build/include" -I"$build/boost/include")
"${CXX:-g++}" "${flags[@]}" "$repo/test_sh/test_storage_observation_gate.cc" -Wl,--wrap=pthread_rwlock_rdlock -o "$scratch/observer"
"$scratch/observer"
"${CXX:-g++}" "${flags[@]}" "$repo/test_sh/hp_storage_gate_probe.cc" "$repo/src/osd/ObjectHeatPredictor.cc" -L"$build/lib" -Wl,-rpath,"$build/lib" -lceph-common -Wl,--wrap=pthread_rwlock_rdlock -o "$scratch/module"
"$scratch/module"
echo 'PASS: disabled storage gate, enable/reset/disable, concurrent commands and shutdown'
