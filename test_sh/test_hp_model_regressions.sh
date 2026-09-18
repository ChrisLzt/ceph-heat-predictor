#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=${CEPH_BUILD_DIR:-$repo/build}
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

flags=(-std=c++17 -O2 -pthread -I"$repo/src" -I"$build/include"
       -I"$build/boost/include" -I"$repo/src/heatpredictor/include")
if [[ -n ${HP_SANITIZERS:-} ]]; then
  flags+=(-O1 -g -fno-omit-frame-pointer "-fsanitize=$HP_SANITIZERS")
fi
for probe in test_hp_tree_split_candidates hp_algorithm_probe hp_online_policy_probe hp_trace_probe test_hp_trace_replay; do
  "${CXX:-g++}" "${flags[@]}" "$repo/test_sh/$probe.cc" -o "$scratch/$probe"
  timeout 120 "$scratch/$probe"
done
