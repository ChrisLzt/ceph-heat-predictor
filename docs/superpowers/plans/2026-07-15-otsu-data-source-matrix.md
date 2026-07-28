# Otsu Data Source Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and execute a reproducible D0/D1/D2 Otsu data-source comparison on the five formal Vdbench workloads.

**Architecture:** Keep threshold smoothing and prediction policy fixed, and select only the Otsu vote source through a separate compile-time profile. Extend the existing fixed time-ring histogram for object-deduplicated and per-I/O modes, then use the shared matrix runner for build metadata, queue drain, 30-second sampling and result aggregation.

**Tech Stack:** C++17, Ceph OSD/MGR perf counters, CMake/Ninja, Bash, Python 3 result summarizer, Oracle Vdbench 5.04.07.

## Global Constraints

- D0/D1/D2 share one prepared dataset and differ only in `HP_OTSU_DATA_SOURCE`.
- H0 fixed EMA is `0.10`; P0 prediction threshold is `0.50`; class weights are `1.0`.
- The label remains future-added heat over a strict 10-second window.
- Every formal workload runs once and MGR status is saved every 30 seconds.
- Luna is read-only monitoring only; Sol owns code, deployment and conclusions.

---

### Task 1: Otsu Data Source Profiles

**Files:**
- Modify: `src/heatpredictor/hp_config.h`
- Modify: `src/osd/CMakeLists.txt`
- Test: `test_sh/hp_algorithm_probe.cc`

**Interfaces:**
- Produces: `HP_OTSU_DATA_SOURCE_OBJECT_ADDED`, `HP_OTSU_DATA_SOURCE_IO_ADDED`,
  `HP_OTSU_DATA_SOURCE_OBJECT_TOTAL`, and validated `HP_OTSU_DATA_SOURCE`.

- [ ] Add probe assertions that profile constants are distinct and valid.
- [ ] Compile the probe with an invalid profile and confirm compilation fails.
- [ ] Add the constants, static assertion and CMake cache validation.
- [ ] Compile all three valid profiles and confirm the probe builds.

### Task 2: Histogram Vote Modes

**Files:**
- Modify: `src/heatpredictor/hp_otsu_histogram.h`
- Modify: `src/heatpredictor/hp_evaluation_queue.h`
- Modify: `test_sh/hp_algorithm_probe.cc`

**Interfaces:**
- Consumes: `HP_OTSU_DATA_SOURCE` constants from Task 1.
- Produces: object replacement for D0/D2, per-I/O append for D1, 60-second
  expiry for both, and `otsu_histogram_vote_count()`.

- [ ] Add failing probes for duplicate-object replacement in D0/D2, duplicate
  retention in D1, one-second gaps and full 60-second expiry.
- [ ] Run each profile probe and verify the expected failure.
- [ ] Implement keyed and unkeyed observation paths over the same fixed bins.
- [ ] Record future-added heat for D0/D1 and deadline total heat clamped to `10`
  for D2.
- [ ] Run all profile probes and the performance probe.

### Task 3: OSD And MGR Statistics

**Files:**
- Modify: `src/heatpredictor/heat_predictor.h`
- Modify: `src/heatpredictor/hp_types.h`
- Modify: `src/osd/ObjectHeatPredictor.cc`
- Modify: `src/mgr/DaemonServer.cc`
- Test: `test_sh/hp_algorithm_probe.cc`

**Interfaces:**
- Produces: `hp_otsu_histogram_vote_count` in OSD live status, perf dump and MGR
  summary with matching enum/register/update/output order.

- [ ] Add a failing source-contract assertion for the generic vote-count API.
- [ ] Rename the retained-count field through algorithm, OSD and MGR layers.
- [ ] Compile and run the algorithm probe; search for obsolete exported names.

### Task 4: Five-Workload Matrix And Sampling

**Files:**
- Modify: `/home/chris/ceph-test/new_workload/run_hp_matrix.sh`
- Modify: `/home/chris/ceph-test/new_workload/tests/test_run_hp_matrix.sh`
- Modify: `/home/chris/ceph-test/new_workload/tools/summarize_hp_matrix.py`

**Interfaces:**
- Produces: profiles `D0`, `D1`, `D2`; five default workloads; per-workload
  `hp_status_30s/*.json`; `hp_status_30s_index.tsv`; final `results.tsv`.

- [ ] Extend the dry-run contract test with D0/D1/D2 and HPC, and verify it fails.
- [ ] Add `HP_OTSU_DATA_SOURCE` cache validation and profile metadata.
- [ ] Add a sampler lifecycle tied to each workload process, with cleanup traps.
- [ ] Extend the summarizer for `hp_otsu_histogram_vote_count`.
- [ ] Run shell syntax, runner contract and `validate_all.sh`.

### Task 5: Build And Dataset Preparation

**Files:**
- Runtime outputs only under `/mnt/cephfs` and the selected report directory.

**Interfaces:**
- Consumes: validated D0 profile and five prepare scripts.
- Produces: five reusable 112.5 GiB anchors and a healthy D0 deployment.

- [ ] Build/install D0, run `ldconfig`, restart OSD/MGR and wait active+clean.
- [ ] Disable Heat Predictor and run all five prepare scripts sequentially.
- [ ] Verify Vdbench completion, anchor sizes, Ceph free space and sample reads.
- [ ] Enable Heat Predictor and verify reset status is zero.

### Task 6: D0/D1/D2 Formal Runs

**Files:**
- Create: `/home/chris/ceph-test/new_workload/hp_runs/reports/<timestamp>_otsu_data_source_matrix/`

**Interfaces:**
- Produces: 15 final statuses, 30-second time series, metadata, raw Vdbench
  outputs, `results.tsv` and a Chinese `REPORT.md`.

- [ ] Run D0 across the five workloads with one Luna monitor.
- [ ] Build/install D1 and run the same five workloads with one Luna monitor.
- [ ] Build/install D2 and run the same five workloads with one Luna monitor.
- [ ] Recompute metrics, validate queue/drop invariants and write the report.
- [ ] Restore the selected final profile explicitly and record it in the report.
