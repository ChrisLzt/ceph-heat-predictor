# Expected Future Heat Feature Implementation Plan

> **Execution note:** Work in the existing dirty `dev` worktree because the
> feature depends on the uncommitted C4/Trace implementation. Do not commit or
> push unless explicitly requested.

**Goal:** Add `expected_future_heat_margin` as a fourth online model feature so
the classifier can distinguish objects whose recent access rate implies future
heating or cooling.

**Architecture:** Keep two time-aware EWMA access rates in each existing
`ObjectHeatState`. On each object access, update both rates from the previous
access interval, extrapolate one constant non-negative future access rate, and
analytically project the existing exponential heat equation to the 10-second
label deadline. Store only the resulting scalar in `PredictionSample`; do not
change real heat, Otsu votes, labels, or introduce timers.

**Tech Stack:** C++17, existing Ceph heat predictor probes, Ninja, systemd,
Vdbench, MGR `osd hp status` snapshots.

---

## Task 1: Lock Down Feature Semantics

**Files:**
- Modify: `test_sh/hp_algorithm_probe.cc`
- Modify: `test_sh/hp_performance_probe.cc`

1. Add failing tests for time-aware EWMA updates, non-negative trend forecast,
   analytical future heat projection, and final feature ordering.
2. Add an integration test proving `EvaluationQueue::prepare_features()`
   produces increasing projected margins for repeated short-interval accesses.
3. Compile/run the probe and confirm it fails because production helpers/state
   do not exist yet.

## Task 2: Implement the Fourth Feature

**Files:**
- Modify: `src/heatpredictor/hp_config.h`
- Modify: `src/heatpredictor/hp_types.h`
- Modify: `src/heatpredictor/hp_features.h`
- Modify: `src/heatpredictor/hp_evaluation_queue.h`
- Modify: `src/heatpredictor/heat_predictor.h`

1. Add fixed constants: 2-second fast EWMA, 10-second slow EWMA, trend gain
   `0.5`, and a fourth base feature.
2. Add two rates to `ObjectHeatState` and one projected-margin scalar to
   `PredictionSample`.
3. Implement numerically stable O(1) rate updates and heat projection.
4. Update rates before replacing `last_access_time_ns`; append the projected
   margin after the existing three C4 inputs.
5. Include all feature-defining constants in the Trace configuration hash.

## Task 3: Verify and Deploy

**Files:**
- Verify: `test_sh/hp_algorithm_probe.cc`
- Verify: `test_sh/hp_performance_probe.cc`
- Verify: `test_sh/test_hp_trace_replay.cc`

1. Run algorithm, performance, replay, and Python analysis tests.
2. Build the affected Ceph target(s), then run the repository's full install,
   `ldconfig`, and service restart workflow.
3. Confirm the cluster is `active+clean`, reset Heat Predictor state, and check
   for dropped evaluations/training samples.

## Task 4: Online Comparison

**Files:**
- Create: `/home/chris/ceph-test/new_workload/hp_runs/reports/<timestamp>_expected_future_heat_feature/REPORT.md`

1. Run each selected formal workload once with the four-feature code, using the
   existing datasets and unchanged workload parameters.
2. Save 10-second MGR snapshots and one final `hp_status.json` per workload.
3. Compare against the existing true online C4 report, including Accuracy,
   Balanced Accuracy, Precision, Recall, predicted/actual hot percentages,
   cold start, hotspot transition, steady state, and high-confidence errors.
4. State explicitly that old three/six-feature binary Trace cannot strictly
   replay the new EWMA state; online comparison therefore includes normal
   single-run variance.

## Task 5: Document the Result

**Files:**
- Modify: `codex_docs/CODEX_CEPH_TODO.md`
- Create: the report from Task 4

1. Record implementation status and exact tested parameters.
2. Accept the feature only if aggregate Accuracy and Balanced Accuracy improve,
   no workload regresses beyond the existing `0.2` percentage-point gate, and
   transition errors do not worsen materially.
