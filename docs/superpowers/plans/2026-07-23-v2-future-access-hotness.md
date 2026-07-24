# V2 Future-Access Hotness Implementation Plan

The approved semantics are defined in
[V2 Future-Access Hotness Design](../specs/2026-07-23-v2-future-access-hotness-design.md).
This document contains only implementation order, affected files, and
verification gates. Repository agent and Git rules come from `AGENTS.md`.

## Goal

For every I/O, predict whether the same object will receive at least the
prediction-time threshold `K` accesses in `(t, t + 10s]`.

Keep one EQ item per I/O, one latest positive Otsu vote per object, and the
existing background ARF training and snapshot architecture. Correct causal and
concurrent behavior is the first acceptance criterion; workload metrics are
reported but are not initial correctness gates.

## Task 0: Establish the Baseline

Confirm `dev` is based on production `main`, identify its dev-only Trace/tests,
record the tested configuration without embedding Git object IDs, and run
`hp_algorithm_probe` plus `hp_performance_probe`. Follow `AGENTS.md` and the
operations manual; do not modify or publish Git history without explicit user
approval.

## Task 1: Add the Future-Access Threshold Module

**Files**

- Create `src/heatpredictor/hp_future_access_threshold.h`
- Modify `src/heatpredictor/hp_config.h`
- Test in `test_sh/hp_algorithm_probe.cc`

Implement a self-contained `HpFutureAccessThreshold` that owns:

- one latest positive observation per object;
- zero-observation removal and latest-deadline batch deduplication;
- the fixed `log2(1 + count)` histogram and upper-clamp accounting;
- Otsu recomputation, score EMA, integer `K` conversion, and
  sparse/tracking/holding state;
- object-order capacity protection, current threshold, maintenance deadline,
  status, and reset.

EQ may submit observations and read threshold/status only. It must not access
the module's containers.

Add deterministic probes for:

- bin boundaries, score conversion, and exact upper-clamp telemetry;
- insert, replace, remove, capacity eviction, and one vote per object;
- same-object batch deduplication by latest deadline;
- sparse `K=1`, readiness, tracking, holding, and hold expiry;
- object-change and elapsed-time recomputation triggers;
- first publication, time-normalized EMA, safe integer conversion, and reset.

## Task 2: Change Samples, Features, and Exact Labels

**Files**

- Modify `src/heatpredictor/hp_types.h`
- Modify `src/heatpredictor/hp_features.h`
- Modify `src/heatpredictor/hp_evaluation_queue.h`
- Modify `src/heatpredictor/heat_predictor.h`
- Modify `src/heatpredictor/hp_config.h`
- Test in `test_sh/hp_algorithm_probe.cc`

Each item stores:

```cpp
uint64_t future_access_threshold_at_prediction;
uint64_t tracked_access_count_after_current_access;
```

The three baseline features are:

```text
log2p1(past_10s_access_count) - log2p1(K_at_prediction)
previous_access_interval_encoded
current_heat_log2p1
```

At item creation:

1. Expire due items.
2. Read current `K`.
3. Sample the strict past-window count and previous-access interval.
4. Account the current access and capture the post-access tracked count.
5. Build features and predict.
6. Enqueue the item and increment its object's pending count.

At deadline:

```text
future_access_count =
    tracked_access_count_at_deadline -
    tracked_access_count_after_current_access
```

Use the item's stored `K` for its label. Every completed item updates metrics
and training independently. Submit completed observations to the threshold
module after per-item evaluation so object-level deduplication cannot alter
I/O-weighted labels.

Add probes for:

- accesses before, exactly at, and after a deadline;
- repeated I/Os with multiple pending items for one object;
- simultaneous expirations for multiple objects;
- threshold changes after enqueue;
- sparse-mode samples and cold-start zero-vote predictions;
- zero observations and same-batch repeated-object completions.

Remove obsolete heat-threshold history and alternate label paths after proving
that no caller remains.

## Task 3: Preserve Runtime and Concurrency Contracts

**Files**

- Modify `src/heatpredictor/heat_predictor.h`
- Modify `src/heatpredictor/hp_evaluation_queue.h`
- Test in `test_sh/hp_algorithm_probe.cc`
- Test in `test_sh/hp_performance_probe.cc`

Keep ARF construction, scaler, 25 trees, class weight `1.0`, prediction
threshold `0.50`, background batch training, and snapshot publication
unchanged.

Verify that:

- sparse and tracking samples use the same prediction/training path;
- an untrained model's valid zero vote predicts cold but remains evaluable;
- reset safely coordinates with in-flight prediction and background training;
- EQ, LRU, and threshold capacities stay independent;
- concurrent update/reset cannot underflow counters, duplicate deletion,
  invalidate iterators, or leave stale pending counts.

## Task 4: Update Telemetry and Lifecycle End to End

**Files**

- Modify `src/heatpredictor/hp_telemetry.h`
- Modify `src/heatpredictor/heat_predictor.h`
- Modify `src/osd/ObjectHeatPredictor.cc`
- Modify `src/osd/ObjectHeatPredictor.h` if required
- Modify `src/mgr/ObjectHeatPredictorStatus.*`
- Modify `src/mgr/DaemonServer.cc`
- Modify `src/mgr/MgrCommands.h`
- Modify `src/osd/OSD.cc` only if lifecycle commands require it

Update the Heat Predictor status, OSD PerfCounter enum/declaration/update/reset
order, MGR parser, aggregation formulas, and JSON grouping together.

Expose:

- approved confusion-matrix and prediction metrics;
- current/candidate `K` and threshold state;
- positive, zero, clamped, sparse, and holding counters;
- cluster K minimum, maximum, average, and state counts;
- existing queue, model, drop, latency, and operation counters.

Remove V1-only threshold fields. Add a status-contract probe that fails when an
approved OSD field is missing or incorrectly grouped by MGR.

Reset, enable, and disable must clear all V2 threshold, EQ, model, and telemetry
state. The existing CLI commands remain unchanged.

## Task 5: Verify Correctness and Integration

Run checks in the levels defined by `AGENTS.md` and the operations manual:

1. Formatting/link checks and deterministic algorithm probes.
2. Sanitizer, concurrency, Trace/replay, and affected-target tests.
3. Full build, install, `ldconfig`, and affected Ceph service restart.
4. Confirm healthy/clean cluster state.
5. Reset and verify sparse `K=1`, empty EQ, zero Otsu population, and aligned
   OSD/MGR status.
6. Generate controlled I/O and verify transition to tracking.
7. Verify reset-on-enable/disable and compare real-time OSD status,
   PerfCounters, and MGR aggregation.

## Task 6: Run Workloads and Finalize Documentation

Use `/home/chris/ceph-test/new_workload/` after correctness checks pass.

1. Validate workload configuration and data consistency.
2. Run each required workload once and capture MGR status at the established
   interval.
3. Capture final status when the configured workload ends; do not add a
   10-second drain. Report final pending items as right-censored.
4. Report Accuracy, Balanced Accuracy, Precision, Recall, predicted/actual hot
   percentages, K/state distribution, drops, capacity events, and latency.
5. Treat anomalous results as findings; do not repeat automatically.

After verification:

- move implemented behavior from `CODEX_CEPH_TODO.md` into `CODEX_CEPH.md`;
- keep exact runtime constants in code and describe only stable behavior in
  documentation;
- update the operations manual only if commands changed;
- run `git diff --check` and validate modified Markdown links;
- leave changes uncommitted unless the user explicitly requests a Git action.
