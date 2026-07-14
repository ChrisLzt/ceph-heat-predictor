# Strict Access Window And EQ Decoupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the pending-EQ proxy with an exact 20-second per-object access count, allow deadline labeling to proceed independently of prediction completion, and raise per-OSD EQ/LRU capacities to 1,000,000.

**Architecture:** Maintain independent 5-second and 20-second timestamp deques with per-object counters. Store pending evaluations in a stable `std::list`; a deadline cursor labels every due node in FIFO order even when prediction is unfinished, while training is emitted only after both prediction and label are complete. The prediction thread retains a stable list iterator, and reset safety continues to rely on the existing shared `reset_mutex` lifetime guard.

**Tech Stack:** C++17, `std::list`, monotonic `steady_clock` nanoseconds, Ceph OSD perf/MGR aggregation, standalone algorithm and performance probes.

## Global Constraints

- The future-label window remains 20 seconds and heat decay remains I/O-based.
- The short feature window remains 5 seconds and excludes the current I/O.
- The exact long feature window is 20 seconds and excludes the current I/O.
- A dropped EQ sample still updates heat and both access windows.
- Label generation must not wait for prediction completion.
- Training, confusion-matrix statistics, and prediction calibration require both label and prediction.
- `HP_PENDING_EVALUATION_CAPACITY` and `HP_LRU_CAPACITY` are both 1,000,000 per OSD.

---

### Task 1: Exact 20-second access count

**Files:**
- Modify: `src/heatpredictor/hp_types.h`
- Modify: `src/heatpredictor/hp_features.h`
- Modify: `src/heatpredictor/hp_evaluation_queue.h`
- Test: `test_sh/hp_algorithm_probe.cc`

**Interfaces:**
- Produces: `PredictionSample::long_window_access_count` and `ObjectHeatState::long_window_access_count`.
- Consumes: `HP_FUTURE_LABEL_WINDOW_NS` as the exact long-window duration.

- [ ] Add a failing probe where an EQ with capacity one drops the second evaluation sample, but a third access still observes two prior accesses in its strict 20-second count.
- [ ] Add a failing boundary probe proving events at age `>= 20s` are excluded and the current I/O is not included.
- [ ] Run the algorithm probe and confirm the failures are caused by the old pending-EQ proxy.
- [ ] Add a 20-second access-event deque, expire it by monotonic timestamp before feature capture, and update per-object long counters independently of EQ admission.
- [ ] Replace pending-count feature inputs with `long_window_access_count`, while retaining `pending_evaluation_count` only for EQ lifecycle ownership.
- [ ] Run the algorithm probe and confirm both exact-window tests pass.

### Task 2: Deadline/prediction rendezvous

**Files:**
- Modify: `src/heatpredictor/hp_evaluation_queue.h`
- Modify: `src/heatpredictor/heat_predictor.h`
- Test: `test_sh/hp_algorithm_probe.cc`
- Test: `test_sh/hp_performance_probe.cc`

**Interfaces:**
- Produces: stable `PredictionReservation` handles, label completion independent of prediction completion, and optional evaluated samples returned by either completion path.
- Consumes: the existing `eq_mutex`, `reset_mutex`, expiry worker, and training queue.

- [ ] Replace the old head-blocking test with a failing probe where the first due sample has no prediction, the second due sample is prediction-complete, and the second sample is emitted on time.
- [ ] Add a failing probe proving a label-complete but prediction-incomplete node no longer consumes EQ admission capacity.
- [ ] Run the algorithm probe and confirm both tests fail under the current FIFO `prediction_complete` gate.
- [ ] Replace the pending deque with a stable list and deadline cursor. Record actual-label fields in the node and decrement EQ/object pending counts when the deadline is processed.
- [ ] Make `complete_prediction()` return an evaluated sample only when the deadline side has already completed; make deadline processing emit only nodes whose prediction side has completed.
- [ ] Update the foreground prediction path so either completion order feeds the same statistics, calibrator, and training queue exactly once.
- [ ] Remove `waiting_prediction` from expiry scheduling; the worker waits only for the next unlabeled deadline.
- [ ] Run algorithm and performance probes, including reset/concurrency coverage.

### Task 3: Capacity and documentation

**Files:**
- Modify: `src/heatpredictor/hp_config.h`
- Modify: `codex_docs/CODEX_CEPH.md`
- Modify: `codex_docs/CODEX_CEPH_TODO.md`
- Test: `test_sh/hp_algorithm_probe.cc`

**Interfaces:**
- Produces: one-million-entry EQ and LRU limits, updated lifecycle documentation, and retained Y0/Y1 future-label experiments.

- [ ] Change both capacity constants and their parameter probes from 100,000 to 1,000,000.
- [ ] Document exact 5/20-second counters, the deadline/prediction rendezvous, memory implications, and the fact that evaluation drops do not drop Ceph I/O.
- [ ] Keep Y0 future-total-heat and Y1 future-incremental-heat as later workload experiments rather than selecting one in this change.
- [ ] Run `git diff --check` and scan for stale pending-count feature names and `waiting_prediction` references.

### Task 4: Build and runtime verification

**Files:**
- Verify: all modified production and probe files.

**Interfaces:**
- Consumes: the completed exact-window and rendezvous implementation.
- Produces: installed OSD/MGR binaries and live status evidence.

- [ ] Compile and run `hp_algorithm_probe` and `hp_performance_probe` from current sources.
- [ ] Run `sudo env CCACHE_TEMPDIR=/tmp ninja -j64`, `sudo ninja install`, and `sudo ldconfig`.
- [ ] Restart OSD and MGR services, wait for all PGs to become `active+clean`, then run `ceph osd hp reset`.
- [ ] Verify `hp_pending_io_count`, `hp_eval_drop_count`, service states, and MGR aggregation.
