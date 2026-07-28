# Time-Domain Heat And Maintenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move heat and freshness semantics to monotonic time, clean access windows without foreground I/O, and make late prediction completion O(1).

**Architecture:** `EvaluationQueue` owns all time-domain object state and exposes one combined maintenance deadline to the existing expiry worker. Stable list iterators rendezvous prediction and labeling. Otsu and snapshot responsiveness use elapsed-time dual triggers while capacities and batching remain count-based.

**Tech Stack:** C++17, `steady_clock`, `std::list`, `condition_variable`, Ceph perf/MGR integration, standalone probes.

## Global Constraints

- Heat retains `1/10` after 20 seconds.
- Short/long access windows remain 5/20 seconds and exclude the current I/O.
- H0/P0 remain active; H1/P1 remain TODO candidates.
- Do not add an `awaiting_prediction` timeout or independent capacity.
- Preserve lock order `reset_mutex -> eq_mutex` and never sleep while holding either lock.

---

### Task 1: Time-domain heat and features

**Files:**
- Modify: `src/heatpredictor/hp_config.h`
- Modify: `src/heatpredictor/hp_types.h`
- Modify: `src/heatpredictor/hp_features.h`
- Modify: `src/heatpredictor/hp_evaluation_queue.h`
- Test: `test_sh/hp_algorithm_probe.cc`

**Interfaces:**
- Produces: `last_access_time_ns`, `time_since_previous_access_ns`, and `decay_heat(...time_ns)`.
- Consumes: `HP_HEAT_DECAY_HORIZON_NS=20s` and monotonic timestamps already supplied to EQ.

- [x] Add probes proving heat retains `1/10` after 20 seconds and recency depends on elapsed time rather than intervening I/O count.
- [x] Run the probe and confirm failures reference missing time-domain names or old I/O-domain results.
- [x] Replace I/O-domain heat state/configuration with nanosecond timestamps and direct exponential decay.
- [x] Convert recency to `log2(1 + elapsed_seconds)` and update constructors/comparison helpers.
- [x] Re-run the complete algorithm probe.

### Task 2: Event-driven access cleanup

**Files:**
- Modify: `src/heatpredictor/hp_evaluation_queue.h`
- Modify: `src/heatpredictor/heat_predictor.h`
- Test: `test_sh/hp_algorithm_probe.cc`

**Interfaces:**
- Produces: a maintenance schedule covering EQ, 5-second, and 20-second deadlines.
- Preserves: foreground cleanup before feature capture and the existing expiry worker.

- [x] Add a probe that advances the worker clock without a new I/O and observes both access counters returning to zero/LRU eligibility.
- [x] Verify the probe fails because the current schedule only contains EQ deadlines.
- [x] Add access deadlines to the schedule and drain due access events in the worker.
- [x] Notify the worker only when an empty access queue gains its first event or an EQ head is created.
- [x] Re-run concurrency and idle-expiry probes.

### Task 3: Otsu and snapshot wall-clock responsiveness

**Files:**
- Modify: `src/heatpredictor/hp_config.h`
- Modify: `src/heatpredictor/hp_evaluation_queue.h`
- Modify: `src/heatpredictor/heat_predictor.h`
- Test: `test_sh/hp_algorithm_probe.cc`

**Interfaces:**
- Produces: score-space threshold state, one-second normalized H0 gain, and snapshot `500 samples OR 1 second` publication.

- [x] Add probes for idle threshold decay, elapsed-time EMA equivalence, time-triggered Otsu recomputation, and time-triggered snapshot publication.
- [x] Verify failures under fixed-per-observation behavior.
- [x] Store threshold heat with its timestamp, derive current heat on reads, and interpolate updates in score space.
- [x] Add one-second reference-interval EMA and Otsu recompute deadline.
- [x] Add the snapshot dual trigger without changing training batch size.
- [x] Re-run all threshold, snapshot, and reset probes.

### Task 4: O(1) prediction reservation

**Files:**
- Modify: `src/heatpredictor/hp_evaluation_queue.h`
- Modify: `src/heatpredictor/heat_predictor.h`
- Test: `test_sh/hp_algorithm_probe.cc`

**Interfaces:**
- Produces: `PredictionReservation` with a stable list iterator and direct completion/cancellation.

- [x] Add compile-time/API assertions that completion consumes an iterator-backed reservation rather than a node pointer.
- [x] Verify the old API fails the probe build.
- [x] Replace pointer lookup with direct iterator finalization and cancellation.
- [x] Keep the node alive until both completion flags are true.
- [x] Re-run late-prediction, fallback, reset, and concurrency probes.

### Task 5: Documentation and L2 verification

**Files:**
- Modify: `codex_docs/CODEX_CEPH.md`
- Modify: `codex_docs/CODEX_CEPH_TODO.md`
- Test: `test_sh/hp_algorithm_probe.cc`
- Test: `test_sh/hp_performance_probe.cc`

**Interfaces:**
- Documents: time-domain baseline and the intentionally deferred awaiting-prediction policy.

- [x] Update stable behavior and remove time-domain work from active TODO.
- [x] Run normal, sanitizer, thread-sanitizer, and performance probes.
- [x] Run `git diff --check` and validate local Markdown links.
- [x] Build/install with the operations manual, run `ldconfig`, restart affected OSD/MGR services sequentially, and wait for `active+clean`.
- [x] Reset Heat Predictor and verify OSD/MGR counters and thresholds.
