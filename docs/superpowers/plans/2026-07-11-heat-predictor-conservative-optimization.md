# Heat Predictor Conservative Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove correctness and synchronization risks and reduce model-neutral overhead before comparing ARF `max_features=3/4/5`.

**Architecture:** Preserve the feature vector, labels, thresholds, model structure, prediction probabilities, and strict EQ index order. Optimize only ADWIN counters, EQ wakeups, training queue lifecycle, and reporting-only future-access quantiles; retain any performance change only after isolated A/B verification.

**Tech Stack:** C++17, Ceph OSD, Adaptive Random Forest, PBDS, algorithm/performance probes, ASan/UBSan/TSan.

## Global Constraints

- Do not change `NUM_FEATURES`, ARF parameters, heat thresholds, labels, or prediction decisions.
- Preserve reset/disable equivalence and prevent pre-reset samples from training a post-reset model.
- Use failing regression tests before production edits.
- Keep only optimizations with identical checksums/results and measurable benefit.

---

### Task 1: ADWIN long-run counter safety

**Files:**
- Modify: `src/heatpredictor/include/drift/ADWIN.h`
- Test: `test_sh/hp_algorithm_probe.cc`

**Interfaces:**
- Produces: `uint64_t AdaptiveWindowing::_calculate_bucket_size(size_t)` and 64-bit `width/tick` counters.

- [x] Add a regression test requiring row 40 to produce `1ULL << 40` and verifying existing detection signatures.
- [x] Run the probe and confirm failure caused by the current 32-bit shift.
- [x] Convert width-related arithmetic to `uint64_t`, remove unused `total_width`, and retain floating-point formulas via explicit conversion.
- [x] Run normal and sanitizer probes and confirm unchanged short-sequence signatures.

### Task 2: Conditional EQ waiter notification

**Files:**
- Modify: `src/heatpredictor/hp_evaluation_queue.h`
- Modify: `src/heatpredictor/heat_predictor.h`
- Modify: `test_sh/hp_performance_probe.cc`
- Test: `test_sh/hp_algorithm_probe.cc`

**Interfaces:**
- Produces: `bool EvaluationQueue::complete_prediction(PendingSlot*, double, int)`; true only when completing the full queue's oldest slot changes the wait predicate.

- [x] Extend staged-slot tests to require no notification for a non-oldest or not-full slot and notification for the full queue's oldest slot.
- [x] Run the probe and confirm the old `void` interface fails compilation.
- [x] Return the predicate transition from `complete_prediction()` while holding `eq_mutex`; notify only when true.
- [x] Extend the serialized/two-phase A/B probe with identical conditional notification.
- [x] Verify ordering, checksums, TSan, and concurrent throughput.

### Task 3: Bounded reset-safe training batches

**Files:**
- Modify: `src/heatpredictor/heat_predictor.h`
- Test: `test_sh/hp_algorithm_probe.cc`

**Interfaces:**
- Produces: training batches of at most `BATCH_SIZE` samples, with `reset_mutex` released between batches.

- [x] Add a test hook/count assertion proving one dequeue batch never exceeds 100 samples.
- [x] Run the probe and confirm the current whole-queue swap violates the bound.
- [x] Move at most `BATCH_SIZE` samples into a local queue under `train_queue_mutex`, using move semantics.
- [x] Preserve FIFO order and keep `reset_mutex` across each local batch so samples cannot cross reset generations.
- [x] Verify concurrent reset/enable tests and training checksums.

### Task 4: Remove model-neutral overhead

**Files:**
- Modify: `src/heatpredictor/heat_predictor.h`
- Modify: `src/heatpredictor/include/HoeffdingTreeClassifier.h`
- Modify: `src/heatpredictor/include/TreeBase.tpp`
- Test: `test_sh/hp_algorithm_probe.cc`

**Interfaces:**
- Removes: unused `HoeffdingTreeClassifier::classes` state.

- [x] Add compile-time/probe coverage showing prediction clones and trained probabilities remain identical without class-set state.
- [x] Move `TrainingSample` into and out of queues and reduce the capacity guard from `while` to `if`.
- [x] Remove the unused `classes` set and stale comments that claim known logical errors.
- [x] Verify model seed, clone, probability, and split tests.

### Task 5: Exact bounded future-access quantiles

**Files:**
- Create: `src/heatpredictor/hp_integer_quantile_window.h`
- Modify: `src/heatpredictor/heat_predictor.h`
- Modify: `test_sh/hp_performance_probe.cc`
- Test: `test_sh/hp_algorithm_probe.cc`

**Interfaces:**
- Produces: `HpIntegerQuantileWindow(capacity, max_value)`, `insert(uint64_t)`, `summary()`, and `clear()` with the same `HpDistributionSummary` output.

- [x] Add equivalence tests against `HpQuantileWindow` for fill, duplicate values, rolling eviction, max, and p50/p90/p95/p99.
- [x] Run the probe and confirm the new type is absent.
- [x] Implement a ring buffer plus Fenwick count tree, rejecting values above `max_value`.
- [x] Replace only hot/cold future-access windows; retain PBDS for future heat.
- [x] Add old/new insertion and summary A/B measurements; revert this task if it is not faster or output differs.

### Task 6: Integrated verification and documentation

**Files:**
- Modify: `CODEX_CEPH.md`
- Modify: `CODEX_CEPH_TODO.md`

- [x] Run algorithm and performance probes, ASan/UBSan, TSan, `git diff --check`, and full `ninja -j64`.
- [x] Install, run `ldconfig`, restart OSD/MGR, and verify HP reset plus cluster health.
- [x] Record retained mechanisms in `CODEX_CEPH.md`, keep it below 15 KB, and leave only `max_features` work in TODO.
