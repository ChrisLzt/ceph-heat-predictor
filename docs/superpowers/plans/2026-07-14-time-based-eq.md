# Time-Based EQ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate every admitted I/O after a fixed 20-second wall-clock window while preserving the existing I/O-index heat model, features, Otsu logic, prediction model, and training policy.

**Architecture:** `EvaluationQueue` stores a monotonic enqueue timestamp in each pending slot and drains all ready slots whose age reached 20 seconds. `HeatPredictor` performs a foreground deadline guard in the same EQ critical section as feature preparation, while a dedicated expiry worker sleeps until the exact head deadline so idle workloads still expire without coupling scheduling to training. A bounded admission limit drops only evaluation samples, never foreground predictions or heat-state updates.

**Tech Stack:** C++17, `std::chrono::steady_clock`, `std::condition_variable`, existing EQ/reset mutexes, Ceph perf counters/MGR formatter, standalone algorithm probe.

## Global Constraints

- Keep `HP_EVALUATION_WINDOW=10000` as the I/O-domain heat-decay scale.
- Add a separate 20-second EQ duration; do not time-normalize heat decay, features, short window, Otsu, or prediction calibration.
- Use `steady_clock`, not wall clock.
- Bound pending evaluation samples at 100000 per OSD; when full, skip only new label/training admission.
- Never block foreground OSD I/O waiting for a 20-second deadline.
- Drain expired entries before applying the current I/O so an access after the deadline is not included in the old label.
- Preserve OSD perf enum/declaration/update/reset/output ordering.
- Do not hold `reset_mutex` or `eq_mutex` while the expiry worker sleeps.
- Preserve lock order `reset_mutex -> eq_mutex` in foreground, expiry, and reset paths.

---

### Task 1: Time-window EQ behavior

**Files:**
- Modify: `src/heatpredictor/hp_config.h`
- Modify: `src/heatpredictor/hp_evaluation_queue.h`
- Test: `test_sh/hp_algorithm_probe.cc`

**Interfaces:**
- Produces: `EvaluationQueue::expire_ready(uint64_t now_ns, uint64_t evaluation_index)` returning `std::vector<EvaluatedItem>`.
- Produces: timestamp-aware `reserve_prediction()`/`enqueue()` and nullable reservations when admission is full.
- Produces: `evaluation_drop_count()`.

- [x] Add probe cases that demonstrate no expiry before the configured duration, expiry exactly at its deadline, batch expiry, out-of-order prediction readiness, and bounded admission.
- [x] Compile and run the probe; verify failure is caused by missing time-window APIs/constants.
- [x] Add:

```cpp
static constexpr uint64_t HP_EVALUATION_WINDOW_NS = 20ULL * 1000 * 1000 * 1000;
static constexpr size_t HP_EVALUATION_MAX_PENDING = 100000;
```

- [x] Store `enqueue_time_ns` in `PendingSlot`, extract existing front-label logic into one helper, and drain every ready expired front entry.
- [x] Restore a non-admitted object's idle LRU state so admission drops cannot leak `heat_map` entries.
- [x] Re-run the complete probe and confirm all EQ tests pass.

### Task 2: Predictor lifecycle and observability

**Files:**
- Modify: `src/heatpredictor/heat_predictor.h`
- Modify: `src/heatpredictor/hp_types.h`
- Modify: `src/osd/ObjectHeatPredictor.cc`
- Modify: `src/mgr/DaemonServer.cc`
- Test: `test_sh/hp_algorithm_probe.cc`

**Interfaces:**
- Produces: `hp_eval_drop_count` in OSD perf and MGR `samples` output.
- Preserves: `hp_io_count = hp_labeled_io_total + hp_pending_io_count + hp_eval_drop_count` after all in-flight predictions complete.

- [x] Add failing lifecycle/concurrency assertions showing that more than 10000 fast I/Os remain pending rather than expiring by count.
- [x] Add a monotonic nanosecond helper and drain expired entries before current feature preparation.
- [x] Add idle expiry scheduling; the final implementation uses the dedicated worker in Task 4.
- [x] Remove the old EQ capacity condition-variable wait; a full evaluation queue must not throttle prediction.
- [x] Export/reset/aggregate `hp_eval_drop_count` in matching OSD perf order and MGR samples order.
- [x] Run the probe and compile `ceph-osd` and `ceph-mgr` affected targets.

### Task 3: Documentation and deployment verification

**Files:**
- Modify: `codex_docs/CODEX_CEPH.md`
- Modify: `codex_docs/CODEX_CEPH_TODO.md`
- Delete: `codex_docs/todo/TIME_BASED_EQ.md` after its completed behavior is moved to stable documentation.

**Interfaces:**
- Documents the hybrid contract: time-based EQ expiry plus I/O-index heat semantics.

- [x] Document the 20-second window, 100000 admission limit, dedicated-worker expiry, and revised conservation formula.
- [x] Remove the completed time-EQ item from the active TODO index.
- [x] Run `git diff --check` and validate local Markdown links.
- [x] Build and run the normal and sanitizer algorithm probes.
- [x] Run the complete build/install flow from `codex_docs/CEPH_OPERATIONS_MANUAL.md`, then `sudo ldconfig` and restart OSD/MGR.
- [x] Wait for `active+clean`, reset Heat Predictor, and verify new OSD/MGR output with no drop under idle conditions.

### Task 4: Dedicated expiry scheduling

**Files:**
- Modify: `src/heatpredictor/heat_predictor.h`
- Modify: `src/heatpredictor/hp_evaluation_queue.h`
- Test: `test_sh/hp_algorithm_probe.cc`
- Document: `codex_docs/CODEX_CEPH.md`

**Interfaces:**
- Produces: `EvaluationQueue::expiry_schedule(uint64_t now_ns)` describing empty,
  waiting-for-deadline, waiting-for-ready, or due state without mutating EQ.
- Produces: `HeatPredictor::expiry_worker()` and a wake-sequence based
  `notify_expiry_worker()`.
- Preserves: one FIFO pending queue, foreground-before-current-I/O expiry, and
  unchanged training samples.

- [x] Add probe assertions for expiry-state transitions and for expiration
  before a current access mutates `heat_map`; run the probe and confirm failure
  due to the missing scheduling interface.
- [x] Add the minimal queue scheduling query without adding a second queue or
  changing label calculation.
- [x] Add the dedicated worker, separate wait mutex/CV, wake sequence, and
  start/stop/reset notifications.
- [x] Merge foreground expiry and current feature preparation into one
  `eq_mutex` critical section; keep snapshot prediction outside the lock.
- [x] Remove the training worker's 50 ms EQ polling and leave it blocked on its
  own training-queue condition variable.
- [x] Run the complete probe, normal compile, ASan/UBSan, and TSan attempt.
- [x] Build/install/restart affected Ceph services and verify idle 20-second
  expiry, reset, status conservation, and `active+clean`.
- [x] Update stable documentation and run `git diff --check` plus Markdown-link
  validation.
