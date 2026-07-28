# Future Added Heat Otsu Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace object-total-heat Otsu with a fixed, time-bounded histogram of completed EQ samples' future-20-second added heat.

**Architecture:** `HpOtsuHistogram` becomes a deep module that owns a fixed 10000-bin aggregate and a 120-slot monotonic-time ring. `EvaluationQueue` labels a due sample before observing it, then lets the histogram update the Otsu candidate without exposing ring maintenance to callers.

**Tech Stack:** C++17, Ceph monotonic nanosecond timestamps, standalone `hp_algorithm_probe`, OSD perf counters and MGR aggregation.

## Global Constraints

- Histogram bins: 10000; `log1p` score width: `0.01`.
- History: 120 one-second slots, keyed by absolute monotonic second.
- Aggregate counts: `uint64_t`; slot counts: `uint32_t`.
- Label window: 20 seconds; label before observe.
- H0 EMA remains `0.10`; P0 threshold remains `0.50`.
- Existing unrelated dirty-worktree changes must remain intact.

---

### Task 1: Timed fixed histogram

**Files:**
- Modify: `src/heatpredictor/hp_config.h`
- Modify: `src/heatpredictor/hp_otsu_histogram.h`
- Test: `test_sh/hp_algorithm_probe.cc`

**Interface:**
- `bool observe(double added_heat, uint64_t sample_time_ns, uint64_t now_ns)`
- `void advance_to(uint64_t now_ns)`
- `std::optional<HpOtsuResult> otsu_result() const`
- `size_t size() const`, `size_t bin_count() const`, `void clear()`

- [ ] Add probe cases for bin mapping, a skipped second, 120-second expiry and late-sample rejection.
- [ ] Run the probe and confirm it fails because the timed fixed interface is absent.
- [ ] Implement fixed heap storage, tagged time slots and three-pass Otsu.
- [ ] Run the probe and confirm the histogram cases pass.

### Task 2: EvaluationQueue integration

**Files:**
- Modify: `src/heatpredictor/hp_evaluation_queue.h`
- Test: `test_sh/hp_algorithm_probe.cc`

**Interface:**
- Completed samples feed `future_window_added_heat` after `actual_label` is set.
- Object heat order statistics remain independent and feed only heat percentile.

- [ ] Add a probe showing the sample that creates a new Otsu threshold was labeled with the preceding threshold.
- [ ] Run the probe and confirm the current object-total-heat implementation fails it.
- [ ] Remove Otsu updates from `record_object_heat` and observe completed added heat in deadline finalization.
- [ ] Convert candidate/effective threshold score handling to `log1p`/`expm1` without idle threshold decay.
- [ ] Verify reset clears the timed histogram and threshold state.

### Task 3: Statistics and documentation

**Files:**
- Modify: `src/heatpredictor/hp_types.h`
- Modify: `src/heatpredictor/heat_predictor.h`
- Modify: `src/osd/ObjectHeatPredictor.cc`
- Modify: `src/mgr/DaemonServer.cc`
- Modify: `codex_docs/CODEX_CEPH.md`
- Modify: `codex_docs/CODEX_CEPH_TODO.md`
- Modify: `codex_docs/todo/OTSU_THRESHOLD.md`
- Test: `test_sh/hp_algorithm_probe.cc`

- [ ] Rename histogram object count to retained sample count without changing perf enum ordering.
- [ ] Document the completed data-source change and leave total-heat/min-10 as a pending control.
- [ ] Scan source and active docs for stale object-total-heat Otsu descriptions.

### Task 4: Verification

- [ ] Run the standalone algorithm probe.
- [ ] Build the affected Ceph target at L1.
- [ ] Run `git diff --check` and verify active Markdown links.
- [ ] Review the final diff for accidental changes outside the approved files.
