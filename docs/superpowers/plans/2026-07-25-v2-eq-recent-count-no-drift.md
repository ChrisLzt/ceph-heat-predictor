# V2 EQ Recent-Count Threshold and No-Drift Implementation Plan

> **For agentic workers:** Execute inline in the main agent. Subagents may only
> monitor long-running commands; they must not write code.

**Goal:** Use the existing EQ as the strict recent-ten-second threshold
population and disable ARF tree replacement.

**Architecture:** `EvaluationQueue` publishes each object's post-admission and
post-removal pending count to the existing object-equal threshold tracker.
Prediction samples the count and K before current admission. A no-op detector
disables ARF warning/drift actions while preserving online tree learning.

**Tech Stack:** C++17, Ceph OSD/MGR, deterministic C++ probes, Vdbench,
Trace/replay Python analysis.

## Global Constraints

- Work only on `dev`.
- Preserve the V2 future-ten-second label and three features.
- Current I/O is excluded from its own recent-count feature.
- EQ capacity remains `1,000,000` per OSD.
- Do not commit or push without an explicit current-task request.

---

### Task 1: Drive K from EQ Counts

**Files:**
- Modify: `src/heatpredictor/hp_evaluation_queue.h`
- Test: `test_sh/hp_algorithm_probe.cc`

- [ ] Add probes for post-admission vote increment, post-expiry decrement,
  cancellation, zero removal and current-I/O exclusion.
- [ ] Run the probe and confirm failure under completed-future-count voting.
- [ ] Publish `pending_evaluation_count` after each successful enqueue,
  expiration and cancellation.
- [ ] Remove completed-future-count threshold observations.
- [ ] Keep future labels based on each item's stored prediction-time K.
- [ ] Re-run algorithm and performance probes.

### Task 2: Disable ARF Drift

**Files:**
- Modify: `src/heatpredictor/include/drift/DetectorConcept.h`
- Modify: `src/heatpredictor/heat_predictor.h`
- Modify: `test_sh/hp_trace_replay.h`
- Test: `test_sh/hp_algorithm_probe.cc`
- Test: `test_sh/test_hp_trace_replay.cc`

- [ ] Add a model probe requiring zero warnings, drifts, promotions and
  discards while predictions and training remain operational.
- [ ] Run it against the current ADWIN production model and confirm failure.
- [ ] Add `NeverDriftDetector` and use it for both production detector slots.
- [ ] Reuse the same detector in the disabled replay profile.
- [ ] Re-run algorithm and replay tests.

### Task 3: Documentation and Correctness Gates

**Files:**
- Modify: `codex_docs/CODEX_CEPH.md`
- Modify: `codex_docs/CODEX_CEPH_TODO.md`
- Modify V2 Trace wording only where the threshold source changed.

- [ ] Update the stable/dev descriptions without changing CLI commands.
- [ ] Run `git diff --check`, algorithm, performance, replay and Python tests.
- [ ] Confirm threshold telemetry changes on EQ admission/removal and all ARF
  adaptation counters stay zero.

### Task 4: Build and Runtime Verification

- [ ] Run affected target build and `sudo ninja -j64`.
- [ ] Run `sudo ninja install`, `sudo ldconfig`, and restart OSD/MGR.
- [ ] Verify `ceph -s`, reset, OSD status, PerfCounters and MGR aggregation.

### Task 5: Five-Workload Comparison

- [ ] Run MapReduce, GraphChi, HPC, AI training and AI inference once.
- [ ] Save MGR status every 30 seconds and final status without extra drain.
- [ ] Compare online metrics with current V2 and offline `recent_disabled`.
- [ ] Save a Chinese report under the workload reports directory.
