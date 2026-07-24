# Future Added Heat Label Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Label each I/O by heat added by later accesses during its 20-second future window and change the five formal Vdbench workloads to 4 MiB requests.

**Architecture:** Keep total object heat as prediction-time state, but store its value at EQ admission and subtract its deadline-decayed contribution from the object's deadline total heat. Expose the resulting quantity consistently as future added heat. Keep the current object-total-heat Otsu source unchanged in this step and document the remaining threshold-population mismatch.

**Tech Stack:** C++17, the standalone `hp_algorithm_probe`, Ceph OSD perf/MGR aggregation, Oracle Vdbench configuration files.

## Global Constraints

- The future label window remains exactly 20 seconds.
- The current I/O and all pre-window heat are excluded from future added heat.
- Heat accumulated by later accesses is decayed to the sample deadline before labeling.
- Otsu remains object-total-heat based until a separate completed-sample threshold design is selected.
- Only the five formal Vdbench test workloads change to `xfersize=4m`; data preparation is unchanged.

---

### Task 1: Future-added-heat label

**Files:**
- Modify: `src/heatpredictor/hp_evaluation_queue.h`
- Modify: `src/heatpredictor/hp_types.h`
- Modify: `src/heatpredictor/heat_predictor.h`
- Test: `test_sh/hp_algorithm_probe.cc`

- [x] Add a failing test proving an I/O with no later access has zero future added heat.
- [x] Add coverage proving a later access contributes only its deadline-decayed heat.
- [x] Subtract the admission heat decayed from admission time to deadline from total deadline heat.
- [x] Rename evaluated-sample and in-memory statistics from future total heat to future added heat.
- [x] Compile and run `hp_algorithm_probe`.

### Task 2: OSD/MGR reporting and documentation

**Files:**
- Modify: `src/osd/ObjectHeatPredictor.cc`
- Modify: `src/mgr/DaemonServer.cc`
- Modify: `codex_docs/CODEX_CEPH.md`
- Modify: `codex_docs/CODEX_CEPH_TODO.md`

- [x] Rename exported fields and descriptions to `future_added_heat` without changing counter order.
- [x] Document the selected future-added label and the pending Otsu input redesign.
- [x] Run source scans for stale `future_total_heat` names.

### Task 3: Formal Vdbench request size

**Files:**
- Modify: the five formal workloads under `/home/chris/ceph-test/new_workload/*/rendered/run_test.vdb` and their generator inputs where present.

- [x] Replace test-path `xfersize=1m` with `xfersize=4m`.
- [x] Confirm preparation configurations were not changed.
- [x] Run each workload's model validator or render check when provided.

### Task 4: Verification

- [x] Run `hp_algorithm_probe` from current sources.
- [x] Run `git diff --check` in both repositories.
- [x] Verify no formal test FWD still declares `xfersize=1m`.
