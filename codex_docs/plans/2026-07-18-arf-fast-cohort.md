# ARF Fast Cohort Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify whether a small rotating cohort of recent-sample ARF trees materially improves cold-to-hot recall when detector sensitivity alone has failed.

**Architecture:** Keep the ensemble at 25 trees and reserve its last five trees as an optional fast cohort. In an enabled experiment, replace one fast tree in round-robin order every `lifetime / 5` labeled samples so that steady-state fast-tree ages span the configured lifetime; keep the first 20 trees, features, threshold, sample weight, snapshot cadence, and normal ADWIN behavior unchanged.

**Tech Stack:** C++17, existing ARF/Trace replay, Ceph local algorithm probes, Python standard-library result aggregation.

## Global Constraints

- Work on `dev`; do not commit or push.
- All edits and result decisions are made by the main agent; a Luna subagent may only monitor a long-running command read-only.
- Default fast-tree count and lifetime are zero, preserving the current baseline.
- A periodic replacement deletes the selected active tree and its background tree, resets its metric and warning/drift detectors, and uses a deterministic new seed.
- Gate: mean Test C2H Recall increases by at least 10 percentage points, Accuracy decreases by no more than 1 percentage point, and Balanced Accuracy does not decrease.

---

### Task 1: Fast-cohort lifecycle

**Files:**
- Modify: `src/heatpredictor/include/ARFClassifier.h`
- Modify: `src/heatpredictor/include/ArfAdaptationTelemetry.h`
- Test: `test_sh/hp_algorithm_probe.cc`

**Interfaces:**
- Consumes: existing ARF constructor parameters and `ArfAdaptationTelemetry`.
- Produces: optional `fast_model_count`, `fast_model_lifetime_samples`, deterministic round-robin replacement, and `fast_model_reset_count` telemetry.

- [x] Add failing tests for invalid configuration, disabled-baseline behavior, and exact round-robin reset count.
- [x] Run the focused algorithm probe and confirm failure because the constructor and telemetry do not yet support the fast cohort.
- [x] Implement validation and replacement with `ceil(lifetime / fast_model_count)` as the rotation interval.
- [x] Re-run the focused probe and confirm all lifecycle tests pass.

### Task 2: Config and replay observability

**Files:**
- Modify: `src/heatpredictor/hp_config.h`
- Modify: `src/heatpredictor/heat_predictor.h`
- Modify: `test_sh/hp_trace_replay.cc`
- Modify: `test_sh/test_hp_trace_replay.cc`

**Interfaces:**
- Consumes: the fast-cohort constructor parameters and telemetry from Task 1.
- Produces: override-safe `HP_ARF_FAST_MODEL_COUNT_VALUE` and `HP_ARF_FAST_MODEL_LIFETIME_SAMPLES_VALUE`, plus final replay summary fields.

- [x] Add failing assertions that the default profile has zero fast trees and zero resets.
- [x] Pass config values through `HeatPredictor::make_model` and print configured count, lifetime, and observed reset count in replay summaries.
- [x] Run replay unit tests and verify deterministic snapshots remain unchanged when the feature is disabled.

### Task 3: Controlled offline profiles

**Files:**
- Create: `/home/chris/ceph-test/new_workload/hp_runs/reports/<timestamp>_arf_fast_cohort/REPORT.md`
- Modify: `codex_docs/CODEX_CEPH_TODO.md`

**Interfaces:**
- Consumes: the six projected no-migration OSD Trace files already archived by the R0/R1/R2 experiment.
- Produces: one result row per profile/workload and a Chinese decision report.

- [x] Build `F0` with no fast trees, `F1` with five fast trees and 5000-sample lifetime, and `F2` with five fast trees and 10000-sample lifetime.
- [x] Replay MapReduce, GraphChi, and HPC WRF exactly once per profile using identical traces and build flags.
- [x] Calculate Accuracy, Balanced Accuracy, C2H Recall, C2C Specificity, predicted-hot ratio, runtime, and reset/adaptation counts.
- [x] Reject candidates that miss any gate or improve C2H only by broadly increasing hot predictions; keep defaults disabled unless a candidate passes.

### Task 4: Verification and decision

**Files:**
- Modify: `codex_docs/CODEX_CEPH_TODO.md`

**Interfaces:**
- Consumes: Task 3 report and fresh verification output.
- Produces: a documented accept/reject decision without publishing Git changes.

- [x] Run focused C++ probes, replay tests, affected Python tests, `ninja ceph-osd ceph-mgr -j64`, and `git diff --check`.
- [x] Confirm `F0` reproduces the prior baseline within deterministic replay expectations.
- [x] Record whether recent-sample forgetting provides a useful upper bound; if not, stop ARF adaptation tuning and return to target/causal-signal design.
