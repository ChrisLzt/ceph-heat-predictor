# HP Trace Baseline Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic schema-v1 ARF replay tool and stop at a strict MapReduce single-OSD parity gate before any feature ablation.

**Architecture:** A C++ deep module reads one fixed-layout Trace, converts evaluated records into prediction/training events, replays the current model and snapshot policy, and returns per-record and aggregate parity. A thin C++ CLI writes deterministic TSV; a Python reporter performs phase joins and emits the gate report.

**Tech Stack:** C++17, existing Heat Predictor model headers, Python 3.10 standard library, existing binary Trace/CSV/metadata.

## Global Constraints

- Work on `/home/chris/ceph-heat-predictor` branch `dev` without committing or pushing.
- Do not modify the online prediction path during schema-v1 replay.
- Replay OSDs independently and reject config/schema mismatches.
- Use TDD: every behavior starts with a failing test.
- Stop after MapReduce `osd.0` if any parity gate fails.

---

### Task 1: Trace reader and deterministic event schedule

**Files:**
- Create: `test_sh/hp_trace_replay.h`
- Create: `test_sh/test_hp_trace_replay.cc`

**Interfaces:**
- Produces `HpReplayTrace read_hp_trace(const std::filesystem::path&)`.
- Produces `std::vector<HpReplayEvent> make_replay_events(const HpReplayTrace&)`.

- [x] Write tests that reject bad magic/schema/record size and accept evaluated records.
- [x] Run the standalone replay test and verify it fails because the module is absent.
- [x] Implement strict binary reading and prediction-before-training tie order.
- [x] Re-run the focused tests and require green.

### Task 2: Model and snapshot replay

**Files:**
- Modify: `test_sh/hp_trace_replay.h`
- Modify: `test_sh/test_hp_trace_replay.cc`

**Interfaces:**
- Produces `HpReplayResult replay_hp_trace(const HpReplayTrace&, const HpReplayOptions&)`.
- Uses the current `HeatPredictor::make_model()` configuration and recorded features.

- [x] Add failing tests for initial cold prediction, training order, 500-sample publish, one-second publish, and per-run model isolation.
- [x] Implement deterministic training and immutable prediction snapshots.
- [x] Re-run all replay tests and the existing `hp_algorithm_probe`.

### Task 3: Parity metrics and CLI

**Files:**
- Create: `test_sh/hp_trace_replay.cc`
- Modify: `test_sh/hp_trace_replay.h`
- Modify: `test_sh/test_hp_trace_replay.cc`

**Interfaces:**
- CLI: `hp_trace_replay TRACE.bin --output osd.N.replay.tsv`.
- Writes aggregate metrics to stdout and one row per evaluated record in original Trace order.

- [x] Add failing tests for class agreement, MAE/RMSE, P95 and Accuracy delta.
- [x] Implement metrics, deterministic TSV output, and nonzero exit on malformed input.
- [x] Compile and run CLI against a synthetic Trace.

### Task 4: Phase reporter and gate

**Files:**
- Create: `test_sh/analyze_hp_replay.py`
- Create: `test_sh/test_analyze_hp_replay.py`
- Modify: `codex_docs/todo/TRACE_DATASET.md`

**Interfaces:**
- CLI: `python3 test_sh/analyze_hp_replay.py --run-root RUN --replay-dir DIR --output-dir OUT`.
- Writes `replay_summary.tsv`, `replay_phase_summary.tsv`, `replay_mismatches.tsv`, and `REPLAY_REPORT.md`.

- [x] Add a failing synthetic stream-join and gate test.
- [x] Implement original-order join, phase mapping, percentiles, and strict gate evaluation.
- [x] Document commands and parity limits.

### Task 5: MapReduce single-OSD stop gate

**Files:**
- Create under existing result tree: `offline_replay_mapreduce_osd0/*`

**Interfaces:**
- Consumes the formal MapReduce `osd.0.bin` from the 20260717 Trace run.
- Produces a Chinese parity report and a pass/fail gate.

- [x] Compile replay tool against the current dev model configuration.
- [x] Replay MapReduce `osd.0` exactly once.
- [x] Verify record/label association, class agreement, MAE, P95, and Accuracy delta.
- [x] Stop and design schema v2 if any gate fails; run remaining OSDs only on pass.

### Task 6: Full replay only after gate pass

**Files:**
- Create under existing result tree: `offline_replay_all5/*`

**Interfaces:**
- Consumes ten OSD Trace files.
- Produces per-workload/per-phase parity tables without changing source code.

- [x] Replay all ten files once only if Task 5 passes.
- [x] Verify raw Trace checksums remain unchanged.
- [x] Run replay/unit/converter tests and `git diff --check`.
- [x] Record whether feature ablation is permitted or schema v2 is required.

## 2026-07-17 Result

- MapReduce `osd.0` gate passed before the remaining files were replayed.
- All 10 OSD files and all 46 workload phases passed the strict parity gate.
- All 1,468,315 source records associated exactly; source Trace checksums remained unchanged.
- Schema v1 is sufficiently faithful for controlled feature ablation. Schema v2 is not required
  for the next experiment, but remains the upgrade path if future model/snapshot changes reduce parity.
- Reports: `hp_runs/reports/20260717_002308_dev_trace_all5/offline_replay_all5/`.
