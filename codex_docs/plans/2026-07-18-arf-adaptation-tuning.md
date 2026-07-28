# ARF Adaptation Observability And Tuning Plan

**Goal:** Measure whether faster tree growth or more sensitive ADWIN replacement improves cold-to-hot recall without hiding adaptation behavior.

**Architecture:** Add a thread-safe telemetry object shared only by the training ARF and expose a plain snapshot through HeatPredictor, replay, OSD perf, and MGR summary. Keep the production defaults unchanged while the `dev` replay build accepts compile-time experiment values, then compare one parameter family at a time on the existing three no-migration traces.

**Tech Stack:** C++17, Ceph PerfCounters/MGR JSON, existing binary Trace replay, Python standard-library reporting.

## Global Constraints

- Work on branch `dev`; do not commit or push.
- Main agent performs every file edit and result review. Subagents may only monitor long-running commands read-only.
- Do not change the default ARF parameters until a candidate passes the gate.
- Use identical traces, feature schema, seed, snapshot cadence, model count, threshold, and sample weight for all profiles.
- Gate: mean Test C2H Recall increases by at least 10 percentage points, Accuracy decreases by no more than 1 percentage point, and Balanced Accuracy does not decrease.

## Task 1: ARF adaptation telemetry

**Files:**
- Create: `src/heatpredictor/include/ArfAdaptationTelemetry.h`
- Modify: `src/heatpredictor/include/ARFClassifier.h`
- Modify: `test_sh/hp_algorithm_probe.cc`

**Behavior:**
- Snapshot fields: warnings, drifts, background promotions, discarded background trees, background-tree training updates, and currently active background trees.
- Warning increments once per warning. Replacing an existing background tree also increments the discard count without changing the active gauge.
- Drift promotes an existing background tree when present; otherwise it installs a fresh tree. Every drift is therefore one current-tree replacement.
- Prediction-only clones do not share or update training telemetry.

**TDD:**
- [x] Add scripted warning/drift detector tests that fail because telemetry is absent.
- [x] Run the focused probe and confirm the expected compile/test failure.
- [x] Implement the atomic telemetry snapshot and ARF event updates.
- [x] Re-run the focused probe and existing ARF parameter tests.

## Task 2: HeatPredictor, replay, OSD and MGR exposure

**Files:**
- Modify: `src/heatpredictor/heat_predictor.h`
- Modify: `test_sh/hp_trace_replay.h`
- Modify: `test_sh/hp_trace_replay.cc`
- Modify: `test_sh/test_hp_trace_replay.cc`
- Modify: `src/osd/ObjectHeatPredictor.cc`
- Modify: `src/mgr/DaemonServer.cc`

**Behavior:**
- HeatPredictor owns one telemetry object, supplies it only to `train_model`, snapshots it lock-free, and clears it on reset.
- Replay writes final telemetry counters in its stdout summary and `HpReplayResult`.
- OSD perf and MGR add one `model_adaptation` group using the exact enum/declaration/update/reset/aggregate/output order required by `AGENTS.md`.
- Fields are summed across OSDs; active background count is the current sum, while all other fields are cumulative since reset.

**TDD:**
- [x] Add replay tests that fail because final adaptation counters are absent.
- [x] Implement HeatPredictor/replay propagation and confirm focused tests pass.
- [x] Add the six OSD fields in contract order and the matching MGR aggregation/output.
- [x] Run `git diff --check` and inspect enum/declaration/update/reset/output ordering manually.

## Task 3: Controlled replay profiles

**Files:**
- Modify: `src/heatpredictor/hp_config.h`
- Modify: `src/heatpredictor/heat_predictor.h`
- Modify: `test_sh/hp_algorithm_probe.cc`
- Create: `ceph-test/new_workload/tools/summarize_hp_arf_adaptation.py`
- Create: `ceph-test/new_workload/tests/test_summarize_hp_arf_adaptation.py`

**Profiles:**
- `R0`: current defaults (`grace=100`, tree `delta=0.001`, `tau=0.05`, warning `delta=0.01`, drift `delta=0.001`).
- `R1`: faster growth only (`grace=50`); all detector parameters unchanged.
- `R2`: more sensitive detectors only (warning `delta=0.02`, drift `delta=0.002`); all tree-growth parameters unchanged.
- `R3`: `R1+R2`, run only if R1 or R2 passes the validation gate.

**Execution:**
- [x] Add override-safe compile-time values on `dev` while preserving R0 defaults.
- [x] Build one replay binary per required profile with the same optimization and include paths.
- [x] Replay both OSD traces for MapReduce, GraphChi, and HPC WRF exactly once per profile.
- [x] Derive Accuracy, Balanced Accuracy, C2H Recall, C2C Specificity, hot ratio, runtime, and adaptation counters.
- [x] Stop without R3 if neither R1 nor R2 passes validation.

## Task 4: Review and documentation

**Files:**
- Create: `ceph-test/new_workload/hp_runs/reports/<timestamp>_arf_adaptation/REPORT.md`
- Modify: `codex_docs/CODEX_CEPH_TODO.md`

**Review:**
- [x] Confirm counters are internally consistent: promotions never exceed drifts, active backgrounds never exceed tree count, and discarded backgrounds require warnings.
- [x] Separate faster splitting effects from whole-tree replacement effects.
- [x] Reject candidates that improve C2H only by broadly predicting more objects hot.
- [x] Keep R0 defaults unless the complete gate passes; record failed directions without adding them to the online baseline.
