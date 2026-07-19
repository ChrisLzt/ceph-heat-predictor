# HP Trace Offline Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dependency-light offline analyzer that validates the five-workload Trace dataset, explains prediction errors by phase and margin, measures probability calibration and feature separation, and emits machine-readable tables plus a Chinese report.

**Architecture:** A Python standard-library CLI streams each OSD CSV once and joins records to the workload's 30-second phase intervals through Trace monotonic-to-wall-clock metadata. Small online accumulators compute confusion matrices, calibration, feature moments/correlation, margin categories, and per-object counts without loading 1.46 million rows into a DataFrame. The first version diagnoses the recorded online policy; exact ARF replay remains a separate follow-up because snapshot-generation timing is not present in schema v1.

**Tech Stack:** Python 3.10 standard library (`argparse`, `csv`, `dataclasses`, `json`, `math`, `unittest`); existing Trace CSV/metadata and phase interval TSV files.

## Global Constraints

- Work on `/home/chris/ceph-heat-predictor` branch `dev`; do not create or push Git commits.
- Never modify raw Trace, workload output, or existing result tables.
- Treat `(workload, osd_id, object_key_hash)` as the object identity.
- Aggregate each OSD independently at input and only merge metric counts afterward.
- Preserve time order; phase assignment uses metadata monotonic/wall-clock anchors and `phase_intervals.tsv`.
- Accuracy is the primary result; Balanced Accuracy, Precision, Recall, calibration, and phase behavior are diagnostics.
- Reject malformed schema, missing required columns, non-evaluated records, label/heat inconsistencies, NaN, and out-of-range probabilities.

---

### Task 1: Core schema, metrics, and interval assignment

**Files:**
- Create: `test_sh/hp_trace_analysis.py`
- Create: `test_sh/analyze_hp_trace.py`
- Create: `test_sh/test_hp_trace_analysis.py`

**Interfaces:**
- Produces `ConfusionMatrix.add(predicted: int, actual: int)` and percentage metrics.
- Produces `TraceMetadata.from_json(path: Path)` and `prediction_wall_time_ns(monotonic_ns: int)`.
- Produces `PhaseIndex.from_tsv(path: Path)` and `lookup(wall_time_ns: int) -> Phase`.

- [ ] Write unit tests for confusion metrics, monotonic-to-wall conversion, before/inside/after phase lookup, and malformed metadata.
- [ ] Run `python3 -m unittest test_sh/test_hp_trace_analysis.py -v` and verify failure because the analyzer module does not exist.
- [ ] Implement the minimal data types, validation, and interval lookup.
- [ ] Re-run the unit tests and require all Task 1 tests to pass.

### Task 2: Streaming diagnostic aggregators

**Files:**
- Modify: `test_sh/hp_trace_analysis.py`
- Modify: `test_sh/test_hp_trace_analysis.py`

**Interfaces:**
- Produces `TraceAnalyzer.consume(row, context)` for one-pass aggregation.
- Produces workload/global confusion, phase/segment confusion, 10-bin calibration with ECE/Brier score, prediction/label margin categories, six-feature class/error moments and Pearson correlations, and per-object access/error counts.

- [ ] Add failing fixture tests with known TP/FP/TN/FN, phase assignments, calibration bins, margin categories, feature means, correlations, and object counts.
- [ ] Run the focused tests and verify expected assertion failures.
- [ ] Implement numerically stable online moments/covariance and bounded aggregate structures.
- [ ] Re-run all analyzer tests and require them to pass.

### Task 3: Run-root discovery and output contract

**Files:**
- Modify: `test_sh/hp_trace_analysis.py`
- Modify: `test_sh/analyze_hp_trace.py`
- Modify: `test_sh/test_hp_trace_analysis.py`
- Modify: `codex_docs/todo/TRACE_DATASET.md`

**Interfaces:**
- CLI: `python3 test_sh/analyze_hp_trace.py --run-root RUN --output-dir OUT`.
- Writes `summary.json`, `workload_summary.tsv`, `phase_error.tsv`, `calibration.tsv`, `margin_analysis.tsv`, `feature_stats.tsv`, `feature_correlation.tsv`, `object_stats.tsv`, and `REPORT.md`.

- [ ] Add a failing end-to-end test using two synthetic OSD CSVs and metadata files.
- [ ] Verify the CLI test fails because discovery/output is absent.
- [ ] Implement deterministic discovery, strict validation, TSV/JSON writers, and concise Chinese report generation.
- [ ] Document the command and output contract in `TRACE_DATASET.md`.
- [ ] Run all analyzer and converter self-tests.

### Task 4: Analyze the formal five-workload dataset

**Files:**
- Create under existing results: `ceph-test/new_workload/hp_runs/reports/20260717_002308_dev_trace_all5/offline_analysis/*`

**Interfaces:**
- Consumes the immutable five-workload run root.
- Produces a diagnostic report linked to the existing collection report.

- [ ] Run the analyzer against all ten CSV files.
- [ ] Verify 5 workloads, 10 OSD files, 1,468,315 evaluated records, zero rejected records, and summary confusion counts equal `results.tsv`.
- [ ] Inspect the report for cold-start/transition/steady behavior, confident errors, calibration, feature redundancy, and object skew.
- [ ] Run `git diff --check`, analyzer unit tests, converter self-test, and confirm raw Trace checksums did not change.

### Task 5: Define the replay gate

**Files:**
- Modify: `codex_docs/todo/TRACE_DATASET.md`

**Interfaces:**
- Records the exact prerequisites for a later per-OSD ARF replay tool.

- [ ] Document prediction-time ordering, delayed training at `label_completion_time_ns`, fixed seed, per-OSD isolation, and online metric aggregation.
- [ ] State the parity gate: replay must first match recorded baseline probabilities/labels closely enough to justify ablation conclusions.
- [ ] Record schema-v1 limitation: exact snapshot parity may require adding snapshot generation/version to each Trace record.
