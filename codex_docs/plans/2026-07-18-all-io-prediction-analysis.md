# All-I/O Prediction Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing Trace analyzer to diagnose every I/O prediction against its 10-second-later label, with steady-state results as the primary evidence.

**Architecture:** Keep `hp_trace_analysis.py` as the single streaming analyzer. Existing all-run accumulators remain compatible; new steady-state accumulators add probability-bin outcomes, exact feature quantiles, per-object confusion, and 30-second time bins. Existing no-migration Trace files are re-analyzed offline, so no Ceph rebuild or Vdbench rerun is required.

**Tech Stack:** Python 3 standard library, `unittest`, CSV/TSV/JSON, existing HPTRACE schema v2.

## Global Constraints

- Analyze MapReduce, GraphChi, and HPC no-migration traces only.
- Use fixed prediction threshold `0.50` and the stored 10-second future label.
- Treat `TP/TN/FP/FN` as prediction outcomes, not object state transitions.
- Use `[120s, 600s]` as the primary steady-state interval; full-run results are secondary.
- Preserve existing output files and add new files without changing Trace schema.
- Do not modify the online model and do not rerun Vdbench.
- Do not create or push Git commits.

---

### Task 1: Define The New Analysis Contract

**Files:**
- Modify: `test_sh/test_hp_trace_analysis.py`
- Modify: `test_sh/hp_trace_analysis.py`

**Interfaces:**
- Extend `ConfusionMatrix` with `false_positive_rate` and `false_negative_rate`.
- Extend `CalibrationTable.add(probability, actual, predicted=None)` so each probability bin can retain its own confusion matrix.
- Add `DistributionSummary` with `add()`, `merge()`, and exact `quantile()` for steady feature profiles.
- Extend `ObjectStats` with TP/FP/TN/FN counts and object accuracy.
- Extend `RecordContext` with `relative_time_ns`.

- [ ] **Step 1: Add failing unit tests**

  Cover FPR/FNR denominators, probability-bin TP/TN/FP/FN counts, exact P10/P50/P90 feature quantiles, per-object confusion, and relative-time context.

- [ ] **Step 2: Verify the tests fail for missing behavior**

  Run: `python3 -m unittest test_sh.test_hp_trace_analysis`

  Expected: failures identify missing rates, bin outcomes, quantiles, object outcomes, or context fields.

- [ ] **Step 3: Implement the minimal accumulators**

  Use `array('d')` for exact finite feature samples and preserve all existing public fields and default call signatures.

- [ ] **Step 4: Verify the unit tests pass**

  Run: `python3 -m unittest test_sh.test_hp_trace_analysis`

  Expected: all tests pass.

### Task 2: Add Steady-State And Time-Bin Aggregation

**Files:**
- Modify: `test_sh/test_hp_trace_analysis.py`
- Modify: `test_sh/hp_trace_analysis.py`

**Interfaces:**
- `WorkloadAggregate.steady_calibration`
- `WorkloadAggregate.steady_feature_by_outcome`
- `WorkloadAggregate.steady_objects`
- `WorkloadAggregate.steady_time_confusion`, keyed by 30-second relative-time bin.

- [ ] **Step 1: Add failing aggregation tests**

  Feed warmup and steady records into one workload and assert only steady records enter the four new accumulators. Assert records at 120s and 149.999s share a bin and a record at 150s enters the next bin.

- [ ] **Step 2: Verify the aggregation tests fail**

  Run: `python3 -m unittest test_sh.test_hp_trace_analysis.TraceAnalyzerTest`

  Expected: failures identify missing steady accumulators or incorrect bin boundaries.

- [ ] **Step 3: Implement aggregation and merging**

  Compute `relative_time_ns = prediction_wall_time_ns - workload_started_at_ns` in `_scan_trace_file`; populate the four accumulators only when `steady_state` is true.

- [ ] **Step 4: Verify all analyzer tests pass**

  Run: `python3 -m unittest test_sh.test_hp_trace_analysis`

  Expected: all tests pass.

### Task 3: Export The Five Diagnostic Tables

**Files:**
- Modify: `test_sh/test_hp_trace_analysis.py`
- Modify: `test_sh/hp_trace_analysis.py`

**Interfaces:**
- `prediction_outcome_summary.tsv`: all/steady matrices, including FPR/FNR and hot ratios.
- `prediction_probability_bins.tsv`: steady 0.1 probability bins with TP/TN/FP/FN.
- `prediction_feature_profiles.tsv`: steady TP/TN/FP/FN feature mean/std/P10/P25/P50/P75/P90.
- `prediction_object_summary.tsv`: steady sample-micro and object-macro accuracy plus top 1/5/10 percent I/O and error concentration.
- `prediction_time_series.tsv`: steady 30-second matrices per workload.

- [ ] **Step 1: Add failing end-to-end contract tests**

  Extend `EXPECTED_OUTPUT_FILES`; assert headers and representative values for each new TSV.

- [ ] **Step 2: Verify the end-to-end test fails**

  Run: `python3 -m unittest test_sh.test_hp_trace_analysis.EndToEndAnalysisTest`

  Expected: the five output files are missing.

- [ ] **Step 3: Implement deterministic TSV row builders and writers**

  Sort workloads, outcomes, features, probability bins, and time bins. Include an aggregate `all` row where schemas match.

- [ ] **Step 4: Verify the complete output contract**

  Run: `python3 -m unittest test_sh.test_hp_trace_analysis`

  Expected: all tests pass and the output directory exactly matches `EXPECTED_OUTPUT_FILES`.

### Task 4: Generate And Independently Verify The Report

**Files:**
- Modify: `test_sh/hp_trace_analysis.py`
- Modify: `codex_docs/CODEX_CEPH_TODO.md`
- Create: `/home/chris/ceph-test/new_workload/hp_runs/reports/<timestamp>_all_io_prediction_analysis/REPORT.md`

**Interfaces:**
- The generated Chinese report treats steady all-I/O prediction as the primary result and full-run results as secondary.
- The active TODO freezes C2H/H2C work and records all-I/O TP/TN/FP/FN diagnosis as the current evidence path.

- [ ] **Step 1: Update report rendering and TODO wording**

  Add steady matrix, probability-bin, feature-profile, object-concentration, and 30-second stability sections. Avoid interpreting TP/TN/FP/FN as state transitions.

- [ ] **Step 2: Run the existing three Trace datasets**

  Run:

  ```bash
  python3 test_sh/analyze_hp_trace.py \
    --run-root /home/chris/ceph-test/new_workload/hp_runs/reports/20260717_190525_no_migration_steady_confidence/D2 \
    --output-dir /home/chris/ceph-test/new_workload/hp_runs/reports/<timestamp>_all_io_prediction_analysis
  ```

  Expected: three workloads, no schema/count/time validation errors, and all output files present.

- [ ] **Step 3: Independently verify conservation identities**

  Check for each workload and scope that `count = TP + TN + FP + FN`, probability-bin counts sum to the steady count, time-bin counts sum to the steady count, and object I/O/error totals equal the steady matrix totals.

- [ ] **Step 4: Run final verification**

  Run: `python3 -m unittest test_sh.test_hp_trace_analysis`

  Expected: all tests pass. Record the generated report path and the principal steady-state findings; leave all changes uncommitted.
