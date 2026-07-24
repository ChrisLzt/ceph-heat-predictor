# No-Migration Steady Confidence Analysis Implementation Plan

> **For agentic workers:** execute tasks in order with test-first RED/GREEN checks. Do not commit or
> push unless the user explicitly requests it.

**Goal:** Run three fixed-hotspot workloads once and report the steady-state FP/FN distribution in
0.1-wide prediction-confidence bins.

**Architecture:** Generalize the existing matrix runner with an optional workload root while keeping
its default behavior unchanged. Extend the offline Trace analyzer with a focused steady-state error
histogram; use run metadata for the exact 120-second warmup boundary and leave OSD Trace schema and
online prediction code unchanged.

**Tech Stack:** Bash runner tests, Python `unittest`, existing binary Trace-to-CSV converter, Ceph MGR
commands, Vdbench.

## Global Constraints

- Run only MapReduce, GraphChi and HPC no-migration workloads, once each.
- Exclude predictions made before `started_at + 120s`.
- For FP use `confidence=p_hot`; for FN use `confidence=1-p_hot`.
- Use bins `[0.5,0.6)`, `[0.6,0.7)`, `[0.7,0.8)`, `[0.8,0.9)`, `[0.9,1.0]`.
- Keep all changes uncommitted.

---

### Task 1: External workload root support

**Files:**
- Modify: `/home/chris/ceph-test/new_workload/run_hp_matrix.sh`
- Test: `/home/chris/ceph-test/new_workload/tests/test_run_hp_matrix.sh`

**Interfaces:**
- Consumes: `--workload-root DIR` and repeated `--workload NAME` arguments.
- Produces: the existing report layout with workload scripts loaded from the selected root.

- [ ] Add a dry-run test selecting a no-migration workload root and verify it fails before support exists.
- [ ] Add `--workload-root`, canonicalize it, validate scripts there, and hash whichever of `configs/`
  and `rendered/` exist.
- [ ] Keep tools, reports and default workload list rooted in `new_workload`.
- [ ] Run the runner contract test and `bash -n`.

### Task 2: Steady-state confidence histogram

**Files:**
- Modify: `/home/chris/ceph-heat-predictor/test_sh/hp_trace_analysis.py`
- Modify: `/home/chris/ceph-heat-predictor/test_sh/test_hp_trace_analysis.py`

**Interfaces:**
- Consumes: workload `metadata.json.started_at`, Trace prediction wall time, predicted hot
  probability, predicted label and actual label.
- Produces: per-workload and aggregate steady confusion matrices plus 0.1-wide FP/FN confidence bins.

- [ ] Add failing unit tests for the exact 120-second boundary, FP/FN confidence conversion and the
  inclusive `1.0` final bin.
- [ ] Add a five-bin histogram type and steady-state confusion state to the workload aggregate.
- [ ] Parse `metadata.json.started_at` and pass the warmup boundary into Trace consumption.
- [ ] Add `steady_error_confidence.tsv` and a Chinese steady-state section to `REPORT.md`.
- [ ] Run all Python analyzer tests.

### Task 3: Local and integration validation

**Files:**
- Verify only; no production source changes.

- [ ] Run `hp_algorithm_probe`, Trace replay tests, Python tests, runner tests and `git diff --check`.
- [ ] Validate all three no-migration workload models and confirm referenced data directories exist.
- [ ] Confirm Ceph is active+clean and the deployed profile matches D2.

### Task 4: Formal no-migration run and report

**Files:**
- Create outputs below `/home/chris/ceph-test/new_workload/hp_runs/reports/<timestamp>_no_migration_steady_confidence/`.

- [ ] Run the three no-migration workloads once with Trace and 10-second MGR snapshots.
- [ ] Wait for pending, awaiting-prediction and training queues to reach zero after every workload.
- [ ] Convert and validate all six Trace files, then run the steady confidence analyzer.
- [ ] Write a Chinese report comparing FP and FN across confidence bins and identifying the dominant
  high-confidence error direction without introducing a separate high-confidence cutoff.
