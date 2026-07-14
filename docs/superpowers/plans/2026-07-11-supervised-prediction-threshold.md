# Supervised Prediction Threshold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace cumulative ratio feedback with an accuracy-optimized supervised probability histogram, set hot training weight to `1.0`, and reduce redundant OSD/MGR output.

**Architecture:** A focused `HpPredictionThresholdCalibrator` owns a bounded FIFO and two fixed histograms for evaluated hot/cold probabilities. `EvaluationQueue` feeds delayed labels into it and reads its effective threshold. OSD exports only raw or non-derivable state; MGR computes global derived metrics from raw counters.

**Tech Stack:** C++17, Ceph perf counters and MGR command formatter, existing standalone `hp_algorithm_probe`.

## Global Constraints

- Calibration window `10000`, update interval `500`, probability bins `1001`.
- Effective threshold starts at `0.50`, is constrained to `0.40~0.60`, and uses EMA alpha `0.10`.
- Candidate threshold maximizes window accuracy; equal scores choose the candidate closest to the current threshold.
- Hot and cold training samples both use weight `1.0`.
- OSD perf enum, declarations, updates, and reset stay in the same order.

---

### Task 1: Supervised probability histogram

**Files:**
- Create: `src/heatpredictor/hp_prediction_threshold.h`
- Modify: `src/heatpredictor/hp_config.h`
- Test: `test_sh/hp_algorithm_probe.cc`

**Interfaces:**
- Produces: `HpPredictionThresholdCalibrator::observe(double, int)`, `threshold()`, `target_threshold()`, `size()`, `current_accuracy()`, and `target_accuracy()`.

- [x] Add failing probe cases for accuracy-optimal selection, tie stability, FIFO eviction, EMA, bounds, and reset defaults.
- [x] Compile and run the probe; confirm failures are caused by the missing calibrator.
- [x] Implement fixed hot/cold histograms and bounded FIFO with O(1) insert/evict and O(1001) periodic scans.
- [x] Re-run the probe and confirm all calibrator cases pass.

### Task 2: EvaluationQueue integration and unit training weight

**Files:**
- Modify: `src/heatpredictor/hp_evaluation_queue.h`
- Modify: `src/heatpredictor/hp_types.h`
- Modify: `src/heatpredictor/heat_predictor.h`
- Modify: `test_sh/hp_algorithm_probe.cc`

**Interfaces:**
- Consumes: `HpPredictionThresholdCalibrator`.
- Produces: calibration sample count, current/target threshold, and current/target window accuracy through `HeatPredictorStats`.

- [x] Replace old ratio-controller tests with failing EQ integration and fixed-unit-weight tests.
- [x] Remove cumulative prediction-balance state and feed each expired `(pred_hot_proba, label)` into the calibrator.
- [x] Read the calibrator threshold before snapshot prediction and export its state through `HeatPredictorStats`.
- [x] Set `HP_HOT_CLASS_WEIGHT=1.0` and verify both labels produce training weight `1.0`.
- [x] Run the complete algorithm probe.

### Task 3: Compact OSD and MGR output

**Files:**
- Modify: `src/osd/ObjectHeatPredictor.cc`
- Modify: `src/mgr/DaemonServer.cc`
- Modify: `CODEX_CEPH.md`
- Modify: `CODEX_CEPH_TODO.md`

**Interfaces:**
- OSD exports raw confusion counts, non-derivable averages, calibration state, heat state, queues, latency, and op counters.
- MGR derives accuracy/precision/recall and predicted/actual hot percentages from summed confusion counts.

- [x] Add calibration perf fields in matching enum/declaration/update/reset order.
- [x] Remove OSD-derived accuracy/precision/recall, predicted/actual percentages and ratio, hot/cold average ratios, quantile-threshold alias, and fixed training-weight output.
- [x] Keep only p99/p95/p50 for each hot/cold future-access and future-heat distribution; remove max/p90 output.
- [x] Remove the MGR predicted/actual ratio and fixed training-weight field, and aggregate calibration values by calibration sample count.
- [x] Update current-state documentation and remove completed P2/P3 work from TODO.

### Task 4: Verification and deployment

**Files:**
- Verify only.

- [x] Build and run `hp_algorithm_probe`.
- [x] Run the full build with `sudo env CCACHE_TEMPDIR=/tmp ninja -j64`.
- [x] Run `sudo ninja install` and `sudo ldconfig`.
- [x] Restart active OSD and MGR services.
- [x] Verify `ceph -s`, OSD perf reset defaults, and `ceph osd hp status -f json-pretty` field shape.
