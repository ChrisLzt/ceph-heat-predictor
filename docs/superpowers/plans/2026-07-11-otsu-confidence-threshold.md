# Otsu Confidence Threshold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace runtime quantile fallback and fixed Otsu EMA with a confidence-controlled effective heat threshold.

**Architecture:** `HpOtsuHistogram` returns one scan result containing the candidate score and confidence inputs. `EvaluationQueue` computes weighted geometric confidence, advances the effective score with a bounded gain, and exposes explicit threshold state. OSD perf and MGR aggregation publish the same state without reconstructing it from unrelated counters.

**Tech Stack:** C++17, PBDS threshold window, bounded Otsu histogram, Ceph perf counters and MGR formatter, standalone algorithm probe.

## Global Constraints

- Initial effective heat threshold is `HP_HEAT_INCREMENT`.
- Sample confidence is `0` at 32 objects and `1` at 1000 objects.
- Sharpness uses the score range with between-class variance at least `0.99` of the optimum and reaches zero at plateau ratio `0.20`.
- Confidence weights are separation `0.65`, sample count `0.20`, and sharpness `0.15`.
- Maximum update gain is `0.10`.
- Runtime quantile fallback, fixed Otsu EMA, category-count confidence, and separation hard rejection are removed.
- Threshold states are `initializing`, `tracking`, and `holding`.
- OSD perf enum, declaration, update, reset, and output orders remain identical.

---

### Task 1: Otsu scan result and confidence math

**Files:**
- Modify: `src/heatpredictor/hp_config.h`
- Modify: `src/heatpredictor/hp_otsu_histogram.h`
- Modify: `test_sh/hp_algorithm_probe.cc`

**Interfaces:**
- Produces: `HpOtsuResult` and `HpOtsuHistogram::otsu_result()`.
- Produces: sample, sharpness, and weighted geometric confidence helpers in `EvaluationQueue`.

- [x] Add failing probe cases for insufficient, constant, weak, monotonic, and clear bimodal samples.
- [x] Verify the probe fails because the new result and constants do not exist.
- [x] Return candidate score, separation, near-optimal range, occupied range, and sample count without hard separation rejection.
- [x] Implement and verify the confidence formulas and numerical bounds.

### Task 2: Confidence-driven effective threshold

**Files:**
- Modify: `src/heatpredictor/hp_evaluation_queue.h`
- Modify: `src/heatpredictor/hp_types.h`
- Modify: `src/heatpredictor/heat_predictor.h`
- Modify: `test_sh/hp_algorithm_probe.cc`

**Interfaces:**
- Consumes: `HpOtsuResult`.
- Produces: effective and candidate thresholds, four confidence values, histogram object count, and method state.

- [x] Add failing tests for initializing, tracking, holding, bounded movement, update cadence, and predictor reset defaults.
- [x] Remove quantile threshold selection and fixed Otsu EMA state.
- [x] Apply `gain = 0.10 * confidence` in time-normalized score space.
- [x] Keep PBDS as threshold-window order statistics storage only.
- [x] Export the new state through `HeatPredictorStats` and pass the full probe.

### Task 3: OSD and MGR observability

**Files:**
- Modify: `src/osd/ObjectHeatPredictor.cc`
- Modify: `src/mgr/DaemonServer.cc`
- Modify: `CODEX_CEPH.md`
- Modify: `CODEX_CEPH_TODO.md`

**Interfaces:**
- OSD exports histogram object count, effective/candidate thresholds, separation, total/sample/sharpness confidence, and method state.
- MGR weights candidate and confidence fields by each OSD histogram object count.

- [x] Add OSD fields in identical enum/declaration/update/reset order.
- [x] Replace method names with `initializing/tracking/holding`.
- [x] Add MGR weighted aggregation and preserve the five requested prediction metrics.
- [x] Update current-state documentation and remove completed P1 from TODO.

### Task 4: Verification and deployment

**Files:**
- Verify only.

- [x] Build and run `hp_algorithm_probe` and its sanitizer variant.
- [x] Run `sudo env CCACHE_TEMPDIR=/tmp ninja -j64` and `sudo ninja install`.
- [x] Run `sudo ldconfig` and restart active OSD/MGR services.
- [x] Verify reset defaults, OSD perf field order, MGR weighted output, and `ceph -s`.
