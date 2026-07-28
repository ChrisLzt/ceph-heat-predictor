# Heat Predictor Architecture Refactor Implementation Plan

> **Execution:** Main agent implements each task inline. Subagents may only monitor long-running builds or tests.

**Goal:** Deepen the evaluation, runtime, telemetry, and MGR aggregation modules without changing heat-prediction behavior.

**Architecture:** Preserve the current prediction and labeling algorithms while reducing the interface exposed by `EvaluationQueue` and `HeatPredictor`. Introduce one shared telemetry contract and move pure cluster aggregation out of `DaemonServer`.

**Tech Stack:** C++17, Ceph PerfCounters, Ceph Formatter, Ninja, temporary standalone probes.

## Global Constraints

- Do not change features, model parameters, heat decay, label generation, Otsu, prediction threshold, or training weights.
- Do not add experiment macros, trace code, or permanent test sources to `main`.
- Do not create commits or push.
- Preserve the current OSD command and MGR JSON field names.
- Keep `PrimaryLogPG` hook placement unchanged.

---

### Task 1: Deepen EvaluationQueue

**Files:**
- Modify: `src/heatpredictor/hp_evaluation_queue.h`
- Modify: `src/heatpredictor/hp_types.h`
- Test: temporary evaluation lifecycle probe under `/tmp`

**Interface:**
- Replace exposed list iterators with a move-only opaque prediction ticket.
- Expose only begin/complete/cancel evaluation, deadline processing, reset construction, and status snapshot behavior.
- Keep heat state, LRU, Otsu, threshold history, pending list, and expiry heap private.

- [x] Write a probe that uses only the intended lifecycle interface and verifies prediction-first, label-first, cancellation, and capacity behavior.
- [x] Compile the probe and verify it fails because the opaque ticket and status snapshot do not exist.
- [x] Introduce the opaque ticket and queue status types.
- [x] Make all state private and adapt the queue implementation.
- [x] Run the lifecycle probe and existing production build targets.

### Task 2: Encapsulate HeatPredictor Runtime

**Files:**
- Modify: `src/heatpredictor/heat_predictor.h`
- Modify: `src/heatpredictor/hp_types.h`
- Modify: `src/osd/ObjectHeatPredictor.cc`
- Test: temporary runtime/status probe under `/tmp`

**Interface:**
- Public operations are prediction, reset, enable/disable, expiry progress notification, and one coherent status snapshot.
- Training model, prediction snapshot, queues, locks, threads, counters, and helper functions are private.
- `ObjectHeatPredictor` consumes one status value instead of many individual getters.

- [x] Write a compile-time probe asserting that runtime internals are not public and a runtime probe covering reset and status snapshots.
- [x] Verify the probe fails against the old interface.
- [x] Introduce the complete `HeatPredictorStatus` value.
- [x] Move runtime machinery to private scope and replace individual getters.
- [x] Adapt the OSD publisher to consume one snapshot.
- [x] Run the probes and affected Ceph targets.

### Task 3: Establish the Telemetry Contract

**Files:**
- Create: `src/heatpredictor/hp_telemetry.h`
- Modify: `src/osd/ObjectHeatPredictor.cc`
- Modify: `src/mgr/DaemonServer.cc`
- Test: temporary telemetry contract probe under `/tmp`

**Interface:**
- Define stable field names, raw units, display units, and cluster aggregation rules in one shared contract.
- Keep Ceph PerfCounter IDs in the OSD adapter.
- Keep typed predictor state independent from Ceph MGR and OSD runtime types.

- [x] Write a probe that checks unique field names and expected aggregation metadata.
- [x] Verify it fails because the shared contract is absent.
- [x] Add constexpr metric descriptors grouped by counters and long-running averages.
- [x] Make the OSD publisher and MGR reader use the shared field names and aggregation rules.
- [x] Verify existing JSON names and units remain unchanged.

### Task 4: Extract Pure MGR Aggregation

**Files:**
- Create: `src/mgr/ObjectHeatPredictorStatus.h`
- Create: `src/mgr/ObjectHeatPredictorStatus.cc`
- Modify: `src/mgr/CMakeLists.txt`
- Modify: `src/mgr/DaemonServer.cc`
- Test: temporary aggregation probe under `/tmp`

**Interface:**
- Input is a collection of typed per-OSD HP snapshots plus up/missing state.
- Output is a typed cluster summary with derived metrics.
- `DaemonServer` remains the Ceph command and Formatter adapter.
- Reset, enable, and disable dispatch remain in `DaemonServer`.

- [x] Write a pure aggregation probe for missing OSDs, sums, weighted averages, percentages, and Otsu method counts.
- [x] Verify it fails because the aggregation module is absent.
- [x] Move aggregation formulas into the new module.
- [x] Keep daemon-state extraction and JSON emission as thin adapters.
- [x] Compile `ceph-mgr`, run the aggregation probe, and compare command output keys.

### Final Verification

- [x] Run all temporary probes.
- [x] Run `git diff --check`.
- [x] Build the affected Ceph targets, then run the required full production build.
- [x] Compare before/after HP status keys and deterministic probe results.
- [x] Confirm no hook, feature, model, threshold, or label behavior changed.

Verification completed on 2026-07-23:

- Four temporary probes passed: evaluation lifecycle, predictor runtime/status,
  telemetry contract, and pure MGR aggregation.
- Full `ninja -j64`, installation, `ldconfig`, and OSD/MGR restart completed.
- The live cluster returned to 145 active+clean PGs; `ceph osd hp status`
  reported both OSDs with the existing JSON field names and units.
