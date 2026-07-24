# V2 Future-Access Hotness Design

**Status:** Approved for implementation on `dev`

**Date:** 2026-07-23

**Baseline:** current production implementation on `main`

**Scope:** Object-level Heat Predictor label definition, features, threshold
maintenance, telemetry, and verification

## 1. Goal

V1 defines an I/O label from the object's decayed total heat and the Otsu heat
threshold. V2 replaces that self-referential target with a direct future-demand
definition:

```text
future_access_count in (t, t + 10s] >= K_at_prediction -> hot
otherwise                                           -> cold
```

The current I/O at time `t` is excluded. Every I/O remains an independent
prediction sample, while the threshold population is object-weighted so a hot
object cannot dominate Otsu merely by producing more I/Os.

The first implementation target is semantic and concurrency correctness. V2 is
not required to beat V1 by a fixed metric before the implementation is
considered correct. Formal workload comparison follows correctness validation.

## 2. Definitions

- **Evaluation window:** the future 10-second interval `(t, t + 10s]`.
- **EQ item:** one prediction sample created for one I/O.
- **Future access count:** accesses to the same object in the item's evaluation
  window, excluding the current I/O.
- **Positive observation:** a completed item whose future access count is
  greater than zero.
- **Object vote:** the latest completed observation retained for one object in
  the Otsu population.
- **K:** the integer future-access threshold used by prediction and labeling.
  Sparse mode publishes the semantic fallback `K=1`; a ready object-weighted
  Otsu tracker may replace it with a dynamic value.
- **K at prediction:** the single threshold value captured when an item enters
  the prediction pipeline.

## 3. Causal Contract

Each EQ item stores `K_at_prediction`. Ten seconds later its label is computed
with that stored value, even if the live Otsu threshold has changed:

```text
actual_hot = future_access_count >= K_at_prediction
```

This rule prevents future threshold updates from retroactively changing the
question that the model answered at prediction time. The same stored K is used
to derive the threshold-relative feature.

K is always available. Construction and reset publish `K=1`, meaning that an
object is hot when it is accessed at least once again in the next 10 seconds.
This gives sparse workloads a stable target even if they never contain enough
objects or distinct count bins for a reliable Otsu split. Sparse-mode items
follow the normal EQ, training, confusion-matrix, and model-quality paths.

## 4. Exact Per-I/O Future Count

Every object state maintains a monotonically increasing tracked access count.
Due items are expired first, then the current I/O increments that count. The
new item captures the post-increment value:

```text
future_access_count =
    tracked_access_count_at_deadline -
    tracked_access_count_after_current_access
```

The post-increment baseline excludes the current I/O from the difference.
Expiring due items before that increment also avoids counting the first access
after an old item's deadline. Together these rules preserve `(t, t + 10s]`.

Multiple pending items for the same object are intentional. Each item has its
own enqueue count, deadline, prediction, label, training sample, and statistics.

## 5. Object-Weighted Otsu Population

### 5.1 Lifecycle

Otsu retains at most one latest positive observation per object:

- A positive observation inserts or replaces the object's vote.
- A zero observation removes the object's old positive vote, if one exists.
- Votes have no semantic TTL. A capacity limit remains only as an abnormal
  memory guard.
- If multiple completed items for one object are applied in a batch, only the
  observation with the latest `label_deadline` updates the object vote. The
  items are still labeled, trained, and counted independently.

This separates sample weighting from threshold weighting:

- model training and metrics are I/O-weighted;
- Otsu is object-weighted.

### 5.2 Score Transform and Histogram

For positive observations:

```text
score = log2(1 + future_access_count)
```

The fixed histogram configuration is:

```text
score_min  = 1.0
bin_width  = 0.01
bin_count  = 2000
score_max  = 21.0
```

The representable count range reaches approximately:

```text
2^21 - 1 = 2,097,151 accesses per 10 seconds
```

Scores above the range are physically clamped into the final bin.
`hp_otsu_upper_clamped_object_count` counts objects whose true score exceeds
the upper bound; it is not merely the occupancy of the final bin.

The histogram is a fixed `uint64_t[2000]`, approximately 16 KiB per OSD.

### 5.3 Otsu Candidate

Otsu scans the occupied histogram distribution and selects the split with the
maximum between-class variance. The chosen score is converted back to an
integer threshold using a conservative ceiling so a published K never admits a
count below the represented split.

The tracker recomputes when either condition is met:

- 100 object-vote changes have accumulated; or
- one second has elapsed since the previous recomputation.

When readiness is reached for the first time, an immediate recomputation is
allowed so sparse mode does not wait for another full interval.

## 6. Threshold State Machine

The threshold tracker has three states:

### Sparse

`K=1` is published when there are fewer than 32 positive objects, fewer than
two occupied bins, or no valid split variance and no held Otsu threshold.
Otsu observations continue to be maintained. Samples remain eligible for
prediction, training, and quality metrics. Enqueue increments
`hp_sparse_threshold_sample_count` without retaining the state in the EQ item.

### Tracking

A valid candidate exists. The candidate score is smoothed before publishing:

```text
effective_gain = 1 - (1 - 0.10)^(elapsed_seconds / 1.0)
published_score =
    old_score + effective_gain * (candidate_score - old_score)
```

The first valid candidate is published directly. Smoothing the score instead of
the integer K avoids quantization-induced sticking.

### Holding

If a previously tracking population temporarily cannot produce a valid split,
the last K is retained for at most 10 seconds. Samples using it increment
`hp_threshold_holding_sample_count`.

After 10 seconds without a valid candidate, the tracker publishes `K=1` and
returns to sparse behavior. A later valid Otsu population may return it to
tracking. Every item keeps only its prediction-time K, so a state
transition never changes an already enqueued item's target.

Each OSD maintains its own threshold. MGR summarizes OSD state and K values but
does not participate in online decisions.

## 7. Baseline Features and Model

V2 starts with exactly three features:

```text
past_access_count_margin =
    log2(1 + past_10s_access_count) -
    log2(1 + K_at_prediction)

previous_access_interval_encoded

current_heat_log2p1 =
    log2(1 + current_heat)
```

`past_10s_access_count` is the number of the object's pending 10-second EQ
items sampled before the current item is enqueued. Because due items are
expired first and every item has the same 10-second lifetime, this value is the
strict past-window access count and excludes the current I/O.

The initial model remains unchanged:

- `StandardScaler` followed by ARF;
- 25 trees and the current ARF detector/tree parameters;
- prediction probability threshold `0.50`;
- hot and cold sample weights both `1.0`;
- current background training and model-snapshot publication;
- EQ, LRU, and threshold object hard limits remain `1,000,000`.

An untrained forest or otherwise valid zero-vote result is a cold-start cold
prediction, not an evaluation drop. Its EQ item is retained; after the
10-second label window it enters the confusion matrix and background training
like every other valid prediction.

V2 deliberately changes the target and threshold-relative feature before
changing the classifier.

## 8. Module Boundary

Add `src/heatpredictor/hp_future_access_threshold.h` as the owner of:

- object vote lifecycle and latest-deadline batch deduplication;
- object-to-bin index and object order used by the hard capacity guard;
- fixed histogram and clamp accounting;
- Otsu recomputation, score EMA, integer K conversion, and state machine;
- the current threshold and threshold telemetry.

Its public surface is limited to reading the current threshold/status and
maintenance deadline, applying a batch of completed observations, and clearing
state. `hp_evaluation_queue.h` owns per-I/O deadlines and exact future counts;
it never reaches into threshold-module containers.

## 9. Telemetry

Retain the current confusion matrix, accuracy, balanced accuracy, precision,
recall, predicted/actual hot percentages, queue/model/drop/latency, and
operation counters.

### Future-Access Threshold

Add:

- `hp_future_access_threshold`
- `hp_future_access_candidate_threshold`
- `hp_threshold_state`
- `hp_otsu_positive_object_count`
- `hp_otsu_zero_observation_count`
- `hp_otsu_upper_clamped_object_count`
- `hp_threshold_holding_sample_count`
- `hp_sparse_threshold_sample_count`

The threshold state is represented consistently as
`sparse/tracking/holding` in human-readable status and as stable numeric
values in PerfCounters.

Remove or rename V1-only heat-threshold fields so OSD status, OSD PerfCounters,
MGR field declarations, derived formulas, and output grouping stay aligned.

MGR reports K minimum, maximum, and average across reporting OSDs plus the count
of OSDs in each threshold state.

## 10. Reset and Enable/Disable

Reset must atomically return the threshold tracker to sparse mode with `K=1`
and clear its histogram, object votes, candidate/publication state, timers,
telemetry, EQ items, and pending counts through the existing reset protocol.

Enable and disable keep the current reset-on-transition contract. Commands do
not change.

## 11. Correctness Verification

Before workload comparison, deterministic tests must cover the causal window,
repeated-object items, stored-K labels, object-vote lifecycle, histogram
boundaries, all threshold-state transitions, cold start, reset/lifecycle,
OSD/MGR telemetry alignment, and concurrent update/reset. Exact cases and
execution order are maintained only in the
[implementation plan](../plans/2026-07-23-v2-future-access-hotness.md).

Formal workloads are then run once for observation and reporting. Algorithm
metrics are not correctness gates for the first V2 implementation. Capture the
final status immediately when the configured workload ends; do not add a
10-second drain or idle period. Items whose label windows are still open are
right-censored, remain visible in the pending count, and are excluded from the
confusion matrix because no complete future label exists.
