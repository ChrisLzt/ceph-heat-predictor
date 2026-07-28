# V2 EQ Recent-Count Threshold and No-Drift Design

**Status:** Approved for implementation on `dev`

**Date:** 2026-07-25

## 1. Goal

Test the best offline combination in the online Heat Predictor:

- derive the dynamic future-access threshold `K` from each object's strict
  trailing 10-second EQ count;
- keep the V2 label as future access count in `(t, t + 10s) >= K_at_prediction`;
- disable ARF warning/background-tree/drift replacement while keeping every
  current tree learning online.

## 2. EQ as the Recent Window

Every admitted EQ item has the same ten-second deadline. Foreground prediction
drains all due items before handling the current I/O. Therefore an object's
`pending_evaluation_count`, sampled before the current item is enqueued, is its
strict trailing count in `[now - 10s, now)`.

Threshold updates follow the same lifecycle:

1. Due EQ items decrement their object count and publish the reduced count.
2. Prediction captures the current count and current `K`.
3. Successful EQ admission increments the object count and publishes the new
   count for later predictions.
4. Cancellation decrements and republishes the count.

The current I/O is excluded from its own feature and included for subsequent
I/Os. An access exactly ten seconds old is expired first.

EQ admission defines the population. Capacity-rejected or cancelled items do
not remain in the recent-count threshold. This is intentional for the current
experiment: only samples retained by the Heat Predictor pipeline influence its
threshold. Existing EQ drop telemetry exposes overload.

## 3. Label and Features

No future-label semantics change:

```text
actual_hot =
    future_access_count_in_(t,t+10s) >= K_at_prediction
```

Feature zero remains:

```text
log2p1(past_10s_eq_count_before_current_io) -
log2p1(K_at_prediction)
```

The previous-access interval and total-heat features remain unchanged.
Every item continues to retain its prediction-time K.

## 4. Threshold Population

The existing object-equal `HpFutureAccessThreshold` is retained. Its vote now
means the object's current positive EQ count instead of its latest completed
future count:

- count greater than zero inserts or replaces the object's vote;
- count zero removes the vote;
- Otsu, EMA, sparse/tracking/holding behavior, capacity, histogram and K
  conversion remain unchanged.

Future label completion no longer updates the threshold.

## 5. Disabled Drift

Production ARF uses a detector whose `update()` never reports drift for both
warning and drift positions:

- no background tree is created;
- no tree is promoted, discarded, or replaced;
- every current tree still receives `learn_one()`;
- scaler updates, batching, snapshots, voting and split logic remain unchanged;
- existing adaptation counters remain present and must stay zero.

Replay retains baseline, conservative and disabled profiles.

## 6. Verification

Deterministic probes cover:

- right-open ten-second boundaries;
- sampling K and count before current EQ admission;
- repeated same-object accesses;
- threshold vote increment, decrement, cancellation and zero removal;
- future label completion not voting its future count;
- reset;
- zero warning/drift telemetry under a changing label stream.

After probes pass:

1. build, install, run `ldconfig`, and restart OSD/MGR;
2. verify cluster state, reset and status;
3. run all five Vdbench workloads once;
4. collect MGR status every 30 seconds and final status without extra drain;
5. compare with offline `recent_disabled` and current online V2 using Accuracy,
   Balanced Accuracy, Precision, Recall, hot ratios, K, drops, adaptation
   counters and prediction latency.
