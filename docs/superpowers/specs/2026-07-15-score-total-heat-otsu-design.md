# Score-Based Total-Heat Otsu Design

## Goal

Restore the score-based, per-object total-heat Otsu algorithm and make the
prediction inputs, dynamic heat threshold and actual label use one consistent
definition: whether an object is hot at the end of the future 10-second window.

## Prediction Semantics

For an I/O arriving at `t0`, the predictor uses state available at `t0` and
predicts whether the object total heat at `deadline = t0 + 10 s` exceeds the
dynamic total-heat threshold at that deadline.

The actual label is:

```text
deadline_total_heat > total_heat_threshold(deadline)
```

`future_window_added_heat` remains available for diagnostics, but it does not
define the label or update Otsu in this version. This prevents a threshold
derived from total heat from being applied directly to an incompatible
future-added-heat distribution.

## Total-Heat Score Window

Each object contributes at most one current-total-heat vote. A new access:

1. decays the object's previous total heat to the current monotonic time;
2. adds `HP_HEAT_INCREMENT`;
3. converts the resulting heat to a time-normalized score;
4. replaces the object's previous score in the threshold window and Otsu
   histogram.

The normalization is:

```text
score = ln(total_heat) - decay_log_factor_per_ns * timestamp_ns
total_heat(score, t) = exp(score + decay_log_factor_per_ns * t)
```

The score lets an inactive object's heat continue to decay without rewriting
every object on every I/O. The threshold window retains the latest score per
object and is bounded by `HP_HEAT_LABEL_THRESHOLD_OBJECT_CAPACITY`. Capacity
eviction removes the oldest updated object.

The physical lower histogram bound is the score corresponding to total heat
`10` at the current time. When that score bin advances, older lower bins are
merged into the new lower-bound bin. Removing or replacing an object whose
stored score is below the current bound erases it from the current lower-bound
bin.

The score histogram is a fixed 800-bin array with logarithmic width `0.01`.
The representable total-heat ratio is therefore `exp(800 * 0.01) = exp(8)`;
with lower bound `10`, the engineering upper bound is about `29810` (reported
as approximately `30000`). Heat above this bound is clamped before converting
it to score. The lower score bin is the moving array origin, so advancing time
shifts the array left and physically merges expired lower bins into bin zero.
Each retained object stores its absolute score and the absolute bin token used
for insertion. Replacement or capacity eviction converts that token against
the current array origin, so a lower-bound shift does not require rewriting all
objects. The token lives in the existing per-object threshold entry; there is
no separate object-to-bin map.

## Otsu And Threshold Smoothing

Otsu scans the occupied score bins and selects the split with maximum
between-class variance. It retains the current separation and sharpness
statistics. The effective threshold remains the fixed-EMA profile with a
one-second reference gain of `0.10`; EMA is performed in score space.

At each Otsu update, the current heat threshold and candidate are converted to
scores at the same monotonic timestamp, smoothed, then converted back to a heat
threshold. The published heat threshold remains fixed between Otsu updates;
only the histogram's score origin moves with time. This matches the original
score-window behavior and prevents the threshold from decaying to its lower
bound between updates.

Published heat thresholds are versioned by effective monotonic time. Feature
generation uses the current version, while delayed label completion uses the
latest version that was already effective at the sample's fixed deadline.
This keeps a later Otsu update from retroactively changing an old label.

The D2 object-total profile becomes the default and uses this score-window
path. The existing D0 object-added and D1 per-I/O-added profiles remain
compile-time experimental alternatives and continue using the 60-second
completed-window histogram. Otsu lifetime in the default D2 path is controlled
only by per-object replacement and threshold-window capacity, matching the
original score-window design.

## Features

Features use only information available at prediction time. The heat inputs
remain:

- total heat immediately after the current access;
- total-heat margin relative to the dynamic total-heat threshold;
- access recency, long-window access count, normalized total heat and
  short-versus-long access-rate change.

No future-added heat enters a feature. Existing names may be clarified where
needed, but this change does not add a new feature dimension.

## Reset And Reporting

Reset clears the per-object score window, histogram, effective threshold state,
pending evaluations and retained object heat state. Existing MGR/OSD threshold,
Otsu confidence, confusion-matrix and future-added-heat diagnostic fields remain
available. Histogram vote count continues to mean the number of retained
objects.

## Added-Heat Otsu Follow-Up

It is mathematically possible to decay a completed future-added-heat vote with
the same normalized-score transform. That would make an old window's measured
increment shrink after its deadline, so it would no longer represent the
original window intensity. It would also accumulate old objects at the lower
bound unless combined with expiry or capacity eviction.

For a future-added-heat threshold, the preferred design is therefore to keep
each completed value unchanged and decay its influence through time expiry or
weighted Otsu. A second option is a two-stage zero-inflated model: separate
zero from positive future heat, then run Otsu only over positive values. Neither
option is part of this first total-heat version.

## Verification

The standalone probe must first fail against the current completed-window Otsu,
then verify:

- score round trips and heat decay at arbitrary timestamps;
- physical lower-bound movement to total heat `10`;
- one latest total-heat vote per object and correct replacement;
- deadline labels compare deadline total heat with the deadline threshold;
- feature heat values and threshold margin use current total heat;
- reset removes all threshold-window and histogram state.

After local probes pass, run the full Ceph build/install process, `ldconfig`,
restart affected daemons, wait for all PGs to become `active+clean`, then verify
live reset and MGR/OSD status. Run all five formal Vdbench workloads once and
sample MGR `hp status` every 30 seconds. Save one final `hp_status.json` and a
Chinese comparison report with elapsed time and Vdbench throughput.
