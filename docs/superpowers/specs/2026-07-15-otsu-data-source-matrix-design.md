# Otsu Data Source Matrix Design

## Goal

Use one immutable five-workload Vdbench dataset to compare three Otsu vote
sources while keeping the 10-second label, fixed hot-threshold EMA, fixed
prediction threshold, features, model and training weights unchanged.

## Profiles

- `D0`: retain the latest completed 10-second future-added-heat vote for each
  object. A newer completion replaces the object's previous vote.
- `D1`: retain every completed evaluation's 10-second future-added-heat vote.
  Votes expire by their completion deadline and are not deduplicated by object.
- `D2`: retain the latest total heat at the evaluation deadline for each object.
  Histogram input is clamped to a minimum heat of `10`; labels still use
  future-added heat.

`HP_OTSU_PROFILE` remains fixed at `HP_OTSU_PROFILE_FIXED_EMA` and
`HP_ENABLE_PREDICTION_CALIBRATION` remains `0`. A new compile-time
`HP_OTSU_DATA_SOURCE` selects D0, D1 or D2 so threshold smoothing is not
conflated with vote-source selection.

## Histogram

All profiles use the same 10,000 `log1p` bins and 60 one-second slots. D0 and
D2 maintain an object-to-latest-vote index so replacement and expiry remain
O(1). D1 only increments slot and aggregate bin counts; slot expiry subtracts
the complete per-second histogram, so memory is fixed even when the I/O rate
increases.

The retained-count perf field is named `hp_otsu_histogram_vote_count` because
its unit is objects in D0/D2 and completed I/O votes in D1. Enum declaration,
perf registration, refresh and MGR aggregation use the same field order.

## Experiment Flow

Validate all five formal Vdbench models, disable and reset Heat Predictor,
prepare the five 112.5 GiB datasets sequentially, verify the resulting anchors,
then enable and reset Heat Predictor. The same prepared data is used for all
profiles.

For D0, D1 and D2, perform a full build/install/restart, wait for all PGs to be
`active+clean`, and run MapReduce, GraphChi, HPC WRF, AI training and AI
inference once. Reset before every workload. Save MGR `hp_status` every 30
seconds and save one final status only after pending evaluation, awaiting
prediction and training queues are empty.

A `gpt-5.6-luna` subagent performs read-only monitoring. It observes the exact
runner process, run log, sample directory, Ceph daemon state and 30-second
sample continuity. It may not modify code, reset Heat Predictor, restart
services or stop a workload.

## Validation And Report

Before formal workloads, run the standalone algorithm and performance probes,
the matrix runner contract test, a full Ceph build/install/restart, and a live
reset/status check. Every formal result must satisfy:

- `hp_labeled_io_total + hp_eval_drop_count == hp_io_count` after drain;
- pending, awaiting prediction and training queue are zero;
- evaluation and training drops are zero;
- all OSDs report and all PGs remain `active+clean`.

The Chinese report compares accuracy, balanced accuracy, precision, recall,
TP/FP/TN/FN, predicted and actual hot percentages, Otsu threshold/confidence,
retained vote count, prediction latency, Vdbench rate and MiB/s. Each workload
runs once; anomalous results are reported without automatic repetition.
