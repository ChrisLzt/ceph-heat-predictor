# Heat Predictor Trace Baseline Replay Design

## Goal

Use the existing schema-v1 Trace to determine whether the recorded online ARF
can be reproduced closely enough for controlled feature ablation. Replay must
stop at an explicit parity gate instead of turning approximate results into
algorithm conclusions.

## Scope

- Work only on `dev`; do not change the online prediction path in phase one.
- Replay each OSD independently with the current model factory and fixed seed.
- Consume evaluated records only. The recorded prediction feature vector and
  final label are authoritative inputs.
- Validate MapReduce `osd.0` before spending time on all OSDs and workloads.
- Keep changes uncommitted unless the user explicitly requests Git operations.

## Architecture

`HpTraceReplay` is a deep module whose interface accepts one binary Trace path
and replay options, then returns parity metrics and per-record replay results.
It hides binary validation, event ordering, model training, snapshot creation,
probability normalization, and metric accumulation.

Each evaluated record becomes a prediction event at `prediction_time_ns` and a
training event at `label_completion_time_ns`. Events are sorted by timestamp;
prediction precedes training on an exact tie to prevent future-label leakage.
Training events update the mutable ARF immediately in deterministic order.
Snapshots are cloned after 500 trained samples or one simulated second, matching
the online policy as closely as schema v1 permits.

The CLI writes replayed probabilities in original binary-record order so the
Python reporter can stream-join them with the existing CSV and phase metadata.

## Parity Gate

- Record and label association: 100%.
- Predicted-class agreement: at least 99%.
- Probability MAE: at most 0.01.
- Probability absolute-error P95: at most 0.05.
- Replayed-vs-recorded Accuracy difference: at most 0.2 percentage points.

Parity is reported per OSD, workload, and workload phase. Failing MapReduce
`osd.0` stops the schema-v1 experiment; feature ablation is forbidden.

## Outputs

- `osd.N.replay.tsv`: recorded/replayed probability and labels per record.
- `replay_summary.tsv`: per OSD parity and metric deltas.
- `replay_phase_summary.tsv`: phase-level parity.
- `replay_mismatches.tsv`: largest probability disagreements.
- `REPLAY_REPORT.md`: Chinese gate result and failure diagnosis.

## Schema-v1 Limitation And Upgrade

Schema v1 does not record when the background worker actually applies a sample
or which snapshot generation serves a prediction. If the parity gate fails,
schema v2 will add `snapshot_generation` to prediction records and a sidecar
event stream containing training application order/time plus snapshot publish
generation/time/trigger. The five workloads are recollected only after that
upgrade is verified with synthetic and single-workload tests.

## Validation

Synthetic tests cover header rejection, event tie ordering, per-OSD isolation,
500-sample and one-second publication, and exact replay of a generated trace.
The first real gate is MapReduce `osd.0`; all five workloads run only after it
passes. No surrogate model is accepted as evidence about the online ARF.
