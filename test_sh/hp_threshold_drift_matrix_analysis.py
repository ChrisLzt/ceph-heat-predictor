#!/usr/bin/env python3
"""Summarize threshold-source and ARF-adaptation replay matrices."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

try:
    from .hp_trace_analysis import ConfusionMatrix, PhaseIndex, TraceMetadata
except ImportError:
    from hp_trace_analysis import ConfusionMatrix, PhaseIndex, TraceMetadata


_REPLAY_FIELDS = {
    "io_sequence",
    "object_key_hash",
    "prediction_time_ns",
    "replay_hot_probability",
    "replay_label",
    "actual_label",
}


@dataclass
class MatrixMetrics:
    profile: str
    scope: str
    workload: str
    segment: str
    confusion: ConfusionMatrix = field(default_factory=ConfusionMatrix)


@dataclass
class AdaptationAggregate:
    profile: str
    workload: str
    records: int = 0
    warnings: int = 0
    drifts: int = 0
    background_promotions: int = 0
    background_discards: int = 0
    background_training_updates: int = 0


@dataclass
class MatrixAnalysis:
    metrics: dict[tuple[str, str, str, str], MatrixMetrics]
    adaptation: dict[tuple[str, str], AdaptationAggregate]


def _integer(row: dict[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid integer field {field!r}") from error


def _read_key_value_log(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read replay log {path}: {error}") from error
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def _metric(
    metrics: dict[tuple[str, str, str, str], MatrixMetrics],
    profile: str,
    scope: str,
    workload: str,
    segment: str,
) -> MatrixMetrics:
    key = (profile, scope, workload, segment)
    if key not in metrics:
        metrics[key] = MatrixMetrics(
            profile, scope, workload, segment
        )
    return metrics[key]


def analyze_matrix(run_root: Path, replay_root: Path) -> MatrixAnalysis:
    workload_root = run_root / "V2"
    workloads = [
        path for path in sorted(workload_root.iterdir())
        if path.is_dir() and (path / "phase_intervals.tsv").is_file()
    ]
    profiles = [path for path in sorted(replay_root.iterdir()) if path.is_dir()]
    if not workloads or not profiles:
        raise ValueError("matrix requires workloads and replay profiles")

    metrics: dict[tuple[str, str, str, str], MatrixMetrics] = {}
    adaptation: dict[tuple[str, str], AdaptationAggregate] = {}
    for profile_path in profiles:
        profile = profile_path.name
        for workload_path in workloads:
            workload = workload_path.name
            phase_index = PhaseIndex.from_tsv(
                workload_path / "phase_intervals.tsv"
            )
            replay_workload = profile_path / workload
            replay_paths = sorted(replay_workload.glob("osd.*.replay.tsv"))
            if not replay_paths:
                raise ValueError(
                    f"missing replay outputs for {profile} {workload}"
                )
            for replay_path in replay_paths:
                osd_name = replay_path.name.removesuffix(".replay.tsv")
                metadata = TraceMetadata.from_json(
                    workload_path / "trace" /
                    f"{osd_name}.csv.metadata.json"
                )
                try:
                    stream = replay_path.open(
                        newline="", encoding="utf-8"
                    )
                except OSError as error:
                    raise ValueError(
                        f"cannot read replay output {replay_path}: {error}"
                    ) from error
                count = 0
                with stream:
                    reader = csv.DictReader(stream, delimiter="\t")
                    missing = _REPLAY_FIELDS - set(reader.fieldnames or ())
                    if missing:
                        raise ValueError(
                            f"replay output {replay_path} is missing "
                            f"{', '.join(sorted(missing))}"
                        )
                    for row in reader:
                        predicted = _integer(row, "replay_label")
                        actual = _integer(row, "actual_label")
                        prediction_time_ns = _integer(
                            row, "prediction_time_ns"
                        )
                        phase = phase_index.lookup(
                            metadata.prediction_wall_time_ns(
                                prediction_time_ns
                            )
                        )
                        for metric in (
                            _metric(
                                metrics, profile, "global", "all", "all"
                            ),
                            _metric(
                                metrics, profile, "workload", workload, "all"
                            ),
                            _metric(
                                metrics, profile, "segment", workload,
                                phase.segment
                            ),
                            _metric(
                                metrics, profile, "phase", workload,
                                phase.phase_name
                            ),
                        ):
                            metric.confusion.add(predicted, actual)
                        count += 1
                if count != metadata.record_count:
                    raise ValueError(
                        f"record count mismatch for {profile} {workload} "
                        f"{osd_name}: {count} != {metadata.record_count}"
                    )

                log_values = _read_key_value_log(
                    replay_workload / f"{osd_name}.replay.log"
                )
                if int(log_values.get("records", "-1")) != count:
                    raise ValueError(
                        f"replay log count mismatch for {replay_path}"
                    )
                key = (profile, workload)
                aggregate = adaptation.setdefault(
                    key, AdaptationAggregate(profile, workload)
                )
                aggregate.records += count
                aggregate.warnings += int(
                    log_values.get("arf_warnings", "0")
                )
                aggregate.drifts += int(
                    log_values.get("arf_drifts", "0")
                )
                aggregate.background_promotions += int(
                    log_values.get("arf_background_promotions", "0")
                )
                aggregate.background_discards += int(
                    log_values.get("arf_background_discards", "0")
                )
                aggregate.background_training_updates += int(
                    log_values.get(
                        "arf_background_training_updates", "0"
                    )
                )
    return MatrixAnalysis(metrics, adaptation)


def write_outputs(analysis: MatrixAnalysis, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = [
            "profile", "scope", "workload", "segment", "samples",
            "tp", "fp", "tn", "fn", "accuracy", "balanced_accuracy",
            "precision", "recall", "specificity", "predicted_hot_ratio",
            "actual_hot_ratio",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for key in sorted(analysis.metrics):
            metric = analysis.metrics[key]
            matrix = metric.confusion
            writer.writerow({
                "profile": metric.profile,
                "scope": metric.scope,
                "workload": metric.workload,
                "segment": metric.segment,
                "samples": matrix.count,
                "tp": matrix.tp,
                "fp": matrix.fp,
                "tn": matrix.tn,
                "fn": matrix.fn,
                "accuracy": matrix.accuracy,
                "balanced_accuracy": matrix.balanced_accuracy,
                "precision": matrix.precision,
                "recall": matrix.recall,
                "specificity": matrix.specificity,
                "predicted_hot_ratio": matrix.predicted_hot_ratio,
                "actual_hot_ratio": matrix.actual_hot_ratio,
            })

    with (output_dir / "adaptation.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = [
            "profile", "workload", "records", "warnings", "drifts",
            "drifts_per_1000_records", "background_promotions",
            "background_discards", "background_training_updates",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for key in sorted(analysis.adaptation):
            row = analysis.adaptation[key]
            writer.writerow({
                "profile": row.profile,
                "workload": row.workload,
                "records": row.records,
                "warnings": row.warnings,
                "drifts": row.drifts,
                "drifts_per_1000_records":
                    row.drifts * 1000.0 / row.records
                    if row.records else 0.0,
                "background_promotions": row.background_promotions,
                "background_discards": row.background_discards,
                "background_training_updates":
                    row.background_training_updates,
            })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--replay-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    analysis = analyze_matrix(args.run_root, args.replay_root)
    write_outputs(analysis, args.output_dir)
    print(
        f"analyzed {len(analysis.metrics)} metric rows and "
        f"{len(analysis.adaptation)} adaptation rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
