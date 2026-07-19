#!/usr/bin/env python3
"""Streaming parity analysis for Heat Predictor baseline replay output."""

from __future__ import annotations

import csv
import heapq
import itertools
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    from .hp_trace_analysis import PhaseIndex, TraceMetadata
except ImportError:
    from hp_trace_analysis import PhaseIndex, TraceMetadata


MIN_CLASS_AGREEMENT = 0.99
MAX_PROBABILITY_MAE = 0.01
MAX_PROBABILITY_P95 = 0.05
MAX_ACCURACY_DELTA_PERCENT_POINTS = 0.2
_REPLAY_NAME = re.compile(r"^osd\.(\d+)\.replay\.tsv$")

_SOURCE_FIELDS = {
    "io_sequence",
    "object_key_hash",
    "prediction_time_ns",
    "label_completion_time_ns",
    "predicted_hot_probability",
    "hot_predict_threshold",
    "predicted_label",
    "actual_label",
}
_REPLAY_FIELDS = {
    "io_sequence",
    "object_key_hash",
    "prediction_time_ns",
    "label_completion_time_ns",
    "online_hot_probability",
    "replay_hot_probability",
    "probability_abs_error",
    "hot_predict_threshold",
    "online_label",
    "replay_label",
    "actual_label",
    "cold_start_fallback",
}


def _float(row: dict[str, str], name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {name}: {row.get(name)!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name}: {value!r}")
    return value


def _integer(row: dict[str, str], name: str) -> int:
    try:
        return int(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {name}: {row.get(name)!r}") from error


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


@dataclass
class ReplayAccumulator:
    source_count: int = 0
    replay_count: int = 0
    association_count: int = 0
    class_agreement_count: int = 0
    online_correct_count: int = 0
    replay_correct_count: int = 0
    cold_start_fallback_count: int = 0
    probability_abs_errors: list[float] = field(default_factory=list)

    def add_missing(self, source_present: bool, replay_present: bool) -> None:
        self.source_count += int(source_present)
        self.replay_count += int(replay_present)

    def add_pair(
        self,
        source: dict[str, str],
        replay: dict[str, str],
    ) -> tuple[bool, float]:
        self.source_count += 1
        self.replay_count += 1
        associated = _rows_are_associated(source, replay)
        if not associated:
            return False, math.inf

        self.association_count += 1
        online_label = _integer(source, "predicted_label")
        replay_label = _integer(replay, "replay_label")
        actual_label = _integer(source, "actual_label")
        if online_label not in (0, 1) or replay_label not in (0, 1) or actual_label not in (0, 1):
            raise ValueError("replay labels must be binary")
        self.class_agreement_count += int(online_label == replay_label)
        self.online_correct_count += int(online_label == actual_label)
        self.replay_correct_count += int(replay_label == actual_label)
        self.cold_start_fallback_count += _integer(replay, "cold_start_fallback")
        absolute_error = abs(
            _float(source, "predicted_hot_probability")
            - _float(replay, "replay_hot_probability")
        )
        self.probability_abs_errors.append(absolute_error)
        return True, absolute_error


@dataclass(frozen=True)
class ReplaySummary:
    workload: str
    osd_id: int
    phase_name: str
    record_count: int
    source_count: int
    replay_count: int
    association_count: int
    association_ratio: float
    class_agreement: float
    probability_mae: float
    probability_rmse: float
    probability_abs_error_p95: float
    online_accuracy: float
    replay_accuracy: float
    accuracy_delta_percent_points: float
    cold_start_fallback_count: int
    gate_passed: bool
    gate_failures: tuple[str, ...]


@dataclass(frozen=True)
class ReplayMismatch:
    workload: str
    osd_id: int
    phase_name: str
    io_sequence: int
    object_key_hash: int
    online_hot_probability: float
    replay_hot_probability: float
    probability_abs_error: float
    online_label: int
    replay_label: int
    actual_label: int


@dataclass
class ReplayRunAnalysis:
    run_root: Path
    replay_root: Path
    summaries: list[ReplaySummary] = field(default_factory=list)
    phase_summaries: list[ReplaySummary] = field(default_factory=list)
    mismatches: list[ReplayMismatch] = field(default_factory=list)

    @property
    def gate_passed(self) -> bool:
        checked = [*self.summaries, *self.phase_summaries]
        return bool(self.summaries) and bool(self.phase_summaries) and all(
            summary.gate_passed for summary in checked
        )


def _rows_are_associated(
    source: dict[str, str], replay: dict[str, str]
) -> bool:
    integer_pairs = (
        ("io_sequence", "io_sequence"),
        ("object_key_hash", "object_key_hash"),
        ("prediction_time_ns", "prediction_time_ns"),
        ("label_completion_time_ns", "label_completion_time_ns"),
        ("predicted_label", "online_label"),
        ("actual_label", "actual_label"),
    )
    if any(_integer(source, left) != _integer(replay, right)
           for left, right in integer_pairs):
        return False
    float_pairs = (
        ("predicted_hot_probability", "online_hot_probability"),
        ("hot_predict_threshold", "hot_predict_threshold"),
    )
    return all(
        math.isclose(
            _float(source, left), _float(replay, right),
            rel_tol=1e-12, abs_tol=1e-15,
        )
        for left, right in float_pairs
    )


def _make_summary(
    workload: str,
    osd_id: int,
    phase_name: str,
    accumulator: ReplayAccumulator,
) -> ReplaySummary:
    record_count = max(accumulator.source_count, accumulator.replay_count)
    associated = accumulator.association_count
    errors = accumulator.probability_abs_errors
    probability_mae = _ratio(sum(errors), len(errors))
    probability_rmse = math.sqrt(_ratio(sum(value * value for value in errors), len(errors)))
    online_accuracy = _ratio(accumulator.online_correct_count, associated)
    replay_accuracy = _ratio(accumulator.replay_correct_count, associated)
    association_ratio = _ratio(accumulator.association_count, record_count)
    class_agreement = _ratio(accumulator.class_agreement_count, associated)
    accuracy_delta = (replay_accuracy - online_accuracy) * 100.0
    p95 = _p95(errors)

    failures = []
    if association_ratio != 1.0:
        failures.append("record_label_association")
    if class_agreement < MIN_CLASS_AGREEMENT:
        failures.append("class_agreement")
    if probability_mae > MAX_PROBABILITY_MAE:
        failures.append("probability_mae")
    if p95 > MAX_PROBABILITY_P95:
        failures.append("probability_abs_error_p95")
    if abs(accuracy_delta) > MAX_ACCURACY_DELTA_PERCENT_POINTS:
        failures.append("accuracy_delta")
    return ReplaySummary(
        workload=workload,
        osd_id=osd_id,
        phase_name=phase_name,
        record_count=record_count,
        source_count=accumulator.source_count,
        replay_count=accumulator.replay_count,
        association_count=accumulator.association_count,
        association_ratio=association_ratio,
        class_agreement=class_agreement,
        probability_mae=probability_mae,
        probability_rmse=probability_rmse,
        probability_abs_error_p95=p95,
        online_accuracy=online_accuracy,
        replay_accuracy=replay_accuracy,
        accuracy_delta_percent_points=accuracy_delta,
        cold_start_fallback_count=accumulator.cold_start_fallback_count,
        gate_passed=not failures,
        gate_failures=tuple(failures),
    )


def _validated_reader(path: Path, required: set[str], delimiter: str):
    stream = path.open(newline="", encoding="utf-8")
    reader = csv.DictReader(stream, delimiter=delimiter)
    missing = required - set(reader.fieldnames or [])
    if missing:
        stream.close()
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return stream, reader


def _analyze_file(
    run_root: Path,
    replay_path: Path,
    workload: str,
    osd_id: int,
) -> tuple[ReplaySummary, list[ReplaySummary], list[ReplayMismatch]]:
    workload_root = run_root / workload
    source_path = workload_root / "trace" / f"osd.{osd_id}.csv"
    metadata_path = source_path.with_suffix(".csv.metadata.json")
    if not source_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"missing source Trace CSV or metadata for {workload} osd.{osd_id}")
    metadata = TraceMetadata.from_json(metadata_path)
    if metadata.osd_id != osd_id or metadata.phase != workload:
        raise ValueError(f"source metadata identity mismatch for {workload} osd.{osd_id}")
    phases = PhaseIndex.from_tsv(workload_root / "phase_intervals.tsv")

    overall = ReplayAccumulator()
    phase_accumulators: dict[tuple[int, str], ReplayAccumulator] = {}
    mismatch_heap: list[tuple[float, int, ReplayMismatch]] = []
    source_stream, source_reader = _validated_reader(source_path, _SOURCE_FIELDS, ",")
    replay_stream, replay_reader = _validated_reader(replay_path, _REPLAY_FIELDS, "\t")
    try:
        for source, replay in itertools.zip_longest(source_reader, replay_reader):
            if source is None or replay is None:
                overall.add_missing(source is not None, replay is not None)
                continue
            prediction_time_ns = _integer(source, "prediction_time_ns")
            phase = phases.lookup(metadata.prediction_wall_time_ns(prediction_time_ns))
            phase_key = (phase.phase_index, phase.phase_name)
            phase_accumulator = phase_accumulators.setdefault(
                phase_key, ReplayAccumulator()
            )
            associated, absolute_error = overall.add_pair(source, replay)
            phase_accumulator.add_pair(source, replay)
            if associated:
                mismatch = ReplayMismatch(
                    workload=workload,
                    osd_id=osd_id,
                    phase_name=phase.phase_name,
                    io_sequence=_integer(source, "io_sequence"),
                    object_key_hash=_integer(source, "object_key_hash"),
                    online_hot_probability=_float(
                        source, "predicted_hot_probability"
                    ),
                    replay_hot_probability=_float(
                        replay, "replay_hot_probability"
                    ),
                    probability_abs_error=absolute_error,
                    online_label=_integer(source, "predicted_label"),
                    replay_label=_integer(replay, "replay_label"),
                    actual_label=_integer(source, "actual_label"),
                )
                heap_entry = (
                    mismatch.probability_abs_error,
                    -mismatch.io_sequence,
                    mismatch,
                )
                if len(mismatch_heap) < 100:
                    heapq.heappush(mismatch_heap, heap_entry)
                elif heap_entry[:2] > mismatch_heap[0][:2]:
                    heapq.heapreplace(mismatch_heap, heap_entry)
    finally:
        source_stream.close()
        replay_stream.close()
    if overall.source_count != metadata.record_count:
        raise ValueError(
            f"source row count {overall.source_count} does not match metadata "
            f"record_count {metadata.record_count} for {workload} osd.{osd_id}"
        )
    summary = _make_summary(workload, osd_id, "all", overall)
    phase_summaries = [
        _make_summary(workload, osd_id, phase_name, accumulator)
        for (_, phase_name), accumulator in sorted(phase_accumulators.items())
    ]
    mismatches = [entry[2] for entry in mismatch_heap]
    return summary, phase_summaries, mismatches


def analyze_replay_run(run_root: Path, replay_root: Path) -> ReplayRunAnalysis:
    run_root = run_root.resolve()
    replay_root = replay_root.resolve()
    if not run_root.is_dir() or not replay_root.is_dir():
        raise ValueError("run root and replay root must be directories")
    replay_paths = sorted(replay_root.rglob("osd.*.replay.tsv"))
    if not replay_paths:
        raise ValueError(f"no replay TSV files found under {replay_root}")

    analysis = ReplayRunAnalysis(run_root=run_root, replay_root=replay_root)
    seen: set[tuple[str, int]] = set()
    for replay_path in replay_paths:
        match = _REPLAY_NAME.match(replay_path.name)
        if match is None:
            continue
        workload = replay_path.parent.name
        osd_id = int(match.group(1))
        identity = (workload, osd_id)
        if identity in seen:
            raise ValueError(f"duplicate replay output for {workload} osd.{osd_id}")
        seen.add(identity)
        summary, phase_summaries, mismatches = _analyze_file(
            run_root, replay_path, workload, osd_id
        )
        analysis.summaries.append(summary)
        analysis.phase_summaries.extend(phase_summaries)
        analysis.mismatches.extend(mismatches)
    analysis.mismatches.sort(
        key=lambda row: (-row.probability_abs_error, row.workload, row.osd_id, row.io_sequence)
    )
    return analysis


_SUMMARY_FIELDS = (
    "workload",
    "osd_id",
    "phase_name",
    "record_count",
    "source_count",
    "replay_count",
    "association_count",
    "association_ratio",
    "class_agreement",
    "probability_mae",
    "probability_rmse",
    "probability_abs_error_p95",
    "online_accuracy",
    "replay_accuracy",
    "accuracy_delta_percent_points",
    "cold_start_fallback_count",
    "gate_passed",
    "gate_failures",
)


def _write_summary_tsv(path: Path, summaries: list[ReplaySummary]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_SUMMARY_FIELDS, delimiter="\t")
        writer.writeheader()
        for summary in summaries:
            row = summary.__dict__.copy()
            row["gate_failures"] = ",".join(summary.gate_failures)
            writer.writerow(row)


def write_replay_outputs(analysis: ReplayRunAnalysis, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_summary_tsv(output_root / "replay_summary.tsv", analysis.summaries)
    _write_summary_tsv(
        output_root / "replay_phase_summary.tsv", analysis.phase_summaries
    )
    mismatch_fields = tuple(ReplayMismatch.__dataclass_fields__)
    with (output_root / "replay_mismatches.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=mismatch_fields, delimiter="\t")
        writer.writeheader()
        for mismatch in analysis.mismatches[:100]:
            writer.writerow(mismatch.__dict__)

    status = "通过" if analysis.gate_passed else "失败"
    passed_phases = sum(
        summary.gate_passed for summary in analysis.phase_summaries
    )
    lines = [
        "# Heat Predictor Trace Replay 一致性报告",
        "",
        f"- Gate 结果：**{status}**",
        f"- Replay 文件数：{len(analysis.summaries)}",
        f"- 总记录数：{sum(row.record_count for row in analysis.summaries)}",
        f"- Phase 通过数：{passed_phases}/{len(analysis.phase_summaries)}",
        "- 阈值：关联 100%，类别一致率 >= 99%，MAE <= 0.01，"
        "P95 <= 0.05，Accuracy 差 <= 0.2 个百分点。",
        "",
        "| 负载 | OSD | 记录数 | 关联率 | 类别一致率 | MAE | P95 | Accuracy 差(pp) | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for summary in analysis.summaries:
        lines.append(
            f"| {summary.workload} | {summary.osd_id} | {summary.record_count} | "
            f"{summary.association_ratio:.5%} | {summary.class_agreement:.5%} | "
            f"{summary.probability_mae:.6f} | "
            f"{summary.probability_abs_error_p95:.6f} | "
            f"{summary.accuracy_delta_percent_points:.4f} | "
            f"{'通过' if summary.gate_passed else '失败'} |"
        )
        if summary.gate_failures:
            lines.append(
                f"\n失败项（{summary.workload} osd.{summary.osd_id}）："
                f"{', '.join(summary.gate_failures)}。"
            )
    lines.extend([
        "",
        "## 结论",
        "",
        "Schema v1 baseline replay 已达到 OSD 与 phase 两级一致性门槛，"
        "可以基于同一批 Trace 开始受控 feature 消融。"
        if analysis.gate_passed
        else "Schema v1 baseline replay 未达到一致性门槛，不得据此进行 feature 消融。",
    ])
    (output_root / "REPLAY_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
