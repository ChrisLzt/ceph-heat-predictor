#!/usr/bin/env python3
"""Compare masked-feature replay profiles against a validated baseline."""

from __future__ import annotations

import csv
import itertools
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

try:
    from .hp_trace_analysis import (
        FEATURE_NAMES,
        ConfusionMatrix,
        PhaseIndex,
        TraceMetadata,
    )
    from .hp_replay_analysis import (
        _REPLAY_FIELDS,
        _SOURCE_FIELDS,
        _rows_are_associated,
        _validated_reader,
    )
except ImportError:
    from hp_trace_analysis import (
        FEATURE_NAMES,
        ConfusionMatrix,
        PhaseIndex,
        TraceMetadata,
    )
    from hp_replay_analysis import (
        _REPLAY_FIELDS,
        _SOURCE_FIELDS,
        _rows_are_associated,
        _validated_reader,
    )

MAX_WORKLOAD_ACCURACY_DROP_PP = 0.2
MAX_MEAN_WORKLOAD_BALANCED_ACCURACY_DROP_PP = 0.2
MAX_PHASE_ACCURACY_DROP_PP = 0.5


def _integer(row: dict[str, str], name: str) -> int:
    try:
        return int(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {name}: {row.get(name)!r}") from error


def _floating(row: dict[str, str], name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {name}: {row.get(name)!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name}: {value!r}")
    return value


@dataclass(frozen=True)
class ProfileDefinition:
    name: str
    disabled_features: tuple[int, ...]

    @property
    def disabled_feature_names(self) -> str:
        if not self.disabled_features:
            return "none"
        return ",".join(FEATURE_NAMES[index] for index in self.disabled_features)


def read_profile_definitions(path: Path) -> list[ProfileDefinition]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
    except OSError as error:
        raise ValueError(f"cannot read profile file {path}: {error}") from error
    if not rows or {"profile", "disabled_features"} - set(rows[0]):
        raise ValueError("profile TSV requires profile and disabled_features")

    profiles = []
    seen_names = set()
    for row in rows:
        name = row["profile"].strip()
        if not name or name in seen_names:
            raise ValueError(f"invalid or duplicate profile: {name!r}")
        seen_names.add(name)
        text = row["disabled_features"].strip()
        if text in ("", "none"):
            disabled = ()
        else:
            try:
                disabled = tuple(sorted(int(value) for value in text.split(",")))
            except ValueError as error:
                raise ValueError(f"invalid feature list for {name}: {text!r}") from error
            if len(disabled) != len(set(disabled)) or any(
                value < 0 or value >= len(FEATURE_NAMES) for value in disabled
            ):
                raise ValueError(f"invalid feature list for {name}: {text!r}")
        profiles.append(ProfileDefinition(name, disabled))
    return profiles


@dataclass
class MetricAccumulator:
    confusion: ConfusionMatrix = field(default_factory=ConfusionMatrix)
    probability_count: int = 0
    brier_sum: float = 0.0
    calibration_count: list[int] = field(default_factory=lambda: [0] * 10)
    calibration_probability_sum: list[float] = field(
        default_factory=lambda: [0.0] * 10
    )
    calibration_actual_sum: list[int] = field(default_factory=lambda: [0] * 10)

    def add(self, probability: float, predicted: int, actual: int) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"candidate probability is out of range: {probability}")
        if predicted not in (0, 1) or actual not in (0, 1):
            raise ValueError("candidate labels must be binary")
        self.confusion.add(predicted, actual)
        self.probability_count += 1
        self.brier_sum += (probability - actual) ** 2
        bin_index = min(int(probability * 10.0), 9)
        self.calibration_count[bin_index] += 1
        self.calibration_probability_sum[bin_index] += probability
        self.calibration_actual_sum[bin_index] += actual

    def merge(self, other: "MetricAccumulator") -> None:
        self.confusion.merge(other.confusion)
        self.probability_count += other.probability_count
        self.brier_sum += other.brier_sum
        for index in range(10):
            self.calibration_count[index] += other.calibration_count[index]
            self.calibration_probability_sum[index] += (
                other.calibration_probability_sum[index]
            )
            self.calibration_actual_sum[index] += other.calibration_actual_sum[index]

    @property
    def brier(self) -> float:
        return self.brier_sum / self.probability_count if self.probability_count else 0.0

    @property
    def ece(self) -> float:
        if not self.probability_count:
            return 0.0
        weighted_error = 0.0
        for count, probability_sum, actual_sum in zip(
            self.calibration_count,
            self.calibration_probability_sum,
            self.calibration_actual_sum,
        ):
            if count:
                weighted_error += count * abs(
                    probability_sum / count - actual_sum / count
                )
        return weighted_error / self.probability_count


@dataclass(frozen=True)
class AblationMetrics:
    profile: str
    workload: str
    phase_name: str
    segment: str
    sample_count: int
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    predicted_hot_ratio: float
    actual_hot_ratio: float
    brier: float
    ece: float
    accuracy_delta_pp: float = 0.0
    balanced_accuracy_delta_pp: float = 0.0
    precision_delta_pp: float = 0.0
    recall_delta_pp: float = 0.0
    predicted_hot_ratio_delta_pp: float = 0.0
    brier_delta: float = 0.0
    ece_delta_pp: float = 0.0


@dataclass(frozen=True)
class ProfileDecision:
    profile: str
    accepted: bool
    reasons: tuple[str, ...]
    global_accuracy_delta_pp: float
    minimum_workload_accuracy_delta_pp: float
    mean_workload_balanced_accuracy_delta_pp: float
    minimum_phase_accuracy_delta_pp: float


@dataclass
class AblationAnalysis:
    profiles: list[ProfileDefinition]
    baseline_profile: str
    global_metrics: dict[str, AblationMetrics]
    workload_metrics: dict[tuple[str, str], AblationMetrics]
    phase_metrics: dict[tuple[str, str, str], AblationMetrics]
    profile_decisions: dict[str, ProfileDecision]
    runtime_seconds: dict[str, float]


def _metrics(
    profile: str,
    workload: str,
    phase_name: str,
    segment: str,
    accumulator: MetricAccumulator,
) -> AblationMetrics:
    matrix = accumulator.confusion
    return AblationMetrics(
        profile=profile,
        workload=workload,
        phase_name=phase_name,
        segment=segment,
        sample_count=matrix.count,
        tp=matrix.tp,
        fp=matrix.fp,
        tn=matrix.tn,
        fn=matrix.fn,
        accuracy=matrix.accuracy,
        balanced_accuracy=matrix.balanced_accuracy,
        precision=matrix.precision,
        recall=matrix.recall,
        predicted_hot_ratio=matrix.predicted_hot_ratio,
        actual_hot_ratio=matrix.actual_hot_ratio,
        brier=accumulator.brier,
        ece=accumulator.ece,
    )


def _with_delta(candidate: AblationMetrics, baseline: AblationMetrics) -> AblationMetrics:
    return replace(
        candidate,
        accuracy_delta_pp=(candidate.accuracy - baseline.accuracy) * 100.0,
        balanced_accuracy_delta_pp=(
            candidate.balanced_accuracy - baseline.balanced_accuracy
        ) * 100.0,
        precision_delta_pp=(candidate.precision - baseline.precision) * 100.0,
        recall_delta_pp=(candidate.recall - baseline.recall) * 100.0,
        predicted_hot_ratio_delta_pp=(
            candidate.predicted_hot_ratio - baseline.predicted_hot_ratio
        ) * 100.0,
        brier_delta=candidate.brier - baseline.brier,
        ece_delta_pp=(candidate.ece - baseline.ece) * 100.0,
    )


def _read_runtime(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    totals: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"profile", "elapsed_seconds"}
        if required - set(reader.fieldnames or []):
            raise ValueError("runtime TSV requires profile and elapsed_seconds")
        for row in reader:
            totals[row["profile"]] = totals.get(row["profile"], 0.0) + float(
                row["elapsed_seconds"]
            )
    return totals


def analyze_ablation_run(
    run_root: Path,
    replay_root: Path,
    profiles_path: Path,
    baseline_profile: str = "A0",
) -> AblationAnalysis:
    profiles = read_profile_definitions(profiles_path)
    profile_names = {profile.name for profile in profiles}
    if baseline_profile not in profile_names:
        raise ValueError(f"baseline profile is missing: {baseline_profile}")

    global_accumulators: dict[str, MetricAccumulator] = {}
    workload_accumulators: dict[tuple[str, str], MetricAccumulator] = {}
    phase_accumulators: dict[
        tuple[str, str, str], tuple[str, MetricAccumulator]
    ] = {}
    workload_paths = [
        path for path in sorted(run_root.iterdir())
        if path.is_dir() and (path / "phase_intervals.tsv").is_file()
        and (path / "trace").is_dir()
    ]
    if not workload_paths:
        raise ValueError(f"no workloads found under {run_root}")

    for profile in profiles:
        global_accumulator = global_accumulators.setdefault(
            profile.name, MetricAccumulator()
        )
        for workload_path in workload_paths:
            workload = workload_path.name
            phases = PhaseIndex.from_tsv(workload_path / "phase_intervals.tsv")
            source_paths = sorted((workload_path / "trace").glob("osd.*.csv"))
            if not source_paths:
                raise ValueError(f"no Trace CSV files for {workload}")
            workload_accumulator = workload_accumulators.setdefault(
                (profile.name, workload), MetricAccumulator()
            )
            for source_path in source_paths:
                metadata = TraceMetadata.from_json(
                    source_path.with_suffix(".csv.metadata.json")
                )
                replay_path = (
                    replay_root / profile.name / workload
                    / f"osd.{metadata.osd_id}.replay.tsv"
                )
                if not replay_path.is_file():
                    raise ValueError(f"missing replay output: {replay_path}")
                source_stream, source_reader = _validated_reader(
                    source_path, _SOURCE_FIELDS, ","
                )
                replay_stream, replay_reader = _validated_reader(
                    replay_path, _REPLAY_FIELDS, "\t"
                )
                count = 0
                try:
                    for source, replay in itertools.zip_longest(
                        source_reader, replay_reader
                    ):
                        if source is None or replay is None:
                            raise ValueError(
                                f"record count mismatch for {profile.name} "
                                f"{workload} osd.{metadata.osd_id}"
                            )
                        if not _rows_are_associated(source, replay):
                            raise ValueError(
                                f"record association mismatch for {profile.name} "
                                f"{workload} osd.{metadata.osd_id} row {count + 2}"
                            )
                        probability = _floating(replay, "replay_hot_probability")
                        predicted = _integer(replay, "replay_label")
                        actual = _integer(source, "actual_label")
                        phase = phases.lookup(metadata.prediction_wall_time_ns(
                            _integer(source, "prediction_time_ns")
                        ))
                        phase_key = (profile.name, workload, phase.phase_name)
                        if phase_key not in phase_accumulators:
                            phase_accumulators[phase_key] = (
                                phase.segment, MetricAccumulator()
                            )
                        phase_accumulator = phase_accumulators[phase_key][1]
                        global_accumulator.add(probability, predicted, actual)
                        workload_accumulator.add(probability, predicted, actual)
                        phase_accumulator.add(probability, predicted, actual)
                        count += 1
                finally:
                    source_stream.close()
                    replay_stream.close()
                if count != metadata.record_count:
                    raise ValueError(
                        f"metadata count mismatch for {profile.name} "
                        f"{workload} osd.{metadata.osd_id}"
                    )

    global_metrics = {
        profile: _metrics(profile, "all", "all", "all", accumulator)
        for profile, accumulator in global_accumulators.items()
    }
    workload_metrics = {
        key: _metrics(key[0], key[1], "all", "all", accumulator)
        for key, accumulator in workload_accumulators.items()
    }
    phase_metrics = {
        key: _metrics(key[0], key[1], key[2], segment, accumulator)
        for key, (segment, accumulator) in phase_accumulators.items()
    }

    baseline_global = global_metrics[baseline_profile]
    for profile in profile_names:
        global_metrics[profile] = _with_delta(
            global_metrics[profile], baseline_global
        )
    for key, candidate in list(workload_metrics.items()):
        baseline = workload_metrics[(baseline_profile, key[1])]
        workload_metrics[key] = _with_delta(candidate, baseline)
    for key, candidate in list(phase_metrics.items()):
        baseline = phase_metrics[(baseline_profile, key[1], key[2])]
        phase_metrics[key] = _with_delta(candidate, baseline)

    decisions = {}
    workload_names = sorted(path.name for path in workload_paths)
    for profile in profiles:
        if profile.name == baseline_profile:
            decisions[profile.name] = ProfileDecision(
                profile.name, True, (), 0.0, 0.0, 0.0, 0.0
            )
            continue
        global_delta = global_metrics[profile.name].accuracy_delta_pp
        workload_rows = [
            workload_metrics[(profile.name, workload)]
            for workload in workload_names
        ]
        phase_rows = [
            row for (candidate_profile, _, _), row in phase_metrics.items()
            if candidate_profile == profile.name
        ]
        min_workload = min(row.accuracy_delta_pp for row in workload_rows)
        mean_workload_bacc = sum(
            row.balanced_accuracy_delta_pp for row in workload_rows
        ) / len(workload_rows)
        min_phase = min(row.accuracy_delta_pp for row in phase_rows)
        reasons = []
        if global_delta < 0.0:
            reasons.append("global_accuracy")
        if min_workload < -MAX_WORKLOAD_ACCURACY_DROP_PP:
            reasons.append("workload_accuracy")
        if mean_workload_bacc < -MAX_MEAN_WORKLOAD_BALANCED_ACCURACY_DROP_PP:
            reasons.append("mean_workload_balanced_accuracy")
        if min_phase < -MAX_PHASE_ACCURACY_DROP_PP:
            reasons.append("phase_accuracy")
        decisions[profile.name] = ProfileDecision(
            profile=profile.name,
            accepted=not reasons,
            reasons=tuple(reasons),
            global_accuracy_delta_pp=global_delta,
            minimum_workload_accuracy_delta_pp=min_workload,
            mean_workload_balanced_accuracy_delta_pp=mean_workload_bacc,
            minimum_phase_accuracy_delta_pp=min_phase,
        )

    return AblationAnalysis(
        profiles=profiles,
        baseline_profile=baseline_profile,
        global_metrics=global_metrics,
        workload_metrics=workload_metrics,
        phase_metrics=phase_metrics,
        profile_decisions=decisions,
        runtime_seconds=_read_runtime(replay_root / "runtime.tsv"),
    )


_METRIC_FIELDS = tuple(AblationMetrics.__dataclass_fields__)


def _write_metrics(path: Path, rows: list[AblationMetrics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_METRIC_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_ablation_outputs(analysis: AblationAnalysis, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    profiles = {profile.name: profile for profile in analysis.profiles}
    summary_fields = (
        "profile", "disabled_features", "disabled_feature_names",
        "sample_count", "accuracy", "balanced_accuracy", "precision",
        "recall", "predicted_hot_ratio", "actual_hot_ratio", "brier", "ece",
        "accuracy_delta_pp", "balanced_accuracy_delta_pp",
        "minimum_workload_accuracy_delta_pp",
        "mean_workload_balanced_accuracy_delta_pp",
        "minimum_phase_accuracy_delta_pp", "runtime_seconds", "accepted",
        "reasons",
    )
    with (output_root / "profile_summary.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields, delimiter="\t")
        writer.writeheader()
        for profile in analysis.profiles:
            metrics = analysis.global_metrics[profile.name]
            decision = analysis.profile_decisions[profile.name]
            writer.writerow({
                "profile": profile.name,
                "disabled_features": ",".join(map(str, profile.disabled_features)) or "none",
                "disabled_feature_names": profile.disabled_feature_names,
                "sample_count": metrics.sample_count,
                "accuracy": metrics.accuracy,
                "balanced_accuracy": metrics.balanced_accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "predicted_hot_ratio": metrics.predicted_hot_ratio,
                "actual_hot_ratio": metrics.actual_hot_ratio,
                "brier": metrics.brier,
                "ece": metrics.ece,
                "accuracy_delta_pp": metrics.accuracy_delta_pp,
                "balanced_accuracy_delta_pp": metrics.balanced_accuracy_delta_pp,
                "minimum_workload_accuracy_delta_pp": decision.minimum_workload_accuracy_delta_pp,
                "mean_workload_balanced_accuracy_delta_pp": decision.mean_workload_balanced_accuracy_delta_pp,
                "minimum_phase_accuracy_delta_pp": decision.minimum_phase_accuracy_delta_pp,
                "runtime_seconds": analysis.runtime_seconds.get(profile.name, 0.0),
                "accepted": decision.accepted,
                "reasons": ",".join(decision.reasons),
            })

    workload_rows = [
        analysis.workload_metrics[key]
        for key in sorted(analysis.workload_metrics)
    ]
    phase_rows = [
        analysis.phase_metrics[key]
        for key in sorted(analysis.phase_metrics)
    ]
    _write_metrics(output_root / "workload_metrics.tsv", workload_rows)
    _write_metrics(output_root / "phase_metrics.tsv", phase_rows)
    _write_metrics(
        output_root / "deltas_vs_baseline.tsv",
        [*analysis.global_metrics.values(), *workload_rows, *phase_rows],
    )

    lines = [
        "# Heat Predictor Feature 消融报告",
        "",
        f"- Baseline：`{analysis.baseline_profile}`",
        f"- 样本数：{analysis.global_metrics[analysis.baseline_profile].sample_count}",
        "- 硬门槛：全局 Accuracy 不下降；任一负载 Accuracy 下降不超过 0.2 pp；",
        "  五负载平均 Balanced Accuracy 下降不超过 0.2 pp；任一 phase Accuracy 下降不超过 0.5 pp。",
        "",
        "| Profile | 禁用 Feature | Accuracy | ΔAccuracy(pp) | BAcc | ΔBAcc(pp) | 最差负载(pp) | 最差 phase(pp) | 耗时(s) | 结论 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for profile in analysis.profiles:
        metrics = analysis.global_metrics[profile.name]
        decision = analysis.profile_decisions[profile.name]
        lines.append(
            f"| {profile.name} | {profile.disabled_feature_names} | "
            f"{metrics.accuracy:.4%} | {metrics.accuracy_delta_pp:+.4f} | "
            f"{metrics.balanced_accuracy:.4%} | "
            f"{metrics.balanced_accuracy_delta_pp:+.4f} | "
            f"{decision.minimum_workload_accuracy_delta_pp:+.4f} | "
            f"{decision.minimum_phase_accuracy_delta_pp:+.4f} | "
            f"{analysis.runtime_seconds.get(profile.name, 0.0):.2f} | "
            f"{'保留候选' if decision.accepted else '淘汰'} |"
        )
    accepted = [
        profile.name for profile in analysis.profiles
        if profile.name != analysis.baseline_profile
        and analysis.profile_decisions[profile.name].accepted
    ]
    lines.extend([
        "",
        "## 结论",
        "",
        "通过门槛的候选：" + (", ".join(accepted) if accepted else "无") + "。",
    ])
    (output_root / "REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
