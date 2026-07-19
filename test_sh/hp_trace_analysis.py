#!/usr/bin/env python3
"""Streaming diagnostics for Heat Predictor trace CSV sessions."""

from __future__ import annotations

import bisect
import csv
import json
import math
from array import array
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


TRACE_MAGIC = "HPTRACE1"
TRACE_SCHEMA_VERSION = 2
TRACE_HEADER_SIZE = 192
TRACE_RECORD_SIZE = 200
TRACE_MAX_FEATURES = 8
FEATURE_NAMES = (
    "current_heat_log_margin",
    "previous_access_interval_encoded",
    "current_heat_log2p1",
    "short_long_access_rate_log_margin",
)
EXPECTED_OUTPUT_FILES = {
    "summary.json",
    "workload_summary.tsv",
    "phase_error.tsv",
    "calibration.tsv",
    "margin_analysis.tsv",
    "feature_stats.tsv",
    "feature_correlation.tsv",
    "object_stats.tsv",
    "steady_error_confidence.tsv",
    "prediction_outcome_summary.tsv",
    "prediction_probability_bins.tsv",
    "prediction_feature_profiles.tsv",
    "prediction_object_summary.tsv",
    "prediction_time_series.tsv",
    "REPORT.md",
}
STEADY_WARMUP_NS = 120 * 1_000_000_000
STEADY_END_NS = 600 * 1_000_000_000
ERROR_CONFIDENCE_EDGES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


@dataclass
class ConfusionMatrix:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def add(self, predicted: int, actual: int) -> None:
        if predicted not in (0, 1) or actual not in (0, 1):
            raise ValueError("predicted and actual labels must be binary")
        if predicted == 1 and actual == 1:
            self.tp += 1
        elif predicted == 1:
            self.fp += 1
        elif actual == 0:
            self.tn += 1
        else:
            self.fn += 1

    def merge(self, other: "ConfusionMatrix") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.tn += other.tn
        self.fn += other.fn

    @property
    def count(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        return _ratio(self.tp + self.tn, self.count)

    @property
    def balanced_accuracy(self) -> float:
        return (self.recall + self.specificity) / 2.0

    @property
    def precision(self) -> float:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def specificity(self) -> float:
        return _ratio(self.tn, self.tn + self.fp)

    @property
    def false_positive_rate(self) -> float:
        return _ratio(self.fp, self.fp + self.tn)

    @property
    def false_negative_rate(self) -> float:
        return _ratio(self.fn, self.fn + self.tp)

    @property
    def predicted_hot_ratio(self) -> float:
        return _ratio(self.tp + self.fp, self.count)

    @property
    def actual_hot_ratio(self) -> float:
        return _ratio(self.tp + self.fn, self.count)

    def as_dict(self) -> dict[str, int | float]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "count": self.count,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "predicted_hot_ratio": self.predicted_hot_ratio,
            "actual_hot_ratio": self.actual_hot_ratio,
        }


@dataclass(frozen=True)
class TraceMetadata:
    feature_count: int
    osd_id: int
    session_id: int
    start_wall_time_ns: int
    start_monotonic_time_ns: int
    config_hash: str
    git_commit: str
    phase: str
    record_count: int

    @classmethod
    def from_json(cls, path: Path) -> "TraceMetadata":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read trace metadata {path}: {error}") from error

        if data.get("magic") != TRACE_MAGIC:
            raise ValueError(
                f"metadata magic={data.get('magic')!r}, expected {TRACE_MAGIC!r}"
            )
        feature_count = data.get("feature_count")
        if not isinstance(feature_count, int) or not 0 <= feature_count <= TRACE_MAX_FEATURES:
            raise ValueError(f"invalid feature_count: {feature_count!r}")
        layout = (
            data.get("schema_version"),
            data.get("header_size"),
            data.get("record_size"),
        )
        valid_layout = (
            layout == (1, TRACE_HEADER_SIZE, 208) and feature_count == 4
        ) or layout == (
            TRACE_SCHEMA_VERSION,
            TRACE_HEADER_SIZE,
            TRACE_RECORD_SIZE,
        )
        if not valid_layout:
            raise ValueError(
                "unsupported trace metadata layout: "
                f"schema/header/record/features={layout + (feature_count,)!r}"
            )

        required = (
            "osd_id",
            "session_id",
            "start_wall_time_ns",
            "start_monotonic_time_ns",
            "config_hash",
            "git_commit",
            "phase",
            "record_count",
        )
        missing = [field for field in required if field not in data]
        if missing:
            raise ValueError(f"metadata is missing required fields: {', '.join(missing)}")
        integer_fields = (
            "osd_id",
            "session_id",
            "start_wall_time_ns",
            "start_monotonic_time_ns",
            "record_count",
        )
        if any(not isinstance(data[field], int) for field in integer_fields):
            raise ValueError("metadata integer fields have invalid types")
        if data["record_count"] < 0:
            raise ValueError("metadata record_count cannot be negative")

        return cls(
            feature_count=feature_count,
            osd_id=data["osd_id"],
            session_id=data["session_id"],
            start_wall_time_ns=data["start_wall_time_ns"],
            start_monotonic_time_ns=data["start_monotonic_time_ns"],
            config_hash=str(data["config_hash"]),
            git_commit=str(data["git_commit"]),
            phase=str(data["phase"]),
            record_count=data["record_count"],
        )

    def prediction_wall_time_ns(self, monotonic_ns: int) -> int:
        if monotonic_ns < 0:
            raise ValueError("monotonic timestamp cannot be negative")
        return self.start_wall_time_ns + (
            monotonic_ns - self.start_monotonic_time_ns
        )


def _iso_time_ns(value: str) -> int:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"invalid ISO timestamp: {value!r}") from error
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp has no timezone: {value!r}")
    delta = parsed.astimezone(timezone.utc) - _EPOCH
    return (
        (delta.days * 86_400 + delta.seconds) * 1_000_000_000
        + delta.microseconds * 1_000
    )


@dataclass(frozen=True)
class Phase:
    phase_index: int
    phase_name: str
    segment: str
    start_ns: int
    end_ns: int


class PhaseIndex:
    def __init__(self, phases: list[Phase]):
        if not phases:
            raise ValueError("phase index cannot be empty")
        phases = sorted(phases, key=lambda phase: phase.start_ns)
        for index, phase in enumerate(phases):
            if phase.end_ns < phase.start_ns:
                raise ValueError(f"phase {phase.phase_name!r} ends before it starts")
            if index and phase.start_ns < phases[index - 1].end_ns:
                raise ValueError("phase intervals overlap")
        self.phases = phases
        self._starts = [phase.start_ns for phase in phases]

    @classmethod
    def from_tsv(cls, path: Path) -> "PhaseIndex":
        try:
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
        except OSError as error:
            raise ValueError(f"cannot read phase intervals {path}: {error}") from error

        required = {
            "sample_start",
            "sample_end",
            "phase_index",
            "phase_name",
            "segment",
        }
        if not rows:
            raise ValueError(f"phase interval file is empty: {path}")
        missing = required - set(rows[0])
        if missing:
            raise ValueError(f"phase intervals are missing: {', '.join(sorted(missing))}")
        phases = []
        for row_number, row in enumerate(rows, start=2):
            try:
                phases.append(
                    Phase(
                        phase_index=int(row["phase_index"]),
                        phase_name=row["phase_name"],
                        segment=row["segment"],
                        start_ns=_iso_time_ns(row["sample_start"]),
                        end_ns=_iso_time_ns(row["sample_end"]),
                    )
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid phase row {row_number}: {error}") from error
        return cls(phases)

    def lookup(self, wall_time_ns: int) -> Phase:
        index = bisect.bisect_right(self._starts, wall_time_ns) - 1
        if index < 0:
            return self.phases[0]
        return self.phases[index]


@dataclass
class RunningMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError("running moment value must be finite")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def merge(self, other: "RunningMoments") -> None:
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean
            self.m2 = other.m2
            self.minimum = other.minimum
            self.maximum = other.maximum
            return
        total = self.count + other.count
        delta = other.mean - self.mean
        self.m2 += other.m2 + delta * delta * self.count * other.count / total
        self.mean += delta * other.count / total
        self.count = total
        self.minimum = min(self.minimum, other.minimum)
        self.maximum = max(self.maximum, other.maximum)

    @property
    def variance(self) -> float:
        return self.m2 / (self.count - 1) if self.count > 1 else 0.0

    @property
    def standard_deviation(self) -> float:
        return math.sqrt(max(0.0, self.variance))

    def as_dict(self) -> dict[str, int | float]:
        return {
            "count": self.count,
            "mean": self.mean if self.count else 0.0,
            "standard_deviation": self.standard_deviation,
            "minimum": self.minimum if self.count else 0.0,
            "maximum": self.maximum if self.count else 0.0,
        }


class DistributionSummary:
    def __init__(self) -> None:
        self.moments = RunningMoments()
        self.values = array("d")

    def add(self, value: float) -> None:
        self.moments.add(value)
        self.values.append(value)

    def merge(self, other: "DistributionSummary") -> None:
        self.moments.merge(other.moments)
        self.values.extend(other.values)

    @property
    def count(self) -> int:
        return self.moments.count

    def quantile(self, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("quantile must be in [0, 1]")
        if not self.values:
            return 0.0
        ordered = sorted(self.values)
        index = max(0, math.ceil(value * len(ordered)) - 1)
        return ordered[index]

    def as_dict(self) -> dict[str, int | float]:
        result = self.moments.as_dict()
        ordered = sorted(self.values)

        def nearest_rank(value: float) -> float:
            if not ordered:
                return 0.0
            index = max(0, math.ceil(value * len(ordered)) - 1)
            return ordered[index]

        result.update({
            "p10": nearest_rank(0.10),
            "p25": nearest_rank(0.25),
            "p50": nearest_rank(0.50),
            "p75": nearest_rank(0.75),
            "p90": nearest_rank(0.90),
        })
        return result


class OnlineCorrelation:
    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("correlation dimension must be positive")
        self.dimension = dimension
        self.count = 0
        self.means = [0.0] * dimension
        self.co_moments = [[0.0] * dimension for _ in range(dimension)]

    def add(self, values: Sequence[float]) -> None:
        if len(values) != self.dimension:
            raise ValueError("correlation vector has the wrong dimension")
        vector = [float(value) for value in values]
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("correlation values must be finite")
        self.count += 1
        delta = [value - mean for value, mean in zip(vector, self.means)]
        for index in range(self.dimension):
            self.means[index] += delta[index] / self.count
        delta_after = [
            value - mean for value, mean in zip(vector, self.means)
        ]
        for row in range(self.dimension):
            for column in range(self.dimension):
                self.co_moments[row][column] += (
                    delta[row] * delta_after[column]
                )

    def merge(self, other: "OnlineCorrelation") -> None:
        if other.dimension != self.dimension:
            raise ValueError("cannot merge correlations with different dimensions")
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.means = list(other.means)
            self.co_moments = [list(row) for row in other.co_moments]
            return
        total = self.count + other.count
        delta = [
            other.means[index] - self.means[index]
            for index in range(self.dimension)
        ]
        factor = self.count * other.count / total
        for row in range(self.dimension):
            for column in range(self.dimension):
                self.co_moments[row][column] += (
                    other.co_moments[row][column]
                    + delta[row] * delta[column] * factor
                )
        for index in range(self.dimension):
            self.means[index] += delta[index] * other.count / total
        self.count = total

    def correlation(self, first: int, second: int) -> float:
        first_variance = self.co_moments[first][first]
        second_variance = self.co_moments[second][second]
        if first_variance <= 0.0 or second_variance <= 0.0:
            return 0.0
        return self.co_moments[first][second] / math.sqrt(
            first_variance * second_variance
        )


@dataclass
class CalibrationBin:
    count: int = 0
    probability_sum: float = 0.0
    actual_hot_count: int = 0
    squared_error_sum: float = 0.0
    confusion: ConfusionMatrix = field(default_factory=ConfusionMatrix)

    def add(self, probability: float, actual: int, predicted: int) -> None:
        self.count += 1
        self.probability_sum += probability
        self.actual_hot_count += actual
        self.squared_error_sum += (probability - actual) ** 2
        self.confusion.add(predicted, actual)

    def merge(self, other: "CalibrationBin") -> None:
        self.count += other.count
        self.probability_sum += other.probability_sum
        self.actual_hot_count += other.actual_hot_count
        self.squared_error_sum += other.squared_error_sum
        self.confusion.merge(other.confusion)

    @property
    def average_probability(self) -> float:
        return _ratio(self.probability_sum, self.count)

    @property
    def actual_hot_ratio(self) -> float:
        return _ratio(self.actual_hot_count, self.count)


class CalibrationTable:
    def __init__(self, bin_count: int = 10):
        if bin_count <= 0:
            raise ValueError("calibration bin count must be positive")
        self.bin_count = bin_count
        self.bins = [CalibrationBin() for _ in range(bin_count)]

    def add(
            self,
            probability: float,
            actual: int,
            predicted: int | None = None) -> None:
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("hot probability must be finite and in [0, 1]")
        if actual not in (0, 1):
            raise ValueError("actual label must be binary")
        if predicted is None:
            predicted = int(probability >= 0.5)
        if predicted not in (0, 1):
            raise ValueError("predicted label must be binary")
        index = min(self.bin_count - 1, int(probability * self.bin_count))
        self.bins[index].add(probability, actual, predicted)

    def merge(self, other: "CalibrationTable") -> None:
        if other.bin_count != self.bin_count:
            raise ValueError("cannot merge calibration tables with different bins")
        for target, source in zip(self.bins, other.bins):
            target.merge(source)

    @property
    def count(self) -> int:
        return sum(entry.count for entry in self.bins)

    @property
    def brier_score(self) -> float:
        return _ratio(
            sum(entry.squared_error_sum for entry in self.bins), self.count
        )

    @property
    def ece(self) -> float:
        if not self.count:
            return 0.0
        return sum(
            entry.count
            * abs(entry.average_probability - entry.actual_hot_ratio)
            for entry in self.bins
        ) / self.count


class FeatureCorrelations:
    def __init__(self, feature_count: int):
        self.feature_count = feature_count
        self.values = OnlineCorrelation(feature_count + 2)

    def add(
            self,
            features: Sequence[float],
            actual_label: int,
            hot_probability: float) -> None:
        self.values.add([*features, float(actual_label), hot_probability])

    def merge(self, other: "FeatureCorrelations") -> None:
        if other.feature_count != self.feature_count:
            raise ValueError("cannot merge different feature correlation schemas")
        self.values.merge(other.values)

    def feature_label_correlation(self, feature_index: int) -> float:
        return self.values.correlation(feature_index, self.feature_count)

    def feature_probability_correlation(self, feature_index: int) -> float:
        return self.values.correlation(feature_index, self.feature_count + 1)

    def feature_pair_correlation(self, first: int, second: int) -> float:
        return self.values.correlation(first, second)


@dataclass
class ObjectStats:
    io_count: int = 0
    error_count: int = 0
    actual_hot_count: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def add(self, predicted: int, actual: int) -> None:
        self.io_count += 1
        self.error_count += int(predicted != actual)
        self.actual_hot_count += actual
        outcome = _outcome_name(predicted, actual)
        setattr(self, outcome, getattr(self, outcome) + 1)

    def merge(self, other: "ObjectStats") -> None:
        self.io_count += other.io_count
        self.error_count += other.error_count
        self.actual_hot_count += other.actual_hot_count
        self.tp += other.tp
        self.fp += other.fp
        self.tn += other.tn
        self.fn += other.fn

    @property
    def accuracy(self) -> float:
        return _ratio(self.tp + self.tn, self.io_count)


def prediction_margin_bucket(margin: float) -> str:
    if margin <= -0.25:
        return "strong_cold"
    if margin < 0.0:
        return "near_cold"
    if margin < 0.25:
        return "near_hot"
    return "strong_hot"


def label_margin_bucket(margin: float) -> str:
    if margin <= -0.5:
        return "strong_cold"
    if margin <= 0.0:
        return "near_cold"
    if margin < 0.5:
        return "near_hot"
    return "strong_hot"


def _outcome_name(predicted: int, actual: int) -> str:
    if predicted == 1:
        return "tp" if actual == 1 else "fp"
    return "tn" if actual == 0 else "fn"


def is_steady_prediction(
        prediction_wall_time_ns: int,
        workload_started_at_ns: int,
        warmup_ns: int = STEADY_WARMUP_NS,
        end_ns: int = STEADY_END_NS) -> bool:
    if warmup_ns < 0:
        raise ValueError("steady warmup cannot be negative")
    if end_ns <= warmup_ns:
        raise ValueError("steady end must be greater than warmup")
    relative_time_ns = prediction_wall_time_ns - workload_started_at_ns
    return warmup_ns <= relative_time_ns < end_ns


class ErrorConfidenceHistogram:
    def __init__(self) -> None:
        self.counts = {
            "fp": [0] * (len(ERROR_CONFIDENCE_EDGES) - 1),
            "fn": [0] * (len(ERROR_CONFIDENCE_EDGES) - 1),
        }

    def add(self, predicted: int, actual: int, hot_probability: float) -> None:
        outcome = _outcome_name(predicted, actual)
        if outcome not in self.counts:
            return
        if not 0.0 <= hot_probability <= 1.0:
            raise ValueError("hot probability must be in [0, 1]")
        confidence = hot_probability if predicted == 1 else 1.0 - hot_probability
        if confidence < ERROR_CONFIDENCE_EDGES[0]:
            raise ValueError(
                "predicted-class confidence below 0.5 requires a non-default "
                "prediction threshold"
            )
        bin_index = bisect.bisect_right(
            ERROR_CONFIDENCE_EDGES[1:-1], confidence
        )
        self.counts[outcome][bin_index] += 1

    def merge(self, other: "ErrorConfidenceHistogram") -> None:
        for outcome in self.counts:
            for index, count in enumerate(other.counts[outcome]):
                self.counts[outcome][index] += count


@dataclass(frozen=True)
class RecordContext:
    workload: str
    osd_id: int
    phase: Phase
    steady_state: bool = False
    relative_time_ns: int = 0


class WorkloadAggregate:
    def __init__(self, feature_count: int):
        self.feature_count = feature_count
        self.confusion = ConfusionMatrix()
        self.calibration = CalibrationTable(10)
        self.feature_by_actual = {
            label: [RunningMoments() for _ in range(feature_count)]
            for label in (0, 1)
        }
        self.feature_by_outcome = {
            outcome: [RunningMoments() for _ in range(feature_count)]
            for outcome in ("tp", "fp", "tn", "fn")
        }
        self.correlations = FeatureCorrelations(feature_count)
        self.objects: dict[tuple[int, int], ObjectStats] = {}
        self.cold_start_fallback_confusion = ConfusionMatrix()
        self.model_prediction_confusion = ConfusionMatrix()
        self.cold_origin_confusion = ConfusionMatrix()
        self.steady_confusion = ConfusionMatrix()
        self.steady_calibration = CalibrationTable(10)
        self.steady_feature_by_outcome = {
            outcome: [DistributionSummary() for _ in range(feature_count)]
            for outcome in ("tp", "fp", "tn", "fn")
        }
        self.steady_objects: dict[tuple[int, int], ObjectStats] = {}
        self.steady_time_confusion: dict[int, ConfusionMatrix] = {}
        self.steady_error_confidence = ErrorConfidenceHistogram()
        self.margin_counts: Counter[tuple[str, str, str]] = Counter()
        self.prediction_margin_by_outcome = {
            outcome: RunningMoments() for outcome in ("tp", "fp", "tn", "fn")
        }
        self.label_margin_by_outcome = {
            outcome: RunningMoments() for outcome in ("tp", "fp", "tn", "fn")
        }

    def add(
            self,
            osd_id: int,
            object_hash: int,
            features: Sequence[float],
            hot_probability: float,
            hot_predict_threshold: float,
            label_heat: float,
            label_heat_threshold: float,
            predicted: int,
            actual: int,
            cold_start_fallback: bool,
            pre_access_cold: bool,
            steady_state: bool,
            relative_time_ns: int) -> None:
        self.confusion.add(predicted, actual)
        outcome = _outcome_name(predicted, actual)
        if steady_state:
            self.steady_confusion.add(predicted, actual)
            self.steady_calibration.add(hot_probability, actual, predicted)
            for index, value in enumerate(features):
                self.steady_feature_by_outcome[outcome][index].add(value)
            self.steady_objects.setdefault(
                (osd_id, object_hash), ObjectStats()
            ).add(predicted, actual)
            time_bin = relative_time_ns // (30 * 1_000_000_000)
            self.steady_time_confusion.setdefault(
                time_bin, ConfusionMatrix()
            ).add(predicted, actual)
            self.steady_error_confidence.add(
                predicted, actual, hot_probability
            )
        if cold_start_fallback:
            self.cold_start_fallback_confusion.add(predicted, actual)
        else:
            self.model_prediction_confusion.add(predicted, actual)
        if pre_access_cold:
            self.cold_origin_confusion.add(predicted, actual)
        self.calibration.add(hot_probability, actual, predicted)
        for index, value in enumerate(features):
            self.feature_by_actual[actual][index].add(value)
            self.feature_by_outcome[outcome][index].add(value)
        self.correlations.add(features, actual, hot_probability)
        object_stats = self.objects.setdefault((osd_id, object_hash), ObjectStats())
        object_stats.add(predicted, actual)

        prediction_margin = hot_probability - hot_predict_threshold
        label_margin = math.log2(1.0 + label_heat) - math.log2(
            1.0 + label_heat_threshold
        )
        self.margin_counts[
            (
                prediction_margin_bucket(prediction_margin),
                label_margin_bucket(label_margin),
                outcome,
            )
        ] += 1
        self.prediction_margin_by_outcome[outcome].add(prediction_margin)
        self.label_margin_by_outcome[outcome].add(label_margin)

    def merge(self, other: "WorkloadAggregate") -> None:
        if other.feature_count != self.feature_count:
            raise ValueError("cannot merge workload aggregates with different schemas")
        self.confusion.merge(other.confusion)
        self.cold_start_fallback_confusion.merge(
            other.cold_start_fallback_confusion
        )
        self.model_prediction_confusion.merge(other.model_prediction_confusion)
        self.cold_origin_confusion.merge(other.cold_origin_confusion)
        self.steady_confusion.merge(other.steady_confusion)
        self.steady_calibration.merge(other.steady_calibration)
        self.steady_error_confidence.merge(other.steady_error_confidence)
        self.calibration.merge(other.calibration)
        for label in (0, 1):
            for target, source in zip(
                    self.feature_by_actual[label], other.feature_by_actual[label]):
                target.merge(source)
        for outcome in ("tp", "fp", "tn", "fn"):
            for target, source in zip(
                    self.feature_by_outcome[outcome],
                    other.feature_by_outcome[outcome]):
                target.merge(source)
            for target, source in zip(
                    self.steady_feature_by_outcome[outcome],
                    other.steady_feature_by_outcome[outcome]):
                target.merge(source)
            self.prediction_margin_by_outcome[outcome].merge(
                other.prediction_margin_by_outcome[outcome]
            )
            self.label_margin_by_outcome[outcome].merge(
                other.label_margin_by_outcome[outcome]
            )
        self.correlations.merge(other.correlations)
        self.margin_counts.update(other.margin_counts)
        for key, stats in other.objects.items():
            self.objects.setdefault(key, ObjectStats()).merge(stats)
        for key, stats in other.steady_objects.items():
            self.steady_objects.setdefault(key, ObjectStats()).merge(stats)
        for time_bin, matrix in other.steady_time_confusion.items():
            self.steady_time_confusion.setdefault(
                time_bin, ConfusionMatrix()
            ).merge(matrix)


def _finite_float(row: Mapping[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {field}") from error
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _integer(row: Mapping[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {field}") from error


def _boolean(row: Mapping[str, str], field: str) -> bool:
    try:
        value = row[field].strip().lower()
    except (KeyError, AttributeError) as error:
        raise ValueError(f"invalid {field}") from error
    if value in ("true", "1"):
        return True
    if value in ("false", "0"):
        return False
    raise ValueError(f"invalid {field}: {row[field]!r}")


class TraceAnalyzer:
    def __init__(self, feature_count: int, heat_increment: float = 100.0):
        if not 0 < feature_count <= TRACE_MAX_FEATURES:
            raise ValueError("feature_count must be in [1, 8]")
        if not math.isfinite(heat_increment) or heat_increment <= 0.0:
            raise ValueError("heat_increment must be positive and finite")
        self.feature_count = feature_count
        self.heat_increment = heat_increment
        self.global_confusion = ConfusionMatrix()
        self.steady_global_confusion = ConfusionMatrix()
        self.workloads: dict[str, WorkloadAggregate] = {}
        self.phase_confusion: dict[
            tuple[str, int, str, str], ConfusionMatrix
        ] = {}
        self.record_count = 0

    def consume(self, row: Mapping[str, str], context: RecordContext) -> None:
        if row.get("outcome") != "evaluated":
            raise ValueError("trace analyzer accepts evaluated records only")
        io_sequence = _integer(row, "io_sequence")
        prediction_time_ns = _integer(row, "prediction_time_ns")
        label_deadline_ns = _integer(row, "label_deadline_ns")
        label_completion_time_ns = _integer(row, "label_completion_time_ns")
        if io_sequence <= 0:
            raise ValueError("io_sequence must be positive")
        if not prediction_time_ns <= label_deadline_ns <= label_completion_time_ns:
            raise ValueError("trace timestamps are out of order")
        for field in (
                "tracked_access_count",
                "time_since_previous_access_ns",
                "long_window_access_count",
                "short_window_access_count",
                "future_window_access_count"):
            if _integer(row, field) < 0:
                raise ValueError(f"{field} cannot be negative")
        heat_after_current_access = _finite_float(
            row, "heat_after_current_access")
        heat_threshold_at_prediction = _finite_float(
            row, "heat_label_threshold_at_prediction")
        if min(
                heat_after_current_access,
                heat_threshold_at_prediction) < 0.0:
            raise ValueError("heat values cannot be negative")
        cold_start_fallback = _boolean(row, "cold_start_fallback")
        if _boolean(row, "evaluation_capacity_drop"):
            raise ValueError("evaluated record cannot be a capacity drop")
        features = [
            _finite_float(row, f"feature_{index}")
            for index in range(self.feature_count)
        ]
        object_hash = _integer(row, "object_key_hash")
        hot_probability = _finite_float(row, "predicted_hot_probability")
        hot_predict_threshold = _finite_float(row, "hot_predict_threshold")
        label_heat = _finite_float(row, "label_heat")
        label_heat_threshold = _finite_float(row, "label_heat_threshold")
        predicted = _integer(row, "predicted_label")
        actual = _integer(row, "actual_label")
        if not 0.0 <= hot_probability <= 1.0:
            raise ValueError("predicted_hot_probability must be in [0, 1]")
        if not 0.0 <= hot_predict_threshold <= 1.0:
            raise ValueError("hot_predict_threshold must be in [0, 1]")
        if label_heat < 0.0 or label_heat_threshold < 0.0:
            raise ValueError("label heat values cannot be negative")
        expected_prediction = int(hot_probability >= hot_predict_threshold)
        if predicted != expected_prediction:
            raise ValueError(
                "predicted_label is inconsistent with probability and threshold"
            )
        expected_actual = int(label_heat > label_heat_threshold)
        if actual != expected_actual:
            raise ValueError(
                "actual_label is inconsistent with label heat and threshold"
            )
        pre_access_heat = max(
            0.0, heat_after_current_access - self.heat_increment)
        pre_access_cold = pre_access_heat <= heat_threshold_at_prediction

        self.global_confusion.add(predicted, actual)
        if context.steady_state:
            self.steady_global_confusion.add(predicted, actual)
        workload = self.workloads.setdefault(
            context.workload, WorkloadAggregate(self.feature_count)
        )
        workload.add(
            context.osd_id,
            object_hash,
            features,
            hot_probability,
            hot_predict_threshold,
            label_heat,
            label_heat_threshold,
            predicted,
            actual,
            cold_start_fallback,
            pre_access_cold,
            context.steady_state,
            context.relative_time_ns,
        )
        phase_key = (
            context.workload,
            context.phase.phase_index,
            context.phase.phase_name,
            context.phase.segment,
        )
        self.phase_confusion.setdefault(phase_key, ConfusionMatrix()).add(
            predicted, actual
        )
        self.record_count += 1

    def merge(self, other: "TraceAnalyzer") -> None:
        if other.feature_count != self.feature_count:
            raise ValueError("cannot merge trace analyzers with different schemas")
        if other.heat_increment != self.heat_increment:
            raise ValueError("cannot merge analyzers with different heat increments")
        self.global_confusion.merge(other.global_confusion)
        self.steady_global_confusion.merge(other.steady_global_confusion)
        for name, aggregate in other.workloads.items():
            self.workloads.setdefault(
                name, WorkloadAggregate(self.feature_count)
            ).merge(aggregate)
        for key, matrix in other.phase_confusion.items():
            self.phase_confusion.setdefault(key, ConfusionMatrix()).merge(matrix)
        self.record_count += other.record_count


@dataclass(frozen=True)
class TraceFileInfo:
    workload: str
    path: str
    osd_id: int
    session_id: int
    record_count: int
    config_hash: str
    git_commit: str


@dataclass
class RunAnalysis:
    run_root: Path
    analyzer: TraceAnalyzer
    trace_files: list[TraceFileInfo]

    @property
    def trace_file_count(self) -> int:
        return len(self.trace_files)

    @property
    def workload_count(self) -> int:
        return len(self.analyzer.workloads)


def _feature_name(index: int) -> str:
    if index < len(FEATURE_NAMES):
        return FEATURE_NAMES[index]
    return f"feature_{index}"


def _required_columns(feature_count: int) -> set[str]:
    return {
        "io_sequence",
        "object_key_hash",
        "prediction_time_ns",
        "label_deadline_ns",
        "label_completion_time_ns",
        *[f"feature_{index}" for index in range(feature_count)],
        "heat_after_current_access",
        "heat_label_threshold_at_prediction",
        "predicted_hot_probability",
        "hot_predict_threshold",
        "label_heat",
        "label_heat_threshold",
        "tracked_access_count",
        "time_since_previous_access_ns",
        "long_window_access_count",
        "short_window_access_count",
        "future_window_access_count",
        "outcome",
        "cold_start_fallback",
        "evaluation_capacity_drop",
        "predicted_label",
        "actual_label",
    }


def _scan_trace_file(
        csv_path: Path,
        metadata: TraceMetadata,
        workload: str,
        phases: PhaseIndex,
        workload_started_at_ns: int) -> TraceAnalyzer:
    local = TraceAnalyzer(metadata.feature_count)
    try:
        with csv_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            missing = _required_columns(metadata.feature_count) - set(
                reader.fieldnames or []
            )
            if missing:
                raise ValueError(
                    f"trace CSV {csv_path} is missing columns: "
                    f"{', '.join(sorted(missing))}"
                )
            for row_number, row in enumerate(reader, start=2):
                try:
                    prediction_time_ns = _integer(row, "prediction_time_ns")
                    prediction_wall_time_ns = metadata.prediction_wall_time_ns(
                        prediction_time_ns
                    )
                    phase = phases.lookup(prediction_wall_time_ns)
                    local.consume(
                        row,
                        RecordContext(
                            workload=workload,
                            osd_id=metadata.osd_id,
                            phase=phase,
                            steady_state=is_steady_prediction(
                                prediction_wall_time_ns,
                                workload_started_at_ns,
                            ),
                            relative_time_ns=(
                                prediction_wall_time_ns
                                - workload_started_at_ns
                            ),
                        ),
                    )
                except ValueError as error:
                    raise ValueError(
                        f"invalid trace row {csv_path}:{row_number}: {error}"
                    ) from error
    except OSError as error:
        raise ValueError(f"cannot read trace CSV {csv_path}: {error}") from error
    if local.record_count != metadata.record_count:
        raise ValueError(
            f"metadata record_count {metadata.record_count} does not match "
            f"{local.record_count} CSV rows in {csv_path}"
        )
    return local


def _discover_workloads(run_root: Path) -> list[Path]:
    if not run_root.is_dir():
        raise ValueError(f"run root is not a directory: {run_root}")
    workloads = [
        path
        for path in sorted(run_root.iterdir())
        if path.is_dir()
        and (path / "phase_intervals.tsv").is_file()
        and (path / "trace").is_dir()
    ]
    if not workloads:
        raise ValueError(f"no workload Trace directories found under {run_root}")
    return workloads


def _trace_phase_matches_workload(phase: str, workload: str) -> bool:
    if phase == workload:
        return True
    repetition_prefix = f"{phase}_r"
    return (
        workload.startswith(repetition_prefix)
        and workload[len(repetition_prefix):].isdigit()
    )


def _workload_started_at_ns(workload_path: Path) -> int:
    metadata_path = workload_path / "metadata.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read workload metadata {metadata_path}: {error}"
        ) from error
    started_at = payload.get("started_at")
    if not isinstance(started_at, str):
        raise ValueError(f"workload metadata has invalid started_at: {started_at!r}")
    return _iso_time_ns(started_at)


def _read_run(run_root: Path) -> RunAnalysis:
    analyzer: TraceAnalyzer | None = None
    trace_files: list[TraceFileInfo] = []
    seen_osds: set[tuple[str, int]] = set()
    for workload_path in _discover_workloads(run_root):
        workload = workload_path.name
        phases = PhaseIndex.from_tsv(workload_path / "phase_intervals.tsv")
        workload_started_at_ns = _workload_started_at_ns(workload_path)
        csv_paths = sorted((workload_path / "trace").glob("osd.*.csv"))
        if not csv_paths:
            raise ValueError(f"no OSD Trace CSV files found for {workload}")
        for csv_path in csv_paths:
            metadata_path = csv_path.with_suffix(".csv.metadata.json")
            metadata = TraceMetadata.from_json(metadata_path)
            if not _trace_phase_matches_workload(metadata.phase, workload):
                raise ValueError(
                    f"metadata phase {metadata.phase!r} does not match {workload!r}"
                )
            osd_key = (workload, metadata.osd_id)
            if osd_key in seen_osds:
                raise ValueError(f"duplicate trace for {workload} osd.{metadata.osd_id}")
            seen_osds.add(osd_key)
            local = _scan_trace_file(
                csv_path,
                metadata,
                workload,
                phases,
                workload_started_at_ns,
            )
            if analyzer is None:
                analyzer = TraceAnalyzer(metadata.feature_count)
            analyzer.merge(local)
            trace_files.append(
                TraceFileInfo(
                    workload=workload,
                    path=str(csv_path),
                    osd_id=metadata.osd_id,
                    session_id=metadata.session_id,
                    record_count=metadata.record_count,
                    config_hash=metadata.config_hash,
                    git_commit=metadata.git_commit,
                )
            )
    if analyzer is None:
        raise ValueError("run root contains no Trace records")
    return RunAnalysis(run_root=run_root, analyzer=analyzer, trace_files=trace_files)


def _percent(value: float) -> float:
    return value * 100.0


def _matrix_output(matrix: ConfusionMatrix) -> dict[str, int | float]:
    return {
        "tp": matrix.tp,
        "fp": matrix.fp,
        "tn": matrix.tn,
        "fn": matrix.fn,
        "count": matrix.count,
        "accuracy_percent": _percent(matrix.accuracy),
        "balanced_accuracy_percent": _percent(matrix.balanced_accuracy),
        "precision_percent": _percent(matrix.precision),
        "recall_percent": _percent(matrix.recall),
        "specificity_percent": _percent(matrix.specificity),
        "false_positive_rate_percent": _percent(matrix.false_positive_rate),
        "false_negative_rate_percent": _percent(matrix.false_negative_rate),
        "predicted_hot_percent": _percent(matrix.predicted_hot_ratio),
        "actual_hot_percent": _percent(matrix.actual_hot_ratio),
    }


def _nearest_rank(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    index = max(0, math.ceil(quantile * len(values)) - 1)
    return sorted(values)[index]


def _object_stats_summary(objects: Sequence[ObjectStats]) -> dict[str, int | float]:
    io_counts = [entry.io_count for entry in objects]
    error_counts = [entry.error_count for entry in objects]
    object_count = len(objects)
    total_io = sum(io_counts)
    total_errors = sum(error_counts)

    def top_share(values: list[int], fraction: float, total: int) -> float:
        if not object_count:
            return 0.0
        top_count = max(1, math.ceil(object_count * fraction))
        return _percent(_ratio(sum(sorted(values, reverse=True)[:top_count]), total))

    result: dict[str, int | float] = {
        "object_count": object_count,
        "single_access_object_count": sum(count == 1 for count in io_counts),
        "io_count": total_io,
        "error_count": total_errors,
        "sample_micro_accuracy_percent": _percent(_ratio(
            total_io - total_errors, total_io
        )),
        "object_macro_accuracy_percent": _percent(_ratio(
            sum(entry.accuracy for entry in objects), object_count
        )),
        "io_per_object_mean": _ratio(total_io, object_count),
        "io_per_object_p50": _nearest_rank(io_counts, 0.50),
        "io_per_object_p90": _nearest_rank(io_counts, 0.90),
        "io_per_object_p99": _nearest_rank(io_counts, 0.99),
        "io_per_object_max": max(io_counts, default=0),
        "mean_object_error_percent": _percent(
            _ratio(
                sum(
                    _ratio(entry.error_count, entry.io_count)
                    for entry in objects
                ),
                object_count,
            )
        ),
    }
    for percent, fraction in ((1, 0.01), (5, 0.05), (10, 0.10)):
        result[f"top_{percent}_percent_object_io_share_percent"] = top_share(
            io_counts, fraction, total_io
        )
        result[f"top_{percent}_percent_object_error_share_percent"] = top_share(
            error_counts, fraction, total_errors
        )
    return result


def _object_summary(aggregate: WorkloadAggregate) -> dict[str, int | float]:
    return _object_stats_summary(list(aggregate.objects.values()))


def _summary_dict(result: RunAnalysis) -> dict[str, object]:
    workloads: dict[str, object] = {}
    for name, aggregate in sorted(result.analyzer.workloads.items()):
        workloads[name] = {
            "confusion": _matrix_output(aggregate.confusion),
            "cold_start_fallback": _matrix_output(
                aggregate.cold_start_fallback_confusion
            ),
            "model_prediction": _matrix_output(
                aggregate.model_prediction_confusion
            ),
            "cold_origin": _matrix_output(
                aggregate.cold_origin_confusion
            ),
            "steady": _matrix_output(aggregate.steady_confusion),
            "calibration": {
                "ece_percent": _percent(aggregate.calibration.ece),
                "brier_score": aggregate.calibration.brier_score,
            },
            "objects": _object_summary(aggregate),
        }
    return {
        "analysis_schema_version": 2,
        "run_root": str(result.run_root),
        "trace_file_count": result.trace_file_count,
        "workload_count": result.workload_count,
        "record_count": result.analyzer.record_count,
        "feature_names": [
            _feature_name(index)
            for index in range(result.analyzer.feature_count)
        ],
        "global": _matrix_output(result.analyzer.global_confusion),
        "steady_global": _matrix_output(
            result.analyzer.steady_global_confusion
        ),
        "workloads": workloads,
        "trace_files": [entry.__dict__ for entry in result.trace_files],
    }


def _write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _workload_rows(result: RunAnalysis) -> list[dict[str, object]]:
    rows = []
    for name, aggregate in sorted(result.analyzer.workloads.items()):
        rows.append({
            "workload": name,
            **_matrix_output(aggregate.confusion),
            "ece_percent": _percent(aggregate.calibration.ece),
            "brier_score": aggregate.calibration.brier_score,
            "cold_start_fallback_count": (
                aggregate.cold_start_fallback_confusion.count
            ),
            "cold_start_fallback_percent": _percent(_ratio(
                aggregate.cold_start_fallback_confusion.count,
                aggregate.confusion.count,
            )),
            "cold_start_fallback_accuracy_percent": _percent(
                aggregate.cold_start_fallback_confusion.accuracy
            ),
            "cold_start_fallback_actual_hot_percent": _percent(
                aggregate.cold_start_fallback_confusion.actual_hot_ratio
            ),
            "model_prediction_accuracy_percent": _percent(
                aggregate.model_prediction_confusion.accuracy
            ),
            "cold_origin_count": aggregate.cold_origin_confusion.count,
            "cold_to_hot_count": (
                aggregate.cold_origin_confusion.tp
                + aggregate.cold_origin_confusion.fn
            ),
            "cold_to_hot_recall_percent": _percent(
                aggregate.cold_origin_confusion.recall
            ),
            "activation_precision_percent": _percent(
                aggregate.cold_origin_confusion.precision
            ),
        })
    return rows


def _phase_rows(result: RunAnalysis) -> list[dict[str, object]]:
    rows = []
    for key, matrix in sorted(result.analyzer.phase_confusion.items()):
        workload, phase_index, phase_name, segment = key
        rows.append({
            "workload": workload,
            "phase_index": phase_index,
            "phase_name": phase_name,
            "segment": segment,
            **_matrix_output(matrix),
        })
    return rows


def _steady_confidence_entries(
        result: RunAnalysis,
) -> list[tuple[str, ConfusionMatrix, ErrorConfidenceHistogram]]:
    entries = [
        (name, aggregate.steady_confusion, aggregate.steady_error_confidence)
        for name, aggregate in sorted(result.analyzer.workloads.items())
    ]
    combined = ErrorConfidenceHistogram()
    for aggregate in result.analyzer.workloads.values():
        combined.merge(aggregate.steady_error_confidence)
    entries.append(("all", result.analyzer.steady_global_confusion, combined))
    return entries


def _steady_error_confidence_rows(
        result: RunAnalysis,
) -> list[dict[str, object]]:
    rows = []
    for workload, matrix, histogram in _steady_confidence_entries(result):
        outcome_totals = {"fp": matrix.fp, "fn": matrix.fn}
        for index in range(len(ERROR_CONFIDENCE_EDGES) - 1):
            bin_total = sum(
                histogram.counts[outcome][index] for outcome in ("fp", "fn")
            )
            for outcome in ("fp", "fn"):
                count = histogram.counts[outcome][index]
                rows.append({
                    "workload": workload,
                    "outcome": outcome,
                    "confidence_lower": ERROR_CONFIDENCE_EDGES[index],
                    "confidence_upper": ERROR_CONFIDENCE_EDGES[index + 1],
                    "upper_inclusive": int(
                        index == len(ERROR_CONFIDENCE_EDGES) - 2
                    ),
                    "count": count,
                    "outcome_percent": _percent(_ratio(
                        count, outcome_totals[outcome]
                    )),
                    "bin_error_percent": _percent(_ratio(count, bin_total)),
                    "all_error_percent": _percent(_ratio(
                        count, matrix.fp + matrix.fn
                    )),
                })
    return rows


def _calibration_rows(result: RunAnalysis) -> list[dict[str, object]]:
    rows = []
    for workload, aggregate in sorted(result.analyzer.workloads.items()):
        for index, entry in enumerate(aggregate.calibration.bins):
            rows.append({
                "workload": workload,
                "bin_index": index,
                "probability_lower": index / aggregate.calibration.bin_count,
                "probability_upper": (index + 1) / aggregate.calibration.bin_count,
                "count": entry.count,
                "average_probability": entry.average_probability,
                "actual_hot_percent": _percent(entry.actual_hot_ratio),
                "absolute_gap_percent": _percent(abs(
                    entry.average_probability - entry.actual_hot_ratio
                )),
            })
    return rows


def _margin_rows(result: RunAnalysis) -> list[dict[str, object]]:
    rows = []
    for workload, aggregate in sorted(result.analyzer.workloads.items()):
        outcome_counts = Counter()
        for (_, _, outcome), count in aggregate.margin_counts.items():
            outcome_counts[outcome] += count
        for key, count in sorted(aggregate.margin_counts.items()):
            prediction_bucket, label_bucket, outcome = key
            rows.append({
                "workload": workload,
                "prediction_margin_bucket": prediction_bucket,
                "label_margin_bucket": label_bucket,
                "outcome": outcome,
                "count": count,
                "workload_percent": _percent(_ratio(
                    count, aggregate.confusion.count
                )),
                "outcome_percent": _percent(_ratio(
                    count, outcome_counts[outcome]
                )),
            })
    return rows


def _feature_stat_rows(result: RunAnalysis) -> list[dict[str, object]]:
    rows = []
    for workload, aggregate in sorted(result.analyzer.workloads.items()):
        groups = [
            ("actual_label", "cold", aggregate.feature_by_actual[0]),
            ("actual_label", "hot", aggregate.feature_by_actual[1]),
            *[
                ("outcome", outcome, aggregate.feature_by_outcome[outcome])
                for outcome in ("tp", "fp", "tn", "fn")
            ],
        ]
        for grouping, group, moments in groups:
            for index, entry in enumerate(moments):
                rows.append({
                    "workload": workload,
                    "grouping": grouping,
                    "group": group,
                    "feature_index": index,
                    "feature_name": _feature_name(index),
                    **entry.as_dict(),
                })
    return rows


def _feature_correlation_rows(result: RunAnalysis) -> list[dict[str, object]]:
    rows = []
    for workload, aggregate in sorted(result.analyzer.workloads.items()):
        for index in range(aggregate.feature_count):
            rows.extend((
                {
                    "workload": workload,
                    "left_feature": _feature_name(index),
                    "right_variable": "actual_label",
                    "correlation": aggregate.correlations.feature_label_correlation(index),
                },
                {
                    "workload": workload,
                    "left_feature": _feature_name(index),
                    "right_variable": "predicted_hot_probability",
                    "correlation": aggregate.correlations.feature_probability_correlation(index),
                },
            ))
        for first in range(aggregate.feature_count):
            for second in range(first + 1, aggregate.feature_count):
                rows.append({
                    "workload": workload,
                    "left_feature": _feature_name(first),
                    "right_variable": _feature_name(second),
                    "correlation": aggregate.correlations.feature_pair_correlation(
                        first, second
                    ),
                })
    return rows


def _object_rows(result: RunAnalysis) -> list[dict[str, object]]:
    return [
        {"workload": workload, **_object_summary(aggregate)}
        for workload, aggregate in sorted(result.analyzer.workloads.items())
    ]


def _combined_workload_aggregate(result: RunAnalysis) -> WorkloadAggregate:
    combined = WorkloadAggregate(result.analyzer.feature_count)
    for aggregate in result.analyzer.workloads.values():
        combined.merge(aggregate)
    return combined


def _prediction_outcome_rows(result: RunAnalysis) -> list[dict[str, object]]:
    rows = []
    for workload, aggregate in sorted(result.analyzer.workloads.items()):
        for scope, matrix in (
                ("all", aggregate.confusion),
                ("steady", aggregate.steady_confusion)):
            rows.append({
                "workload": workload,
                "scope": scope,
                **_matrix_output(matrix),
            })
    for scope, matrix in (
            ("all", result.analyzer.global_confusion),
            ("steady", result.analyzer.steady_global_confusion)):
        rows.append({"workload": "all", "scope": scope, **_matrix_output(matrix)})
    return rows


def _prediction_probability_bin_rows(
        result: RunAnalysis) -> list[dict[str, object]]:
    combined = _combined_workload_aggregate(result)
    entries = [
        (workload, aggregate.steady_calibration)
        for workload, aggregate in sorted(result.analyzer.workloads.items())
    ]
    entries.append(("all", combined.steady_calibration))
    rows = []
    for workload, calibration in entries:
        for index, entry in enumerate(calibration.bins):
            rows.append({
                "workload": workload,
                "bin_index": index,
                "probability_lower": index / calibration.bin_count,
                "probability_upper": (index + 1) / calibration.bin_count,
                "upper_inclusive": int(index == calibration.bin_count - 1),
                "count": entry.count,
                "average_probability": entry.average_probability,
                "actual_hot_percent": _percent(entry.actual_hot_ratio),
                **_matrix_output(entry.confusion),
            })
    return rows


def _prediction_feature_profile_rows(
        result: RunAnalysis) -> list[dict[str, object]]:
    combined = _combined_workload_aggregate(result)
    entries = [
        (workload, aggregate)
        for workload, aggregate in sorted(result.analyzer.workloads.items())
    ]
    entries.append(("all", combined))
    rows = []
    for workload, aggregate in entries:
        for outcome in ("tp", "fp", "tn", "fn"):
            for index, summary in enumerate(
                    aggregate.steady_feature_by_outcome[outcome]):
                rows.append({
                    "workload": workload,
                    "outcome": outcome,
                    "feature_index": index,
                    "feature_name": _feature_name(index),
                    **summary.as_dict(),
                })
    return rows


def _prediction_object_summary_rows(
        result: RunAnalysis) -> list[dict[str, object]]:
    rows = [
        {
            "workload": workload,
            **_object_stats_summary(list(aggregate.steady_objects.values())),
        }
        for workload, aggregate in sorted(result.analyzer.workloads.items())
    ]
    all_objects = [
        entry
        for aggregate in result.analyzer.workloads.values()
        for entry in aggregate.steady_objects.values()
    ]
    rows.append({"workload": "all", **_object_stats_summary(all_objects)})
    return rows


def _prediction_time_series_rows(
        result: RunAnalysis) -> list[dict[str, object]]:
    combined_bins: dict[int, ConfusionMatrix] = {}
    rows = []
    for workload, aggregate in sorted(result.analyzer.workloads.items()):
        for time_bin, matrix in sorted(aggregate.steady_time_confusion.items()):
            rows.append({
                "workload": workload,
                "window_start_seconds": time_bin * 30,
                "window_end_seconds": (time_bin + 1) * 30,
                **_matrix_output(matrix),
            })
            combined_bins.setdefault(time_bin, ConfusionMatrix()).merge(matrix)
    for time_bin, matrix in sorted(combined_bins.items()):
        rows.append({
            "workload": "all",
            "window_start_seconds": time_bin * 30,
            "window_end_seconds": (time_bin + 1) * 30,
            **_matrix_output(matrix),
        })
    return rows


def _segment_summary(result: RunAnalysis) -> dict[str, ConfusionMatrix]:
    segments: dict[str, ConfusionMatrix] = {}
    for (*_, segment), matrix in result.analyzer.phase_confusion.items():
        segments.setdefault(segment, ConfusionMatrix()).merge(matrix)
    return segments


def _error_profile(aggregate: WorkloadAggregate) -> dict[str, int | float]:
    errors = aggregate.confusion.fp + aggregate.confusion.fn
    confident_errors = 0
    clear_label_errors = 0
    for (prediction_bucket, label_bucket, outcome), count in (
            aggregate.margin_counts.items()):
        if outcome not in ("fp", "fn"):
            continue
        if (
                (outcome == "fp" and prediction_bucket == "strong_hot")
                or (outcome == "fn" and prediction_bucket == "strong_cold")):
            confident_errors += count
        if (
                (outcome == "fp" and label_bucket == "strong_cold")
                or (outcome == "fn" and label_bucket == "strong_hot")):
            clear_label_errors += count
    return {
        "errors": errors,
        "fn_share": _ratio(aggregate.confusion.fn, errors),
        "confident_error_share": _ratio(confident_errors, errors),
        "clear_label_error_share": _ratio(clear_label_errors, errors),
    }


def _report_text(result: RunAnalysis) -> str:
    lines = [
        "# Heat Predictor Trace 离线分析报告",
        "",
        "## 数据完整性",
        "",
        f"- 负载数：{result.workload_count}",
        f"- OSD Trace 文件数：{result.trace_file_count}",
        f"- 已评估记录数：{result.analyzer.record_count}",
        "- 所有 CSV 均通过 schema、记录数、时间顺序和标签一致性校验。",
        "",
        "## 总体结果",
        "",
        "| 负载 | 样本 | Accuracy | Balanced Accuracy | Precision | Recall | 实际热比例 | ECE | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for workload, aggregate in sorted(result.analyzer.workloads.items()):
        matrix = aggregate.confusion
        lines.append(
            f"| {workload} | {matrix.count} | {_percent(matrix.accuracy):.2f}% | "
            f"{_percent(matrix.balanced_accuracy):.2f}% | "
            f"{_percent(matrix.precision):.2f}% | {_percent(matrix.recall):.2f}% | "
            f"{_percent(matrix.actual_hot_ratio):.2f}% | "
            f"{_percent(aggregate.calibration.ece):.2f}% | "
            f"{aggregate.calibration.brier_score:.4f} |"
        )
    lines.extend((
        "",
        "## 稳态全 I/O 预测结果",
        "",
        "每条样本表示预测时刻的冷热判断与10秒后标签的对照。TP/TN/FP/FN 是"
        "预测结果，不表示 object 状态迁移。稳态区间为负载开始后 `[120s,600s)`。",
        "",
        "| 负载 | 样本 | TN | FP | FN | TP | Accuracy | BAcc | Precision | Recall | FPR | FNR | 实际热 | 预测热 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for workload, matrix, _ in _steady_confidence_entries(result):
        lines.append(
            f"| {workload} | {matrix.count} | {matrix.tn} | {matrix.fp} | "
            f"{matrix.fn} | {matrix.tp} | {_percent(matrix.accuracy):.2f}% | "
            f"{_percent(matrix.balanced_accuracy):.2f}% | "
            f"{_percent(matrix.precision):.2f}% | "
            f"{_percent(matrix.recall):.2f}% | "
            f"{_percent(matrix.false_positive_rate):.2f}% | "
            f"{_percent(matrix.false_negative_rate):.2f}% | "
            f"{_percent(matrix.actual_hot_ratio):.2f}% | "
            f"{_percent(matrix.predicted_hot_ratio):.2f}% |"
        )
    lines.extend((
        "",
        "## 稳态预测概率分箱",
        "",
        "概率以0.1为宽度分箱。只展示非空分箱，用于判断错误是否集中在0.5边界附近。",
        "",
        "| 负载 | 概率区间 | 样本 | 平均概率 | 实际热 | TN | FP | FN | TP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for row in _prediction_probability_bin_rows(result):
        if int(row["count"]) == 0:
            continue
        right = "]" if int(row["upper_inclusive"]) else ")"
        interval = (
            f"[{float(row['probability_lower']):.1f},"
            f"{float(row['probability_upper']):.1f}{right}"
        )
        lines.append(
            f"| {row['workload']} | {interval} | {row['count']} | "
            f"{_percent(float(row['average_probability'])):.2f}% | "
            f"{float(row['actual_hot_percent']):.2f}% | {row['tn']} | "
            f"{row['fp']} | {row['fn']} | {row['tp']} |"
        )
    lines.extend((
        "",
        "## 稳态 FP/FN 置信度分布",
        "",
        "FP 的置信度为 `p_hot`，FN 的置信度为 `1-p_hot`。前四个区间左闭右开，"
        "最后一个 `[0.9,1.0]` 两端闭合。",
        "",
        "| 负载 | 置信度 | FP | FP 内占比 | FN | FN 内占比 | 区间内 FP 占比 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ))
    for workload, matrix, histogram in _steady_confidence_entries(result):
        for index in range(len(ERROR_CONFIDENCE_EDGES) - 1):
            fp_count = histogram.counts["fp"][index]
            fn_count = histogram.counts["fn"][index]
            right_bracket = "]" if index == len(ERROR_CONFIDENCE_EDGES) - 2 else ")"
            interval = (
                f"[{ERROR_CONFIDENCE_EDGES[index]:.1f},"
                f"{ERROR_CONFIDENCE_EDGES[index + 1]:.1f}{right_bracket}"
            )
            lines.append(
                f"| {workload} | {interval} | {fp_count} | "
                f"{_percent(_ratio(fp_count, matrix.fp)):.2f}% | {fn_count} | "
                f"{_percent(_ratio(fn_count, matrix.fn)):.2f}% | "
                f"{_percent(_ratio(fp_count, fp_count + fn_count)):.2f}% |"
            )
    lines.extend((
        "",
        "## Cold-start fallback",
        "",
        "| 负载 | Fallback 样本 | 占比 | Fallback Accuracy | Fallback 实际热比例 | 模型可用时 Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ))
    for workload, aggregate in sorted(result.analyzer.workloads.items()):
        fallback = aggregate.cold_start_fallback_confusion
        model = aggregate.model_prediction_confusion
        lines.append(
            f"| {workload} | {fallback.count} | "
            f"{_percent(_ratio(fallback.count, aggregate.confusion.count)):.2f}% | "
            f"{_percent(fallback.accuracy):.2f}% | "
            f"{_percent(fallback.actual_hot_ratio):.2f}% | "
            f"{_percent(model.accuracy):.2f}% |"
        )
    lines.extend((
        "",
        "## 阶段表现",
        "",
        "| 阶段类型 | 样本 | Accuracy | Balanced Accuracy |",
        "|---|---:|---:|---:|",
    ))
    for segment, matrix in sorted(_segment_summary(result).items()):
        lines.append(
            f"| {segment} | {matrix.count} | {_percent(matrix.accuracy):.2f}% | "
            f"{_percent(matrix.balanced_accuracy):.2f}% |"
        )
    lines.extend((
        "",
        "## 错误结构",
        "",
        "预测 margin 为 `probability - 0.5`，绝对值不小于 0.25 视为高置信；"
        "标签 margin 为 `log2p1(label_heat) - log2p1(label_threshold)`，绝对值"
        "不小于 0.5 视为位于明确标签一侧。",
        "",
        "| 负载 | 预测热-实际热 | FN 占错误 | 高置信错误 | 明确标签错误 |",
        "|---|---:|---:|---:|---:|",
    ))
    for workload, aggregate in sorted(result.analyzer.workloads.items()):
        profile = _error_profile(aggregate)
        lines.append(
            f"| {workload} | "
            f"{_percent(aggregate.confusion.predicted_hot_ratio - aggregate.confusion.actual_hot_ratio):+.2f} pp | "
            f"{_percent(float(profile['fn_share'])):.2f}% | "
            f"{_percent(float(profile['confident_error_share'])):.2f}% | "
            f"{_percent(float(profile['clear_label_error_share'])):.2f}% |"
        )
    lines.extend((
        "",
        "## Feature 相关性诊断",
        "",
        "| 负载 | 标签相关性最强 feature | 相关系数 | 热度两项互相关 |",
        "|---|---|---:|---:|",
    ))
    heat_feature_indices = (0, 2)
    for workload, aggregate in sorted(result.analyzer.workloads.items()):
        label_correlations = [
            aggregate.correlations.feature_label_correlation(index)
            for index in range(aggregate.feature_count)
        ]
        strongest = max(
            range(aggregate.feature_count),
            key=lambda index: abs(label_correlations[index]),
        )
        heat_correlations = [
            abs(aggregate.correlations.feature_pair_correlation(first, second))
            for position, first in enumerate(heat_feature_indices)
            for second in heat_feature_indices[position + 1:]
            if first < aggregate.feature_count and second < aggregate.feature_count
        ]
        lines.append(
            f"| {workload} | {_feature_name(strongest)} | "
            f"{label_correlations[strongest]:+.3f} | "
            f"{max(heat_correlations, default=0.0):.3f} |"
        )
    lines.extend((
        "",
        "## 稳态 TP/TN/FP/FN Feature 分布",
        "",
        "完整分位数见 `prediction_feature_profiles.tsv`；此处列出各组中位数和P90。",
        "",
        "| 负载 | 结果 | Feature | 样本 | P50 | P90 |",
        "|---|---|---|---:|---:|---:|",
    ))
    for row in _prediction_feature_profile_rows(result):
        if row["workload"] == "all" or int(row["count"]) == 0:
            continue
        lines.append(
            f"| {row['workload']} | {str(row['outcome']).upper()} | "
            f"{row['feature_name']} | {row['count']} | "
            f"{float(row['p50']):.4f} | {float(row['p90']):.4f} |"
        )
    lines.extend((
        "",
        "## 稳态 Object 误差集中度",
        "",
        "Object macro Accuracy 先计算每个 object 的准确率再平均；它不受高频 object "
        "样本量直接支配。",
        "",
        "| 负载 | Object 数 | 样本 Accuracy | Object macro Accuracy | Top 1% 错误 | Top 5% 错误 | Top 10% 错误 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ))
    for objects in _prediction_object_summary_rows(result):
        lines.append(
            f"| {objects['workload']} | {objects['object_count']} | "
            f"{float(objects['sample_micro_accuracy_percent']):.2f}% | "
            f"{float(objects['object_macro_accuracy_percent']):.2f}% | "
            f"{float(objects['top_1_percent_object_error_share_percent']):.2f}% | "
            f"{float(objects['top_5_percent_object_error_share_percent']):.2f}% | "
            f"{float(objects['top_10_percent_object_error_share_percent']):.2f}% |"
        )
    lines.extend((
        "",
        "## 稳态 30 秒时序",
        "",
        "窗口按负载开始后的相对时间划分，用于识别总体均值掩盖的局部退化。",
        "",
        "| 负载 | 时间窗口 | 样本 | Accuracy | BAcc | FPR | FNR | 实际热 | 预测热 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for row in _prediction_time_series_rows(result):
        if row["workload"] == "all":
            continue
        lines.append(
            f"| {row['workload']} | [{row['window_start_seconds']},"
            f"{row['window_end_seconds']})s | {row['count']} | "
            f"{float(row['accuracy_percent']):.2f}% | "
            f"{float(row['balanced_accuracy_percent']):.2f}% | "
            f"{float(row['false_positive_rate_percent']):.2f}% | "
            f"{float(row['false_negative_rate_percent']):.2f}% | "
            f"{float(row['actual_hot_percent']):.2f}% | "
            f"{float(row['predicted_hot_percent']):.2f}% |"
        )
    _, combined_matrix, combined_confidence = _steady_confidence_entries(result)[-1]
    high_confidence_errors = sum(
        combined_confidence.counts[outcome][index]
        for outcome in ("fp", "fn")
        for index in range(2, len(ERROR_CONFIDENCE_EDGES) - 1)
    )
    combined_objects = _prediction_object_summary_rows(result)[-1]
    lines.extend((
        "",
        "## 本轮核心发现",
        "",
        f"- 稳态共 {combined_matrix.count} 条样本，Accuracy="
        f"{_percent(combined_matrix.accuracy):.2f}%，BAcc="
        f"{_percent(combined_matrix.balanced_accuracy):.2f}%；FN="
        f"{combined_matrix.fn}，FP={combined_matrix.fp}。",
        f"- 置信度不低于0.7的错误占全部错误 "
        f"{_percent(_ratio(high_confidence_errors, combined_matrix.fp + combined_matrix.fn)):.2f}%，"
        "因此问题不能只归因于0.5附近的预测阈值边界。",
        f"- Top 10% object 集中了 "
        f"{float(combined_objects['top_10_percent_object_error_share_percent']):.2f}% "
        "的稳态错误，后续应区分少量困难 object 与普遍 feature 不可分。",
    ))
    for workload, aggregate in sorted(result.analyzer.workloads.items()):
        matrix = aggregate.steady_confusion
        lines.append(
            f"- `{workload}`：FPR={_percent(matrix.false_positive_rate):.2f}%，"
            f"FNR={_percent(matrix.false_negative_rate):.2f}%，"
            f"预测热-实际热={_percent(matrix.predicted_hot_ratio - matrix.actual_hot_ratio):+.2f} pp。"
        )
    time_rows = _prediction_time_series_rows(result)
    for workload in sorted(result.analyzer.workloads):
        accuracy_values = [
            float(row["accuracy_percent"])
            for row in time_rows
            if row["workload"] == workload
        ]
        if accuracy_values:
            lines.append(
                f"- `{workload}` 的30秒 Accuracy 范围为 "
                f"{min(accuracy_values):.2f}%~{max(accuracy_values):.2f}%，"
                "说明总体均值下仍存在局部波动。"
            )
    lines.extend((
        "",
        "## 结论与下一步",
        "",
        "1. 首要诊断对象是全部 I/O 的10秒预测结果；TP/TN/FP/FN 不应解释为 object "
        "状态迁移。",
        "2. 相关性只能说明 feature 与标签共同变化，不能证明因果收益；是否保留 feature"
        " 必须使用通过 parity gate 的同一 Trace 配对回放。",
        "3. 高置信错误和明确标签错误用于定位问题，不应单独作为调整预测阈值或 Otsu"
        " 阈值的依据。",
        "4. 下一步根据概率分箱、feature 分位数、object 集中度和30秒时序选择一个"
        "证据最强的瓶颈，再做单变量受控实验。",
        "",
        "## 诊断索引",
        "",
        "- `phase_error.tsv`：按负载、热点阶段和阶段类型定位错误。",
        "- `calibration.tsv`：检查模型热概率是否可信；ECE 越低越好。",
        "- `margin_analysis.tsv`：区分边界错误和高置信错误。",
        "- `feature_stats.tsv`：比较真实冷热类与 TP/FP/TN/FN 的特征分布。",
        "- `feature_correlation.tsv`：查找特征冗余及其与标签、模型输出的关系。",
        "- `object_stats.tsv`：检查访问和错误是否集中在少量 object。",
        "- `prediction_outcome_summary.tsv`：全程与稳态 TP/TN/FP/FN 主表。",
        "- `prediction_probability_bins.tsv`：稳态概率分箱及每箱四类结果。",
        "- `prediction_feature_profiles.tsv`：稳态四类结果的 feature 精确分位数。",
        "- `prediction_object_summary.tsv`：稳态 object macro 指标和错误集中度。",
        "- `prediction_time_series.tsv`：稳态30秒窗口的局部指标。",
        "",
        "## 使用边界",
        "",
        "本报告解释已记录在线策略，不进行 ARF 精确回放。Schema v1 未记录"
        "预测快照代次，因此离线消融必须先通过 baseline 概率/标签一致性门槛。",
        "",
    ))
    return "\n".join(lines)


def _write_outputs(result: RunAnalysis, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(_summary_dict(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    workload_rows = _workload_rows(result)
    _write_tsv(
        output_dir / "workload_summary.tsv",
        list(workload_rows[0]),
        workload_rows,
    )
    phase_rows = _phase_rows(result)
    _write_tsv(output_dir / "phase_error.tsv", list(phase_rows[0]), phase_rows)
    calibration_rows = _calibration_rows(result)
    _write_tsv(
        output_dir / "calibration.tsv",
        list(calibration_rows[0]),
        calibration_rows,
    )
    margin_rows = _margin_rows(result)
    _write_tsv(
        output_dir / "margin_analysis.tsv",
        list(margin_rows[0]),
        margin_rows,
    )
    feature_rows = _feature_stat_rows(result)
    _write_tsv(
        output_dir / "feature_stats.tsv",
        list(feature_rows[0]),
        feature_rows,
    )
    correlation_rows = _feature_correlation_rows(result)
    _write_tsv(
        output_dir / "feature_correlation.tsv",
        list(correlation_rows[0]),
        correlation_rows,
    )
    object_rows = _object_rows(result)
    _write_tsv(
        output_dir / "object_stats.tsv",
        list(object_rows[0]),
        object_rows,
    )
    steady_confidence_rows = _steady_error_confidence_rows(result)
    _write_tsv(
        output_dir / "steady_error_confidence.tsv",
        list(steady_confidence_rows[0]),
        steady_confidence_rows,
    )
    prediction_outcome_rows = _prediction_outcome_rows(result)
    _write_tsv(
        output_dir / "prediction_outcome_summary.tsv",
        list(prediction_outcome_rows[0]),
        prediction_outcome_rows,
    )
    prediction_probability_bin_rows = _prediction_probability_bin_rows(result)
    _write_tsv(
        output_dir / "prediction_probability_bins.tsv",
        list(prediction_probability_bin_rows[0]),
        prediction_probability_bin_rows,
    )
    prediction_feature_profile_rows = _prediction_feature_profile_rows(result)
    _write_tsv(
        output_dir / "prediction_feature_profiles.tsv",
        list(prediction_feature_profile_rows[0]),
        prediction_feature_profile_rows,
    )
    prediction_object_summary_rows = _prediction_object_summary_rows(result)
    _write_tsv(
        output_dir / "prediction_object_summary.tsv",
        list(prediction_object_summary_rows[0]),
        prediction_object_summary_rows,
    )
    prediction_time_series_rows = _prediction_time_series_rows(result)
    time_series_fields = [
        "workload",
        "window_start_seconds",
        "window_end_seconds",
        *_matrix_output(ConfusionMatrix()).keys(),
    ]
    _write_tsv(
        output_dir / "prediction_time_series.tsv",
        time_series_fields,
        prediction_time_series_rows,
    )
    (output_dir / "REPORT.md").write_text(_report_text(result), encoding="utf-8")


def analyze_run_root(run_root: Path, output_dir: Path) -> RunAnalysis:
    result = _read_run(run_root.resolve())
    _write_outputs(result, output_dir.resolve())
    return result
