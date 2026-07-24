#!/usr/bin/env python3
"""Unit tests for the Heat Predictor trace analyzer."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import test_sh.hp_trace_analysis as hp_trace_analysis
from test_sh.hp_trace_analysis import (
    CalibrationTable,
    ConfusionMatrix,
    EXPECTED_OUTPUT_FILES,
    OnlineCorrelation,
    PhaseIndex,
    RecordContext,
    RunningMoments,
    TraceAnalyzer,
    TraceMetadata,
    analyze_run_root,
    label_margin_bucket,
    prediction_margin_bucket,
)


class ConfusionMatrixTest(unittest.TestCase):
    def test_metrics_use_the_expected_denominators(self) -> None:
        matrix = ConfusionMatrix()
        for predicted, actual in (
            [(1, 1)] * 2
            + [(1, 0)]
            + [(0, 0)] * 3
            + [(0, 1)]
        ):
            matrix.add(predicted, actual)

        self.assertEqual((matrix.tp, matrix.fp, matrix.tn, matrix.fn), (2, 1, 3, 1))
        self.assertAlmostEqual(matrix.accuracy, 5 / 7)
        self.assertAlmostEqual(matrix.balanced_accuracy, ((2 / 3) + (3 / 4)) / 2)
        self.assertAlmostEqual(matrix.precision, 2 / 3)
        self.assertAlmostEqual(matrix.recall, 2 / 3)
        self.assertAlmostEqual(matrix.false_positive_rate, 1 / 4)
        self.assertAlmostEqual(matrix.false_negative_rate, 1 / 3)
        self.assertAlmostEqual(matrix.predicted_hot_ratio, 3 / 7)
        self.assertAlmostEqual(matrix.actual_hot_ratio, 3 / 7)

    def test_invalid_binary_labels_are_rejected(self) -> None:
        matrix = ConfusionMatrix()
        with self.assertRaisesRegex(ValueError, "binary"):
            matrix.add(2, 0)


class SteadyErrorConfidenceTest(unittest.TestCase):
    def test_steady_boundary_is_inclusive_at_120_seconds(self) -> None:
        second_ns = 1_000_000_000
        started_at_ns = 10 * second_ns
        self.assertFalse(hp_trace_analysis.is_steady_prediction(
            started_at_ns + 120 * second_ns - 1,
            started_at_ns,
        ))
        self.assertTrue(hp_trace_analysis.is_steady_prediction(
            started_at_ns + 120 * second_ns,
            started_at_ns,
        ))
        self.assertTrue(hp_trace_analysis.is_steady_prediction(
            started_at_ns + 600 * second_ns - 1,
            started_at_ns,
        ))
        self.assertFalse(hp_trace_analysis.is_steady_prediction(
            started_at_ns + 600 * second_ns,
            started_at_ns,
        ))

    def test_fp_and_fn_use_predicted_class_confidence_bins(self) -> None:
        histogram = hp_trace_analysis.ErrorConfidenceHistogram()
        histogram.add(predicted=1, actual=0, hot_probability=0.60)
        histogram.add(predicted=0, actual=1, hot_probability=0.21)
        histogram.add(predicted=1, actual=0, hot_probability=1.00)

        self.assertEqual(histogram.counts["fp"], [0, 1, 0, 0, 1])
        self.assertEqual(histogram.counts["fn"], [0, 0, 1, 0, 0])


class TraceMetadataTest(unittest.TestCase):
    def metadata(self, **overrides: object) -> dict[str, object]:
        values: dict[str, object] = {
            "magic": "HPTRACE1",
            "schema_version": 2,
            "header_size": 192,
            "record_size": 200,
            "feature_count": 6,
            "osd_id": 1,
            "session_id": 9,
            "start_wall_time_ns": 1_000,
            "start_monotonic_time_ns": 100,
            "config_hash": "0123456789abcdef",
            "git_commit": "abc123",
            "phase": "fixture",
            "record_count": 7,
        }
        values.update(overrides)
        return values

    def write_metadata(self, root: Path, **overrides: object) -> Path:
        path = root / "trace.csv.metadata.json"
        path.write_text(json.dumps(self.metadata(**overrides)), encoding="utf-8")
        return path

    def test_converts_monotonic_prediction_time_to_wall_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metadata = TraceMetadata.from_json(
                self.write_metadata(Path(directory))
            )
        self.assertEqual(metadata.prediction_wall_time_ns(145), 1_045)

    def test_rejects_incompatible_trace_schema(self) -> None:
        invalid_cases = (
            ("magic", "OTHER"),
            ("schema_version", 3),
            ("header_size", 191),
            ("record_size", 199),
            ("feature_count", 9),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for field, value in invalid_cases:
                with self.subTest(field=field):
                    with self.assertRaises(ValueError):
                        TraceMetadata.from_json(
                            self.write_metadata(root, **{field: value})
                        )

    def test_accepts_the_known_schema_v1_csv_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metadata = TraceMetadata.from_json(
                self.write_metadata(
                    Path(directory),
                    schema_version=1,
                    record_size=208,
                    feature_count=4,
                )
            )

        self.assertEqual(metadata.feature_count, 4)

    def test_rejects_a_mixed_schema_v1_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                TraceMetadata.from_json(
                    self.write_metadata(
                        Path(directory),
                        schema_version=1,
                        record_size=200,
                        feature_count=4,
                    )
                )


class PhaseIndexTest(unittest.TestCase):
    def write_phases(self, root: Path) -> Path:
        path = root / "phase_intervals.tsv"
        path.write_text(
            "profile\tworkload\trepetition\tsample_start\tsample_end\t"
            "predicted_at\tphase_index\tphase_name\tsegment\tduration_seconds\n"
            "p\tw\t1\t2026-07-17T00:00:00+00:00\t"
            "2026-07-17T00:00:10+00:00\t2026-07-17T00:00:05+00:00\t"
            "1\tfirst\tcold_start\t10\n"
            "p\tw\t1\t2026-07-17T00:00:20+00:00\t"
            "2026-07-17T00:00:30+00:00\t2026-07-17T00:00:25+00:00\t"
            "2\tsecond\tsteady\t10\n",
            encoding="utf-8",
        )
        return path

    @staticmethod
    def ns(second: int) -> int:
        return int(
            datetime(2026, 7, 17, 0, 0, second, tzinfo=timezone.utc).timestamp()
            * 1_000_000_000
        )

    def test_lookup_handles_boundaries_gaps_and_outside_times(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            phases = PhaseIndex.from_tsv(self.write_phases(Path(directory)))

        self.assertEqual(phases.lookup(self.ns(0) - 1).phase_name, "first")
        self.assertEqual(phases.lookup(self.ns(5)).phase_name, "first")
        self.assertEqual(phases.lookup(self.ns(15)).phase_name, "first")
        self.assertEqual(phases.lookup(self.ns(25)).phase_name, "second")
        self.assertEqual(phases.lookup(self.ns(30) + 1).phase_name, "second")


class StreamingAccumulatorTest(unittest.TestCase):
    def test_running_moments_and_correlation_are_stable(self) -> None:
        moments = RunningMoments()
        for value in (1.0, 2.0, 3.0):
            moments.add(value)
        self.assertEqual(moments.count, 3)
        self.assertAlmostEqual(moments.mean, 2.0)
        self.assertAlmostEqual(moments.variance, 1.0)

        correlation = OnlineCorrelation(2)
        for value in ((1.0, 2.0), (2.0, 4.0), (3.0, 6.0)):
            correlation.add(value)
        self.assertAlmostEqual(correlation.correlation(0, 1), 1.0)

    def test_calibration_uses_probability_not_decision_threshold(self) -> None:
        calibration = CalibrationTable(10)
        for probability, predicted, actual in (
            (0.8, 1, 1),
            (0.7, 1, 0),
            (0.2, 0, 0),
            (0.3, 0, 1),
        ):
            calibration.add(probability, actual, predicted)
        self.assertAlmostEqual(calibration.brier_score, 0.265)
        self.assertAlmostEqual(calibration.ece, 0.45)
        self.assertEqual(calibration.bins[8].confusion.tp, 1)
        self.assertEqual(calibration.bins[7].confusion.fp, 1)
        self.assertEqual(calibration.bins[2].confusion.tn, 1)
        self.assertEqual(calibration.bins[3].confusion.fn, 1)

    def test_distribution_summary_reports_exact_nearest_rank_quantiles(self) -> None:
        summary = hp_trace_analysis.DistributionSummary()
        for value in (1.0, 2.0, 3.0, 4.0, 100.0):
            summary.add(value)

        self.assertEqual(summary.count, 5)
        self.assertEqual(summary.quantile(0.10), 1.0)
        self.assertEqual(summary.quantile(0.50), 3.0)
        self.assertEqual(summary.quantile(0.90), 100.0)


class TraceAnalyzerTest(unittest.TestCase):
    @staticmethod
    def row(
        sequence: int,
        object_hash: int,
        feature_0: float,
        probability: float,
        predicted: int,
        label_heat: float,
        actual: int,
        cold_start_fallback: bool = False,
        heat_after_current_access: float = 100.0,
        heat_threshold_at_prediction: float = 100.0,
    ) -> dict[str, str]:
        values: dict[str, object] = {
            "io_sequence": sequence,
            "object_key_hash": object_hash,
            "prediction_time_ns": 1_000 + sequence,
            "label_deadline_ns": 2_000 + sequence,
            "label_completion_time_ns": 2_001 + sequence,
            "heat_after_current_access": heat_after_current_access,
            "heat_label_threshold_at_prediction": heat_threshold_at_prediction,
            "predicted_hot_probability": probability,
            "hot_predict_threshold": 0.5,
            "label_heat": label_heat,
            "label_heat_threshold": 100.0,
            "tracked_access_count": 1,
            "time_since_previous_access_ns": 0,
            "long_window_access_count": 0,
            "short_window_access_count": 0,
            "future_window_access_count": 1,
            "outcome": "evaluated",
            "cold_start_fallback": cold_start_fallback,
            "evaluation_capacity_drop": "False",
            "predicted_label": predicted,
            "actual_label": actual,
        }
        for index in range(6):
            values[f"feature_{index}"] = feature_0 + index
        return {key: str(value) for key, value in values.items()}

    @staticmethod
    def context(
        osd_id: int = 0,
        steady_state: bool = False,
        relative_time_ns: int = 0,
    ) -> RecordContext:
        return RecordContext(
            workload="fixture",
            osd_id=osd_id,
            phase=PhaseIndexTestPhase.phase(),
            steady_state=steady_state,
            relative_time_ns=relative_time_ns,
        )

    def test_consumes_rows_into_all_primary_aggregates(self) -> None:
        analyzer = TraceAnalyzer(feature_count=6)
        rows = (
            self.row(1, 10, 4.0, 0.8, 1, 200.0, 1),
            self.row(2, 10, 1.0, 0.7, 1, 50.0, 0),
            self.row(3, 20, 2.0, 0.2, 0, 50.0, 0),
            self.row(4, 30, 5.0, 0.3, 0, 200.0, 1),
        )
        for row in rows:
            analyzer.consume(row, self.context())

        workload = analyzer.workloads["fixture"]
        self.assertEqual(
            (workload.confusion.tp, workload.confusion.fp,
             workload.confusion.tn, workload.confusion.fn),
            (1, 1, 1, 1),
        )
        self.assertAlmostEqual(workload.calibration.brier_score, 0.265)
        self.assertEqual(workload.feature_by_actual[1][0].count, 2)
        self.assertAlmostEqual(workload.feature_by_actual[1][0].mean, 4.5)
        self.assertAlmostEqual(workload.feature_by_actual[0][0].mean, 1.5)
        self.assertGreater(workload.correlations.feature_label_correlation(0), 0.8)
        self.assertEqual(workload.objects[(0, 10)].io_count, 2)
        self.assertEqual(workload.objects[(0, 10)].error_count, 1)
        self.assertEqual(len(analyzer.phase_confusion), 1)

    def test_separates_cold_start_fallback_from_model_predictions(self) -> None:
        analyzer = TraceAnalyzer(feature_count=6)
        analyzer.consume(
            self.row(1, 10, 1.0, 0.0, 0, 200.0, 1, True),
            self.context(),
        )
        analyzer.consume(
            self.row(2, 20, 2.0, 0.8, 1, 200.0, 1),
            self.context(),
        )
        workload = analyzer.workloads["fixture"]
        self.assertEqual(workload.cold_start_fallback_confusion.count, 1)
        self.assertEqual(workload.model_prediction_confusion.count, 1)
        self.assertEqual(workload.cold_start_fallback_confusion.fn, 1)

    def test_steady_state_excludes_warmup_and_tracks_error_confidence(self) -> None:
        analyzer = TraceAnalyzer(feature_count=6)
        analyzer.consume(
            self.row(1, 10, 1.0, 0.55, 1, 50.0, 0),
            self.context(steady_state=False),
        )
        analyzer.consume(
            self.row(2, 20, 1.0, 0.65, 1, 50.0, 0),
            self.context(steady_state=True),
        )
        analyzer.consume(
            self.row(3, 30, 1.0, 0.05, 0, 200.0, 1),
            self.context(steady_state=True),
        )

        workload = analyzer.workloads["fixture"]
        self.assertEqual(workload.confusion.fp, 2)
        self.assertEqual(
            (workload.steady_confusion.fp, workload.steady_confusion.fn),
            (1, 1),
        )
        self.assertEqual(
            workload.steady_error_confidence.counts["fp"],
            [0, 1, 0, 0, 0],
        )
        self.assertEqual(
            workload.steady_error_confidence.counts["fn"],
            [0, 0, 0, 0, 1],
        )
        self.assertEqual(workload.steady_calibration.count, 2)
        self.assertEqual(workload.steady_feature_by_outcome["fp"][0].count, 1)
        self.assertEqual(workload.steady_feature_by_outcome["fn"][0].count, 1)
        self.assertEqual(len(workload.steady_objects), 2)

    def test_steady_time_bins_use_30_second_relative_boundaries(self) -> None:
        analyzer = TraceAnalyzer(feature_count=6)
        second_ns = 1_000_000_000
        rows = (
            self.row(1, 10, 1.0, 0.8, 1, 200.0, 1),
            self.row(2, 20, 1.0, 0.2, 0, 50.0, 0),
            self.row(3, 30, 1.0, 0.8, 1, 50.0, 0),
        )
        relative_times = (
            120 * second_ns,
            150 * second_ns - 1,
            150 * second_ns,
        )
        for row, relative_time_ns in zip(rows, relative_times):
            analyzer.consume(
                row,
                self.context(
                    steady_state=True,
                    relative_time_ns=relative_time_ns,
                ),
            )

        time_bins = analyzer.workloads["fixture"].steady_time_confusion
        self.assertEqual(time_bins[4].count, 2)
        self.assertEqual(time_bins[5].count, 1)

    def test_steady_object_stats_keep_the_full_confusion_matrix(self) -> None:
        analyzer = TraceAnalyzer(feature_count=6)
        rows = (
            self.row(1, 10, 1.0, 0.8, 1, 200.0, 1),
            self.row(2, 10, 1.0, 0.8, 1, 50.0, 0),
            self.row(3, 10, 1.0, 0.2, 0, 50.0, 0),
            self.row(4, 10, 1.0, 0.2, 0, 200.0, 1),
        )
        for row in rows:
            analyzer.consume(row, self.context(steady_state=True))

        stats = analyzer.workloads["fixture"].steady_objects[(0, 10)]
        self.assertEqual((stats.tp, stats.fp, stats.tn, stats.fn), (1, 1, 1, 1))
        self.assertAlmostEqual(stats.accuracy, 0.5)

    def test_object_identity_includes_osd_id(self) -> None:
        analyzer = TraceAnalyzer(feature_count=6)
        analyzer.consume(self.row(1, 10, 1.0, 0.8, 1, 200.0, 1), self.context(0))
        analyzer.consume(self.row(2, 10, 1.0, 0.8, 1, 200.0, 1), self.context(1))
        self.assertEqual(len(analyzer.workloads["fixture"].objects), 2)

    def test_cold_origin_confusion_measures_activation_quality(self) -> None:
        analyzer = TraceAnalyzer(feature_count=4, heat_increment=100.0)
        rows = (
            self.row(1, 10, 1.0, 0.8, 1, 200.0, 1,
                     heat_threshold_at_prediction=50.0),
            self.row(2, 20, 1.0, 0.2, 0, 200.0, 1,
                     heat_threshold_at_prediction=50.0),
            self.row(3, 30, 1.0, 0.8, 1, 50.0, 0,
                     heat_threshold_at_prediction=50.0),
            self.row(4, 40, 1.0, 0.8, 1, 200.0, 1,
                     heat_after_current_access=200.0,
                     heat_threshold_at_prediction=50.0),
        )
        for row in rows:
            analyzer.consume(row, self.context())

        cold_origin = analyzer.workloads["fixture"].cold_origin_confusion
        self.assertEqual(
            (cold_origin.tp, cold_origin.fp, cold_origin.tn, cold_origin.fn),
            (1, 1, 0, 1),
        )
        self.assertAlmostEqual(cold_origin.recall, 0.5)
        self.assertAlmostEqual(cold_origin.precision, 0.5)

    def test_rejects_inconsistent_labels_and_non_evaluated_rows(self) -> None:
        analyzer = TraceAnalyzer(feature_count=6)
        inconsistent_actual = self.row(1, 10, 1.0, 0.8, 1, 50.0, 1)
        with self.assertRaisesRegex(ValueError, "actual_label"):
            analyzer.consume(inconsistent_actual, self.context())

        inconsistent_prediction = self.row(1, 10, 1.0, 0.2, 1, 200.0, 1)
        with self.assertRaisesRegex(ValueError, "predicted_label"):
            analyzer.consume(inconsistent_prediction, self.context())

        dropped = self.row(1, 10, 1.0, 0.8, 1, 200.0, 1)
        dropped["outcome"] = "evaluation_capacity_drop"
        with self.assertRaisesRegex(ValueError, "evaluated"):
            analyzer.consume(dropped, self.context())

    def test_margin_buckets_keep_boundary_errors_visible(self) -> None:
        self.assertEqual(prediction_margin_bucket(-0.3), "strong_cold")
        self.assertEqual(prediction_margin_bucket(-0.1), "near_cold")
        self.assertEqual(prediction_margin_bucket(0.1), "near_hot")
        self.assertEqual(prediction_margin_bucket(0.3), "strong_hot")
        self.assertEqual(label_margin_bucket(-0.6), "strong_cold")
        self.assertEqual(label_margin_bucket(-0.1), "near_cold")
        self.assertEqual(label_margin_bucket(0.1), "near_hot")
        self.assertEqual(label_margin_bucket(0.6), "strong_hot")


class PhaseIndexTestPhase:
    @staticmethod
    def phase():
        from test_sh.hp_trace_analysis import Phase

        return Phase(
            phase_index=1,
            phase_name="fixture_phase",
            segment="steady",
            start_ns=0,
            end_ns=10_000,
        )


class EndToEndAnalysisTest(unittest.TestCase):
    def write_fixture(self, root: Path) -> Path:
        workload = root / "fixture_workload"
        trace_dir = workload / "trace"
        trace_dir.mkdir(parents=True)
        phase_path = workload / "phase_intervals.tsv"
        phase_path.write_text(
            "profile\tworkload\trepetition\tsample_start\tsample_end\t"
            "predicted_at\tphase_index\tphase_name\tsegment\tduration_seconds\n"
            "p\tfixture_workload\t1\t2026-07-17T00:00:00+00:00\t"
            "2026-07-17T00:00:10+00:00\t2026-07-17T00:00:05+00:00\t"
            "1\tfixture_phase\tsteady\t10\n",
            encoding="utf-8",
        )
        (workload / "metadata.json").write_text(
            json.dumps({"started_at": "2026-07-17T00:00:00+00:00"}),
            encoding="utf-8",
        )
        wall_start = PhaseIndexTest.ns(0)
        rows = (
            TraceAnalyzerTest.row(1, 10, 4.0, 0.8, 1, 200.0, 1),
            TraceAnalyzerTest.row(2, 10, 2.0, 0.2, 0, 50.0, 0),
        )
        for index, row in enumerate(rows):
            prediction_time_ns = 1_000 + (120 + index * 30) * 1_000_000_000
            row["prediction_time_ns"] = str(prediction_time_ns)
            row["label_deadline_ns"] = str(
                prediction_time_ns + 10 * 1_000_000_000
            )
            row["label_completion_time_ns"] = str(
                prediction_time_ns + 10 * 1_000_000_000 + 1
            )
        for osd_id, row in enumerate(rows):
            csv_path = trace_dir / f"osd.{osd_id}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            metadata = TraceMetadataTest().metadata(
                feature_count=3,
                osd_id=osd_id,
                start_wall_time_ns=wall_start,
                start_monotonic_time_ns=1_000,
                phase="fixture_workload",
                record_count=1,
            )
            csv_path.with_suffix(".csv.metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
        return root

    def test_run_root_analysis_writes_the_complete_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "analysis"
            result = analyze_run_root(self.write_fixture(root / "run"), output)

            self.assertEqual(result.analyzer.record_count, 2)
            self.assertEqual(result.trace_file_count, 2)
            self.assertEqual(
                {path.name for path in output.iterdir()}, EXPECTED_OUTPUT_FILES
            )
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["global"]["count"], 2)
            self.assertEqual(summary["global"]["tp"], 1)
            self.assertEqual(summary["global"]["tn"], 1)
            self.assertEqual(summary["trace_file_count"], 2)
            for filename in (
                "prediction_outcome_summary.tsv",
                "prediction_probability_bins.tsv",
                "prediction_feature_profiles.tsv",
                "prediction_object_summary.tsv",
                "prediction_time_series.tsv",
            ):
                self.assertTrue((output / filename).is_file())

            def read_tsv(filename: str) -> list[dict[str, str]]:
                with (output / filename).open(newline="", encoding="utf-8") as stream:
                    return list(csv.DictReader(stream, delimiter="\t"))

            outcomes = read_tsv("prediction_outcome_summary.tsv")
            steady = next(
                row for row in outcomes
                if row["workload"] == "fixture_workload"
                and row["scope"] == "steady"
            )
            self.assertEqual(
                (steady["tp"], steady["fp"], steady["tn"], steady["fn"]),
                ("1", "0", "1", "0"),
            )

            probability_bins = read_tsv("prediction_probability_bins.tsv")
            self.assertEqual(
                sum(
                    int(row["count"])
                    for row in probability_bins
                    if row["workload"] == "fixture_workload"
                ),
                2,
            )

            feature_profiles = read_tsv("prediction_feature_profiles.tsv")
            tp_feature_0 = next(
                row for row in feature_profiles
                if row["workload"] == "fixture_workload"
                and row["outcome"] == "tp"
                and row["feature_index"] == "0"
            )
            self.assertEqual(tp_feature_0["p50"], "4.0")

            object_summary = read_tsv("prediction_object_summary.tsv")
            fixture_objects = next(
                row for row in object_summary
                if row["workload"] == "fixture_workload"
            )
            self.assertEqual(fixture_objects["object_count"], "2")

            time_series = read_tsv("prediction_time_series.tsv")
            fixture_bins = [
                row for row in time_series
                if row["workload"] == "fixture_workload"
            ]
            self.assertEqual(
                [(row["window_start_seconds"], row["count"]) for row in fixture_bins],
                [("120", "1"), ("150", "1")],
            )
            report = (output / "REPORT.md").read_text(encoding="utf-8")
            self.assertNotIn("short_long_access_rate_log_margin", report)
            self.assertNotIn("## 冷转热识别", report)
            self.assertIn("## 稳态全 I/O 预测结果", report)
            self.assertIn("## 稳态预测概率分箱", report)
            self.assertIn("## 稳态 Object 误差集中度", report)
            self.assertIn("## 稳态 30 秒时序", report)
            self.assertIn("## 本轮核心发现", report)

    def test_metadata_record_count_must_match_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.write_fixture(Path(directory) / "run")
            metadata_path = root / "fixture_workload/trace/osd.0.csv.metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["record_count"] = 2
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "record_count"):
                analyze_run_root(root, Path(directory) / "analysis")

    def test_repetition_suffix_does_not_change_trace_phase_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.write_fixture(Path(directory) / "run")
            (root / "fixture_workload").rename(root / "fixture_workload_r1")

            result = analyze_run_root(root, Path(directory) / "analysis")

            self.assertEqual(result.analyzer.record_count, 2)
            self.assertIn("fixture_workload_r1", result.analyzer.workloads)


if __name__ == "__main__":
    unittest.main()
