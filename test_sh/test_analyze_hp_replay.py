#!/usr/bin/env python3

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hp_replay_analysis import analyze_replay_run, write_replay_outputs


class ReplayAnalysisTest(unittest.TestCase):
    def test_repetition_suffix_does_not_change_trace_phase_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "run"
            replay_root = root / "replay"
            workload = "fixture_workload"
            repeated_workload = f"{workload}_r1"
            trace_dir = run_root / repeated_workload / "trace"
            replay_dir = replay_root / repeated_workload
            trace_dir.mkdir(parents=True)
            replay_dir.mkdir(parents=True)

            (run_root / repeated_workload / "phase_intervals.tsv").write_text(
                "sample_start\tsample_end\tphase_index\tphase_name\tsegment\n"
                "1970-01-01T00:00:00+00:00\t"
                "1970-01-01T00:00:10+00:00\t0\tsteady\tsteady\n",
                encoding="utf-8",
            )
            metadata = {
                "magic": "HPTRACE1",
                "schema_version": 4,
                "header_size": 192,
                "record_size": 184,
                "feature_count": 3,
                "osd_id": 0,
                "session_id": 1,
                "start_wall_time_ns": 1_000_000_000,
                "start_monotonic_time_ns": 100,
                "config_hash": "1",
                "git_commit": "fixture",
                "phase": workload,
                "record_count": 1,
            }
            (trace_dir / "osd.0.csv.metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            (trace_dir / "osd.0.csv").write_text(
                "io_sequence,object_key_hash,prediction_time_ns,"
                "label_completion_time_ns,predicted_hot_probability,"
                "hot_predict_threshold,predicted_label,actual_label\n"
                "1,10,100,200,0.8,0.5,1,1\n",
                encoding="utf-8",
            )
            (replay_dir / "osd.0.replay.tsv").write_text(
                "io_sequence\tobject_key_hash\tprediction_time_ns\t"
                "label_completion_time_ns\tonline_hot_probability\t"
                "replay_hot_probability\tprobability_abs_error\t"
                "hot_predict_threshold\tonline_label\treplay_label\t"
                "actual_label\tcold_start_fallback\n"
                "1\t10\t100\t200\t0.8\t0.8\t0\t0.5\t1\t1\t1\t0\n",
                encoding="utf-8",
            )

            analysis = analyze_replay_run(run_root, replay_root)

            self.assertEqual(len(analysis.summaries), 1)
            self.assertTrue(analysis.summaries[0].gate_passed)

    def test_stream_join_phase_metrics_and_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "run"
            replay_root = root / "replay"
            output_root = root / "output"
            workload = "fixture_workload"
            trace_dir = run_root / workload / "trace"
            replay_dir = replay_root / workload
            trace_dir.mkdir(parents=True)
            replay_dir.mkdir(parents=True)

            (run_root / workload / "phase_intervals.tsv").write_text(
                "sample_start\tsample_end\tphase_index\tphase_name\tsegment\n"
                "1970-01-01T00:00:00+00:00\t"
                "1970-01-01T00:00:10+00:00\t0\tsteady\tsteady\n",
                encoding="utf-8",
            )
            metadata = {
                "magic": "HPTRACE1",
                "schema_version": 4,
                "header_size": 192,
                "record_size": 184,
                "feature_count": 3,
                "osd_id": 0,
                "session_id": 1,
                "start_wall_time_ns": 1_000_000_000,
                "start_monotonic_time_ns": 100,
                "config_hash": "1",
                "git_commit": "fixture",
                "phase": workload,
                "record_count": 2,
            }
            (trace_dir / "osd.0.csv.metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            source_fields = [
                "io_sequence",
                "object_key_hash",
                "prediction_time_ns",
                "label_completion_time_ns",
                "predicted_hot_probability",
                "hot_predict_threshold",
                "predicted_label",
                "actual_label",
            ]
            source_rows = [
                ["1", "10", "100", "200", "0.8", "0.5", "1", "1"],
                ["2", "20", "200", "300", "0.2", "0.5", "0", "0"],
            ]
            with (trace_dir / "osd.0.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.writer(stream)
                writer.writerow(source_fields)
                writer.writerows(source_rows)

            replay_fields = [
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
            ]
            replay_rows = [
                ["1", "10", "100", "200", "0.8", "0.8", "0", "0.5", "1", "1", "1", "0"],
                ["2", "20", "200", "300", "0.2", "0.2", "0", "0.5", "0", "0", "0", "0"],
            ]
            with (replay_dir / "osd.0.replay.tsv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(replay_fields)
                writer.writerows(replay_rows)

            analysis = analyze_replay_run(run_root, replay_root)
            self.assertEqual(len(analysis.summaries), 1)
            summary = analysis.summaries[0]
            self.assertEqual(summary.record_count, 2)
            self.assertEqual(summary.association_ratio, 1.0)
            self.assertEqual(summary.class_agreement, 1.0)
            self.assertEqual(summary.probability_mae, 0.0)
            self.assertTrue(summary.gate_passed)
            self.assertEqual(analysis.phase_summaries[0].phase_name, "steady")

            write_replay_outputs(analysis, output_root)
            self.assertTrue((output_root / "replay_summary.tsv").is_file())
            self.assertTrue((output_root / "replay_phase_summary.tsv").is_file())
            self.assertTrue((output_root / "replay_mismatches.tsv").is_file())
            report = (output_root / "REPLAY_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("通过", report)

            replay_rows[1][5] = "0.9"
            replay_rows[1][6] = "0.7"
            replay_rows[1][9] = "1"
            with (replay_dir / "osd.0.replay.tsv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(replay_fields)
                writer.writerows(replay_rows)
            failed = analyze_replay_run(run_root, replay_root).summaries[0]
            self.assertFalse(failed.gate_passed)
            self.assertIn("class_agreement", failed.gate_failures)
            self.assertIn("probability_mae", failed.gate_failures)
            self.assertIn("probability_abs_error_p95", failed.gate_failures)
            self.assertIn("accuracy_delta", failed.gate_failures)


if __name__ == "__main__":
    unittest.main()
