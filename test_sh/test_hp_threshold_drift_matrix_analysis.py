#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hp_threshold_drift_matrix_analysis import analyze_matrix, write_outputs


class ThresholdDriftMatrixAnalysisTest(unittest.TestCase):
    def test_replay_labels_and_adaptation_logs_drive_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "run"
            replay_root = root / "replay"
            workload = "fixture_r1"
            trace_dir = run_root / "V2" / workload / "trace"
            trace_dir.mkdir(parents=True)
            (run_root / "V2" / workload / "phase_intervals.tsv").write_text(
                "sample_start\tsample_end\tphase_index\tphase_name\tsegment\n"
                "1970-01-01T00:00:00+00:00\t"
                "1970-01-01T00:00:10+00:00\t0\tsteady_0\tsteady\n",
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

            destination = replay_root / "recent_disabled" / workload
            destination.mkdir(parents=True)
            (destination / "osd.0.replay.tsv").write_text(
                "io_sequence\tobject_key_hash\tprediction_time_ns\t"
                "label_completion_time_ns\tonline_hot_probability\t"
                "replay_hot_probability\tprobability_abs_error\t"
                "hot_predict_threshold\tonline_label\treplay_label\t"
                "actual_label\tcold_start_fallback\n"
                "1\t10\t100\t200\t0.2\t0.8\t0.6\t0.5\t0\t1\t1\t0\n"
                "2\t20\t200\t300\t0.8\t0.2\t0.6\t0.5\t1\t0\t0\t0\n",
                encoding="utf-8",
            )
            (destination / "osd.0.replay.log").write_text(
                "adaptation_profile=disabled\n"
                "records=2\n"
                "arf_warnings=0\n"
                "arf_drifts=0\n"
                "arf_background_promotions=0\n"
                "arf_background_discards=0\n"
                "arf_background_training_updates=0\n",
                encoding="utf-8",
            )

            analysis = analyze_matrix(run_root, replay_root)
            metrics = analysis.metrics[
                ("recent_disabled", "workload", workload, "all")
            ]
            self.assertEqual(metrics.confusion.tp, 1)
            self.assertEqual(metrics.confusion.tn, 1)
            self.assertEqual(metrics.confusion.count, 2)
            adaptation = analysis.adaptation[
                ("recent_disabled", workload)
            ]
            self.assertEqual(adaptation.records, 2)
            self.assertEqual(adaptation.drifts, 0)

            output_dir = root / "output"
            write_outputs(analysis, output_dir)
            self.assertTrue((output_dir / "metrics.tsv").is_file())
            self.assertTrue((output_dir / "adaptation.tsv").is_file())


if __name__ == "__main__":
    unittest.main()
