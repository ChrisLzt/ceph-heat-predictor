#!/usr/bin/env python3

import csv
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from project_hp_trace_v1_csv import HEADER, RECORD, project_trace


class ProjectHpTraceV1CsvTest(unittest.TestCase):
    def test_projects_first_three_features_and_drops_future_heat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "osd.0.csv"
            output = root / "osd.0.bin"
            metadata = {
                "magic": "HPTRACE1",
                "schema_version": 1,
                "header_size": 192,
                "record_size": 208,
                "feature_count": 4,
                "osd_id": 0,
                "session_id": 7,
                "start_wall_time_ns": 100,
                "start_monotonic_time_ns": 200,
                "git_commit": "fixture",
                "phase": "steady",
                "record_count": 1,
            }
            source.with_suffix(".csv.metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            fields = [
                "io_sequence", "object_key_hash", "prediction_time_ns",
                "label_deadline_ns", "label_completion_time_ns",
                "feature_0", "feature_1", "feature_2", "feature_3",
                "heat_after_current_access",
                "heat_label_threshold_at_prediction",
                "predicted_hot_probability", "hot_predict_threshold",
                "label_heat", "label_heat_threshold",
                "future_window_added_heat", "tracked_access_count",
                "time_since_previous_access_ns", "long_window_access_count",
                "short_window_access_count", "future_window_access_count",
                "outcome", "cold_start_fallback",
                "evaluation_capacity_drop", "predicted_label", "actual_label",
            ]
            row = [
                1, 2, 300, 400, 500,
                1.25, 2.5, 3.75, 99.0,
                100.0, 50.0, 0.75, 0.5, 120.0, 60.0,
                20.0, 3, 4, 5, 6, 7,
                "evaluated", "True", "False", 1, 1,
            ]
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(fields)
                writer.writerow(row)

            count = project_trace(source, output, config_hash=0x1234)

            self.assertEqual(count, 1)
            raw = output.read_bytes()
            header = HEADER.unpack(raw[: HEADER.size])
            record = RECORD.unpack(raw[HEADER.size :])
            self.assertEqual(header[1:5], (2, 192, 200, 3))
            self.assertEqual(header[10], 0x1234)
            self.assertEqual(record[5:13], (1.25, 2.5, 3.75, 0, 0, 0, 0, 0))
            self.assertEqual(record[13:19], (100.0, 50.0, 0.75, 0.5, 120.0, 60.0))
            self.assertEqual(record[19:24], (3, 4, 5, 6, 7))
            self.assertEqual(record[24:28], (0, 1, 1, 1))

    def test_rejects_non_evaluated_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "osd.0.csv"
            output = root / "osd.0.bin"
            source.with_suffix(".csv.metadata.json").write_text(
                json.dumps({
                    "magic": "HPTRACE1",
                    "schema_version": 1,
                    "header_size": 192,
                    "record_size": 208,
                    "feature_count": 4,
                    "osd_id": 0,
                    "session_id": 1,
                    "start_wall_time_ns": 1,
                    "start_monotonic_time_ns": 1,
                    "git_commit": "fixture",
                    "phase": "steady",
                    "record_count": 1,
                }), encoding="utf-8"
            )
            source.write_text(
                "io_sequence,outcome\n1,prediction_error\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "evaluated"):
                project_trace(source, output, config_hash=1)


if __name__ == "__main__":
    unittest.main()
