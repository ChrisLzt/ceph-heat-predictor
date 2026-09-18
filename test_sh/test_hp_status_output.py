"""Run against hp_status_output_probe; no running Ceph cluster is required."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

PROBE = os.environ["HP_STATUS_OUTPUT_PROBE"]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/pybind"))
from ceph_argparse import parse_funcsig, validate_command


def output(scenario="healthy", fmt="json", mode="simple"):
    return subprocess.check_output([PROBE, scenario, fmt, mode], text=True)


def summary(scenario="healthy", mode="simple"):
    return json.loads(output(scenario, mode=mode))["summary"]


class StatusOutputTests(unittest.TestCase):
    def test_simple_keeps_accuracy_and_uses_global_counts(self):
        s = summary()
        self.assertEqual(set(s), {"osds", "samples", "prediction", "latency"})
        self.assertEqual(s["osds"], {
            "enabled_osds": 2, "reporting_osds": 2, "up_osds": 2})
        self.assertEqual(s["samples"], {"hp_labeled_io_total": 100})
        self.assertEqual(set(s["prediction"]), {
            "hp_hot_accuracy", "hp_hot_precision", "hp_hot_recall",
            "hp_eval_pred_hot_percent", "hp_eval_actual_hot_percent"})
        self.assertEqual(s["prediction"]["hp_hot_accuracy"], 90)
        self.assertEqual(s["prediction"]["hp_hot_precision"], 93.75)
        self.assertAlmostEqual(s["prediction"]["hp_hot_recall"], 90.909, places=3)
        self.assertEqual(s["prediction"]["hp_eval_pred_hot_percent"], 64)
        self.assertEqual(s["prediction"]["hp_eval_actual_hot_percent"], 66)
        self.assertEqual(s["latency"], {"hp_predict_latency": {"avgtime_us": 89}})

    def test_plain_is_short_and_accuracy_precedes_precision(self):
        text = output(fmt="plain")
        self.assertEqual(len(text.splitlines()), 5)
        self.assertIn("Accuracy: 90.00%", text)
        self.assertLess(text.index("Accuracy:"), text.index("Precision:"))
        self.assertIn("89.00 us", text)
        self.assertNotIn("ALERT", text)  # Ordinary queue occupancy is not an error.

    def test_absent_samples_and_timing_are_not_reported_as_zero(self):
        s = summary("empty")
        self.assertTrue(all(v is None for v in s["prediction"].values()))
        self.assertIsNone(s["latency"]["hp_predict_latency"]["avgtime_us"])
        self.assertIn("no evaluated samples", output("empty", "plain"))
        self.assertIn("no timing samples", output("empty", "plain"))
        self.assertIsNone(summary("no_timing")["latency"]["hp_predict_latency"]["avgtime_us"])

    def test_zero_class_denominators_are_unavailable(self):
        p = summary("cold_only")["prediction"]
        self.assertEqual(p["hp_hot_accuracy"], 100)
        self.assertIsNone(p["hp_hot_precision"])
        self.assertIsNone(p["hp_hot_recall"])
        self.assertEqual(p["hp_eval_actual_hot_percent"], 0)

    def test_alerts_are_conditional_and_reports_exclude_torn_osds(self):
        self.assertEqual(summary("errors")["alerts"], {
            "hp_predict_error_count": 2, "hp_background_error_count": 4,
            "hp_eval_drop_count": 6, "hp_train_drop_count": 8})
        missing = summary("missing")
        self.assertEqual(missing["alerts"], {"missing_osds": [1]})
        self.assertEqual(missing["samples"]["hp_labeled_io_total"], 30)
        self.assertEqual(missing["osds"]["reporting_osds"], 1)
        self.assertIn("missing OSDs: 1", output("missing", "plain"))
        self.assertEqual(summary("disabled")["alerts"], {"disabled_osds": 1})
        self.assertEqual(summary("no_osds")["alerts"], {"no_up_osds": True})

    def test_detail_retains_machine_contract_and_nanosecond_units(self):
        s = summary(mode="detail")
        self.assertEqual(set(s), {
            "osds", "samples", "heat_state", "confusion_matrix",
            "actual_behavior", "prediction", "training", "model_adaptation",
            "trace", "latency", "read_ops", "write_ops"})
        self.assertEqual(s["samples"]["hp_io_count"], 104)
        self.assertEqual(s["samples"]["hp_pending_io_count"], 4)
        self.assertEqual(s["latency"]["hp_predict_latency"], {
            "avgcount": 4, "sum_ns": 356000, "avgtime_ns": 89000})
        self.assertEqual(s["heat_state"]["future_access_threshold"]["avg"], 3)
        self.assertIn("hp_hot_balanced_accuracy", s["prediction"])
        self.assertEqual(json.loads(output(fmt="plain", mode="detail"))["summary"], s)

    def test_json_pretty_keeps_the_selected_content(self):
        for mode in ("simple", "detail"):
            self.assertEqual(json.loads(output(fmt="json-pretty", mode=mode))["summary"],
                             summary(mode=mode))

    def test_real_command_signature_accepts_detail_flag(self):
        tokens = output("command").split()
        sig = [dict(part.split("=", 1) for part in token.split(","))
               if token.startswith("name=") else token for token in tokens]
        commands = {"hp": {"sig": parse_funcsig(sig), "help": "HP status"}}
        parsed = validate_command(commands, ["osd", "hp", "status", "--detail"])
        self.assertIs(parsed.get("detail"), True)
        parsed = validate_command(commands, ["osd", "hp", "status"])
        self.assertFalse(parsed.get("detail", False))


if __name__ == "__main__":
    unittest.main()
