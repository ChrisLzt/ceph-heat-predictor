import unittest
from export_results import short_window_diagnostics


class RealtimeDiagnosticsTests(unittest.TestCase):
    def test_bins_weight_counts_not_percentages(self):
        rows = [
            {'stage': 'lru', 'onode_hits': '0', 'onode_misses': '100', 'begin_wall': 100, 'end_wall': 102},
            {'stage': 's3fifo', 'onode_hits': '99', 'onode_misses': '1', 'begin_wall': 282, 'end_wall': 284},
            {'stage': 's3fifo', 'onode_hits': '1', 'onode_misses': '1', 'begin_wall': 284, 'end_wall': 286},
        ]
        result = short_window_diagnostics(rows, 100)
        self.assertEqual(result['minimum_valid_interval_percent'], 50)
        self.assertAlmostEqual(result['minimum_30s_bin_percent'], 10000 / 102)
        self.assertEqual(result['observed_30s_bins'], 1)

    def test_no_queries_remain_unavailable(self):
        result = short_window_diagnostics([{'stage': 's3fifo', 'onode_hits': '0', 'onode_misses': '0'}], 100)
        self.assertIsNone(result['minimum_valid_interval_percent'])
        self.assertIsNone(result['minimum_30s_bin_percent'])
        self.assertEqual(result['valid_nonzero_s3fifo_intervals'], 0)
