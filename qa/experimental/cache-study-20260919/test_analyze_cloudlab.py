import copy
import json
from pathlib import Path
import tempfile
import unittest
from collections import Counter
from analyze_cloudlab import analyze_case, classify, delta, interval, metrics, percent, suite_completion
from run_cloudlab_study import CASES


def sample(wall, hits=10, misses=2, policy='lru', generation=0):
    osds = {}
    for i in ('0', '1', '2'):
        osds[i] = {
            'begin_wall': wall, 'end_wall': wall + .01, 'begin_mono': wall,
            'cache': {'cache_instance': i, 'policy_generation': generation,
                      'effective_policy': policy, 'shards': [{'effective_policy': policy}],
                      'onode_hits': hits, 'onode_misses': misses},
            'hp': {'enabled': policy == 's3fifo'},
            'bluestore': {'onode_shard_hits': hits, 'onode_shard_misses': misses,
                          'buffer_hit_bytes': hits * 4096, 'buffer_miss_bytes': misses * 4096},
            'osd': {'object_ctx_cache_hit': hits, 'object_ctx_cache_total': hits + misses,
                    'op_r': hits + misses, 'op_w': 0, 'op_rw': 0},
            'mds': {'mds': {'traverse_hit': hits, 'traverse': hits + misses}},
        }
    return {'osds': osds}


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.timing = {'start_wall': 100, 'switch_request_wall': 280, 'switch_confirmed_wall': 281}

    def test_zero_is_unavailable(self):
        self.assertIsNone(percent(0, 0))
        self.assertIsNone(metrics(Counter())['strict_gt95'])

    def test_threshold_is_strict(self):
        result = metrics(Counter(onode_hits=95, onode_misses=5))
        self.assertFalse(result['strict_gt95'])
        self.assertTrue(metrics(Counter(onode_hits=97, onode_misses=3))['strict_gt96'])

    def test_weight_by_query_not_osd(self):
        a, b = sample(101), sample(103, 100, 12)
        b['osds']['2']['cache']['onode_hits'] = 910
        counts = interval(a, b)
        self.assertEqual(counts['onode_hits'], 1080)
        self.assertEqual(counts['onode_misses'], 30)
        self.assertAlmostEqual(metrics(counts)['onode_percent'], 1080 / 1110 * 100)

    def test_baseline(self):
        self.assertEqual(classify(sample(101), sample(103), self.timing, 700), ('lru', None))

    def test_after_switch(self):
        self.assertEqual(classify(sample(282, policy='s3fifo', generation=1),
                                 sample(284, policy='s3fifo', generation=1), self.timing, 700), ('s3fifo', None))

    def test_switch_window_excluded_from_both(self):
        self.assertEqual(classify(sample(279), sample(282, policy='s3fifo', generation=1),
                                 self.timing, 700), ('transition', None))

    def test_missing_osd(self):
        b = sample(103)
        del b['osds']['2']
        self.assertEqual(classify(sample(101), b, self.timing, 700)[1], 'missing-osd')

    def test_restart(self):
        b = sample(103)
        b['osds']['1']['cache']['cache_instance'] = 'new'
        self.assertEqual(classify(sample(101), b, self.timing, 700)[1], 'osd-restart')

    def test_reset(self):
        with self.assertRaises(ValueError):
            delta({'hit': 100}, {'hit': 2}, 'hit')

    def test_gap(self):
        self.assertEqual(classify(sample(101), sample(112), self.timing, 700)[1], 'collection-gap')

    def test_end_boundary(self):
        self.assertEqual(classify(sample(699), sample(701), self.timing, 700)[1], 'outside-measurement-window')

    def test_no_mixing_of_cache_layers(self):
        a, b = sample(101), sample(103, 110, 102)
        b['osds']['0']['bluestore']['buffer_hit_bytes'] = 1000000000000000
        self.assertEqual(metrics(interval(a, b))['onode_percent'], 50)

    def test_mds_reset_does_not_discard_onode(self):
        a, b = sample(101), sample(103, 110, 102)
        b['osds']['0']['mds']['mds']['traverse'] = 0
        counts = interval(a, b)
        self.assertEqual(metrics(counts)['onode_percent'], 50)
        self.assertEqual(counts['mds_invalid_windows'], 1)
        self.assertIsNone(metrics(counts)['mds_traversal_percent'])

    def test_missing_diagnostic_osd_does_not_sum_partial_cluster(self):
        a, b = sample(101), sample(103, 110, 102)
        del b['osds']['1']['bluestore']['buffer_hit_bytes']
        counts = interval(a, b)
        self.assertEqual(counts['buffer_invalid_windows'], 1)
        self.assertEqual(counts['buffer_hit_bytes'], 0)
        self.assertIsNone(metrics(counts)['buffer_byte_percent'])
        self.assertEqual(metrics(counts)['onode_percent'], 50)

    def test_complete_case_report_with_empty_measurement_window(self):
        final = sample(101)['osds']
        for item in final.values():
            item['hp'].update({key: 0 for key in ('hp_labeled_io_total', 'hp_trained_sample_count',
                               'hp_io_count', 'hp_pending_io_count', 'hp_awaiting_prediction_count',
                               'hp_train_queue_length')})
            names = ('true_positive_count', 'false_positive_count', 'true_negative_count',
                     'false_negative_count', 'eval_drop_count', 'predict_error_count',
                     'train_drop_count', 'background_error_count',
                     'status_publish_generation_begin', 'status_publish_generation_end')
            item['object_hp_status'] = {'hp_' + key: 0 for key in names}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in {
                'COMPLETE.json': {}, 'final-samples.json': final,
                'timing.json': {**self.timing, 'nominal_end_wall': 700, 'last_rd_nominal_end_wall': 700},
            }.items():
                (root / name).write_text(json.dumps(value))
            (root / 'samples.jsonl').write_text('')
            report = analyze_case(root)
            self.assertIsNone(report['hot_cold']['accuracy_percent'])
            self.assertTrue(report['hot_cold']['per_osd_accounting']['0']['labels_match_confusion'])
            self.assertEqual((root / 'intervals.csv').read_text().strip(), 'begin_wall,end_wall,stage')


class SuiteCompletionTests(unittest.TestCase):
    def check(self, requested, completed):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            if requested is not None:
                (root / 'COMPLETE.json').write_text(json.dumps({'cases': requested}))
            return suite_completion(root, completed)

    def test_selected_cases_are_not_a_full_suite(self):
        result = self.check(CASES[:3], CASES[:3])
        self.assertTrue(result['requested_cases_complete'])
        self.assertFalse(result['complete_suite'])

    def test_all_five_complete(self):
        result = self.check(CASES, CASES)
        self.assertTrue(result['requested_cases_complete'])
        self.assertTrue(result['complete_suite'])

    def test_missing_case_is_incomplete(self):
        result = self.check(CASES, CASES[:3])
        self.assertFalse(result['requested_cases_complete'])
        self.assertFalse(result['complete_suite'])

    def test_missing_marker_is_incomplete(self):
        self.assertFalse(self.check(None, CASES)['complete_suite'])

    def test_invalid_selection_is_not_complete(self):
        self.assertFalse(self.check([CASES[0], CASES[0]], [CASES[0]])['requested_cases_complete'])
        self.assertFalse(self.check(['unknown'], ['unknown'])['requested_cases_complete'])


if __name__ == '__main__':
    unittest.main()
