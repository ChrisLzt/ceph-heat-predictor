import unittest
from analyze_windows import windows
from test_analyze_cloudlab import sample


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.timing = {'start_wall': 0, 'switch_request_wall': 180,
                       'switch_confirmed_wall': 181, 'nominal_end_wall': 600}
        self.rows = [sample(t, hits=t * 97, misses=t * 3, policy='s3fifo', generation=1)
                     for t in range(180, 602, 2)]

    def test_transition_and_end_are_not_silently_counted_as_pass(self):
        result = windows(self.rows, self.timing)
        self.assertEqual(result['valid_windows'], 40)
        self.assertEqual(result['strict_gt96_windows'], 40)
        self.assertFalse(result['all_42_windows_proven_gt96'])

    def test_equal96_is_not_strictly_above96(self):
        rows = [sample(t, hits=t * 96, misses=t * 4, policy='s3fifo', generation=1)
                for t in range(180, 602, 2)]
        result = windows(rows, self.timing)
        self.assertEqual(result['minimum_percent'], 96)
        self.assertEqual(result['strict_gt96_windows'], 0)

    def test_gap_is_not_interpolated(self):
        result = windows([r for r in self.rows if not 300 <= r['osds']['0']['begin_wall'] <= 330], self.timing)
        self.assertLess(result['valid_windows'], 40)
        self.assertFalse(result['all_42_windows_proven_gt96'])


if __name__ == '__main__':
    unittest.main()
