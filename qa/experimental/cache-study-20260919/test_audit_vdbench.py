import unittest
from audit_vdbench import histogram_counts


class HistogramTests(unittest.TestCase):
    def test_integer_counts_not_rounded_percentages(self):
        text = '''Total of all requested operations since warmup:
min < max count pct

0.000 < 1.000 1,002 99.99 99.99
1.000 < max 1 0.01 100.00

other unrelated numeric rows
0.000 < max 900 100 100
'''
        self.assertEqual(histogram_counts(text), [1003])

    def test_missing_buckets_rejected(self):
        with self.assertRaises(ValueError):
            histogram_counts('Total of all requested operations since warmup: invalid')

    def test_zero_operation_histogram(self):
        self.assertEqual(histogram_counts('Total of all requested operations since warmup:\n0.000 < max 0 0 0'), [0])


if __name__ == '__main__':
    unittest.main()
