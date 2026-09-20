#!/usr/bin/env python3
"""Show short-window failures instead of hiding them in a stage average."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cache-study-20260919'))
from analyze_cloudlab import OSDS, classify, interval, metrics


def windows(rows, timing):
    centers = [sum(row['osds'][o]['begin_wall'] for o in OSDS) / len(OSDS) for row in rows]
    result = []
    for second in range(180, 600, 10):
        start, end = timing['start_wall'] + second, timing['start_wall'] + second + 10
        lo, hi = [min(range(len(rows)), key=lambda i: abs(centers[i] - target)) for target in (start, end)]
        item = {'nominal_start_second': second, 'nominal_end_second': second + 10,
                'actual_start_second': centers[lo] - timing['start_wall'],
                'actual_end_second': centers[hi] - timing['start_wall']}
        reason = None
        counts = Counter()
        if hi <= lo or abs(centers[lo] - start) > 3 or abs(centers[hi] - end) > 3:
            reason = 'missing-endpoint-within-3s'
        else:
            for a, b in zip(rows[lo:hi], rows[lo + 1:hi + 1]):
                stage, rejected = classify(a, b, timing, timing['nominal_end_wall'])
                if rejected or stage != 's3fifo':
                    reason = rejected or 'transition-or-wrong-stage'
                    break
                try:
                    counts.update(interval(a, b))
                except (KeyError, ValueError) as error:
                    reason = str(error)
                    break
        item['valid'] = reason is None
        if reason:
            item['reason'] = reason
        else:
            item.update(metrics(counts))
        result.append(item)
    valid = [x for x in result if x['valid'] and x['onode_percent'] is not None]
    return {
        'method': 'Counters at nearest sample endpoints within 3s; actual bounds reported. No interpolation.',
        'expected_windows': 42, 'valid_windows': len(valid),
        'strict_gt96_windows': sum(x['onode_percent'] > 96 for x in valid),
        'minimum_percent': min((x['onode_percent'] for x in valid), default=None),
        'all_42_windows_proven_gt96': len(valid) == 42 and all(x['onode_percent'] > 96 for x in valid),
        'windows': result,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('case', type=Path)
    args = p.parse_args()
    assert (args.case / 'COMPLETE.json').is_file()
    timing = json.loads((args.case / 'timing.json').read_text())
    rows = [json.loads(line) for line in (args.case / 'samples.jsonl').read_text().splitlines()]
    result = windows(rows, timing)
    (args.case / 'windows10s.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'windows'}, indent=2))


if __name__ == '__main__':
    main()
