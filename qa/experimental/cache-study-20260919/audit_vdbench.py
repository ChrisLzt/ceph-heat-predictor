#!/usr/bin/env python3
"""Recompute realized FWD shares from integer histogram counts, not skew.html."""
import argparse
import json
from pathlib import Path
import re

BUCKET = re.compile(r'^\s*[\d.]+\s*<\s*(?:[\d.]+|max)\s+([\d,]+)\s')
MARKER = 'Total of all requested operations since warmup:'


def histogram_counts(text):
    totals = []
    for block in text.split(MARKER)[1:]:
        count = 0
        found = False
        for line in block.splitlines():
            match = BUCKET.match(line)
            if match:
                found = True
                count += int(match[1].replace(',', ''))
            elif found:
                break
        if not found:
            raise ValueError('Histogram section has no buckets')
        totals.append(count)
    return totals


def audit(model, output):
    phases = []
    summary_counts = histogram_counts((output / 'histogram.html').read_text())
    assert len(summary_counts) == len(model['profiles']['current']), summary_counts
    for phase_index, phase in enumerate(model['profiles']['current']):
        rows = []
        total_weight = sum(lane['weight'] for lane in phase['lanes'])
        for lane_index, lane in enumerate(phase['lanes']):
            name = f'p{phase_index:03d}_{lane_index:04d}'
            text = (output / f'{name}.histogram.html').read_text()
            counts = histogram_counts(text)
            if not counts:
                assert f'Performance histogram for FWD={name}' in text and 'Starting RD=' in text
                # Vdbench emits only the RD header for a zero-operation FWD.
                # The independent summary total below catches missing nonzero data.
                counts = [0]
            assert len(counts) == 1, (name, counts)
            rows.append({'fwd': name, 'bin': lane['bin'], 'operations': counts[0],
                         'requested_percent': lane['weight'] / total_weight * 100})
        total = sum(row['operations'] for row in rows)
        assert total == summary_counts[phase_index], (phase_index, total, summary_counts)
        assert total > 0
        for row in rows:
            row['actual_percent'] = row['operations'] / total * 100
            row['delta_percentage_points'] = row['actual_percent'] - row['requested_percent']
        phases.append({
            'phase': phase_index, 'operations': total,
            'histogram_sum_matches_summary': True,
            'max_absolute_delta_pp': max(abs(row['delta_percentage_points']) for row in rows),
            'total_variation_percent': sum(abs(row['delta_percentage_points']) for row in rows) / 2,
            'zero_operation_fwds': sum(row['operations'] == 0 for row in rows),
            'actual_percent_sum': sum(row['actual_percent'] for row in rows),
            'rows': rows,
        })
    skew = (output / 'skew.html').read_text()
    reported = re.findall(r'p(\d{3})_\d+\s+\d+\s+([\d.]+)%\s+([\d.]+)%', skew)
    reported_sums = {}
    for phase, requested, actual in reported:
        reported_sums[phase] = reported_sums.get(phase, 0) + float(actual)
    return {'method': 'Integer FWD histogram counts / sum of all FWD histogram counts in each RD',
            'histogram_scope': 'Vdbench non-warmup intervals, usually avg_2-end, not Onode counter scope',
            'skew_report_actual_percent_sums': reported_sums,
            'phases': phases}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('result', type=Path)
    args = parser.parse_args()
    result = audit(json.loads(args.model.read_text()), args.output)
    args.result.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({**result, 'phases': [{k: v for k, v in p.items() if k != 'rows'} for p in result['phases']]}, indent=2))


if __name__ == '__main__':
    main()
