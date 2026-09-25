#!/usr/bin/env python3
"""Plot complete raw-counter analyses, using query-weighted 30-second bins."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CASES = [
    ('bigdata_baleen_v2', 'Baleen'),
    ('graph_graphchi_psw_v2', 'GraphChi'),
    ('hpc_wrf_continuous_v2', 'WRF'),
    ('ai_training_ses_v2', 'AI training'),
    ('ai_inference_ses_v2', 'AI inference'),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    suite = json.loads((args.run / 'analysis.json').read_text())
    assert suite['complete_suite'] and len(suite['cases']) == 5
    fig, axes = plt.subplots(5, 2, figsize=(12, 15), constrained_layout=True)
    for index, (case, label) in enumerate(CASES):
        report = suite['cases'][case]
        start = report['timing']['start_wall']
        bins = defaultdict(lambda: defaultdict(float))
        gaps = []
        previous_end = None
        with (args.run / case / 'intervals.csv').open() as stream:
            for row in csv.DictReader(stream):
                begin, end = float(row['begin_wall']), float(row['end_wall'])
                if previous_end is not None and begin - previous_end > 10:
                    gaps.append((previous_end - start, begin - start))
                previous_end = end
                midpoint = (float(row['begin_wall']) + float(row['end_wall'])) / 2 - start
                bucket = min(19, int(midpoint // 30))
                for key in ('onode_hits', 'onode_misses', 'op_r', 'op_w', 'op_rw', 'seconds'):
                    bins[bucket][key] += float(row.get(key) or 0)
        times = [(key + .5) * 30 for key in sorted(bins)]
        counts = [bins[key] for key in sorted(bins)]
        ratios = [100 * row['onode_hits'] / (row['onode_hits'] + row['onode_misses'])
                  if row['onode_hits'] + row['onode_misses'] else float('nan') for row in counts]
        left, right = axes[index]
        left.plot(times, ratios, color='#146e57', marker='.', linewidth=1.7)
        left.axhline(95, color='#bf3939', linestyle='--', linewidth=1, label='95% threshold')
        left.axhline(96, color='#947100', linestyle=':', linewidth=1, label='96% target')
        left.set_ylabel(f'{label}\nOnode lookup hits (%)')
        left.set_ylim(max(0, min([95] + [v for v in ratios if v == v]) - 2), 100.5)
        s3 = report['nominal600']['stages']['s3fifo']
        left.set_title(f'S3FIFO phase: {s3["onode_percent"]:.6f}%', fontsize=10)
        right.plot(times, [row['op_r'] / row['seconds'] for row in counts], color='#225e9b', label='OSD read ops/s')
        right.plot(times, [(row['op_w'] + row['op_rw']) / row['seconds'] for row in counts],
                   color='#b15359', label='OSD write/mixed ops/s')
        right.set_ylabel('OSD operations / second')
        right.set_ylim(bottom=0)
        for axis in (left, right):
            axis.axvspan(0, 180, color='#c9c9c9', alpha=.20)
            axis.axvline(180, color='#555555', linewidth=1)
            for begin, end in gaps:
                axis.axvspan(begin, end, facecolor='none', edgecolor='#a66c12', hatch='////', linewidth=0)
            axis.set_xlim(0, 600)
            axis.grid(alpha=.20)
            axis.set_xlabel('Seconds after first RD starts')
        if index == 0:
            left.legend(fontsize=8, loc='lower right')
            right.legend(fontsize=8)
    fig.suptitle('CloudLab: complete fixed-size workloads, persistent post-preparation state\n'
                 '30-second query-weighted bins; shaded 0-180s LRU/HP off; hatching marks excluded telemetry gaps', fontsize=12)
    fig.savefig(args.run / 'cloudlab-timeseries.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
