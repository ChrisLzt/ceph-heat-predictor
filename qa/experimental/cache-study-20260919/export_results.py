#!/usr/bin/env python3
"""Export separate result families without combining unrelated cache layers."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

CASES = ['bigdata_baleen_v2', 'graph_graphchi_psw_v2', 'hpc_wrf_continuous_v2',
         'ai_training_ses_v2', 'ai_inference_ses_v2']


def table(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def short_window_diagnostics(rows, start):
    windows, bins = [], defaultdict(lambda: [0, 0])
    for row in rows:
        if row['stage'] != 's3fifo':
            continue
        hits, misses = int(row['onode_hits']), int(row['onode_misses'])
        if not hits + misses:
            continue
        windows.append(100 * hits / (hits + misses))
        midpoint = (float(row['begin_wall']) + float(row['end_wall'])) / 2 - start
        bucket = min(19, int(midpoint // 30))
        bins[bucket][0] += hits
        bins[bucket][1] += misses
    return {'valid_nonzero_s3fifo_intervals': len(windows),
            'minimum_valid_interval_percent': min(windows) if windows else None,
            'minimum_30s_bin_percent': min(100 * h / (h + m) for h, m in bins.values()) if bins else None,
            'observed_30s_bins': len(bins),
            'scope': 'Secondary diagnostic; excluded telemetry gaps remain unobserved; bins use interval midpoints'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    args = ap.parse_args()
    suite = json.loads((args.run / 'analysis.json').read_text())
    assert suite['complete_suite'] and set(suite['cases']) == set(CASES)
    onode, diagnostics, hot_cold, quality, distribution, realtime = [], [], [], [], [], []
    for case in CASES:
        report = suite['cases'][case]
        stages = report['nominal600']['stages']
        s3, lru = stages['s3fifo'], stages['lru']
        t = report['timing']
        with (args.run / case / 'intervals.csv').open() as stream:
            realtime.append({'case': case, **short_window_diagnostics(csv.DictReader(stream), t['start_wall'])})
        onode.append({'case': case, 'lru_percent': lru['onode_percent'],
                      's3fifo_percent': s3['onode_percent'], 'hits': s3['onode_hits'],
                      'misses': s3['onode_misses'], 'valid_seconds': s3['seconds'],
                      'strict_gt95': s3['strict_gt95'], 'strict_gt96': s3['strict_gt96'],
                      'full_rd_s3fifo_percent': report['full_rd_schedule']['stages']['s3fifo']['onode_percent']})
        diagnostics.append({'case': case, **{key: s3.get(key) for key in (
            'extent_query_percent', 'buffer_byte_percent', 'mds_traversal_percent',
            'object_context_query_percent', 'onode_shard_hits', 'onode_shard_misses',
            'buffer_hit_bytes', 'buffer_miss_bytes', 'mds_traverse', 'mds_traverse_hit',
            'object_ctx_cache_hit', 'object_ctx_cache_total')}})
        hp = report['hot_cold']
        hot_cold.append({'case': case, **{key: value for key, value in hp.items() if not isinstance(value, dict)},
                         'all_osd_labels_match_confusion': all(x['labels_match_confusion'] for x in hp['per_osd_accounting'].values()),
                         'all_osd_status_generations_coherent': all(x['status_generation_coherent'] for x in hp['per_osd_accounting'].values())})
        quality.append({'case': case, 'switch_request_seconds': t['switch_request_wall'] - t['start_wall'],
                        'switch_confirm_seconds': t['switch_confirmed_wall'] - t['start_wall'],
                        'full_rd_duration_seconds': t['last_rd_nominal_end_wall'] - t['start_wall'],
                        'lru_valid_seconds': lru['seconds'], 'transition_seconds': stages['transition']['seconds'],
                        's3fifo_valid_seconds': s3['seconds'],
                        's3fifo_coverage_of_nominal420_percent': s3['seconds'] / 420 * 100,
                        'rejected_intervals': json.dumps(report['nominal600']['rejected_intervals'], sort_keys=True)})
        audit_path = args.run / case / 'workload-skew-audit.json'
        audit = json.loads(audit_path.read_text())
        for phase in audit['phases']:
            distribution.append({'case': case, **{k: v for k, v in phase.items() if k != 'rows'},
                                 'vdbench_reported_percent_sum': audit['skew_report_actual_percent_sums'].get(f'{phase["phase"]:03d}')})
    for name, rows in [('onode-results.csv', onode), ('cache-diagnostics.csv', diagnostics),
                       ('hot-cold-results.csv', hot_cold), ('sampling-quality.csv', quality),
                       ('workload-distribution.csv', distribution), ('realtime-diagnostics.csv', realtime)]:
        table(args.run / name, rows)
    hits = sum(row['hits'] for row in onode)
    misses = sum(row['misses'] for row in onode)
    summary = {'run_id': args.run.name, 'complete_cases': len(onode),
               'all_cases_strict_gt95': all(row['strict_gt95'] for row in onode),
               'all_cases_strict_gt96': all(row['strict_gt96'] for row in onode),
               'pooled_hits': hits, 'pooled_misses': misses,
               'pooled_onode_percent': hits / (hits + misses) * 100,
               'equal_case_mean_percent': sum(row['s3fifo_percent'] for row in onode) / len(onode),
               'certification_claim': False,
               'scope': 'Onode lookup ratio on the disclosed CloudLab FUSE path and fixed cache budget'}
    (args.run / 'result-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
