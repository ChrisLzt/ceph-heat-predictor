#!/usr/bin/env python3
"""Read complete case evidence; never pool away a failing workload."""
import argparse
import csv
import json
from pathlib import Path

PREFETCH_COUNTERS = ('loaded', 'used', 'unused_removed', 'scanned', 'db_reads',
                     'encoded_bytes', 'errors', 'oversized', 'queue_full', 'pressure_pauses')
HP_ERRORS = ('hp_eval_drop_count', 'hp_predict_error_count',
             'hp_train_drop_count', 'hp_background_error_count')


def read(path):
    return json.loads(path.read_text())


def summarize_case(root, phase):
    assert (root / 'COMPLETE.json').exists(), root
    report = read(root / 'analysis.json')
    windows = read(root / 'windows10s.json')
    rows = [json.loads(line) for line in (root / 'samples.jsonl').read_text().splitlines()]
    start, end = rows[0]['osds'], rows[-1]['osds']
    quality = []
    for osd in ('0', '1', '2'):
        instances = {r['osds'][osd]['cache']['cache_instance'] for r in rows}
        if len(instances) != 1:
            quality.append('osd-restart-' + osd)
        for row in rows:
            cache = row['osds'][osd]['cache']
            prefetch = cache['prefetch']
            if prefetch['configured'] != (phase == 'on'):
                quality.append('wrong-prefetch-configuration-' + osd)
                break
            if cache['effective_policy'] == 'lru' and prefetch['active']:
                quality.append('prefetch-active-during-lru-' + osd)
                break
    pf = {key: sum(end[osd]['cache']['prefetch'][key] - start[osd]['cache']['prefetch'][key]
                   for osd in ('0', '1', '2')) for key in PREFETCH_COUNTERS}
    restored = read(root.parent / 'restored-baseline.json')
    pf_restored = {key: sum(restored[osd]['cache']['prefetch'][key] -
                           start[osd]['cache']['prefetch'][key]
                           for osd in ('0', '1', '2')) for key in PREFETCH_COUNTERS}
    if any(value < 0 for value in pf.values()):
        quality.append('prefetch-counter-reset')
    if phase == 'off' and any(pf.values()):
        quality.append('unexpected-prefetch-work-while-off')
    hp = report['hot_cold']
    for key in HP_ERRORS:
        if hp[key]:
            quality.append(key)
    for osd, values in hp['per_osd_accounting'].items():
        if (not values['status_generation_coherent'] or not values['labels_match_confusion'] or
                values['labeled'] != values['trained'] or
                any(values[key] for key in ('pending', 'awaiting_prediction', 'training_queue'))):
            quality.append('hp-accounting-' + osd)
    for reason, count in report['nominal600']['rejected_intervals'].items():
        if count and reason != 'outside-measurement-window':
            quality.append('rejected-intervals:' + reason)
    tail = []
    for window in reversed(windows['windows']):
        if not window.get('valid') or not window.get('strict_gt96'):
            break
        tail.append(window)
    tail.reverse()
    convergence = None
    if tail:
        actual_switch = report['timing']['switch_confirmed_wall'] - report['timing']['start_wall']
        convergence = {'nominal_start_second': tail[0]['nominal_start_second'],
                       'actual_start_second': tail[0]['actual_start_second'],
                       'seconds_after_switch': tail[0]['actual_start_second'] - actual_switch,
                       'consecutive_windows': len(tail),
                       'minimum_percent': min(w['onode_percent'] for w in tail)}
    stages = report['nominal600']['stages']
    result = {
        'path': str(root), 'phase': phase,
        'stages': {name: {'hit_percent': value.get('onode_percent'),
                         'hits': value.get('onode_hits', 0), 'misses': value.get('onode_misses', 0),
                         'seconds': value.get('seconds', 0),
                         'osd_read_iops': value.get('op_r', 0) / value['seconds'] if value.get('seconds') else None}
                   for name, value in stages.items()},
        'windows': {k: v for k, v in windows.items() if k != 'windows'},
        'invalid_windows': [w for w in windows['windows'] if not w['valid']],
        'trailing_gt96': convergence, 'hot_cold': hp,
        'prefetch_full_sample_span_delta': pf,
        'prefetch_until_lru_restore_delta': pf_restored,
        'prefetch_used_percent': 100 * pf['used'] / pf['loaded'] if pf['loaded'] else None,
        'memory': report['sampled_memory'],
        'switch_seconds': report['timing']['switch_confirmed_wall'] - report['timing']['start_wall'],
        'quality_flags': sorted(set(quality)),
        'inventory_unchanged': read(root / 'workload-complete.json')['inventory_unchanged'],
    }
    assert result['inventory_unchanged']
    return result


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    plan = read(args.root / 'plan.json')
    result = {'planned': len(plan), 'completed': 0, 'cases': {}}
    caches, heat, costs = [], [], []
    for item in plan:
        case, phase = item['case'], item['phase']
        root = args.root / 'runs' / item['run_id'] / case
        if args.partial and not (root / 'windows10s.json').exists():
            continue
        value = summarize_case(root, phase)
        result['cases'].setdefault(case, {})[phase] = value
        result['completed'] += 1
        stages, windows = value['stages'], value['windows']
        caches.append({'case': case, 'prefetch': phase,
                       'baseline_percent': stages['lru']['hit_percent'],
                       's3fifo_percent': stages['s3fifo']['hit_percent'],
                       'hits': stages['s3fifo']['hits'], 'misses': stages['s3fifo']['misses'],
                       'valid_seconds': stages['s3fifo']['seconds'],
                       'valid_10s_windows': windows['valid_windows'],
                       'gt96_10s_windows': windows['strict_gt96_windows'],
                       'minimum_10s_percent': windows['minimum_percent'],
                       'osd_read_iops': stages['s3fifo']['osd_read_iops'],
                       'quality_flags': ';'.join(value['quality_flags'])})
        heat.append({'case': case, 'prefetch': phase,
                     **{k: v for k, v in value['hot_cold'].items() if k != 'per_osd_accounting'}})
        cost = value['prefetch_until_lru_restore_delta']
        costs.append({'case': case, 'prefetch': phase,
                      'scope': 'first sample through restore-to-LRU acknowledgement', **cost,
                      'first_use_percent': 100 * cost['used'] / cost['loaded'] if cost['loaded'] else None})
    (args.root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    if caches:
        write_csv(args.root / 'cache-hit-rates.csv', caches)
        write_csv(args.root / 'hot-cold-accuracy.csv', heat)
        write_csv(args.root / 'prefetch-cost.csv', costs)
    print(json.dumps({'planned': result['planned'], 'completed': result['completed'],
                      'cases': {case: {phase: {'hit_percent': v['stages']['s3fifo']['hit_percent'],
                                  'hp_accuracy_percent': v['hot_cold']['accuracy_percent'],
                                  'minimum_10s_percent': v['windows']['minimum_percent'],
                                  'quality_flags': v['quality_flags']}
                               for phase, v in variants.items()}
                          for case, variants in result['cases'].items()}}, indent=2))


if __name__ == '__main__':
    main()
