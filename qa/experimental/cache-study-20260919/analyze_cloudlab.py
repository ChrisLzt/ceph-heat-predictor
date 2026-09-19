#!/usr/bin/env python3
"""Auditable disjoint cache metrics from complete, stable per-OSD intervals."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

OSDS = ('0', '1', '2')


def percent(hits, total):
    return 100.0 * hits / total if total else None


def delta(before, after, key):
    value = after[key] - before[key]
    if value < 0:
        raise ValueError('counter-reset:' + key)
    return value


def classify(before, after, timing, end_wall):
    if set(before['osds']) != set(OSDS) or set(after['osds']) != set(OSDS):
        return 'invalid', 'missing-osd'
    policies = set()
    for osd in OSDS:
        a, b = before['osds'][osd], after['osds'][osd]
        if a['begin_wall'] < timing['start_wall'] or b['end_wall'] > end_wall:
            return 'excluded', 'outside-measurement-window'
        if not 0 < b['begin_mono'] - a['begin_mono'] <= 10:
            return 'invalid', 'collection-gap'
        if a['cache']['cache_instance'] != b['cache']['cache_instance']:
            return 'invalid', 'osd-restart'
        if a['cache']['policy_generation'] != b['cache']['policy_generation']:
            policies.add('transition')
        for sample in (a, b):
            p = sample['cache']['effective_policy']
            if any(s['effective_policy'] != p for s in sample['cache']['shards']):
                policies.add('transition')
            if bool(sample['hp']['enabled']) != (p == 's3fifo'):
                policies.add('transition')
            policies.add(p)
    if policies == {'lru'} and max(after['osds'][o]['end_wall'] for o in OSDS) <= timing['switch_request_wall']:
        return 'lru', None
    if policies == {'s3fifo'} and min(before['osds'][o]['begin_wall'] for o in OSDS) >= timing['switch_confirmed_wall']:
        return 's3fifo', None
    return 'transition', None


def interval(before, after):
    totals = Counter()
    for osd in OSDS:
        a, b = before['osds'][osd], after['osds'][osd]
        for field in ('onode_hits', 'onode_misses'):
            totals[field] += delta(a['cache'], b['cache'], field)
    # Optional diagnostics must not veto an otherwise valid Onode interval.
    groups = {
        'extent': ('bluestore', ('onode_shard_hits', 'onode_shard_misses')),
        'buffer': ('bluestore', ('buffer_hit_bytes', 'buffer_miss_bytes')),
        'object_context': ('osd', ('object_ctx_cache_hit', 'object_ctx_cache_total')),
        'osd_ops': ('osd', ('op_r', 'op_w', 'op_rw')),
    }
    for name, (section, fields) in groups.items():
        values = Counter()
        try:
            for osd in OSDS:
                a, b = before['osds'][osd][section], after['osds'][osd][section]
                for field in fields:
                    values[field] += delta(a, b, field)
            if name == 'object_context' and values['object_ctx_cache_hit'] > values['object_ctx_cache_total']:
                raise ValueError('hits-exceed-total')
        except (ValueError, KeyError):
            totals[name + '_invalid_windows'] = 1
        else:
            totals.update(values)
            totals[name + '_valid_windows'] = 1
    try:
        a, b = before['osds']['0']['mds']['mds'], after['osds']['0']['mds']['mds']
        traverse, hits = (delta(a, b, field) for field in ('traverse', 'traverse_hit'))
        if hits > traverse:
            raise ValueError('hits-exceed-total')
    except (ValueError, KeyError):
        totals['mds_invalid_windows'] = 1
    else:
        totals.update(mds_traverse=traverse, mds_traverse_hit=hits, mds_valid_windows=1)
    totals['seconds'] = sum(after['osds'][o]['begin_mono'] - before['osds'][o]['begin_mono'] for o in OSDS) / len(OSDS)
    totals['windows'] = 1
    return totals


def metrics(counts):
    result = dict(counts)
    result.update(
        onode_percent=percent(counts['onode_hits'], counts['onode_hits'] + counts['onode_misses']),
        extent_query_percent=percent(counts['onode_shard_hits'], counts['onode_shard_hits'] + counts['onode_shard_misses']),
        buffer_byte_percent=percent(counts['buffer_hit_bytes'], counts['buffer_hit_bytes'] + counts['buffer_miss_bytes']),
        mds_traversal_percent=percent(counts['mds_traverse_hit'], counts['mds_traverse']),
        object_context_query_percent=percent(counts['object_ctx_cache_hit'], counts['object_ctx_cache_total']),
    )
    result['strict_gt95'] = None if result['onode_percent'] is None else result['onode_percent'] > 95
    result['strict_gt96'] = None if result['onode_percent'] is None else result['onode_percent'] > 96
    return result


def analyze_case(root):
    assert (root / 'COMPLETE.json').is_file(), 'Incomplete measurement: ' + str(root)
    timing = json.loads((root / 'timing.json').read_text())
    rows = [json.loads(line) for line in (root / 'samples.jsonl').read_text().splitlines()]
    report = {'timing': timing}
    csv_rows = []
    for window, end in [('nominal600', timing['nominal_end_wall']),
                        ('full_rd_schedule', timing['last_rd_nominal_end_wall'])]:
        stages = {name: Counter() for name in ('lru', 'transition', 's3fifo')}
        rejected = Counter()
        for a, b in zip(rows, rows[1:]):
            stage, reason = classify(a, b, timing, end)
            if reason:
                rejected[reason] += 1
                continue
            try:
                counts = interval(a, b)
            except (ValueError, KeyError, AssertionError) as error:
                rejected[str(error) or type(error).__name__] += 1
                continue
            stages[stage].update(counts)
            if window == 'nominal600':
                csv_rows.append({'begin_wall': a['osds']['0']['begin_wall'],
                                 'end_wall': b['osds']['0']['end_wall'], 'stage': stage, **metrics(counts)})
        report[window] = {'stages': {k: metrics(v) for k, v in stages.items()},
                          'rejected_intervals': dict(rejected)}
    final = json.loads((root / 'final-samples.json').read_text())
    cm = Counter()
    for sample in final.values():
        for name in ('hp_true_positive_count', 'hp_false_positive_count', 'hp_true_negative_count',
                     'hp_false_negative_count', 'hp_eval_drop_count', 'hp_predict_error_count',
                     'hp_train_drop_count', 'hp_background_error_count'):
            cm[name] += sample['object_hp_status'][name]
    tp, fp, tn, fn = [cm['hp_' + n + '_count'] for n in ('true_positive', 'false_positive', 'true_negative', 'false_negative')]
    accounting = {}
    for osd, sample in final.items():
        hp, status = sample['hp'], sample['object_hp_status']
        confusion_total = sum(status['hp_' + name + '_count'] for name in
                              ('true_positive', 'false_positive', 'true_negative', 'false_negative'))
        accounting[osd] = {
            'labeled': hp['hp_labeled_io_total'], 'confusion_total': confusion_total,
            'trained': hp['hp_trained_sample_count'],
            'io_count': hp['hp_io_count'], 'pending': hp['hp_pending_io_count'],
            'awaiting_prediction': hp['hp_awaiting_prediction_count'],
            'training_queue': hp['hp_train_queue_length'],
            'status_generation_coherent': status['hp_status_publish_generation_begin'] == status['hp_status_publish_generation_end'],
            'labels_match_confusion': hp['hp_labeled_io_total'] == confusion_total,
        }
    report['hot_cold'] = {**dict(cm), 'per_osd_accounting': accounting,
                           'accuracy_percent': percent(tp + tn, tp + fp + tn + fn),
                           'precision_percent': percent(tp, tp + fp), 'recall_percent': percent(tp, tp + fn),
                           'specificity_percent': percent(tn, tn + fp)}
    report['sampled_memory'] = {}
    for osd in OSDS:
        inside = [r['osds'][osd] for r in rows
                  if timing['start_wall'] <= r['osds'][osd]['begin_wall'] <= timing['last_rd_nominal_end_wall']]
        values = {
            'resident_onodes': [sum(shard['resident_onodes'] for shard in s['cache']['shards']) for s in inside],
            'tracked_mempool_bytes': [s['mempools']['mempool']['total']['bytes'] for s in inside],
            'onode_struct_bytes': [s['mempools']['mempool']['by_pool']['bluestore_cache_onode']['bytes'] for s in inside],
        }
        report['sampled_memory'][osd] = {k: {'min': min(v), 'max': max(v)} for k, v in values.items() if v}
    with (root / 'intervals.csv').open('w') as stream:
        fields = list(dict.fromkeys(key for row in csv_rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields or ['begin_wall', 'end_wall', 'stage'])
        writer.writeheader()
        writer.writerows(csv_rows)
    (root / 'analysis.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    args = ap.parse_args()
    cases = {p.name: analyze_case(p) for p in sorted(args.run.iterdir()) if p.is_dir() and (p / 'COMPLETE.json').exists()}
    pooled = Counter()
    for case in cases.values():
        s = case['nominal600']['stages']['s3fifo']
        for key in ('onode_hits', 'onode_misses'):
            pooled[key] += s.get(key, 0)
    result = {'complete_suite': (args.run / 'COMPLETE.json').exists(), 'cases': cases,
              'pooled_onode_percent': percent(pooled['onode_hits'], pooled['onode_hits'] + pooled['onode_misses'])}
    (args.run / 'analysis.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v['nominal600']['stages']['s3fifo'] for k, v in cases.items()}, indent=2))


if __name__ == '__main__':
    main()
