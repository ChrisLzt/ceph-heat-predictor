#!/usr/bin/env python3
"""Independently audit supplied files and recompute the frozen stage metrics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

CASES = ['bigdata_baleen_v2', 'graph_graphchi_psw_v2',
         'hpc_wrf_continuous_v2', 'ai_training_ses_v2', 'ai_inference_ses_v2']


def percent(hit, miss):
    return 100 * hit / (hit + miss) if hit + miss else None


def analyze(root):
    integrity = {'files': 0, 'bytes': 0, 'bad': []}
    for row in csv.DictReader((root / 'FILES.csv').open()):
        data = (root / row['path']).read_bytes()
        integrity['files'] += 1
        integrity['bytes'] += len(data)
        if len(data) != int(row['bytes']) or hashlib.sha256(data).hexdigest() != row['sha256']:
            integrity['bad'].append(row['path'])
    assert not integrity['bad'], integrity['bad']
    result = {'integrity': integrity, 'cases': {}, 'notes': [
        'Historical evidence only, not a new CloudLab experiment.',
        'Perf counters were collected slightly after cache status; metric scopes and timing differ.',
        'No MDS counters or per-object cache-miss causes were supplied.',
        'Shared capacity, warm history and changing phases prevent causal strategy comparison.',
    ]}
    test = root / '01-current-joint-test'
    for case in CASES:
        directory = test / case
        samples = {s['seq']: s for s in map(json.loads, (directory / 'samples.jsonl').open())}
        stages = {}
        for row in csv.DictReader((directory / 'cache-intervals.csv').open()):
            stage = row['stage']
            if stage not in ('lru', 's3fifo', 'transition'):
                continue
            a, b = samples[int(row['seq0'])], samples[int(row['seq1'])]
            acc = stages.setdefault(stage, dict(hits=0, misses=0, windows=0, seconds=0,
                extent_hits=0, extent_misses=0, buffer_hit_bytes=0, buffer_miss_bytes=0))
            hits = misses = 0
            for osd in a['osds']:
                old, new = a['osds'][osd], b['osds'][osd]
                assert old['cache']['cache_instance'] == new['cache']['cache_instance']
                dh = new['cache']['onode_hits'] - old['cache']['onode_hits']
                dm = new['cache']['onode_misses'] - old['cache']['onode_misses']
                assert dh >= 0 and dm >= 0
                hits += dh
                misses += dm
                if stage != 'transition':
                    assert old['cache']['policy_generation'] == new['cache']['policy_generation']
                    assert new['cache']['effective_policy'] == stage
                for key, source in [('extent_hits', 'onode_shard_hits'),
                                    ('extent_misses', 'onode_shard_misses'),
                                    ('buffer_hit_bytes', 'buffer_hit_bytes'),
                                    ('buffer_miss_bytes', 'buffer_miss_bytes')]:
                    delta = new['bluestore'][source] - old['bluestore'][source]
                    assert delta >= 0
                    acc[key] += delta
            assert hits == int(row['hits']) and misses == int(row['misses'])
            acc['hits'] += hits
            acc['misses'] += misses
            acc['windows'] += 1
            acc['seconds'] += float(row['mean_osd_interval_seconds'])
        for acc in stages.values():
            acc['onode_percent'] = percent(acc['hits'], acc['misses'])
            acc['extent_percent'] = percent(acc['extent_hits'], acc['extent_misses'])
            acc['buffer_byte_percent'] = percent(acc['buffer_hit_bytes'], acc['buffer_miss_bytes'])
        manifest = json.loads((test / 'configs' / case / 'rendered/manifest.json').read_text())
        cfg = json.loads((directory / 'cache-config.json').read_text())
        result['cases'][case] = {'stages': stages, 'bytes': manifest['capacity_bytes'],
            'files': manifest['file_count'], 'osd_config': cfg,
            'onode_gt95': stages['s3fifo']['onode_percent'] > 95,
            'onode_gt96': stages['s3fifo']['onode_percent'] > 96}
    result['dataset_bytes'] = sum(c['bytes'] for c in result['cases'].values())
    result['dataset_files'] = sum(c['files'] for c in result['cases'].values())
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('materials', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    data = analyze(args.materials)
    args.output.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({'integrity': data['integrity'], 'dataset_bytes': data['dataset_bytes'],
        'dataset_files': data['dataset_files'],
        's3fifo': {k: v['stages']['s3fifo'] for k, v in data['cases'].items()}}, indent=2))
