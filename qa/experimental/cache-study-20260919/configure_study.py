#!/usr/bin/env python3
"""Freeze common cache budgets before workload preparation or measurement."""
import json
from pathlib import Path
import subprocess
import time

OUT = Path('/lab/integrated-20260919/results')


def ceph(*args):
    p = subprocess.run(['ceph', *args, '-f', 'json'], text=True,
                       capture_output=True, check=True, timeout=40)
    return json.loads(p.stdout) if p.stdout.strip() else None


assert ceph('osd', 'ls') == [0, 1, 2]
settings = {
    'bluestore_cache_autotune': 'false',
    'bluestore_cache_size': '8589934592',
    'bluestore_cache_meta_ratio': '0.70',
    'bluestore_cache_kv_ratio': '0.20',
    'osd_memory_target': '17179869184',
}
before = {str(i): {k: ceph('tell', f'osd.{i}', 'config', 'get', k)
                   for k in settings} for i in range(3)}
(OUT / 'cache-budget-before.json').write_text(json.dumps(before, indent=2))
for key, value in settings.items():
    ceph('config', 'set', 'osd', key, value)
    for i in range(3):
        result = ceph('tell', f'osd.{i}', 'config', 'set', key, value)
        assert 'error' not in json.dumps(result).lower(), result
ceph('osd', 'pool', 'set', 'labfs_data', 'pg_num', '128')
ceph('osd', 'pool', 'set', 'labfs_data', 'pgp_num', '128')
for _ in range(240):
    status = ceph('status')
    pgs = status.get('pgmap', {}).get('pgs_by_state', [])
    if pgs and all(p['state_name'] == 'active+clean' for p in pgs):
        break
    time.sleep(2)
else:
    raise RuntimeError('Cluster did not settle; no workload may start')
after = {str(i): {k: ceph('tell', f'osd.{i}', 'config', 'get', k)
                  for k in settings} for i in range(3)}
(OUT / 'cache-budget-frozen.json').write_text(json.dumps({
    'settings': settings, 'effective': after, 'status': status,
    'versions': ceph('versions'), 'osd_df': ceph('osd', 'df'),
    'note': '8 GiB total BlueStore cache, not 8 GiB exclusive Onode cache; metadata 70%, KV 20%.'
}, indent=2))
print(json.dumps({'ready': True, 'effective': after}))
