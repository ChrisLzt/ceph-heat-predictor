#!/usr/bin/env python3
"""Prepare disjoint remaining datasets concurrently; measurement remains serial."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import time
from workload_common.single_v2.core import validate_bundle, json_text
from workload_common.single_v2.lifecycle import preflight, verify_inventory, check_ready, INTENT, READY

BASE = Path('/mnt/ceph-lab/cache-study-20260919')
VDB = BASE / 'scoutfs-vdbench-d904d81d565c5882d0aabf2434eef46851954767/vdbench'
CASES = ['bigdata_baleen_v2', 'graph_graphchi_psw_v2', 'hpc_wrf_continuous_v2',
         'ai_training_ses_v2', 'ai_inference_ses_v2']


def prepare(item):
    port, case = item
    bundle = BASE / 'adapted-configs' / case
    manifest = validate_bundle(bundle)
    root = Path(manifest['data_root'])
    if (root / READY).exists():
        return check_ready(bundle)
    report = preflight(bundle)
    dest = BASE / 'preparation' / case
    dest.mkdir(exist_ok=False)
    root.mkdir(parents=True, exist_ok=True)
    marker = {'id': case, 'layout_sha256': manifest['layout_sha256'],
              'capacity_bytes': manifest['capacity_bytes'], 'state': 'preparing',
              'started_epoch': time.time(), 'generation_parallelism': 'four independent other cases'}
    with (root / INTENT).open('x') as stream:
        stream.write(json_text(marker))
    print(json_text({'begin': case, 'preflight': report}), flush=True)
    argv = [str(VDB), f'-p{port}', '-f', str(bundle / 'prepare_data.vdb'), '-o', str(dest / 'vdbench')]
    (dest / 'command.json').write_text(json_text({'argv': argv}))
    with (dest / 'console.log').open('w') as log:
        subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
    for marker_path in root.rglob('no_dismount.txt'):
        assert marker_path.read_text().strip() == 'This file was created to keep anchor busy and prevent auto-dismount'
        target = dest / 'archived-markers' / marker_path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(marker_path), str(target))
    inventory = verify_inventory(manifest)
    marker.update(state='ready', inventory=inventory, completed_epoch=time.time())
    (root / READY).write_text(json_text(marker))
    (dest / 'verified.json').write_text(json_text(marker))
    print(json_text(marker), flush=True)
    return marker


if __name__ == '__main__':
    assert not (BASE / 'preparation/ALL_DATA_READY.json').exists()
    # The first case is already owned by the original preparation process.
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(prepare, enumerate(CASES[1:], 5581)))
    first = Path('/mnt/ceph-lab/cephfs/cache-study-20260919') / CASES[0] / READY
    while not first.exists():
        time.sleep(30)
    verified = {case: check_ready(BASE / 'adapted-configs' / case) for case in CASES}
    (BASE / 'preparation/ALL_DATA_READY.json').write_text(json_text({
        'completed_epoch': time.time(), 'cases': CASES, 'verification': verified,
        'note': 'Data preparation overlapped across cases; formal measurement is serial.'}))
    print('ALL FIVE PERSISTENT DATASETS VERIFIED', flush=True)
