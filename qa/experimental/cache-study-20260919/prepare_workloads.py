#!/usr/bin/env python3
"""Render path-only adaptations and persist all five complete datasets."""
import json
from pathlib import Path
import shutil
import subprocess
import time
from workload_common.single_v2.core import render, validate_bundle, json_text
from workload_common.single_v2.lifecycle import (
    preflight, verify_inventory, check_ready, INTENT, READY,
)

BASE = Path('/mnt/ceph-lab/cache-study-20260919')
VDB = BASE / 'scoutfs-vdbench-d904d81d565c5882d0aabf2434eef46851954767/vdbench'
CASES = ['bigdata_baleen_v2', 'graph_graphchi_psw_v2', 'hpc_wrf_continuous_v2',
         'ai_training_ses_v2', 'ai_inference_ses_v2']


def archive_markers(root, destination):
    for marker in root.rglob('no_dismount.txt'):
        assert marker.read_text().strip() == 'This file was created to keep anchor busy and prevent auto-dismount'
        target = destination / marker.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists()
        shutil.move(str(marker), str(target))


manifests = {}
for case in CASES:
    original = BASE / 'configs' / case / 'rendered'
    before = validate_bundle(original)
    model = json.loads((original / 'model.json').read_text())
    root = Path('/mnt/ceph-lab/cephfs/cache-study-20260919') / case
    output = BASE / 'adapted-configs' / case
    after = render(model, data_root=root, output=output, rate=before['rate'])
    for name in before['config_sha256']:
        assert (output / name).read_text() == (original / name).read_text().replace(before['data_root'], str(root)), name
    assert before['model_sha256'] == after['model_sha256']
    manifests[case] = after
results = BASE / 'preparation'
results.mkdir(exist_ok=True)
(results / 'path-only-adaptation.json').write_text(json_text(manifests))
assert sum(m['capacity_bytes'] for m in manifests.values()) == 706704703488
for case, manifest in manifests.items():
    root = Path(manifest['data_root'])
    bundle = BASE / 'adapted-configs' / case
    if (root / READY).exists():
        print(json_text(check_ready(bundle)), flush=True)
        continue
    report = preflight(bundle)
    dest = results / case
    dest.mkdir(exist_ok=False)
    root.mkdir(parents=True, exist_ok=True)
    marker = {'id': case, 'layout_sha256': manifest['layout_sha256'],
              'capacity_bytes': manifest['capacity_bytes'], 'state': 'preparing', 'started_epoch': time.time()}
    with (root / INTENT).open('x') as stream:
        stream.write(json_text(marker))
    print(json_text({'begin': case, 'preflight': report}), flush=True)
    with (dest / 'console.log').open('w') as log:
        subprocess.run([str(VDB), '-f', str(bundle / 'prepare_data.vdb'), '-o', str(dest / 'vdbench')],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    archive_markers(root, dest / 'archived-markers')
    inventory = verify_inventory(manifest)
    marker.update(state='ready', inventory=inventory, completed_epoch=time.time())
    (root / READY).write_text(json_text(marker))
    (dest / 'verified.json').write_text(json_text(marker))
    print(json_text(marker), flush=True)
(results / 'ALL_DATA_READY.json').write_text(json_text({'completed_epoch': time.time(), 'cases': CASES}))
