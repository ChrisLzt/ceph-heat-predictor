#!/usr/bin/env python3
"""Initialize only the empty, explicitly provisioned SSD loop-backed OSD."""
import json
import os
from pathlib import Path
import subprocess

root = Path('/lab/cache-study-20260919')
osd = root / 'osd'
identity = json.loads((osd / 'identity.json').read_text())
assert not (osd / 'whoami').exists(), 'Existing OSD; initialization forbidden'
assert set(p.name for p in osd.iterdir()) == {'identity.json', 'ceph.conf', 'keyring'}
assert Path('/dev/loop0').is_block_device()
assert not subprocess.check_output(['wipefs', '-n', '--noheadings', '--output', 'TYPE', '/dev/loop0']).strip()
(osd / 'block').symlink_to('/dev/loop0')
env = dict(os.environ)
env['CEPH_CONF'] = str(osd / 'ceph.conf')
env['LD_LIBRARY_PATH'] = str(root / 'runtime/build/lib')
binary = str(root / 'runtime/build/bin/ceph-osd')
subprocess.run([binary, '--mkfs', '-i', str(identity['id']), '--osd-uuid', identity['uuid']],
               check=True, env=env)
print(json.dumps({'initialized': identity['id'], 'uuid': identity['uuid']}))
