#!/usr/bin/env python3
"""Allocate a new SSD-backed OSD file without altering existing partitions."""
import json
import os
from pathlib import Path
import subprocess

root = Path('/mnt/ceph-lab')
target = root / 'cache-study-20260919/osd-block.img'
size = 320 * 2**30
assert os.geteuid() == 0
assert os.path.ismount(root)
mount = json.loads(subprocess.check_output(['findmnt', '-J', str(root)], text=True))['filesystems'][0]
assert mount['source'] == '/dev/sda4' and mount['fstype'] == 'ext4', mount
assert not target.exists(), 'Refuse to overwrite an existing OSD file'
fs = os.statvfs(root)
assert fs.f_bavail * fs.f_frsize > size + 20 * 2**30, 'Insufficient local SSD headroom'
target.parent.mkdir(exist_ok=True)
subprocess.run(['fallocate', '-l', str(size), str(target)], check=True)
target.chmod(0o600)
assert target.stat().st_size == size and target.stat().st_blocks * 512 >= size
loop = subprocess.check_output(['losetup', '--find', '--show', '--direct-io=on',
                                str(target)], text=True).strip()
assert Path(loop).is_block_device()
info = json.loads(subprocess.check_output(['losetup', '-J', '-l', loop], text=True))
(target.parent / 'block-identity.json').write_text(json.dumps(info, indent=2))
print(json.dumps(info))
