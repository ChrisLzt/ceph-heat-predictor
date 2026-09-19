#!/usr/bin/env python3
"""Prepare an isolated incremental CloudLab build, leaving live files intact."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

BASE = Path('/mnt/ceph-lab')
DEST = BASE / 'integrated-20260919'
SHA = 'fbfd7114508d14b7e582263bd8a4fbc3883fa39f'
EXPECTED = {
    'codex_docs/README.md': '2e2fde10fe0b04eba3aa2fd673a86abc697bdaf8932857ece699d266435f7fce',
    'codex_docs/ONODE_CACHE_OPERATIONS.md': '47d7f5dda30fbbe1f327d70c9c843c6ac34959cb1d287fd0502152924898fd3f',
    'src/os/ObjectStore.h': 'bbfe28c774a91f18d7409dec9a185d2297a1b26c93af23d2eec82bcc79b6cdd2',
    'src/os/bluestore/BlueStore.cc': '499e058052992db746152bad17d3edbd60f4144df5e2d4f42e1331ac5fd52503',
    'src/os/bluestore/BlueStore.h': '166b00cde08e8224373e0f797cc0d1a7dad9fe7b85a85735c7c01775235c307d',
    'src/osd/OSD.cc': 'f793d62dfd28a83d3cae49b061db69c630e91f9657ec72f2c2e23806525392a0',
    'src/test/objectstore/CMakeLists.txt': '222835c3fd1f970f7cdc63f474a5490d102f09bed5b002dd88a39040f37adb98',
    'src/test/objectstore/test_bluestore_onode_cache.cc': 'e553427628c0469612b1225c8b05745f8adfaa9185f5118e2e413633de1e1741',
}


def run(*args, **kw):
    return subprocess.run(args, check=True, text=True, **kw)


assert os.geteuid() == 0
assert os.path.ismount(BASE)
assert not DEST.exists(), 'Do not overwrite an existing build'
for name, digest in EXPECTED.items():
    assert hashlib.sha256((BASE / 'src' / name).read_bytes()).hexdigest() == digest, name
DEST.mkdir()
(DEST / 'results').mkdir()
for name in ('src', 'build', 'build-mgr'):
    run('cp', '-a', '--reflink=auto', str(BASE / name), str(DEST / name))
src = DEST / 'src'
git = ['git', '-c', f'safe.directory={src}']
run(*git, 'apply', '-R', '--check', str(BASE / 'onode-online-switch.patch'), cwd=src)
run(*git, 'apply', '-R', str(BASE / 'onode-online-switch.patch'), cwd=src)
assert not subprocess.check_output(git + ['status', '--porcelain'], cwd=src).strip()
run(*git, 'fetch', 'origin', SHA, cwd=src)
run(*git, 'switch', '--detach', SHA, cwd=src)
assert subprocess.check_output(git + ['rev-parse', 'HEAD'], cwd=src, text=True).strip() == SHA
for path in [DEST, src]:
    run('chown', '-R', 'wzp:nfs-ganesha-PG0', str(path))
(DEST / 'source.json').write_text(json.dumps({'commit': SHA, 'production_changes': False}, indent=2))
print(DEST, flush=True)
