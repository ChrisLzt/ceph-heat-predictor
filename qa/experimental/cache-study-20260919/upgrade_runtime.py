#!/usr/bin/env python3
"""Upgrade only this disposable lab's OSD/MGR; preserve disks and old binaries."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

OLD = Path('/lab')
NEW = OLD / 'integrated-20260919'
CLUSTER = OLD / 'cluster'
EVIDENCE = NEW / 'results'


def ceph(*args):
    p = subprocess.run(['ceph', *args, '-f', 'json'], text=True,
                       capture_output=True, check=True, timeout=30)
    return json.loads(p.stdout) if p.stdout.strip() else None


def stop(binary):
    pids = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            exe = (path / 'exe').resolve()
        except OSError:
            continue
        if exe.name == binary:
            assert str(exe).startswith('/lab/'), str(exe)
            pids.append(int(path.name))
    assert len(pids) == 1, (binary, pids)
    os.kill(pids[0], signal.SIGTERM)
    for _ in range(120):
        if not Path('/proc', str(pids[0]), 'exe').exists():
            return
        time.sleep(.5)
    raise RuntimeError(f'{binary} did not exit cleanly')


def launch(binary, name, build):
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = str(NEW / build / 'lib')
    env['PYTHONPATH'] = ':'.join(map(str, [NEW / build / 'lib/cython_modules/lib.3',
                                          NEW / 'src/src/pybind', NEW / 'src/src/python-common']))
    env['CEPH_CONF'] = str(CLUSTER / 'ceph.conf')
    log = open(EVIDENCE / f'{binary}-runtime.log', 'a')
    args = [str(NEW / build / 'bin' / binary), '-f', '-i', name]
    args += ['--erasure-code-dir', str(NEW / 'build/lib'),
             '--plugin-dir', str(NEW / 'build/lib'),
             '--osd-class-dir', str(NEW / 'build/lib')]
    if binary == 'ceph-mgr':
        args += ['--mgr-module-path', str(NEW / 'src/src/pybind/mgr')]
    process = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    (EVIDENCE / f'{binary}.pid').write_text(str(process.pid))


assert subprocess.check_output([str(NEW / 'build/bin/ceph-osd'), '--version'],
                               text=True).find('fbfd7114508d14b7e582263bd8a4fbc3883fa39f') >= 0
assert not (EVIDENCE / 'runtime-upgraded.json').exists()
if not (EVIDENCE / 'pre-upgrade-status.json').exists():
    (EVIDENCE / 'pre-upgrade-status.json').write_text(json.dumps(ceph('status'), indent=2))
ceph('osd', 'set', 'noout')
try:
    current = ceph('versions').get('osd', {})
    if not any('fbfd7114508d14b7e582263bd8a4fbc3883fa39f' in v for v in current):
        stop('ceph-osd')
        launch('ceph-osd', '0', 'build')
    for _ in range(90):
        try:
            z = ceph('tell', 'osd.0', 'version')
            if z.get('version') == 'bfd711':
                break
        except (subprocess.SubprocessError, ValueError):
            pass
        time.sleep(1)
    else:
        raise RuntimeError('New OSD did not become ready; old binary retained')
    stop('ceph-mgr')
    launch('ceph-mgr', 'lab', 'build-mgr')
    for _ in range(90):
        s = ceph('status')
        if s.get('mgrmap', {}).get('available'):
            try:
                hp = ceph('osd', 'hp', 'status', '--detail')
                break
            except subprocess.SubprocessError:
                pass
        time.sleep(1)
    else:
        raise RuntimeError('New MGR did not become ready')
    identity = {'osd': ceph('tell', 'osd.0', 'version'),
                'hp': ceph('tell', 'osd.0', 'object_hp', 'status'),
                'versions': ceph('versions'), 'status': ceph('status')}
    (EVIDENCE / 'runtime-upgraded.json').write_text(json.dumps(identity, indent=2))
    print(json.dumps(identity['versions']))
finally:
    ceph('osd', 'unset', 'noout')
