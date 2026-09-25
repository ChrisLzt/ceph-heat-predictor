#!/usr/bin/env python3
"""Disposable three-node lab adapter for the frozen stock SINGLE workload."""
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import subprocess
import sys
import time

import study_agent as agent

SOURCE = Path('/mnt/ceph-lab/single-workload-20260919/source')
BASE = Path('/mnt/ceph-lab/single-workload-20260920')
DATA = Path('/mnt/ceph-lab/cephfs/cache-study-20260919')
COMMIT = '303e43e2e1c98cb74ec156af75ce2546faaa72eb'
JAR_SHA = '8d53b728baf4e3eb28b538765b81de606ce2b1dfca39c66822d02704b462304a'


def inventory(case):
    sys.path.insert(0, str(SOURCE))
    from workload_common.single_v2.lifecycle import check_ready
    info = check_ready(BASE / 'configs' / case / 'rendered')
    root = DATA / case
    records = {str(p.relative_to(root)): [s.st_ino, s.st_size]
               for p in sorted(root.rglob('*')) if p.is_file() for s in [p.stat()]}
    return {**info, 'identity_sha256': hashlib.sha256(
        json.dumps(records, sort_keys=True).encode()).hexdigest(), 'entries': len(records)}


def runtime():
    assert agent.OSD == 1
    assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    assert not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True).strip()
    preflight = json.loads((BASE / 'preflight.json').read_text())
    assert preflight['workload_commit'] == COMMIT
    jar = Path(preflight['runtime']['path']) / 'vdbench.jar'
    assert hashlib.sha256(jar.read_bytes()).hexdigest() == JAR_SHA
    sys.path.insert(0, str(SOURCE))
    from workload_common.single_v2.core import validate_bundle
    manifests = {}
    for case, info in preflight['cases'].items():
        manifest = validate_bundle(BASE / 'configs' / case / 'rendered')
        assert manifest['model_sha256'] == info['model_sha256']
        assert manifest['config_sha256'] == info['config_sha256']
        manifests[case] = manifest
    return {'kind': 'stock-vdbench-single', 'runtime': preflight['runtime'],
            'preflight': preflight, 'adaptation': manifests}


def start(case, run_id):
    assert agent.OSD == 1 and (agent.WORKLOAD is None or agent.WORKLOAD.poll() is not None)
    assert re.fullmatch('[a-z0-9_-]+', case) and re.fullmatch('[a-z0-9_-]+', run_id)
    identity = runtime()
    assert case in identity['adaptation']
    mounted = json.loads(subprocess.check_output([
        'findmnt', '--json', '-T', str(DATA), '-o', 'TARGET,FSTYPE'], text=True))['filesystems'][0]
    assert mounted == {'target': '/mnt/ceph-lab/cephfs', 'fstype': 'ceph'}, mounted
    agent.RUN = BASE / 'runs' / run_id / case
    agent.RUN.mkdir(parents=True, exist_ok=False)
    agent.archive_markers(DATA / case)
    before = inventory(case)
    (agent.RUN / 'inventory-before.json').write_text(json.dumps(before, indent=2))
    config = BASE / 'configs' / case / 'rendered/run_current.vdb'
    args = ['runuser', '-u', 'wzp', '--', str(Path(identity['runtime']['path']) / 'vdbench'),
            '-f', str(config), '-o', str(agent.RUN / 'vdbench')]
    user = pwd.getpwnam('wzp')
    os.chown(agent.RUN, user.pw_uid, user.pw_gid)
    env = dict(os.environ, TZ='UTC', JAVA_TOOL_OPTIONS='-Duser.timezone=UTC')
    env.pop('LD_LIBRARY_PATH', None)
    with (agent.RUN / 'console.log').open('w') as log:
        agent.WORKLOAD = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, cwd=BASE)
    return {'launch_wall': time.time(), 'argv': args, 'pid': agent.WORKLOAD.pid, 'inventory': before}


def dispatch(req):
    action = req['action']
    if action == 'runtime':
        return runtime()
    if action == 'preparation':
        return {'all_ready': (BASE / 'preflight.json').exists(), 'wall': time.time()}
    if action == 'start':
        return start(req['case'], req['run_id'])
    if action == 'finish':
        assert agent.WORKLOAD is not None and agent.WORKLOAD.poll() == 0
        agent.archive_markers(DATA / req['case'])
        after = inventory(req['case'])
        before = json.loads((agent.RUN / 'inventory-before.json').read_text())
        assert before == after, 'Persistent dataset changed'
        (agent.RUN / 'inventory-after.json').write_text(json.dumps(after, indent=2))
        return {'workload': agent.workload_status(), 'inventory_unchanged': True, 'inventory': after}
    return agent.dispatch(req)


if __name__ == '__main__':
    agent.serve(dispatch)
