#!/usr/bin/env python3
"""Bounded JSON RPC over an existing SSH stream; no network listener or keys."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
import traceback

OSD = int(sys.argv[1])
BASE = Path('/mnt/ceph-lab/cache-study-20260919')
ASOK = (Path('/mnt/ceph-lab/cluster/run/osd.0.asok') if OSD == 0
        else BASE / 'osd' / f'osd.{OSD}.asok')
WORKLOAD = None
RUN = None
RD = re.compile(r'(\d\d:\d\d:\d\d\.\d+) Starting RD=([^;]+); elapsed=(\d+);')


def recv(sock, count):
    result = b''
    while len(result) < count:
        part = sock.recv(count - len(result))
        if not part:
            raise RuntimeError('short admin-socket response')
        result += part
    return result


def admin(prefix, path=ASOK, **fields):
    with socket.socket(socket.AF_UNIX) as sock:
        sock.settimeout(15)
        sock.connect(str(path))
        sock.sendall(json.dumps({'prefix': prefix, 'format': 'json', **fields}).encode() + b'\0')
        size = struct.unpack('!I', recv(sock, 4))[0]
        assert size < 64 * 1024 * 1024, size
        data = json.loads(recv(sock, size))
        if isinstance(data, dict) and 'error' in data:
            raise RuntimeError(data)
        return data


def cli(*args):
    assert OSD == 0
    p = subprocess.run(['docker', 'exec', 'ceph-lab-server', 'ceph', *args, '-f', 'json'],
                       text=True, capture_output=True, check=True, timeout=40)
    return json.loads(p.stdout) if p.stdout.strip() else None


def sample():
    result = {'begin_wall': time.time(), 'begin_mono': time.monotonic(), 'osd_id': OSD}
    result['cache'] = admin('onode_cache status')
    result['hp'] = admin('object_hp status')
    perf = admin('perf dump')
    result['mempools'] = admin('dump_mempools')
    for name in ('bluestore', 'object_hp_status', 'osd', 'bluestore_cache_meta', 'bluestore_cache_data'):
        if name in perf:
            result[name] = perf[name]
    if OSD == 0:
        mds = admin('perf dump', path=Path('/mnt/ceph-lab/cluster/run/mds.lab.asok'))
        result['mds'] = {k: v for k, v in mds.items() if k in ('mds', 'mds_cache', 'mds_server', 'mds_mem')}
    result['end_wall'] = time.time()
    result['end_mono'] = time.monotonic()
    return result


def control(policy, enabled):
    assert policy in ('lru', 's3fifo') and isinstance(enabled, bool)
    result = {'requested_wall': time.time()}
    result['cache_reply'] = admin('onode_cache policy', policy=policy)
    result['cache_return_wall'] = time.time()
    result['hp_reply'] = admin('object_hp enable' if enabled else 'object_hp disable')
    result['cache'] = admin('onode_cache status')
    result['hp'] = admin('object_hp status')
    assert result['cache']['effective_policy'] == policy
    assert all(s['effective_policy'] == policy for s in result['cache']['shards'])
    assert result['hp']['enabled'] == enabled
    assert result['hp']['hp_feature_policy'] == 'C4' and result['hp']['hp_feature_count'] == 7
    result['confirmed_wall'] = time.time()
    return result


def workload_status():
    assert WORKLOAD is not None
    console = RUN / 'ses/vdbench-console.log'
    if not console.exists():
        console = RUN / 'console.log'
    text = console.read_text(errors='replace')
    phases = []
    now = datetime.datetime.now(datetime.timezone.utc)
    for match in RD.finditer(text):
        tod = datetime.datetime.strptime(match[1], '%H:%M:%S.%f').time()
        candidates = [datetime.datetime.combine(now.date() + datetime.timedelta(days=d), tod,
                                                tzinfo=datetime.timezone.utc).timestamp() for d in (-1, 0, 1)]
        phases.append({'rd': match[2], 'seconds': int(match[3]),
                       'start_wall': min(candidates, key=lambda t: abs(time.time() - t))})
    return {'pid': WORKLOAD.pid, 'returncode': WORKLOAD.poll(), 'phases': phases, 'wall': time.time()}


def archive_markers(root):
    for marker in root.rglob('no_dismount.txt'):
        assert marker.read_text().strip() == 'This file was created to keep anchor busy and prevent auto-dismount'
        dest = RUN / 'archived-markers' / marker.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest = dest.with_name(dest.name + '.' + str(time.time_ns()))
        shutil.move(str(marker), str(dest))


def data_inventory(case):
    sys.path.insert(0, str(BASE))
    from workload_common.single_v2.lifecycle import check_ready
    info = check_ready(BASE / 'adapted-configs' / case)
    root = Path('/mnt/ceph-lab/cephfs/cache-study-20260919') / case
    records = {str(p.relative_to(root)): [s.st_ino, s.st_size] for p in sorted(root.rglob('*'))
               if p.is_file() for s in [p.stat()]}
    encoded = json.dumps(records, sort_keys=True).encode()
    return {**info, 'identity_sha256': hashlib.sha256(encoded).hexdigest(), 'entries': len(records)}


def start(case, run_id):
    global WORKLOAD, RUN
    assert OSD == 1 and (WORKLOAD is None or WORKLOAD.poll() is not None)
    assert re.fullmatch('[a-z0-9_-]+', case) and re.fullmatch('[a-z0-9_-]+', run_id)
    assert (BASE / 'preparation/ALL_DATA_READY.json').is_file()
    assert (BASE / 'fractional-runtime-identity.json').is_file()
    RUN = BASE / 'runs' / run_id / case
    RUN.mkdir(parents=True, exist_ok=False)
    root = Path('/mnt/ceph-lab/cephfs/cache-study-20260919') / case
    archive_markers(root)
    before = data_inventory(case)
    (RUN / 'inventory-before.json').write_text(json.dumps(before, indent=2))
    config = BASE / 'adapted-configs' / case / 'run_current.vdb'
    runtime = BASE / 'vdbench-cloudlab-fractional-v2'
    args = ['runuser', '-u', 'wzp', '--', str(runtime / 'vdbench'), '-f', str(config), '-o', str(RUN / 'vdbench')]
    if case.startswith('ai_'):
        assert (BASE / 'ses-preflight.json').is_file()
        args = ['runuser', '-u', 'wzp', '--', str(BASE / 'venv-ses/bin/python'),
                str(BASE / 'run_ses_case.py'), '--case', case, '--output', str(RUN / 'ses')]
    user = pwd.getpwnam('wzp')
    os.chown(RUN, user.pw_uid, user.pw_gid)
    env = dict(os.environ, TZ='UTC', JAVA_TOOL_OPTIONS='-Duser.timezone=UTC')
    env.pop('LD_LIBRARY_PATH', None)
    log = (RUN / 'console.log').open('w')
    WORKLOAD = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, cwd=BASE)
    log.close()
    return {'launch_wall': time.time(), 'argv': args, 'pid': WORKLOAD.pid, 'inventory': before}


def finish(case):
    assert WORKLOAD is not None and WORKLOAD.poll() == 0
    archive_markers(Path('/mnt/ceph-lab/cephfs/cache-study-20260919') / case)
    after = data_inventory(case)
    before = json.loads((RUN / 'inventory-before.json').read_text())
    assert before == after, 'Persistent dataset identity/size changed'
    (RUN / 'inventory-after.json').write_text(json.dumps(after, indent=2))
    return {'workload': workload_status(), 'inventory_unchanged': True, 'inventory': after}


def dispatch(req):
    action = req['action']
    if action == 'sample':
        return sample()
    if action == 'control':
        return control(req['policy'], req['enabled'])
    if action == 'cluster':
        return {k: cli(*args) for k, args in {
            'status': ['status'], 'versions': ['versions'], 'osd_df': ['osd', 'df'],
            'hp_mgr': ['osd', 'hp', 'status', '--detail'],
        }.items()}
    if action == 'budget':
        config = admin('config show')
        return {k: v for k, v in config.items() if k.startswith(('bluestore_cache', 'osd_memory'))}
    if action == 'start':
        return start(req['case'], req['run_id'])
    if action == 'workload':
        return workload_status()
    if action == 'finish':
        return finish(req['case'])
    if action == 'preparation':
        marker = BASE / 'preparation/ALL_DATA_READY.json'
        return {'all_ready': marker.exists(), 'wall': time.time()}
    if action == 'runtime':
        assert OSD == 1
        return {'runtime': json.loads((BASE / 'fractional-runtime-identity.json').read_text()),
                'adaptation': json.loads((BASE / 'preparation/path-only-adaptation.json').read_text()),
                'ses': json.loads((BASE / 'ses-preflight.json').read_text())}
    if action == 'ping':
        return {'wall': time.time(), 'osd': OSD, 'hostname': socket.gethostname()}
    raise ValueError('Unsupported action')


def terminate_owned_tree(pid):
    children = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            ppid = int((path / 'stat').read_text().split(') ', 1)[1].split()[1])
            if ppid == pid:
                children.append(int(path.name))
        except (OSError, ValueError):
            continue
    for child in children:
        terminate_owned_tree(child)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


try:
    for line in sys.stdin:
        try:
            response = {'ok': True, 'data': dispatch(json.loads(line))}
        except Exception:
            response = {'ok': False, 'error': traceback.format_exc()}
        print(json.dumps(response), flush=True)
finally:
    if WORKLOAD is not None and WORKLOAD.poll() is None:
        terminate_owned_tree(WORKLOAD.pid)
        os.killpg(WORKLOAD.pid, signal.SIGTERM)
        try:
            WORKLOAD.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(WORKLOAD.pid, signal.SIGKILL)
            WORKLOAD.wait()
