#!/usr/bin/env python3
"""Register only the two new lab OSDs and export their least-privilege keys."""
import json
import os
from pathlib import Path
import subprocess
import tarfile
import uuid

new = Path('/lab/integrated-20260919')
export = new / 'export'
export.mkdir(exist_ok=True)


def ceph(*args, **kw):
    return subprocess.check_output(['ceph', *args], text=True, **kw).strip()


fsid = ceph('fsid')
assert json.loads(ceph('osd', 'ls', '-f', 'json')) == [0]
for expected, host, ip in [(1, 'hp118', '10.10.1.2'), (2, 'hp081', '10.10.1.3')]:
    dst = export / host
    assert not dst.exists()
    dst.mkdir(mode=0o700)
    secret = subprocess.check_output(['ceph-authtool', '--gen-print-key'], text=True).strip()
    osd_uuid = str(uuid.uuid4())
    osd_id = int(ceph('osd', 'new', osd_uuid, '-i', '-', input=json.dumps({'cephx_secret': secret})))
    assert osd_id == expected
    (dst / 'keyring').write_text(f'[osd.{osd_id}]\nkey = {secret}\n')
    (dst / 'identity.json').write_text(json.dumps({'id': osd_id, 'uuid': osd_uuid, 'host': host}))
    (dst / 'ceph.conf').write_text(f'''[global]
fsid = {fsid}
mon_host = [v2:10.10.1.1:3300,v1:10.10.1.1:6789]
public_network = 10.10.1.0/24
cluster_network = 10.10.1.0/24
public_addr = {ip}
auth_cluster_required = cephx
auth_service_required = cephx
auth_client_required = cephx
keyring = /lab/cache-study-20260919/osd/keyring
admin_socket = /lab/cache-study-20260919/osd/$name.asok
log_file = /lab/cache-study-20260919/osd/$name.log
pid_file = /lab/cache-study-20260919/osd/$name.pid
osd_data = /lab/cache-study-20260919/osd
osd_objectstore = bluestore
osd_crush_location = root=default host={host}
erasure_code_dir = /lab/cache-study-20260919/runtime/build/lib
plugin_dir = /lab/cache-study-20260919/runtime/build/lib
osd_class_dir = /lab/cache-study-20260919/runtime/build/lib
bluestore_cache_type = 2q
bluestore_cache_autotune = false
bluestore_cache_size = 8589934592
bluestore_cache_meta_ratio = 0.70
bluestore_cache_kv_ratio = 0.20
osd_memory_target = 17179869184
''')
    archive = export / f'{host}-osd-private.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for path in dst.iterdir():
            tar.add(path, arcname=path.name)
    archive.chmod(0o600)
    os.chown(archive, 20001, 5117)
    print(json.dumps({'host': host, 'osd': osd_id, 'private_archive': str(archive)}))
with tarfile.open(export / 'osd-runtime.tar.gz', 'w:gz') as tar:
    for path in sorted((new / 'build/lib').glob('*.so*')):
        tar.add(path, arcname=str(path.relative_to(new)), recursive=False)
    tar.add(new / 'build/bin/ceph-osd', arcname='build/bin/ceph-osd')
os.chown(export / 'osd-runtime.tar.gz', 20001, 5117)
print('runtime archive complete')
