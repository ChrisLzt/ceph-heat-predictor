#!/usr/bin/env python3
"""Build a separately identified fractional-transfer compatibility runtime."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

base = Path('/mnt/ceph-lab/cache-study-20260919')
original = base / 'scoutfs-vdbench-d904d81d565c5882d0aabf2434eef46851954767'
fixed = base / 'vdbench-cloudlab-fractional-v2'
assert not fixed.exists()
shutil.copytree(original, fixed)
classes = fixed / 'classes'
exports = ['--add-exports', 'java.base/jdk.internal.org.objectweb.asm=ALL-UNNAMED']
subprocess.run(['javac', *exports, '-d', str(base / 'runtime-patcher'),
                str(base / 'PatchFractional.java')], check=True)
subprocess.run(['java', *exports, '-cp', f'{base / "runtime-patcher"}:{fixed / "vdbench.jar"}',
                'PatchFractional', str(fixed / 'vdbench.jar'), str(classes)], check=True)
subprocess.run(['javac', '--release', '8', '-d', str(classes),
                str(base / 'FractionalSampler.java')], check=True)
subprocess.run(['javac', '-cp', f'{classes}:{fixed / "vdbench.jar"}',
                '-d', str(base / 'runtime-tests'), str(base / 'FractionalXferTest.java')], check=True)
signatures = []
for cp in [str(original / 'vdbench.jar'), f'{classes}:{fixed / "vdbench.jar"}']:
    signatures.append(subprocess.check_output(['javap', '-p', '-classpath', cp, 'Vdb.FwgEntry'], text=True))
assert signatures[0] == signatures[1], 'Binary API differs from original runtime'
for name, cp in [('original', str(original / 'vdbench.jar')),
                 ('fixed', f'{classes}:{fixed / "vdbench.jar"}')]:
    (base / f'FwgEntry-{name}-bytecode.txt').write_text(subprocess.check_output(
        ['javap', '-p', '-c', '-classpath', cp, 'Vdb.FwgEntry'], text=True))
tests = subprocess.check_output(['java', '-cp', f'{base / "runtime-tests"}:{classes}:{fixed / "vdbench.jar"}',
                                  'Vdb.FractionalXferTest'], text=True)
(base / 'fractional-runtime-tests.txt').write_text(tests)
print(tests)
files = [fixed / 'vdbench.jar', classes / 'Vdb/FwgEntry.class',
         classes / 'Vdb/FractionalSampler.class', base / 'FractionalSampler.java',
         base / 'PatchFractional.java', base / 'FractionalXferTest.java']
identity = {
    'runtime': 'vdbench-cloudlab-fractional-v2',
    'original_binary_repository': 'versity/scoutfs-vdbench',
    'original_binary_commit': 'd904d81d565c5882d0aabf2434eef46851954767',
    'patch': 'Only FwgEntry.getXferSize bytecode replaced; original journal-recovery branch preserved. Other methods retained. Fractional CDF in a new helper.',
    'not_identical_to_colleague_missing_runtime': True,
    'binary_signatures_match': True,
    'files': {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    'test_output': tests,
}
(base / 'fractional-runtime-identity.json').write_text(json.dumps(identity, indent=2))
