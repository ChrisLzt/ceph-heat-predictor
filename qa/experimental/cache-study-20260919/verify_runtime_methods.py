#!/usr/bin/env python3
import json
from pathlib import Path
import re

base = Path('/mnt/ceph-lab/cache-study-20260919')
normalized = []
for name in ('original', 'fixed'):
    text = (base / f'FwgEntry-{name}-bytecode.txt').read_text()
    text, n = re.subn(r'  public int getXferSize\(\);.*?(?=  public int getMaxXfersize\(\);)',
                      '', text, flags=re.S)
    assert n == 1
    text = text.replace('// Field Vdb/FwgEntry.', '// Field ')
    text = text.replace('// Method Vdb/FwgEntry.', '// Method ')
    normalized.append(text)
assert normalized[0] == normalized[1], 'Unexpected changes outside getXferSize'
path = base / 'fractional-runtime-identity.json'
identity = json.loads(path.read_text())
identity['other_methods_identical_after_javap_owner_normalization'] = True
path.write_text(json.dumps(identity, indent=2))
print('PASS: all original fields, signatures and other method bytecode retained')
