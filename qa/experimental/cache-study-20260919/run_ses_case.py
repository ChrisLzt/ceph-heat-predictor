#!/usr/bin/env python3
"""Run the frozen AI input through the hash-verified SES research lifecycle."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from workload_common.single_v2.core import validate_bundle
from workload_common.single_v2.lifecycle import check_ready
import ses_adapter_cloudlab

BASE = Path('/mnt/ceph-lab/cache-study-20260919')
SES = BASE / '002_storage_evaluation_system-src-v1.2.0/storage_evaluation_system'


def preflight():
    source = ses_adapter_cloudlab.preflight(SES)
    assert not source['missing_dependencies'], source
    # Import only after checking all pinned file hashes.
    ses_adapter_cloudlab._load_framework(source['source_root'])
    for case in ('ai_training_ses_v2', 'ai_inference_ses_v2'):
        bundle = BASE / 'adapted-configs' / case
        manifest = validate_bundle(bundle)
        assert manifest['provenance']['current_suite']['source_profile'] == 'phase_zipf099'
        ses_adapter_cloudlab._validate_measurement_config((bundle / 'run_current.vdb').read_text())
    source['imports_executed'] = True
    source['pip_freeze'] = subprocess.check_output([str(BASE / 'venv-ses/bin/python'), '-m', 'pip', 'freeze'], text=True).splitlines()
    (BASE / 'ses-preflight.json').write_text(json.dumps(source, indent=2))
    print(json.dumps(source, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--case', choices=['ai_training_ses_v2', 'ai_inference_ses_v2'])
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if args.preflight:
        preflight()
        return
    assert args.case and args.output
    bundle = BASE / 'adapted-configs' / args.case
    manifest = validate_bundle(bundle)
    check_ready(bundle)
    report = ses_adapter_cloudlab.run(
        config=bundle / 'run_current.vdb', output=args.output,
        vdbench=BASE / 'vdbench-cloudlab-fractional-v2/vdbench', ses_root=SES,
        metadata={**manifest, 'model_id': args.case, 'profile': 'current',
                  'logical_bytes': manifest['capacity_bytes'],
                  'manifest_sha256': hashlib.sha256((bundle / 'manifest.json').read_bytes()).hexdigest(),
                  'hardware': {'environment': 'CloudLab 3-node SSD CephFS',
                               'replicas': 1, 'osd_count': 3,
                               'client': 'hp118', 'research_only': True},
                  'runtime_difference': 'CloudLab validated fractional sampler; colleague runtime binary unavailable.'})
    print(json.dumps({'status': report['status'], 'report': report['report']}))


if __name__ == '__main__':
    main()
