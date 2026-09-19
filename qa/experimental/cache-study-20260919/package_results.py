#!/usr/bin/env python3
"""Validate and package the completed study, without credentials or data files."""
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'cloudlab-runs/fixed8g-ses-postprepare-001'
ARCHIVE = ROOT / 'server-results/fixed8g-ses-postprepare-001-workloads.tar.gz'
ARCHIVE_SHA = '6bd5e9dc84dd14f2459291f2ae02c1f1f589efecbb8a3b911e23ce2c3506169a'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert digest(ARCHIVE) == ARCHIVE_SHA, 'Remote workload report archive is incomplete or changed'
    with tarfile.open(ARCHIVE) as archive:
        members = archive.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            assert not name.is_absolute() and '..' not in name.parts, member.name
            assert member.isfile() or member.isdir(), member.name
        assert sum(member.size for member in members) < 2 * 1024**3
        archived_audits = 0
        for audit in RUN.glob('*/workload-skew-audit.json'):
            archived = json.load(archive.extractfile(f'{RUN.name}/{audit.parent.name}/{audit.name}'))
            assert archived == json.loads(audit.read_text())
            archived_audits += 1
        assert archived_audits == 5
        for case in ('ai_training_ses_v2', 'ai_inference_ses_v2'):
            report = json.load(archive.extractfile(f'{RUN.name}/{case}/ses/ses-research-result.json'))
            assert report['status'] == 'completed'
    summary = json.loads((RUN / 'result-summary.json').read_text())
    assert summary['complete_cases'] == 5 and not summary['certification_claim']
    with (RUN / 'onode-results.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5 and len({row['case'] for row in rows}) == 5
    hits, misses = 0, 0
    for row in rows:
        hit, miss = int(row['hits']), int(row['misses'])
        ratio = 100 * hit / (hit + miss)
        assert abs(ratio - float(row['s3fifo_percent'])) < 1e-10
        assert ratio > 96 and row['strict_gt95'] == row['strict_gt96'] == 'True'
        complete = json.loads((RUN / row['case'] / 'workload-complete.json').read_text())
        assert complete['inventory_unchanged'] and complete['workload']['returncode'] == 0
        assert (RUN / row['case'] / 'COMPLETE.json').is_file()
        hits += hit
        misses += miss
    assert hits == summary['pooled_hits'] and misses == summary['pooled_misses']
    assert abs(100 * hits / (hits + misses) - summary['pooled_onode_percent']) < 1e-10
    row_counts = {}
    for name in ('cache-diagnostics', 'hot-cold-results', 'sampling-quality', 'workload-distribution', 'realtime-diagnostics'):
        with (RUN / f'{name}.csv').open() as stream:
            row_counts[name] = len(list(csv.DictReader(stream)))
        assert row_counts[name] == (14 if name == 'workload-distribution' else 5)
    assert (RUN / 'cloudlab-timeseries.png').stat().st_size > 10000

    files = set()
    for pattern in ('*.md', '*.py', 'original-*-skew-audit.json'):
        files.update(ROOT.glob(pattern))
    for name in ('materials-audit.json', 'fractional-runtime-identity.json',
                 'ses-preflight.json', 'SkewReport.javap.txt'):
        files.add(ROOT / name)
    files.update(path for path in RUN.rglob('*') if path.is_file())
    for name in ('build-and-tests.log', 'onode-tests.xml', 'bluestore-types.xml',
                 'cache-budget-frozen.json', 'ALL_DATA_READY.json',
                 'runtime-upgraded.json', 'after-study-status.json', ARCHIVE.name):
        files.add(ROOT / 'server-results' / name)
    manifest = []
    for path in sorted(files):
        assert path.is_file() and not path.is_symlink(), str(path)
        manifest.append({'path': str(path.relative_to(ROOT)), 'size': path.stat().st_size,
                         'sha256': digest(path)})
    qa = {'run_id': RUN.name, 'cases': len(rows), 'independent_csv_recalculation': True,
          'all_application_returncodes_zero': True, 'all_file_inventories_unchanged': True,
          'csv_row_counts': row_counts, 'remote_archive_sha256': ARCHIVE_SHA,
          'remote_archive_members': len(members), 'packaged_files': len(manifest),
          'archived_histogram_audits_match': archived_audits,
          'ses_completed_reports': 2,
          'certification_claim': False}
    output = ROOT / 'cache-study-results-20260919.tar.gz'
    prefix = 'cache-study-results-20260919'
    with tarfile.open(output, 'w:gz') as bundle:
        for path in sorted(files):
            bundle.add(path, arcname=f'{prefix}/{path.relative_to(ROOT)}', recursive=False)
        for name, value in [('MANIFEST.json', manifest), ('DELIVERY-QA.json', qa)]:
            data = (json.dumps(value, indent=2) + '\n').encode()
            info = tarfile.TarInfo(f'{prefix}/{name}')
            info.size = len(data)
            info.mode = 0o644
            bundle.addfile(info, io.BytesIO(data))
    with tarfile.open(output) as bundle:
        for entry in manifest:
            data = bundle.extractfile(f'{prefix}/{entry["path"]}').read()
            assert len(data) == entry['size']
            assert hashlib.sha256(data).hexdigest() == entry['sha256']
    qa.update({'bundle': output.name, 'bundle_sha256': digest(output), 'roundtrip_hashes_valid': True})
    (ROOT / 'DELIVERY-QA.json').write_text(json.dumps(qa, indent=2) + '\n')
    print(json.dumps(qa, indent=2))


if __name__ == '__main__':
    main()
