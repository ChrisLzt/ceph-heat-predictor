"""Offline adapter tests: actual SES lifecycle, stub executor, no storage I/O."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from workload_common.single_v2.tests.test_ses_adapter import SESAdapterTest
import ses_adapter_cloudlab as adapter


class CloudLabSESTest(SESAdapterTest):
    def module(self):
        return adapter

    def test_current_requires_verified_alias(self):
        metadata = {'model_id': 'ai_inference_ses_v2', 'profile': 'current'}
        with self.assertRaisesRegex(ValueError, 'Unverified current-profile alias'):
            adapter.run('/missing', '/missing', '/missing', '/missing', metadata)

    def test_current_alias_reaches_source_validation_without_changing_config(self):
        valid = ('fsd=b,anchor=/no-storage-access,files=4,size=1g\n'
                 'fwd=r,fsd=b,operation=read,fileio=random,fileselect=random,threads=1,xfersize=64k\n'
                 'rd=measure,fwd=r,fwdrate=max,elapsed=600,interval=1,format=no\n')
        metadata = {'model_id': 'ai_inference_ses_v2', 'profile': 'current',
                    'manifest_sha256': 'a' * 64, 'logical_bytes': 123480309760,
                    'provenance': {'current_suite': {'source_profile': 'phase_zipf099', 'revision': '2026-09-17'}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'measurement.vdb'
            config.write_text(valid)
            executable = root / 'never-execute'
            executable.touch(mode=0o700)
            with patch.object(adapter, 'preflight', side_effect=RuntimeError('source-validation-reached')):
                with self.assertRaisesRegex(RuntimeError, 'source-validation-reached'):
                    adapter.run(config, root / 'output', executable, root / 'ses', metadata)
            self.assertEqual(config.read_text(), valid)


if __name__ == '__main__':
    unittest.main()
