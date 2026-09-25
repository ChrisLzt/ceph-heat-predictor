import unittest
from run_cloudlab_study import ready_cluster


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.cluster = {
            'status': {'osdmap': {'num_up_osds': 3, 'num_in_osds': 3},
                       'pgmap': {'pgs_by_state': [{'state_name': 'active+clean'}]},
                       'health': {'checks': {'POOL_NO_REDUNDANCY': {}, 'OSDMAP_FLAGS': {}}}},
            'osdmap': {'flags': 'noscrub,nodeep-scrub,sortbitwise,recovery_deletes,purged_snapdirs,pglog_hardlimit'}}

    def test_scrub_flags_require_explicit_opt_in(self):
        with self.assertRaises(AssertionError):
            ready_cluster(self.cluster)
        ready_cluster(self.cluster, True)

    def test_other_flags_and_other_health_warnings_still_fail(self):
        self.cluster['osdmap']['flags'] += ',noout'
        with self.assertRaises(AssertionError):
            ready_cluster(self.cluster, True)
        self.cluster['osdmap']['flags'] = 'noscrub,nodeep-scrub'
        self.cluster['status']['health']['checks']['OSD_DOWN'] = {}
        with self.assertRaises(AssertionError):
            ready_cluster(self.cluster, True)

    def test_active_scrub_and_missing_flag_are_not_accepted(self):
        self.cluster['status']['pgmap']['pgs_by_state'][0]['state_name'] = 'active+clean+scrubbing+deep'
        with self.assertRaises(AssertionError):
            ready_cluster(self.cluster, True)
        self.cluster['status']['pgmap']['pgs_by_state'][0]['state_name'] = 'active+clean'
        self.cluster['osdmap']['flags'] = 'noscrub'
        with self.assertRaises(AssertionError):
            ready_cluster(self.cluster, True)


if __name__ == '__main__':
    unittest.main()
