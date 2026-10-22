import unittest
from user_soft import behavior_only_shuffle as s, behavior_only_runner as b
from user_soft.behavior_only_six_seed import limited_configs


class BehaviorShuffleQualification(unittest.TestCase):
    def test_exact_matched_design(self):
        expected = limited_configs(b.enumerate_configs())
        actual = s.configs()
        self.assertEqual(len(actual), 144)
        for (a,c), (a0,c0) in zip(actual,expected):
            self.assertEqual(a,a0)
            for key,value in c0.items():
                if key != 'condition':
                    self.assertEqual(c[key],value)
            self.assertEqual(c['shuffle_seed'],c['seed']+10000)
            self.assertEqual(c['metadata_specs'],b.SPECS)

    def test_wait_gate_requires_all_records_and_unlock(self):
        state = dict(status='complete',completed=144,records={str(i):dict(status='complete') for i in range(144)})
        self.assertTrue(s.ready(state,False))
        self.assertFalse(s.ready(state,True))
        state['records']['0']['status']='fit_complete'
        self.assertFalse(s.ready(state,False))

    def test_failed_or_paused_never_launches(self):
        for status in ('failed','running','stopped_after_current','interrupted'):
            self.assertFalse(s.ready(dict(status=status,completed=144,records={}),False))
