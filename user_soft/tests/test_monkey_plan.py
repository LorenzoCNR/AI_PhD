import unittest
from user_soft.monkey_plan import roster,require_target_freeze


class MonkeyPlanTests(unittest.TestCase):
    def test_exact_approved_count(self):
        jobs=roster()
        self.assertEqual(len(jobs),250)
        self.assertEqual(sum(j['role']=='observed' for j in jobs),150)
        self.assertEqual(sum(j['role']=='behavior_shuffle' for j in jobs),100)
        self.assertEqual(len({j['trial_id'] for j in jobs}),250)
        self.assertEqual({j['training_seed'] for j in jobs},set(range(42,47)))
        self.assertEqual({j['embedding_dimension'] for j in jobs},{3,4,8,16,32})

    def test_controls_match_observed_and_no_time_redundant_fit(self):
        jobs=roster()
        observed={(j['architecture'],j['condition'],j['embedding_dimension'],j['training_seed']) for j in jobs if j['role']=='observed'}
        for j in jobs:
            if j['role']=='behavior_shuffle':
                self.assertNotEqual(j['condition'],'time_only')
                self.assertIn((j['architecture'],j['condition'],j['embedding_dimension'],j['training_seed']),observed)
                self.assertEqual(j['shuffle_seed'],j['training_seed']+10000)

    def test_no_silent_behavioral_defaults(self):
        for value in ({},{'approved':True},{'behavior_target':'position_xy'}):
            with self.assertRaises(RuntimeError):require_target_freeze(value)
