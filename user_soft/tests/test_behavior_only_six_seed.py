"""Budget filtering must never change scientific configs or previous records."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from user_soft import behavior_only_runner as campaign
from user_soft import behavior_only_six_seed as cap


class SixSeedBudgetTests(unittest.TestCase):
    def test_exact144_subset_seeds42_to47(self):
        original = campaign.enumerate_configs()
        before = copy.deepcopy(original)
        selected = cap.limited_configs(original)
        self.assertEqual(original, before)
        self.assertEqual(len(selected),144)
        self.assertEqual({c['seed'] for _,c in selected},set(range(42,48)))
        self.assertEqual(selected,[(a,c) for a,c in before if c['seed'] < 48])

    def test_each_architecture_has72_and_each_seed24(self):
        selected = cap.limited_configs(campaign.enumerate_configs())
        for architecture in ('cnn','transformer'):
            self.assertEqual(sum(a==architecture for a,c in selected),72)
        for seed in range(42,48):
            self.assertEqual(sum(c['seed']==seed for a,c in selected),24)

    def test_budget_state_preserves_records_without_mutating_input(self):
        with tempfile.TemporaryDirectory() as directory:
            directive=Path(directory)/'directive.json'
            directive.write_text(json.dumps({'seeds':list(range(42,48))}))
            state=dict(planned=168,completed=8,records={'cnn/achilles/fold_0/soft_v1_stable/seed_42':{'status':'complete','checkpoint_sha256':'fixed'}})
            before=copy.deepcopy(state)
            adjusted=cap.budget_state(state,directive)
            self.assertEqual(state,before)
            self.assertEqual(adjusted['records'],before['records'])
            self.assertEqual(adjusted['planned'],144)
            self.assertEqual(adjusted['original_planned'],168)
            self.assertEqual(adjusted['completed'],8)

    def test_duplicate_or_incomplete_plan_is_rejected(self):
        original=campaign.enumerate_configs()
        with self.assertRaises(ValueError):cap.limited_configs(original[:10])
        changed=copy.deepcopy(original)
        changed[1]=changed[0]
        with self.assertRaises(ValueError):cap.limited_configs(changed)


if __name__=='__main__':
    unittest.main()
