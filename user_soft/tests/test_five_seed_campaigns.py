"""Five-seed campaign qualification, preserving older sealed seven-seed roots."""
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from user_soft import transformer_campaign as transformer
from user_soft import shuffle_runner as shuffle
from user_soft import transformer_runner as initial
from user_soft import rat_runner as runner
from user_soft.data import config_hash


class FiveSeedQualification(unittest.TestCase):
    def test_transformer_exact_240_and_seed_set(self):
        cells=transformer.enumerate_configs(5)
        self.assertEqual(len(cells),240)
        self.assertEqual(Counter(condition for condition,_ in cells),{'joint':120,'time_only':120})
        self.assertEqual({c['seed'] for _,c in cells},set(range(42,47)))
        self.assertEqual({c['fold'] for _,c in cells},{0,1,2})
        self.assertEqual(len({(condition,c['rat'],c['fold'],c['seed'],c['version']) for condition,c in cells}),240)
    def test_shuffle_exact_240_and_sixty_shared_permutations(self):
        cells=shuffle.build_configs(5)
        self.assertEqual(len(cells),240)
        self.assertEqual(Counter(architecture for architecture,_ in cells),{'cnn':120,'transformer':120})
        self.assertEqual({c['seed'] for _,c in cells},set(range(42,47)))
        self.assertEqual(len({(c['rat'],c['fold'],c['seed']) for _,c in cells}),60)
    def test_default_seven_designs_retained_and_invalid_count_rejected(self):
        self.assertEqual(len(transformer.enumerate_configs()),336)
        self.assertEqual(len(shuffle.build_configs()),336)
        for count in (0,3,6,8):
            with self.assertRaises(ValueError):transformer.enumerate_configs(count)
            with self.assertRaises(ValueError):shuffle.build_configs(count)
    def test_exact_initial_joint_identity_and_existing_full_architecture_gate(self):
        sources=transformer.sources()
        for config in initial.initial_configs():
            self.assertEqual(transformer.config_for(config['rat'],0,42,config['version'],'joint'),config)
        self.assertEqual(transformer.training_sources('joint',sources),initial.sources())
        gate=json.loads((initial.DEFAULT_OUTPUT/'qualification'/'GATE.json').read_text())
        self.assertEqual(gate['decision'],'GO')
        self.assertEqual(gate['training_source_signature'],config_hash(initial.sources()))
        for _,config in transformer.enumerate_configs(5):
            self.assertEqual(config['transformer_settings'],initial.ARCHITECTURE)
    def test_shuffle_correct_new_240_predecessor_and_original_time_168(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            new=root/'results'/'user_soft_rat_transformer_5seed_v1'
            for folder,count in ((root/'results'/'user_soft_rat_time_only_v1',168),(new,240)):
                folder.mkdir(parents=True)
                (folder/'CAMPAIGN_STATE.json').write_text(json.dumps(dict(status='requested_limit',completed=count,
                    records={str(i):dict(status='complete') for i in range(count)})))
            with patch.object(runner,'ROOT',root):
                shuffle.prerequisites(new,240)
                with self.assertRaises(RuntimeError):shuffle.prerequisites(new,336)
                (new/'RUNNING.lock').write_text('active owner')
                with self.assertRaises(RuntimeError):shuffle.prerequisites(new,240)


if __name__=='__main__':unittest.main()
