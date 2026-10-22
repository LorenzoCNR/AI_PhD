"""Matched TRAIN row-shuffle software qualification; CPU/generated data only."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from user_soft import shuffle_runner as shuffled_runner
from user_soft import rat_runner as runner,transformer_runner as transformer
from user_soft.data import save_rows,config_hash
from user_soft.tests import test_runner as runner_tests


class ShuffleQualification(runner_tests.RunnerQualification):
    def setUp(self):
        super().setUp()
        self.original_train=self.splits['train']
        self.splits['train'],self.permutation=shuffled_runner.shuffle_train(self.original_train,42)
        self.config=shuffled_runner.frozen_config('cnn','toy',0,42,'soft_v1_stable',updates=4)
        self.config.update(batch_size=32,checkpoint_interval=2,diagnostic_interval=2)
    def test_exact_official_permutation_and_untouched_neural_metadata(self):
        expected=np.random.default_rng(42+10000).permutation(48)
        np.testing.assert_array_equal(self.permutation,expected)
        np.testing.assert_array_equal(self.splits['train'].behavior,self.original_train.behavior[expected])
        for key in ('neural','time_id','trial_id','source_id','session_id'):
            np.testing.assert_array_equal(getattr(self.splits['train'],key),getattr(self.original_train,key))
        self.assertEqual(self.config['shuffle_seed'],10042)
    def test_permutation_shared_across_architectures_versions_and_original_unchanged(self):
        original=self.original_train.behavior.copy()
        for architecture in ('cnn','transformer'):
            for version in ('soft_v1_frozen','soft_v1_stable'):
                config=shuffled_runner.frozen_config(architecture,'toy',0,42,version,updates=4)
                rows,permutation=shuffled_runner.shuffle_train(self.original_train,config['seed'])
                np.testing.assert_array_equal(permutation,self.permutation)
                np.testing.assert_array_equal(rows.behavior,self.splits['train'].behavior)
        np.testing.assert_array_equal(self.original_train.behavior,original)
    def test_validation_test_cannot_be_shuffled(self):
        for split in ('valid','test'):
            before=self.splits[split].behavior.copy()
            with self.assertRaises(ValueError):shuffled_runner.shuffle_train(self.splits[split],42)
            np.testing.assert_array_equal(self.splits[split].behavior,before)
    def test_exact_336_design(self):
        cells=shuffled_runner.build_configs()
        self.assertEqual(len(cells),336)
        self.assertEqual(len({(a,c['rat'],c['fold'],c['seed'],c['version']) for a,c in cells}),336)
        self.assertEqual({c['seed'] for _,c in cells},set(range(42,49)))
        self.assertTrue(all(c['condition']=='position_direction_shuffled' for _,c in cells))
    def test_unchanged_parent_scales_and_finite_both_losses_architectures(self):
        before=copy.deepcopy(self.scales)
        values=runner.tensors(self.splits['train'],self.device)
        for architecture in ('cnn','transformer'):
            for version in ('soft_v1_frozen','soft_v1_stable'):
                config=shuffled_runner.frozen_config(architecture,'toy',0,42,version,updates=4)
                if architecture=='transformer':model=transformer.make_encoder(4,config,self.device)
                else:model=runner.make_encoder(4,config,self.device)
                result=runner.objective(model,values,torch.arange(16),config,self.scales)
                result.loss.backward()
                self.assertTrue(torch.isfinite(result.loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertEqual(self.scales,before)
    def test_embedding_metadata_and_resume(self):
        record,_=self.fit()
        data_folder=self.root/'original_data';data_folder.mkdir()
        for split,rows in self.splits.items():
            save_rows(data_folder/f'{split}.npz',self.original_train if split=='train' else rows)
        output=self.root/'output'
        folder,manifest=runner.export_cell(output,self.root/'fit',data_folder,self.config,record,self.device)
        for split,rows in self.splits.items():
            expected=self.original_train if split=='train' else rows
            with np.load(folder/f'{split}_embedding.npz',allow_pickle=False) as archive:
                self.assertEqual(archive['embedding_raw'].shape,(48,32))
                np.testing.assert_array_equal(archive['behavior'],expected.behavior)
                np.testing.assert_array_equal(archive['source_id'],expected.source_id)
                np.testing.assert_array_equal(archive['trial_id'],expected.trial_id)
        repeated,second=runner.export_cell(output,self.root/'fit',data_folder,self.config,record,self.device)
        self.assertEqual(folder,repeated)
        self.assertEqual(manifest,second)
    def test_diagnostics_use_shuffled_targets_and_exact_donor_ids(self):
        embedding=self.root/'embedding';embedding.mkdir()
        rng=np.random.default_rng(1)
        np.savez_compressed(embedding/'train_embedding.npz',embedding=rng.normal(size=(48,32)).astype(np.float32),
                            behavior=self.original_train.behavior,source_id=self.original_train.source_id)
        output=self.root/'diagnostics'
        shuffled_runner.diagnostic_cell(output,embedding,self.original_train,self.splits['train'],
            self.permutation,dict(train_scales=self.scales),self.config,dict(signature='immutable_toy_shuffle'))
        path=output/'stage04_diagnostics'/'toy'/'fold_0'/'soft_v1_stable'/'seed_42'
        stage=json.loads((path/'STAGE_SIGNATURE.json').read_text())
        ids=np.asarray(stage['original_train_row_indices'])
        with np.load(path/'TRAIN_REFERENCE_W_Q_P.npz',allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['behavior'],self.original_train.behavior[self.permutation[ids]])
            np.testing.assert_array_equal(archive['behavior_donor_source_id'],self.original_train.source_id[self.permutation[ids]])
            np.testing.assert_array_equal(archive['time_id'],self.original_train.time_id[ids])
    def test_prerequisites_accept_fully_finished_limit_and_reject_active_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,count in (('user_soft_rat_time_only_v1',168),('user_soft_rat_transformer_campaign_v1',336)):
                path=root/'results'/name;path.mkdir(parents=True)
                record=dict(status='requested_limit',completed=count,planned=count,
                    records={str(i):dict(status='complete') for i in range(count)})
                (path/'CAMPAIGN_STATE.json').write_text(json.dumps(record))
            with patch.object(runner,'ROOT',root):
                shuffled_runner.prerequisites()
                active=root/'results'/'user_soft_rat_time_only_v1'/'RUNNING.lock'
                active.write_text('owner')
                with self.assertRaises(RuntimeError):shuffled_runner.prerequisites()
                active.unlink()
                path=root/'results'/'user_soft_rat_time_only_v1'/'CAMPAIGN_STATE.json'
                state=json.loads(path.read_text());state['records']['0']['status']='failed'
                path.write_text(json.dumps(state))
                with self.assertRaises(RuntimeError):shuffled_runner.prerequisites()


if __name__=='__main__':unittest.main()
