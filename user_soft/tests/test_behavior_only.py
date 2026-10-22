"""CPU qualification of behavior-only metadata and the reused fit/export transaction."""
import copy
import math
import unittest
from unittest import mock
import numpy as np
import torch
from user_soft import behavior_only_runner as behavior
from user_soft import rat_runner as rr, transformer_runner as tr
from user_soft.data import save_rows, eligible_centers
from user_soft.loss import evaluate_soft, fit_distance_scales
from user_soft.tests.test_runner import RunnerQualification


class BehaviorMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        batch = dict(behavior=torch.tensor([[0.,1.,0.],[.4,1.,0.],[0.,0.,1.],[.4,0.,1.],[0.,1.,0.]]),
                     source_id=torch.arange(5), time_id=torch.tensor([0,1,50,2,500]), trial_id=torch.arange(5))
        scales = fit_distance_scales(batch, behavior.SPECS, split='train', source_ids=batch['source_id'])
        return batch, scales

    def test_independent_euclidean_formula_and_normalization(self):
        batch, scales = self.fixture()
        z = torch.randn(5, 7, generator=torch.Generator().manual_seed(42))
        result = evaluate_soft(z, batch, behavior.SPECS, scales=scales)
        b = batch['behavior'].double()
        distance = ((b[:, None] - b[None, :]).square().sum(-1)).sqrt()
        self.assertAlmostEqual(float(distance[0,2]), math.sqrt(2), places=6)
        self.assertAlmostEqual(float(distance[0,1]), .4, places=6)
        logits = -distance / scales['scales'][0]['scale'] / .5
        logits.fill_diagonal_(-torch.inf)
        torch.testing.assert_close(result.Q.double(), torch.softmax(logits, 1), rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(result.Q.sum(1), torch.ones(5))
        torch.testing.assert_close(result.P.sum(1), torch.ones(5))
        self.assertTrue(torch.equal(result.Q.diag(), torch.zeros(5)))
        self.assertTrue(torch.equal(result.P.diag(), torch.zeros(5)))

    def test_time_and_trial_changes_cannot_change_target_or_loss(self):
        batch, scales = self.fixture()
        z = torch.randn(5, 9)
        changed = {k: v.clone() for k, v in batch.items()}
        changed['time_id'] = torch.tensor([9000,0,400,-50,1])
        changed['trial_id'] = torch.zeros(5, dtype=torch.long)
        first = evaluate_soft(z, batch, behavior.SPECS, scales=scales)
        second = evaluate_soft(z, changed, behavior.SPECS, scales=scales)
        torch.testing.assert_close(first.Q, second.Q, rtol=0, atol=0)
        torch.testing.assert_close(first.loss, second.loss, rtol=0, atol=0)
        self.assertGreater(first.Q[0,4], first.Q[0,2])

    def test_exact_parent_scale_projection_and_train_firewall(self):
        batch, _ = self.fixture()
        parent = fit_distance_scales(batch, rr.SPECS, split='train', source_ids=batch['source_id'])
        before = copy.deepcopy(parent)
        projected = behavior.project_scales(parent)
        self.assertEqual(projected['scales'], [s for s in parent['scales'] if s['key']=='behavior'])
        self.assertEqual(projected['train_id_sha256'], parent['train_id_sha256'])
        self.assertEqual(projected['specs'], behavior.SPECS)
        self.assertEqual(parent, before)
        parent['split'] = 'test'
        with self.assertRaises(ValueError):
            behavior.project_scales(parent)

    def test_plan_has_only_stable_seven_seeds_and_unchanged_training(self):
        configs = behavior.enumerate_configs()
        self.assertEqual(len(configs), 168)
        self.assertEqual({c['seed'] for _,c in configs}, set(range(42,49)))
        self.assertEqual({c['version'] for _,c in configs}, {'soft_v1_stable'})
        for architecture,c in configs:
            base = (rr.frozen_config if architecture=='cnn' else tr.frozen_config)(c['rat'], c['fold'], c['seed'], c['version'])
            for key in ('batch_size','max_updates','learning_rate','embedding_temperature','metadata_temperature',
                        'output_dimension','offset_left','offset_right','early_stopping','optimizer'):
                self.assertEqual(c[key],base[key])
            self.assertEqual(c['metadata_specs'], behavior.SPECS)
            self.assertEqual(c['condition'],'behavior_only')


class BehaviorCNNQualification(RunnerQualification):
    architecture = 'cnn'

    def setUp(self):
        super().setUp()
        self.config = behavior.frozen_config(self.architecture,'toy',0,42,updates=4)
        self.config.update(batch_size=32,checkpoint_interval=2,diagnostic_interval=2)
        self.scales = fit_distance_scales(self.splits['train'].metadata(),behavior.SPECS,
            split='train',source_ids=self.splits['train'].source_id)

    def fit(self,folder=None,config=None):
        with behavior.context_bindings(self.architecture):
            return super().fit(folder,config)

    def test_binding_and_finite_encoder_gradients(self):
        original_specs,original_factory=rr.SPECS,rr.make_encoder
        with behavior.context_bindings(self.architecture):
            self.assertEqual(rr.SPECS,behavior.SPECS)
            model=rr.make_encoder(4,self.config,self.device)
            self.assertEqual(type(model).__name__,'Offset10Model' if self.architecture=='cnn' else 'TransformerInputAdapter')
            result=rr.objective(model,rr.tensors(self.splits['train'],self.device),torch.arange(16),self.config,self.scales)
            result.loss.backward()
            gradients=[p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients),0)
        self.assertIs(rr.SPECS,original_specs)
        self.assertIs(rr.make_encoder,original_factory)

    def test_embedding_metadata_and_resume(self):
        record,_=self.fit()
        data_folder=self.root/'data'
        data_folder.mkdir()
        for split,rows in self.splits.items():save_rows(data_folder/f'{split}.npz',rows)
        exporter=rr.export_cell if self.architecture=='cnn' else tr.export_cell
        with behavior.context_bindings(self.architecture):
            folder,manifest=exporter(self.root/'output',self.root/'fit',data_folder,self.config,record,self.device)
        before=(folder/'manifest.json').read_bytes()
        for split,rows in self.splits.items():
            with np.load(folder/f'{split}_embedding.npz',allow_pickle=False) as data:
                self.assertEqual(data['embedding_raw'].shape,(48,32))
                for key in ('behavior','source_id','trial_id','time_id','session_id','valid_mask'):
                    self.assertEqual(len(data[key]),48)
                np.testing.assert_array_equal(data['behavior'],rows.behavior)
                np.testing.assert_array_equal(data['trial_id'],rows.trial_id)
                self.assertEqual(data['valid_mask'].sum(),len(eligible_centers(rows)))
        with behavior.context_bindings(self.architecture),mock.patch.object(rr,'make_encoder',side_effect=AssertionError('No refit')):
            _,reused=exporter(self.root/'output',self.root/'fit',data_folder,self.config,record,self.device)
        self.assertEqual(reused,manifest)
        self.assertEqual((folder/'manifest.json').read_bytes(),before)


class BehaviorTransformerQualification(BehaviorCNNQualification):
    architecture='transformer'


if __name__=='__main__':
    unittest.main()
