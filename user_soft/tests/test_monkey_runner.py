"""Independent CPU qualification of the bounded active-monkey Soft campaign."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest import mock
import numpy as np
import torch
from user_soft import monkey_runner as m, rat_runner as rr
from user_soft.data import TrialRows, eligible_centers
from user_soft.loss import evaluate_soft, fit_distance_scales


class MonkeyRunnerQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='monkey_soft_cpu_')
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(42)
        self.rows = {}
        for j, split in enumerate(('train', 'valid', 'test')):
            t = np.tile(np.arange(24), 2)
            xy = np.column_stack((t / 23., np.sin(t / 5.))).astype(np.float32)
            neural = np.column_stack((xy, np.cos(t / 5.), t / 23.))
            neural = (neural + .01 * rng.normal(size=neural.shape)).astype(np.float32)
            self.rows[split] = TrialRows(neural, xy, np.arange(j * 100, j * 100 + 48),
                np.repeat([j * 10, j * 10 + 1], 24), t, np.zeros(48, dtype=np.int64), split)
        self.parent = fit_distance_scales(self.rows['train'].metadata(), m.specs_for('joint'),
            split='train', source_ids=self.rows['train'].source_id)
        self.device = torch.device('cpu')

    def tearDown(self):
        self.temp.cleanup()

    def config(self, architecture='cnn', condition='joint'):
        config = m.frozen_config(architecture, condition, 3, 42, updates=4)
        config.update(batch_size=16, checkpoint_interval=2, diagnostic_interval=2)
        return config

    def fit(self, folder, config):
        with m.context_bindings(config):
            return rr.fit_cell(folder, self.rows['train'], self.rows['valid'], config,
                m.project_scales(self.parent, config['condition']), 'immutable_toy_parent',
                {'qualification': 'toy_only'}, self.device)

    def test_roster_is_exact_250_unique_paired_trials(self):
        roster = m.enumerate_configs()
        self.assertEqual(len(roster), 250)
        self.assertEqual(len({c['trial_id'] for _, c in roster}), 250)
        self.assertEqual(sum(not c['shuffle'] for _, c in roster), 150)
        self.assertEqual(sum(c['shuffle'] for _, c in roster), 100)
        self.assertEqual({c['seed'] for _, c in roster}, set(range(42, 47)))
        self.assertEqual({c['output_dimension'] for _, c in roster}, {3, 4, 8, 16, 32})
        for condition, expected in [('time_only', 50), ('joint', 100), ('behavior_only', 100)]:
            self.assertEqual(sum(c['condition'] == condition for _, c in roster), expected)
        for _, c in roster:
            self.assertEqual(c['population'], 'active')
            self.assertEqual(c['split_universe'], 'official_active_train_valid_test')
            if c['shuffle']:
                self.assertEqual(c['shuffle_seed'], c['seed'] + 10000)
        with self.assertRaises(ValueError):
            m.frozen_config('cnn', 'time_only', 3, 42, shuffle=True)

    def test_settings_match_frozen_soft_except_sampling_units_and_dimension(self):
        for architecture in m.ARCHITECTURES:
            c = m.frozen_config(architecture, 'joint', 8, 42)
            old = rr.frozen_config('toy', 0, 42, 'soft_v1_stable')
            for key in ('batch_size', 'learning_rate', 'max_updates', 'optimizer', 'betas',
                        'eps', 'weight_decay', 'early_stopping', 'embedding_temperature',
                        'metadata_temperature', 'input_policy', 'offset_left', 'offset_right'):
                self.assertEqual(c[key], old[key])
            self.assertEqual(c['time_bin_ms'], 1)
            self.assertEqual(c['output_dimension'], 8)
            self.assertEqual(c['behavior_target'], 'position_xy_joint')
            self.assertEqual(c['version'], 'soft_v1_stable')

    def test_joint_distance_formula_and_q_p_normalization(self):
        specs = m.specs_for('joint')
        self.assertEqual([s['weight'] for s in specs], [.5, .5])
        batch = self.rows['train'].metadata(np.arange(8))
        scales = m.project_scales(self.parent, 'joint')
        z = torch.randn(8, 3, generator=torch.Generator().manual_seed(42))
        result = evaluate_soft(z, batch, specs, scales=scales)
        bounds = {s['key']: s['scale'] for s in scales['scales']}
        time = batch['time_id'].float()
        xy = batch['behavior'].float()
        distance = .5 * (time[:, None] - time[None, :]).abs() / bounds['time_id']
        distance += .5 * torch.cdist(xy, xy) / bounds['behavior']
        logits = -distance / .5
        logits.fill_diagonal_(-torch.inf)
        torch.testing.assert_close(result.Q, logits.softmax(1), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(result.Q.sum(1), torch.ones(8))
        torch.testing.assert_close(result.P.sum(1), torch.ones(8))
        self.assertTrue(torch.equal(result.Q.diag(), torch.zeros(8)))
        self.assertTrue(torch.equal(result.P.diag(), torch.zeros(8)))

    def test_time_only_ignores_xy_and_behavior_only_ignores_time(self):
        batch = self.rows['train'].metadata(np.arange(8))
        z = torch.randn(8, 3)
        for condition, irrelevant in [('time_only', 'behavior'), ('behavior_only', 'time_id')]:
            scales = m.project_scales(self.parent, condition)
            changed = {k: v.clone() for k, v in batch.items()}
            changed[irrelevant] = changed[irrelevant] * 100 + 73
            a = evaluate_soft(z, batch, m.specs_for(condition), scales=scales)
            b = evaluate_soft(z, changed, m.specs_for(condition), scales=scales)
            torch.testing.assert_close(a.Q, b.Q, rtol=0, atol=0)
            torch.testing.assert_close(a.loss, b.loss, rtol=0, atol=0)

    def test_scales_are_train_frozen_and_not_minibatch_refitted(self):
        before = copy.deepcopy(self.parent)
        for condition in m.CONDITIONS:
            projected = m.project_scales(self.parent, condition)
            self.assertEqual(projected['train_id_sha256'], self.parent['train_id_sha256'])
            for scale in projected['scales']:
                self.assertIn(scale, self.parent['scales'])
        self.assertEqual(self.parent, before)
        for split in ('valid', 'test'):
            invalid = dict(self.parent, split=split)
            with self.assertRaises(ValueError):
                m.project_scales(invalid, 'joint')

    def test_shuffle_joint_rows_train_only_preserves_neural_time_and_ids(self):
        original = self.rows['train']
        shuffled, permutation = m.shuffled_train(original, 42)
        np.testing.assert_array_equal(np.sort(permutation), np.arange(48))
        np.testing.assert_array_equal(shuffled.behavior, original.behavior[permutation])
        for key in ('neural', 'source_id', 'trial_id', 'time_id', 'session_id'):
            np.testing.assert_array_equal(getattr(shuffled, key), getattr(original, key))
        again, second = m.shuffled_train(original, 42)
        np.testing.assert_array_equal(permutation, second)
        np.testing.assert_array_equal(shuffled.behavior, again.behavior)
        for split in ('valid', 'test'):
            with self.assertRaises(ValueError):
                m.shuffled_train(self.rows[split], 42)

    def test_batch_permutation_and_cross_trial_candidates(self):
        indices = np.array([0, 1, 2, 24, 25, 26])
        batch = self.rows['train'].metadata(indices)
        z = torch.randn(6, 3)
        specs = m.specs_for('joint')
        scales = m.project_scales(self.parent, 'joint')
        a = evaluate_soft(z, batch, specs, scales=scales)
        permutation = torch.tensor([4, 0, 2, 5, 1, 3])
        b = evaluate_soft(z[permutation], {k: v[permutation] for k, v in batch.items()}, specs, scales=scales)
        torch.testing.assert_close(b.Q, a.Q[permutation][:, permutation])
        torch.testing.assert_close(b.loss, a.loss)
        self.assertTrue(a.candidate_mask[0, 3])
        self.assertGreater(float(a.Q[0, 3]), 0)

    def test_both_encoders_ten_bins_have_finite_gradients_and_restore_bindings(self):
        original_specs, original_factory = rr.SPECS, rr.make_encoder
        for architecture in m.ARCHITECTURES:
            config = self.config(architecture)
            with m.context_bindings(config):
                model = rr.make_encoder(4, config, self.device)
                z = model(torch.randn(8, 4, 10))
                self.assertEqual(z.shape, (8, 3))
                batch = self.rows['train'].metadata(np.arange(8))
                result = evaluate_soft(z, batch, config['metadata_specs'], scales=m.project_scales(self.parent, 'joint'))
                result.loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
            self.assertIs(rr.SPECS, original_specs)
            self.assertIs(rr.make_encoder, original_factory)

    def test_toy_fit_and_checkpoint_reuse_without_training(self):
        for architecture in m.ARCHITECTURES:
            with self.subTest(architecture=architecture):
                config = self.config(architecture)
                folder = self.root / architecture
                record, created = self.fit(folder, config)
                self.assertTrue(created)
                self.assertEqual(record['actual_updates'], 4)
                self.assertNotEqual(record['initial_parameter_hash'], record['final_parameter_hash'])
                before = (folder / 'checkpoint.pt').read_bytes()
                with mock.patch.object(rr, 'objective', side_effect=AssertionError('Unexpected refit')):
                    reused, created = self.fit(folder, config)
                self.assertFalse(created)
                self.assertEqual(reused['final_parameter_hash'], record['final_parameter_hash'])
                self.assertEqual(before, (folder / 'checkpoint.pt').read_bytes())

    def test_interruption_exact_resume_and_config_invalidation(self):
        config = self.config('transformer')
        reference, _ = self.fit(self.root / 'reference', config)
        objective, count = rr.objective, 0
        def interrupted(model, values, indices, cfg, scales):
            nonlocal count
            if model.training:
                count += 1
                if count == 3:
                    raise KeyboardInterrupt('CPU qualification interruption')
            return objective(model, values, indices, cfg, scales)
        with mock.patch.object(rr, 'objective', side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.fit(self.root / 'resumed', config)
        self.assertEqual(torch.load(self.root / 'resumed/checkpoint.pt', weights_only=False)['step'], 2)
        resumed, _ = self.fit(self.root / 'resumed', config)
        self.assertEqual(reference['final_parameter_hash'], resumed['final_parameter_hash'])
        changed = dict(config, embedding_temperature=.2)
        before = (self.root / 'resumed/checkpoint.pt').read_bytes()
        with self.assertRaises(RuntimeError):
            self.fit(self.root / 'resumed', changed)
        self.assertEqual(before, (self.root / 'resumed/checkpoint.pt').read_bytes())

    def test_export_cardinality_labels_unit_norms_and_trial_safe_mask(self):
        rows = self.rows['test']
        labels = dict(position_xy=rows.behavior, direction=np.zeros(48, dtype=np.int64))
        for architecture in m.ARCHITECTURES:
            config = self.config(architecture)
            with m.context_bindings(config):
                model = rr.make_encoder(4, config, self.device)
                arrays = m.exported_arrays(rows, labels, config, model, self.device)
            self.assertEqual(arrays['embedding_raw'].shape, (48, 3))
            for key in ('embedding_unit', 'behavior', 'trial_id', 'time_id', 'source_id', 'valid_mask', 'position_xy'):
                self.assertEqual(len(arrays[key]), 48)
            np.testing.assert_array_equal(np.flatnonzero(arrays['valid_mask']), eligible_centers(rows))
            np.testing.assert_array_equal(arrays['position_xy'], rows.behavior)
            np.testing.assert_allclose(np.linalg.norm(arrays['embedding_unit'], axis=1), 1, rtol=1e-5)
            self.assertEqual(str(arrays['split']), 'test')

    def test_export_rejects_mismatched_label_cardinality_and_xy_width(self):
        rows = self.rows['test']
        config = self.config('cnn')
        with m.context_bindings(config):
            model = rr.make_encoder(4, config, self.device)
        for labels in (dict(position_xy=rows.behavior, direction=np.zeros(47)),
                       dict(position_xy=np.zeros((48, 3)), direction=np.zeros(48))):
            with self.assertRaises((ValueError, RuntimeError)):
                m.exported_arrays(rows, labels, config, model, self.device)


if __name__ == '__main__':
    unittest.main()
