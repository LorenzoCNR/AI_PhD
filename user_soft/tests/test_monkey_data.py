"""CPU-only qualification of the frozen monkey input boundary; no fitting."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import joblib
import numpy as np
from user_soft import monkey_data as md
from user_soft.data import sha256, windows


class MonkeyDataQualification(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data_root, self.manifest = md._paths(self.root)
        self.data_root.mkdir(parents=True)
        self.manifest.parent.mkdir(parents=True)
        record = dict(trial_length_bins=600, bin_ms=1, neural_channels=65,
                      all_trial_indices_by_split={'train': [2, 7], 'valid': [9], 'test': [11]},
                      session_trial_indices_by_split={}, source_file_hashes={}, trial_counts={})
        self.sources = {}
        for split, ids in record['all_trial_indices_by_split'].items():
            n = len(ids)
            rng = np.random.default_rng(42 + n)
            value = dict(num_trials=n, trial_len=600,
                         spikes=rng.normal(size=(n * 600, 65)).astype('float32'),
                         pos=rng.normal(size=(n * 600, 2)).astype('float32'),
                         vel=rng.normal(size=(n * 600, 2)).astype('float32'),
                         movement_dir=np.arange(n) % 8,
                         movement_dir_actpas=np.arange(n) % 8,
                         passive=np.zeros(n, dtype=bool))
            p = self.data_root / f'all_{split}.jl'
            joblib.dump(value, p)
            self.sources[split] = value
            record['source_file_hashes'][p.name] = {'sha256': sha256(p)}
            record['trial_counts'][p.name] = n
        self.record = record
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.record), encoding='utf-8')

    def test_equivalence_ids_validity_and_explicit_target(self):
        splits = md.load_train_validation(self.root, 'all')
        for split, rows in splits.items():
            for output, source in [('neural', 'spikes'), ('position_xy', 'pos'), ('velocity_xy', 'vel')]:
                np.testing.assert_array_equal(getattr(rows, output), self.sources[split][source])
            np.testing.assert_array_equal(rows.source_id, rows.trial_id * 600 + rows.time_id)
            expected = (rows.time_id >= 5) & (rows.time_id <= 595)
            np.testing.assert_array_equal(rows.valid_mask, expected)
            self.assertFalse(hasattr(rows, 'behavior'))
            compatible = rows.as_trial_rows(rows.position_xy)
            ix = np.flatnonzero(rows.valid_mask)[:2]
            self.assertEqual(tuple(windows(compatible, ix).shape), (2, 65, 10))
            self.assertFalse(rows.provenance['preprocessing_estimated'])
        self.assertFalse(set(splits['train'].trial_id) & set(splits['valid'].trial_id))

    def test_train_validation_never_opens_test_or_full_container(self):
        original_load = joblib.load
        opened = []
        def guarded(path, *args, **kwargs):
            opened.append(Path(path).name)
            self.assertNotIn('_test', str(path))
            self.assertNotIn('_all.jl', str(path))
            return original_load(path, *args, **kwargs)
        with mock.patch.object(md.joblib, 'load', side_effect=guarded):
            md.load_train_validation(self.root, 'all')
        self.assertEqual(opened, ['all_train.jl', 'all_valid.jl'])

    def test_test_requires_explicit_export_api(self):
        rows = md.load_test_for_export(self.root, 'all')
        self.assertEqual(rows.split, 'test')
        self.assertTrue(rows.provenance['test_container_opened'])
        self.assertEqual(set(rows.trial_id), {11})

    def test_hash_tamper_rejected_before_deserialization(self):
        with (self.data_root / 'all_train.jl').open('ab') as stream:
            stream.write(b'tamper')
        with mock.patch.object(md.joblib, 'load') as load:
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                md.load_train_validation(self.root, 'all')
            load.assert_not_called()

    def test_partition_overlap_rejected(self):
        self.record['all_trial_indices_by_split']['valid'] = [2]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'partition overlap'):
            md.load_train_validation(self.root, 'all')

    def test_nonfinite_raw_rejected(self):
        source = copy.deepcopy(self.sources['train'])
        source['pos'][0, 0] = np.nan
        with mock.patch.object(md.joblib, 'load', return_value=source):
            with self.assertRaisesRegex(ValueError, 'position_xy'):
                md.load_train_validation(self.root, 'all')

    def test_sampling_contract_rejected(self):
        self.record['bin_ms'] = 25
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'sampling mismatch'):
            md.load_train_validation(self.root, 'all')


class ActualFrozenMonkeyQualification(unittest.TestCase):
    def test_real_train_valid_exact_equivalence_and_firewall(self):
        root = Path(__file__).resolve().parents[2]
        data_root, manifest = md._paths(root)
        self.assertTrue(manifest.is_file(), 'Frozen real split provenance required')
        mapping = json.loads(manifest.read_text(encoding='utf-8'))
        original_load = joblib.load
        for population in ('all', 'active', 'passive'):
            opened = []
            def guarded(path, *args, **kwargs):
                opened.append(Path(path).name)
                self.assertNotIn('_test', str(path))
                self.assertNotIn('_all.jl', str(path))
                return original_load(path, *args, **kwargs)
            with mock.patch.object(md.joblib, 'load', side_effect=guarded):
                rows = md.load_train_validation(root, population)
            self.assertEqual(opened, [f'{population}_train.jl', f'{population}_valid.jl'])
            for split in ('train', 'valid'):
                expected = original_load(data_root / f'{population}_{split}.jl')
                for key, raw_key in [('neural', 'spikes'), ('position_xy', 'pos'), ('velocity_xy', 'vel')]:
                    np.testing.assert_array_equal(getattr(rows[split], key), expected[raw_key])
                for key, raw_key in [('direction', 'movement_dir'), ('direction_active_passive', 'movement_dir_actpas'), ('passive', 'passive')]:
                    np.testing.assert_array_equal(getattr(rows[split], key), np.repeat(expected[raw_key], 600))
                ids = (mapping['all_trial_indices_by_split'][split] if population == 'all'
                       else mapping['session_trial_indices_by_split'][population][split])
                np.testing.assert_array_equal(rows[split].trial_id, np.repeat(ids, 600))
                np.testing.assert_array_equal(rows[split].time_id, np.tile(np.arange(600), len(ids)))
                self.assertEqual(rows[split].neural.shape[1], 65)
                self.assertEqual(rows[split].provenance['bin_ms'], 1)
                self.assertEqual(int(rows[split].valid_mask.sum()), int(expected['num_trials']) * 591)
                self.assertFalse(rows[split].provenance['test_container_opened'])


if __name__ == '__main__':
    unittest.main()
