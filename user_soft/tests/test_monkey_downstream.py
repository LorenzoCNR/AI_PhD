"""CPU tests of monkey XY probes and strict selection/test ordering."""
import copy
import unittest
from collections.abc import Mapping
from unittest import mock
import numpy as np
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from user_soft import monkey_downstream as md


def fixture():
    rng = np.random.default_rng(42)
    result = {}
    for i, split in enumerate(('train', 'valid', 'test')):
        n = 72 if split == 'train' else 24
        z = rng.normal(size=(n, 3)) + .1
        y = np.stack((2 * z[:, 0] - z[:, 1], z[:, 2] + 3 * z[:, 1]), axis=1)
        result[split] = dict(embedding_raw=z, embedding_unit=z / np.linalg.norm(z, axis=1, keepdims=True),
            position_xy=y, direction=np.arange(n) % 8, passive=np.zeros(n, dtype=bool),
            trial_id=np.full(n, i + 1, dtype=np.int64), time_id=np.arange(n),
            source_id=np.arange(n) + i * 1000, valid_mask=np.ones(n, dtype=bool))
        result[split]['valid_mask'][0] = False
    return result


class GuardedTest(Mapping):
    def __init__(self, values, selected):
        self.values, self.selected = values, selected

    def __getitem__(self, key):
        if self.selected() < 6:
            raise AssertionError('TEST opened before both3-alpha searches completed')
        return self.values[key]

    def __iter__(self):
        raise AssertionError('Do not eagerly enumerate TEST')

    def __len__(self):
        return len(self.values)


class MonkeyDownstreamTests(unittest.TestCase):
    def test_joint_xy_ridge_and_raw_unit_separate(self):
        result = md.score_arrays(fixture())
        self.assertEqual(len(result['records']), 8)
        self.assertEqual(set(result['selections']), set(md.REPRESENTATIONS))
        for row in result['records']:
            prefix = f"{row['representation']}_{row['split']}_"
            y, predicted = result['predictions'][prefix + 'position_true'], result['predictions'][prefix + row['decoder'] + '_position']
            self.assertEqual(y.shape[1], 2)
            self.assertAlmostEqual(row['position_r2'], r2_score(y, predicted, multioutput='uniform_average'))
            self.assertAlmostEqual(row['position_mae'], np.abs(y - predicted).mean())
            self.assertEqual(row['position_units'], 'native_official_position_xy_no_conversion')
        raw = next(r for r in result['records'] if r['representation'] == 'embedding_raw' and r['split'] == 'test' and r['decoder'] == 'ridge')
        self.assertGreater(raw['position_r2'], .99)

    def test_test_fields_unopened_until_both_alpha_selections(self):
        arrays = fixture()
        calls = []
        original = md.r2_score
        def tracked(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)
        arrays['test'] = GuardedTest(arrays['test'], lambda: len(calls))
        with mock.patch.object(md, 'r2_score', side_effect=tracked):
            result = md.score_arrays(arrays)
        self.assertEqual(result['provenance']['optimizer_updates'], 0)

    def test_knn_matches_sklearn_uniform_cosine(self):
        arrays = fixture()
        result = md.score_arrays(arrays)
        for rep in md.REPRESENTATIONS:
            train, test = arrays['train'], arrays['test']
            x, y, d = train[rep][train['valid_mask']], train['position_xy'][train['valid_mask']], train['direction'][train['valid_mask']]
            query = test[rep][test['valid_mask']]
            p = KNeighborsRegressor(n_neighbors=36, metric='cosine').fit(x, y).predict(query)
            c = KNeighborsClassifier(n_neighbors=36, metric='cosine').fit(x, d).predict(query)
            np.testing.assert_allclose(p, result['predictions'][rep + '_test_knn_position'])
            np.testing.assert_array_equal(c, result['predictions'][rep + '_test_knn_direction'])

    def test_disjoint_source_and_trials_required(self):
        for field in ('source_id', 'trial_id'):
            arrays = fixture()
            arrays['test'][field][1] = arrays['train'][field][1]
            with self.assertRaises(ValueError):
                md.score_arrays(arrays)

    def test_bad_unit_and_shape_rejected(self):
        arrays = fixture()
        arrays['valid']['embedding_unit'][1] *= 2
        with self.assertRaisesRegex(ValueError, 'normalized raw'):
            md.score_arrays(arrays)
        arrays = fixture()
        arrays['valid']['position_xy'] = arrays['valid']['position_xy'][:, :1]
        with self.assertRaisesRegex(ValueError, 'XY'):
            md.score_arrays(arrays)

    def test_support_identity_and_reproducibility(self):
        arrays = fixture()
        first, second = md.score_arrays(arrays), md.score_arrays(copy.deepcopy(arrays))
        self.assertEqual(first['records'], second['records'])
        for split in ('valid', 'test'):
            np.testing.assert_array_equal(first['predictions'][f'embedding_raw_{split}_source_id'], arrays[split]['source_id'][arrays[split]['valid_mask']])
            np.testing.assert_array_equal(first['predictions'][f'embedding_raw_{split}_source_id'], first['predictions'][f'embedding_unit_{split}_source_id'])
        for key in first['predictions']:
            np.testing.assert_array_equal(first['predictions'][key], second['predictions'][key])

    def test_scaler_train_only_and_no_test_effect_on_selection(self):
        arrays = fixture()
        first = md.score_arrays(arrays)
        changed = copy.deepcopy(arrays)
        changed['test']['position_xy'] += 100000
        self.assertEqual(first['selections'], md.score_arrays(changed)['selections'])
        original = md.StandardScaler.fit
        seen = []
        def track(scaler, x, *args, **kwargs):
            seen.append(np.array(x))
            return original(scaler, x, *args, **kwargs)
        with mock.patch.object(md.StandardScaler, 'fit', track):
            md.score_arrays(arrays)
        self.assertEqual(len(seen), 6)
        for ix, value in enumerate(seen):
            rep = md.REPRESENTATIONS[ix // 3]
            np.testing.assert_array_equal(value, arrays['train'][rep][arrays['train']['valid_mask']])


if __name__ == '__main__':
    unittest.main()
