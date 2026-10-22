"""Validation-selected probes on frozen monkey embeddings, with joint XY targets.

No encoder fit and no behavioral-target construction. TEST archive fields are
unopened until Ridge alpha is fixed for BOTH raw and unit representations.
Position units remain native to the official position_xy source (no conversion).
"""
from collections.abc import Mapping
from pathlib import Path
import hashlib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RIDGE_ALPHA_GRID = (0.1, 1.0, 10.0)
KNN_NEIGHBORS = 36
REPRESENTATIONS = ('embedding_raw', 'embedding_unit')


class LazyArchive(Mapping):
    """NPZ fields loaded on demand; construction itself reads no archive."""
    def __init__(self, path):
        self.path = Path(path)

    def __getitem__(self, key):
        with np.load(self.path, allow_pickle=False) as archive:
            return archive[key].copy()

    def __iter__(self):
        with np.load(self.path, allow_pickle=False) as archive:
            return iter(tuple(archive.files))

    def __len__(self):
        with np.load(self.path, allow_pickle=False) as archive:
            return len(archive.files)


def _support(data):
    ids = np.asarray(data['source_id'])
    mask = np.asarray(data['valid_mask'])
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique integer source_id required')
    if mask.shape != ids.shape or mask.dtype != np.bool_ or not mask.any():
        raise ValueError('Explicit nonempty Boolean valid_mask required')
    for key in ('trial_id', 'time_id'):
        value = np.asarray(data[key])
        if value.shape != ids.shape or not np.issubdtype(value.dtype, np.integer) or np.any(value < 0):
            raise ValueError(f'Invalid {key} metadata')
    coordinates = np.stack((data['trial_id'], data['time_id']), axis=1)
    if len(np.unique(coordinates, axis=0)) != len(ids):
        raise ValueError('Repeated trial/time coordinates')
    payload = np.ascontiguousarray(np.stack((ids[mask], data['trial_id'][mask], data['time_id'][mask]), axis=1), dtype=np.int64)
    return {'ids': ids, 'mask': mask, 'trials': np.asarray(data['trial_id']),
            'support_sha256': hashlib.sha256(payload.tobytes()).hexdigest()}


def _disjoint(supports):
    names = list(supports)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            if np.intersect1d(supports[first]['ids'], supports[second]['ids']).size:
                raise ValueError('Train/validation/test source rows overlap')
            if np.intersect1d(supports[first]['trials'], supports[second]['trials']).size:
                raise ValueError('Train/validation/test trial identities overlap')


def _rows(data, representation, support):
    raw = np.asarray(data['embedding_raw'])
    mask = support['mask']
    n = len(mask)
    if raw.ndim != 2 or raw.shape[0] != n or raw.shape[1] < 1:
        raise ValueError('Embedding must be N x D')
    raw = raw[mask].astype(np.float64, copy=False)
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite valid embedding')
    x = raw
    if representation == 'embedding_unit':
        stored = np.asarray(data['embedding_unit'])
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        if stored.shape != (n, raw.shape[1]) or np.any(norms <= 0):
            raise ValueError('Invalid unit embedding shape or zero norm')
        x = stored[mask].astype(np.float64, copy=False)
        if not np.isfinite(x).all() or not np.allclose(x, raw / norms, rtol=1e-5, atol=1e-6):
            raise ValueError('Unit embedding differs from normalized raw')
    position = np.asarray(data['position_xy'])
    direction, passive = np.asarray(data['direction']), np.asarray(data['passive'])
    if position.shape != (n, 2) or direction.shape != (n,) or passive.shape != (n,):
        raise ValueError('Joint XY and direction/AP cardinalities required')
    y = position[mask].astype(np.float64, copy=False)
    d = direction[mask]
    if not np.isfinite(y).all() or not np.isin(d, np.arange(8)).all() or not np.isin(passive, (0, 1)).all():
        raise ValueError('Invalid position/direction/AP values')
    return x, y, d.astype(np.int64)


def _knn_predict(neighbors, x, train_xy, train_direction):
    """Same uniform kNN36/cosine predictions; share lookup and bound query memory."""
    chunks = [neighbors.kneighbors(x[start:start + 256], return_distance=False)
              for start in range(0, len(x), 256)]
    indices = np.concatenate(chunks, axis=0)
    position = train_xy[indices].mean(axis=1)
    classes = np.unique(train_direction)
    votes = np.stack([(train_direction[indices] == label).sum(axis=1) for label in classes], axis=1)
    return position, classes[votes.argmax(axis=1)]


def score_arrays(arrays, *, ridge_alphas=RIDGE_ALPHA_GRID):
    alphas = tuple(float(a) for a in ridge_alphas)
    if not alphas or len(set(alphas)) != len(alphas) or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError('Finite positive distinct Ridge alpha grid required')
    supports = {s: _support(arrays[s]) for s in ('train', 'valid')}
    _disjoint(supports)
    fitted, selections = {}, {}
    for representation in REPRESENTATIONS:
        x_train, y_train, d_train = _rows(arrays['train'], representation, supports['train'])
        x_valid, y_valid, _ = _rows(arrays['valid'], representation, supports['valid'])
        if x_train.shape[1] != x_valid.shape[1] or len(x_train) < KNN_NEIGHBORS or len(x_valid) < 2:
            raise ValueError('Compatible dimension,36 TRAIN and2 VALID rows required')
        best_score, best_model, best_alpha, candidates = -np.inf, None, None, []
        for alpha in alphas:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x_train, y_train)
            score = float(r2_score(y_valid, model.predict(x_valid), multioutput='uniform_average'))
            if not np.isfinite(score):
                raise ValueError('Nonfinite validation XY R2')
            candidates.append({'alpha': alpha, 'validation_position_r2': score})
            if score > best_score:
                best_score, best_model, best_alpha = score, model, alpha
        selections[representation] = dict(selected_alpha=best_alpha, validation_candidates=candidates,
                                          tie_break='first fixed-grid entry', selection_split='valid')
        neighbors = NearestNeighbors(n_neighbors=KNN_NEIGHBORS, metric='cosine', algorithm='brute', n_jobs=1).fit(x_train)
        fitted[representation] = (neighbors, y_train, d_train, best_model, x_train.shape[1])
    # TEST is first dereferenced only after BOTH alpha decisions are immutable.
    supports['test'] = _support(arrays['test'])
    _disjoint(supports)
    records, predictions = [], {}
    for representation in REPRESENTATIONS:
        neighbors, train_xy, train_d, ridge, dimension = fitted[representation]
        for split in ('valid', 'test'):
            support = supports[split]
            x, y, d = _rows(arrays[split], representation, support)
            if x.shape[1] != dimension or len(x) < 2:
                raise ValueError('Evaluation dimension mismatch or fewer than2 rows')
            knn_xy, knn_direction = _knn_predict(neighbors, x, train_xy, train_d)
            prefix = f'{representation}_{split}_'
            predictions.update({prefix + 'position_true': y, prefix + 'direction_true': d,
                                prefix + 'knn_direction': knn_direction,
                                prefix + 'source_id': support['ids'][support['mask']],
                                prefix + 'trial_id': support['trials'][support['mask']],
                                prefix + 'time_id': np.asarray(arrays[split]['time_id'])[support['mask']]})
            for decoder, predicted in (('knn', knn_xy), ('ridge', ridge.predict(x))):
                predictions[prefix + decoder + '_position'] = predicted
                records.append(dict(representation=representation, split=split, decoder=decoder,
                    n_samples=len(x), n_trials=len(np.unique(support['trials'][support['mask']])),
                    position_r2=float(r2_score(y, predicted, multioutput='uniform_average')),
                    position_mae=float(np.abs(predicted - y).mean()),
                    position_coordinate_mae=np.abs(predicted - y).mean(axis=0).tolist(),
                    position_units='native_official_position_xy_no_conversion',
                    position_r2_aggregation='uniform_average_over_XY',
                    direction_accuracy=float(accuracy_score(d, knn_direction)) if decoder == 'knn' else None,
                    direction_balanced_accuracy=float(balanced_accuracy_score(d, knn_direction)) if decoder == 'knn' else None,
                    selected_alpha=selections[representation]['selected_alpha'] if decoder == 'ridge' else None,
                    knn_neighbors=KNN_NEIGHBORS if decoder == 'knn' else None,
                    support_sha256=support['support_sha256']))
    return dict(records=records, predictions=predictions, selections=selections,
                provenance=dict(encoder_fitted=False, optimizer_updates=0,
                    scaler_fit_split='train_only', ridge_fit_split='train_only_no_refit_after_selection',
                    test_access='after_both_raw_unit_alpha_selections', ridge_alpha_grid=list(alphas),
                    knn_metric='cosine', knn_neighbors=KNN_NEIGHBORS,
                    source_hashes_required_from_caller=True,
                    support={s: v['support_sha256'] for s, v in supports.items()}))
