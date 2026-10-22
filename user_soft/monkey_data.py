"""Deterministic frozen monkey input boundary; no behavioral-target choice.

Train/validation preparation opens only the requested source splits. TEST has
a separate explicit export API. No preprocessing, scaling or model selection
is estimated here. Global trial IDs come from the frozen exact-content map.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import joblib
import numpy as np
from .data import TrialRows, assert_disjoint, eligible_centers, sha256


@dataclass
class MonkeyRows:
    neural: np.ndarray
    position_xy: np.ndarray
    velocity_xy: np.ndarray
    direction: np.ndarray
    direction_active_passive: np.ndarray
    passive: np.ndarray
    trial_id: np.ndarray
    time_id: np.ndarray
    source_id: np.ndarray
    session_id: np.ndarray
    split: str
    population: str
    provenance: dict

    @property
    def valid_mask(self):
        """Strict trial-interior ten-bin support [-5,+4]; no padding."""
        mask = np.zeros(len(self.neural), dtype=bool)
        mask[eligible_centers(self, left=5, right=5)] = True
        return mask

    def as_trial_rows(self, behavior):
        """Caller explicitly supplies its scientific target; no concatenation here."""
        return TrialRows(self.neural, np.asarray(behavior), self.source_id,
                         self.trial_id, self.time_id, self.session_id, self.split)


def _paths(root):
    root = Path(root).resolve()
    return (root / 'data/monkey_reaching_preload_smth_40',
            root / 'results/monkey_figure3_v1/stage00_data/DATA_AND_SPLIT_PROVENANCE.json')


def _integer_ids(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or not (np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_):
        raise ValueError(f'Invalid {name}')
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise ValueError(f'Noninteger {name}')
    return array.astype(np.int64)


def _load_split(root, session, split):
    if session not in ('all', 'active', 'passive') or split not in ('train', 'valid', 'test'):
        raise ValueError('Explicit frozen population and split required')
    data_root, manifest = _paths(root)
    record = json.loads(manifest.read_text(encoding='utf-8'))
    if (record['trial_length_bins'], record['bin_ms'], record['neural_channels']) != (600, 1, 65):
        raise ValueError('Frozen monkey dimensions/sampling mismatch')
    mapping = (record['all_trial_indices_by_split'] if session == 'all'
               else record['session_trial_indices_by_split'][session])
    ids = _integer_ids(mapping[split], 'global trial IDs')
    if len(np.unique(ids)) != len(ids) or np.any(ids < 0):
        raise ValueError('Duplicate/negative global trial IDs')
    # Metadata partitions only: no excluded source container is opened.
    for other in ('train', 'valid', 'test'):
        if other != split and np.intersect1d(ids, _integer_ids(mapping[other], 'partition IDs')).size:
            raise ValueError('Frozen trial partition overlap')
    source = data_root / f'{session}_{split}.jl'
    expected_hash = record['source_file_hashes'][source.name]['sha256']
    actual_hash = sha256(source)
    if actual_hash != expected_hash:
        raise ValueError(f'Frozen source hash mismatch: {source.name}')
    data = joblib.load(source)
    n_trials, trial_length = int(data['num_trials']), int(data['trial_len'])
    if trial_length != 600 or n_trials != len(ids) or n_trials != record['trial_counts'][source.name]:
        raise ValueError('Frozen trial count/length mismatch')
    n = n_trials * 600
    arrays = {name: np.asarray(data[key]) for name, key in
              (('neural', 'spikes'), ('position_xy', 'pos'), ('velocity_xy', 'vel'))}
    for name, width in (('neural', 65), ('position_xy', 2), ('velocity_xy', 2)):
        if arrays[name].shape != (n, width) or not np.isfinite(arrays[name]).all():
            raise ValueError(f'Invalid {name} dimensions/values')
    labels = {}
    for name, key, maximum in (('direction', 'movement_dir', 7),
                               ('direction_active_passive', 'movement_dir_actpas', 15),
                               ('passive', 'passive', 1)):
        value = _integer_ids(data[key], key)
        if value.shape != (n_trials,) or np.any((value < 0) | (value > maximum)):
            raise ValueError(f'Invalid {key} labels')
        labels[name] = np.repeat(value, 600)
    if session != 'all' and not np.all(labels['passive'] == (session == 'passive')):
        raise ValueError('Population/passive label mismatch')
    time_id = np.tile(np.arange(600, dtype=np.int64), n_trials)
    trial_id = np.repeat(ids, 600)
    provenance = dict(source_path=str(source), source_sha256=actual_hash,
                      split_provenance_path=str(manifest), split_provenance_sha256=sha256(manifest),
                      adapter_sha256=sha256(Path(__file__)), split=split, population=session,
                      original_global_trial_ids=ids.tolist(), rows=n, trials=n_trials,
                      bin_ms=1, trial_length_bins=600, neural_channels=65,
                      preprocessing_estimated=False, behavior_target_defined=False,
                      source_containers_opened=[source.name], test_container_opened=split == 'test')
    return MonkeyRows(**arrays, **labels, trial_id=trial_id, time_id=time_id,
                      source_id=trial_id * 600 + time_id,
                      session_id=np.zeros(n, dtype=np.int64), split=split,
                      population=session, provenance=provenance)


def load_train_validation(root, session):
    """Load frozen TRAIN/VALID only; TEST source and full container stay closed."""
    rows = {split: _load_split(root, session, split) for split in ('train', 'valid')}
    assert_disjoint(rows)
    return rows


def load_test_for_export(root, session):
    """Explicit downstream TEST boundary; never called by TRAIN/VALID preparation."""
    return _load_split(root, session, 'test')
