"""Exact official rat split identities and explicit temporal context construction."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def config_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@dataclass
class TrialRows:
    neural: np.ndarray
    behavior: np.ndarray
    source_id: np.ndarray
    trial_id: np.ndarray
    time_id: np.ndarray
    session_id: np.ndarray
    split: str

    def __post_init__(self):
        for key in ('neural', 'behavior', 'source_id', 'trial_id', 'time_id', 'session_id'):
            setattr(self, key, np.asarray(getattr(self, key)))
        n = len(self.neural)
        if n < 10 or self.neural.ndim != 2 or self.behavior.ndim != 2:
            raise ValueError('Expected at least 10 rows, neural/behavior N x features')
        if any(len(getattr(self, k)) != n for k in ('behavior', 'source_id', 'trial_id', 'time_id', 'session_id')):
            raise ValueError('Metadata cardinality mismatch')
        if not np.isfinite(self.neural).all() or not np.isfinite(self.behavior).all():
            raise ValueError('Nonfinite neural/behavior data')
        if self.split not in ('train', 'valid', 'test'):
            raise ValueError('Explicit split required')
        if len(np.unique(self.source_id)) != n:
            raise ValueError('Duplicate source IDs within one session split')
        if not np.isfinite(self.time_id).all() or np.any(self.time_id < 0):
            raise ValueError('Invalid trial-relative times')

    def metadata(self, centers=None):
        centers = np.arange(len(self.neural)) if centers is None else np.asarray(centers)
        return {k: torch.as_tensor(getattr(self, k)[centers]) for k in
                ('behavior', 'time_id', 'trial_id', 'session_id', 'source_id')}


def assert_disjoint(splits):
    keys = list(splits)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            first, second = splits[a], splits[b]
            ids1 = set(zip(first.session_id.tolist(), first.source_id.tolist()))
            ids2 = set(zip(second.session_id.tolist(), second.source_id.tolist()))
            trials1 = set(zip(first.session_id.tolist(), first.trial_id.tolist()))
            trials2 = set(zip(second.session_id.tolist(), second.trial_id.tolist()))
            if ids1 & ids2 or trials1 & trials2:
                raise ValueError(f'Split row/trial overlap: {a}/{b}')


def eligible_centers(rows, left=5, right=5):
    """Return indices with the complete [-left,right-1] support in one trial."""
    n = len(rows.neural)
    centers = np.arange(left, n - right + 1)
    if not len(centers):
        return centers
    offsets = np.arange(-left, right)
    indices = centers[:, None] + offsets
    ok = ((rows.trial_id[indices] == rows.trial_id[centers, None]).all(1) &
          (rows.session_id[indices] == rows.session_id[centers, None]).all(1) &
          (rows.time_id[indices] == rows.time_id[centers, None]+offsets).all(1))
    return centers[ok]


def windows(rows, centers, left=5, right=5):
    centers = np.asarray(centers, dtype=np.int64)
    if not np.isin(centers, eligible_centers(rows, left, right)).all():
        raise ValueError('Trial-safe context crosses a trial/session boundary')
    indices = centers[:, None] + np.arange(-left, right)[None, :]
    return torch.as_tensor(rows.neural[indices].transpose(0, 2, 1), dtype=torch.float32)


def legacy_windows(rows, centers, left=5, right=5):
    """Exact CEBRA expand_index clamp rule on the split concatenation."""
    centers = np.asarray(centers, dtype=np.int64)
    if np.any(centers < 0) or np.any(centers >= len(rows.neural)):
        raise ValueError('Center outside split')
    indices = np.clip(centers, left, len(rows.neural)-right)[:, None] + np.arange(-left, right)
    return torch.as_tensor(rows.neural[indices].transpose(0, 2, 1), dtype=torch.float32)


def train_batches(rows, batch_size, seed):
    if rows.split != 'train':
        raise ValueError('Training sampler accepts train rows only')
    if batch_size < 2:
        raise ValueError('Soft training needs at least two candidates')
    generator = torch.Generator().manual_seed(seed)
    while True:
        yield torch.randint(len(rows.neural), (batch_size,), generator=generator)


def load_official_splits(root, rat, fold):
    from cebra.datasets.hippocampus import SingleRatTrialSplitDataset
    session = ('achilles', 'buddy', 'cicero', 'gatsby').index(rat)
    complete = SingleRatTrialSplitDataset(name=rat, split_no=fold, split='all', root=Path(root)/'data')
    trial_lookup = {(s.start, s.stop): i for i, s in enumerate(complete.selected_indices)}
    splits = {}
    for split in ('train', 'valid', 'test'):
        dataset = SingleRatTrialSplitDataset(name=rat, split_no=fold, split=split, root=Path(root)/'data')
        slices = dataset.selected_indices
        source_id = np.concatenate([np.arange(s.start, s.stop) for s in slices])
        trial_id = np.concatenate([np.full(s.stop-s.start, trial_lookup[(s.start,s.stop)]) for s in slices])
        time_id = np.concatenate([np.arange(s.stop-s.start) for s in slices])
        splits[split] = TrialRows(dataset.neural.numpy(), dataset.continuous_index.numpy(),
                                 source_id, trial_id, time_id, np.full(len(source_id), session), split)
        if not np.array_equal(complete.neural.numpy()[source_id], splits[split].neural):
            raise ValueError('Prepared neural rows differ from exact official source')
        if not np.array_equal(complete.continuous_index.numpy()[source_id], splits[split].behavior):
            raise ValueError('Prepared behavior differs from exact official source')
    assert_disjoint(splits)
    return splits


def save_rows(path, rows):
    mask = np.zeros(len(rows.neural), dtype=bool)
    mask[eligible_centers(rows)] = True
    np.savez_compressed(path, **{k: getattr(rows,k) for k in
        ('neural','behavior','source_id','trial_id','time_id','session_id','split')}, valid_mask=mask)


def load_rows(path):
    with np.load(path, allow_pickle=False) as d:
        return TrialRows(**{k: d[k].copy() for k in
            ('neural','behavior','source_id','trial_id','time_id','session_id')}, split=str(d['split']))
