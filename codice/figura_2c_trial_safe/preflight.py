"""Verify every prepared trial-safe window before any real Figure 2c fit."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from cebra.data import Offset
from cebra_trial_safe import TrialSafeDataset
import soft_trial_safe
from prepare_data import DEFAULT_OUTPUT, RATS, SPLITS, sha256, _check_prepared


def make_trials(rows: dict) -> list[dict]:
    trial_ids = rows['trial_id']
    starts = np.r_[0, np.flatnonzero(trial_ids[1:] != trial_ids[:-1]) + 1]
    stops = np.r_[starts[1:], len(trial_ids)]
    return [dict(neural=rows['neural'][start:stop],
                 position=rows['behavior'][start:stop, :1],
                 direction=rows['behavior'][start:stop, 1].astype(np.int64))
            for start, stop in zip(starts, stops)]


def check_one(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as f:
        rows = {key:f[key].copy() for key in f.files}
    x, trial, source, time = (rows[k] for k in ('neural', 'trial_id', 'source_id', 'time_id'))
    n = len(x)
    ds = TrialSafeDataset(x, (rows['behavior'][:, :1], rows['behavior'][:, 1].astype(np.int64)))
    ds.set_trials(trial)
    ds.offset = Offset(5, 5)
    all_centers = torch.arange(n)
    windows = ds.expand_index(all_centers).cpu().numpy()
    if windows.shape != (n, 10):
        raise AssertionError('One ten-bin window per center is required')
    if not np.all(trial[windows] == trial[:, None]):
        raise AssertionError('A window crosses a trial boundary')
    if not np.all(np.diff(windows, axis=1) == 1):
        raise AssertionError('Window is padded or contains repeated indices')
    if not np.all(np.diff(source[windows], axis=1) == 1):
        raise AssertionError('Window crosses a source-recording discontinuity')
    if not np.all(np.diff(time[windows], axis=1) == 1):
        raise AssertionError('Window crosses a time discontinuity')
    if not np.array_equal(ds.continuous_index.cpu().numpy(), rows['behavior'][:, :1]):
        raise AssertionError('Position target shifted away from original bin')
    if not np.array_equal(ds.discrete_index.cpu().numpy(), rows['behavior'][:, 1].astype(np.int64)):
        raise AssertionError('Direction target shifted away from original bin')
    starts = np.r_[0, np.flatnonzero(trial[1:] != trial[:-1]) + 1]
    stops = np.r_[starts[1:], n]
    edge_centers = np.unique(np.concatenate([np.r_[np.arange(a, a+5), np.arange(b-5, b)]
                                             for a,b in zip(starts, stops)]))
    consecutive_trial_ids = np.repeat(np.arange(len(starts)), stops-starts)
    official = ds.expand_index_in_trial(torch.as_tensor(edge_centers),
                                        torch.as_tensor(consecutive_trial_ids),
                                        torch.as_tensor(np.r_[starts, n])).cpu().numpy()
    if not np.array_equal(official, windows[edge_centers]):
        raise AssertionError('Trial-edge policy differs from CEBRA expand_index_in_trial')
    soft = soft_trial_safe.prepare_trials(make_trials(rows), condition='Position+Direction')
    expected = torch.as_tensor(x[windows])
    if not torch.equal(soft['windows'], expected):
        raise AssertionError('Soft and CEBRA use different neural windows')
    if not np.array_equal(soft['position'].numpy(), rows['behavior'][:, :1]):
        raise AssertionError('Soft position metadata moved with the window')
    if not np.array_equal(soft['direction'].numpy(), rows['behavior'][:, 1].astype(np.int64)):
        raise AssertionError('Soft direction metadata moved with the window')
    shifted = np.any(windows != np.arange(n)[:, None] + np.arange(-5, 5), axis=1)
    return dict(rows=n, trial_count=len(starts), windows=n, context_bins=10,
                shifted_centers=int(shifted.sum()), boundary_crossings=0,
                repeated_indices=0, source_discontinuities=0,
                official_edge_cases_compared=len(edge_centers),
                original_label_alignment=True, cebra_soft_windows_identical=True)


def main():
    output = DEFAULT_OUTPUT
    prepared = json.loads((output/'data'/'PREPARED.json').read_text(encoding='utf-8'))
    _check_prepared(output, prepared)
    results = {}
    for scope in ('split', 'full'):
        for rat in RATS:
            for split in (SPLITS if scope == 'split' else ('full',)):
                key = f'{scope}/{rat}/{split}'
                results[key] = check_one(output/'data'/scope/rat/f'{split}.npz')
                print(key, results[key], flush=True)
    info = dict(status='pass', rule='shift entire [-5,+4] window inside same trial; no padding',
                evaluation='all input bins preserved; metadata at original bin',
                window_scope='window extraction only; original CEBRA pair sampler unchanged',
                cebra_adapter_sha256=sha256(Path(__file__).with_name('cebra_trial_safe.py')),
                soft_adapter_sha256=sha256(Path(__file__).with_name('soft_trial_safe.py')),
                prepared_manifest_sha256=sha256(output/'data'/'PREPARED.json'),
                datasets=results)
    path = output/'data'/'PREFLIGHT.json'
    path.write_text(json.dumps(info, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
