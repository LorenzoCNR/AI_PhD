"""Freeze the Figure 2c split, full-recording rows and shared permutations.

This module only prepares the small, explicit inputs for the new campaign.
It does not run an encoder or modify the previous campaign.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
OLD = PROJECT / 'results' / 'figure2c_shuffle_diagnostic_2026-11-02'
DEFAULT_OUTPUT = PROJECT / 'results' / 'figura_2_2026-10-28' / '2c'
RATS = ('achilles', 'gatsby')
SPLITS = ('train', 'valid', 'test')
SHUFFLED = ('position_shuffled', 'direction_shuffled', 'position_direction_shuffled')
CONDITIONS = ('position', 'direction', 'position_direction') + SHUFFLED
MODELS = ('cebra_cnn', 'soft_stable_cnn', 'soft_stable_transformer')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_rows(rows: dict) -> None:
    """Validate contiguous, complete trials; each center must admit 10 bins."""
    n = len(rows['source_id'])
    if rows['neural'].ndim != 2 or rows['behavior'].shape != (n, 3):
        raise ValueError('Unexpected neural or behavioral dimensions')
    if any(len(rows[k]) != n for k in ('neural', 'trial_id', 'time_id')):
        raise ValueError('Metadata and neural rows are misaligned')
    if len(np.unique(rows['source_id'])) != n:
        raise ValueError('Repeated source rows')
    if not np.isfinite(rows['neural']).all() or not np.isfinite(rows['behavior']).all():
        raise ValueError('Nonfinite values')
    trial = np.asarray(rows['trial_id'])
    source = np.asarray(rows['source_id'])
    time = np.asarray(rows['time_id'])
    starts = np.r_[0, np.flatnonzero(trial[1:] != trial[:-1]) + 1]
    stops = np.r_[starts[1:], n]
    if len(np.unique(trial[starts])) != len(starts):
        raise ValueError('A trial appears in more than one block')
    for start, stop in zip(starts, stops):
        if stop - start < 10:
            raise ValueError(f'Trial {trial[start]} has fewer than ten bins')
        if not np.array_equal(np.diff(time[start:stop]), np.ones(stop-start-1, dtype=time.dtype)):
            raise ValueError(f'Nonconsecutive time within trial {trial[start]}')
        if not np.array_equal(np.diff(source[start:stop]), np.ones(stop-start-1, dtype=source.dtype)):
            raise ValueError(f'Nonconsecutive source rows within trial {trial[start]}')


def _read_legacy_rows(rat: str, split: str, raw: dict) -> dict:
    path = OLD / 'fits' / rat / 'soft_stable_cnn' / 'position' / f'{split}_embedding.npz'
    with np.load(path, allow_pickle=False) as saved:
        rows = {k: saved[k].copy() for k in ('source_id', 'trial_id', 'time_id', 'behavior')}
    ids = rows['source_id'].astype(np.int64)
    rows['neural'] = np.asarray(raw['spikes'], dtype=np.float32)[ids]
    if not np.array_equal(rows['behavior'], np.asarray(raw['position'], dtype=np.float32)[ids]):
        raise ValueError(f'Behavior mismatch for {rat}/{split}')
    validate_rows(rows)
    return rows


def _write_npz(path: Path, rows: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.partial.npz')
    np.savez_compressed(tmp, **rows)
    tmp.replace(path)


def _read_permutation(path: Path, ids: np.ndarray) -> np.ndarray:
    with path.open(newline='', encoding='utf-8') as f:
        data = list(csv.DictReader(f))
    recipients = np.array([int(x['recipient_source_id']) for x in data], dtype=np.int64)
    donors = np.array([int(x['donor_source_id']) for x in data], dtype=np.int64)
    if not np.array_equal(recipients, ids) or set(donors) != set(ids):
        raise ValueError(f'Permutation IDs do not match split: {path}')
    lookup = {int(source_id): i for i, source_id in enumerate(ids)}
    permutation = np.array([lookup[int(source_id)] for source_id in donors], dtype=np.int64)
    if len(np.unique(permutation)) != len(ids):
        raise ValueError(f'Permutation is not bijective: {path}')
    return permutation


def _write_permutation(path: Path, ids: np.ndarray, permutation: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(('recipient_source_id', 'donor_source_id'))
        writer.writerows(zip(ids.tolist(), ids[permutation].tolist()))


def _check_prepared(output: Path, manifest: dict) -> None:
    for rel, expected in manifest['artifacts'].items():
        path = output / rel
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f'Prepared input missing or changed: {path}')


def prepare(output: Path = DEFAULT_OUTPUT) -> dict:
    output = Path(output).resolve()
    marker = output / 'data' / 'PREPARED.json'
    if marker.exists():
        manifest = json.loads(marker.read_text(encoding='utf-8'))
        _check_prepared(output, manifest)
        return manifest
    if not (OLD / 'permutations' / 'manifest.json').is_file():
        raise FileNotFoundError('Previous split/permutation provenance is unavailable')
    old_perm = json.loads((OLD / 'permutations' / 'manifest.json').read_text(encoding='utf-8'))
    manifest = dict(scope='Figure 2c trial-safe split and full recording',
                    policy='10 bins [-5,+4]; shift window within trial; target remains original bin',
                    model_seed=42, split_fold=0, rats={}, permutations={}, artifacts={},
                    input_provenance={})
    for rat in RATS:
        raw_path = PROJECT / 'data' / 'rat_hippocampus' / f'{rat}.jl'
        raw = joblib.load(raw_path)
        split_rows = {s: _read_legacy_rows(rat, s, raw) for s in SPLITS}
        sets = {s: set(split_rows[s]['source_id'].tolist()) for s in SPLITS}
        trials = {s: set(split_rows[s]['trial_id'].tolist()) for s in SPLITS}
        for i, a in enumerate(SPLITS):
            for b in SPLITS[i+1:]:
                if sets[a] & sets[b] or trials[a] & trials[b]:
                    raise ValueError(f'Leak between {rat}/{a} and {b}')
        ids = np.concatenate([split_rows[s]['source_id'] for s in SPLITS])
        if not np.array_equal(np.sort(ids), np.arange(len(raw['spikes']))):
            raise ValueError(f'{rat}: splits do not cover the recording exactly')
        for model in MODELS:
            for condition in CONDITIONS:
                for split in SPLITS:
                    path = OLD / 'fits' / rat / model / condition / f'{split}_embedding.npz'
                    with np.load(path, allow_pickle=False) as previous:
                        for k in ('source_id', 'trial_id', 'time_id'):
                            if not np.array_equal(previous[k], split_rows[split][k]):
                                raise ValueError(f'Legacy identity mismatch: {path}, {k}')
        combined = {k: np.concatenate([split_rows[s][k] for s in SPLITS], axis=0)
                    for k in split_rows['train']}
        order = np.argsort(combined['source_id'])
        full = {k: value[order] for k, value in combined.items()}
        validate_rows(full)
        manifest['rats'][rat] = dict(neurons=int(full['neural'].shape[1]),
                                      full_rows=len(full['source_id']),
                                      splits={s:len(split_rows[s]['source_id']) for s in SPLITS},
                                      trial_counts={s:int(len(np.unique(split_rows[s]['trial_id']))) for s in SPLITS})
        manifest['input_provenance'][rat] = dict(source=str(raw_path.relative_to(PROJECT)),
                                                  source_sha256=sha256(raw_path))
        for scope, group in (('split', split_rows), ('full', {'full':full})):
            for split, rows in group.items():
                path = output / 'data' / scope / rat / f'{split}.npz'
                _write_npz(path, rows)
                manifest['artifacts'][path.relative_to(output).as_posix()] = sha256(path)
            perm_ids = group['train']['source_id'] if scope == 'split' else group['full']['source_id']
            for condition in SHUFFLED:
                seed = int(old_perm[f'{rat}/{condition}']['seed'])
                if scope == 'split':
                    old_path = OLD / 'permutations' / rat / f'{condition}.csv'
                    perm = _read_permutation(old_path, perm_ids)
                    if not np.array_equal(perm, np.random.default_rng(seed).permutation(len(perm_ids))):
                        raise ValueError(f'Legacy RNG/seed mismatch: {old_path}')
                else:
                    perm = np.random.default_rng(seed).permutation(len(perm_ids))
                path = output / 'permutations' / scope / rat / f'{condition}.csv'
                _write_permutation(path, perm_ids, perm)
                rel = path.relative_to(output).as_posix()
                manifest['artifacts'][rel] = sha256(path)
                manifest['permutations'][f'{scope}/{rat}/{condition}'] = dict(
                    seed=seed, algorithm='numpy.random.default_rng(seed).permutation(N)',
                    rows=len(perm_ids), file=rel, shared_by=list(MODELS),
                    source='previous exact mapping' if scope == 'split' else 'full recording; same seed and algorithm')
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = prepare(args.output)
    print(json.dumps({'status':'prepared', 'output':str(args.output), 'rats':manifest['rats']}, indent=2))


if __name__ == '__main__':
    main()
