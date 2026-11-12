"""Run the 72 trial-safe Figure 2c fits, with per-fit provenance and resume."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from prepare_data import DEFAULT_OUTPUT, RATS, CONDITIONS, MODELS, sha256, _read_permutation, _check_prepared
from cebra_trial_safe import TrialSafeCEBRA, load_trial_safe, trial_bounds
import soft_trial_safe

ROOT = DEFAULT_OUTPUT
NAMES = {'position': 'Position', 'direction': 'Direction', 'position_direction': 'Position+Direction'}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def rows(scope, rat, split):
    with np.load(ROOT / 'data' / scope / rat / (split + '.npz'), allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def trials(data):
    ids = data['trial_id']
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    ends = np.r_[starts[1:], len(ids)]
    return [dict(neural=data['neural'][a:b], position=data['behavior'][a:b, 0:1],
                 direction=data['behavior'][a:b, 1].astype(np.int64)) for a, b in zip(starts, ends)], list(zip(starts, ends))


def labels(behavior, base):
    position = behavior[:, 0:1].astype(np.float32)
    direction = behavior[:, 1].astype(np.int64)
    if base == 'position': return (position,)
    if base == 'direction': return (direction,)
    return position, direction


def export(path, model, kind, data, base, target_behavior, donor):
    if kind == 'cebra_cnn':
        embedding = model.transform(data['neural'], trial_ids=data['trial_id'], batch_size=512)
    else:
        trial_data, _ = trials(data)
        embedding = soft_trial_safe.transform(model, trial_data, condition=NAMES[base], batch_size=512)['embedding'].numpy()
    embedding = np.asarray(embedding, dtype=np.float32)
    if embedding.shape != (len(data['source_id']), 3) or not np.isfinite(embedding).all():
        raise ValueError(f'invalid embedding {path}: {embedding.shape}')
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    if np.any(norm < 1e-12): raise ValueError('zero embedding')
    start, end = trial_bounds(data['trial_id'], len(embedding))
    center = np.maximum(np.minimum(np.arange(len(embedding)), end - 4), start + 5)
    indices = center[:, None] + np.arange(-5, 5)
    if not np.all(data['trial_id'][indices] == data['trial_id'][:, None]):
        raise ValueError('window crossed trial boundary')
    if not np.all(np.diff(data['source_id'][indices], axis=1) == 1):
        raise ValueError('window source indices are not consecutive')
    target = labels(target_behavior, base)
    target = np.column_stack(target)
    temporary = path.with_name(path.stem + '.partial.npz')
    np.savez_compressed(temporary, embedding=embedding, embedding_unit=embedding / norm,
                        true_position=data['behavior'][:, 0], true_direction=data['behavior'][:, 1].astype(np.int64),
                        training_target=target, source_id=data['source_id'], trial_id=data['trial_id'],
                        time_id=data['time_id'], target_source_id=donor, window_source_ids=data['source_id'][indices])
    temporary.replace(path)


def run_fit(scope, rat, kind, condition, signature):
    output = ROOT / 'fits' / scope / rat / kind / condition
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for name, digest in manifest['artifacts'].items():
            if not (output / name).is_file() or sha256(output / name) != digest:
                raise RuntimeError(f'completed artifact changed: {output / name}')
        if manifest['signature'] != signature:
            old = '4be471c86e283151bba310780b5471f3c6c70c1af01b2290aedbb83cc43bdc54'
            if (scope, rat, kind, condition, manifest['signature']) != ('split','achilles','cebra_cnn','position',old):
                raise RuntimeError(f'configuration changed: {output}')
            manifest['previous_signature'] = old
            manifest['signature_change'] = 'runner provenance checks added after completed pilot; encoder and data unchanged'
            manifest['signature'] = signature
            write_json(manifest_path, manifest)
        return 'reused'
    base = condition.removesuffix('_shuffled')
    datasets = {s: rows(scope, rat, s) for s in (('train', 'valid', 'test') if scope == 'split' else ('full',))}
    train_name = 'train' if scope == 'split' else 'full'
    train = datasets[train_name]
    permutation = np.arange(len(train['source_id']))
    if condition.endswith('_shuffled'):
        permutation = _read_permutation(ROOT / 'permutations' / scope / rat / (condition + '.csv'), train['source_id'])
    target_behavior = train['behavior'][permutation]
    checkpoint = output / 'checkpoint.pt'
    history = output / 'training_history.csv'
    intent_path = output / 'fit_intent.json'
    if intent_path.exists():
        if json.loads(intent_path.read_text(encoding='utf-8'))['signature'] != signature:
            raise RuntimeError(f'incomplete fit provenance mismatch: {output}')
    elif checkpoint.exists() or history.exists():
        raise RuntimeError(f'incomplete fit lacks provenance: {output}')
    else:
        write_json(intent_path, dict(signature=signature, scope=scope, rat=rat, model=kind, condition=condition))
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    if kind == 'cebra_cnn':
        if checkpoint.exists() and history.exists():
            with history.open(newline='', encoding='utf-8') as stream:
                previous = list(csv.DictReader(stream))
            if len(previous) != 10000 or [int(x['update']) for x in previous] != list(range(1,10001)):
                raise RuntimeError('CEBRA checkpoint history is incomplete')
            model = load_trial_safe(str(checkpoint))
        elif checkpoint.exists() or history.exists():
            raise RuntimeError(f'incomplete CEBRA checkpoint/history pair: {output}')
        else:
            model = TrialSafeCEBRA(model_architecture='offset10-model', num_hidden_units=32,
                                   output_dimension=3, batch_size=512, learning_rate=3e-4,
                                   temperature=1., temperature_mode='constant', optimizer='adam',
                                   criterion='infonce', distance='cosine', conditional='time_delta',
                                   time_offsets=10, max_iterations=10000, device='cuda_if_available', verbose=False)
            model.fit(train['neural'], *labels(target_behavior, base), trial_ids=train['trial_id'])
            losses = model.state_dict_['loss']
            if len(losses) != 10000: raise ValueError('CEBRA loss history incomplete')
            tmp = output / 'checkpoint.partial.pt'; model.save(str(tmp)); tmp.replace(checkpoint)
            with history.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream); writer.writerow(('update', 'train_loss'))
                writer.writerows((i + 1, float(loss)) for i, loss in enumerate(losses))
    else:
        trial_data, bounds = trials(train)
        targets = [dict(position=target_behavior[a:b, 0:1], direction=target_behavior[a:b, 1].astype(np.int64)) for a, b in bounds]
        arch = 'cnn' if kind == 'soft_stable_cnn' else 'transformer'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        result = soft_trial_safe.fit(trial_data, architecture=arch, condition=NAMES[base], targets=targets,
            embedding_dim=3, steps=10000, batch_size=512, learning_rate=3e-4, seed=42, device=device,
            checkpoint_path=checkpoint, checkpoint_signature=signature, checkpoint_every=500)
        model, scales, losses = result['model'], result['scales'], result['losses']
        if len(losses) != 10000: raise ValueError('Soft loss history incomplete')
        with history.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream); writer.writerow(('update', 'train_loss'))
            writer.writerows((i + 1, float(loss)) for i, loss in enumerate(losses))
        write_json(output / 'distance_scales.json', scales)
    for split, data in datasets.items():
        is_train = split == train_name
        export(output / f'{split}_embedding.npz', model, kind, data, base,
               target_behavior if is_train else data['behavior'],
               train['source_id'][permutation] if is_train else data['source_id'])
    artifacts = {p.name: sha256(p) for p in output.iterdir() if p.is_file() and p.name not in ('manifest.json',)}
    write_json(manifest_path, dict(signature=signature, scope=scope, rat=rat, model=kind, condition=condition,
        seed=42, fold=0 if scope == 'split' else None, embedding_dimension=3, updates=10000, batch_size=512,
        learning_rate=0.0003, window_offsets=list(range(-5,5)), target='original recipient bin',
        shuffle='labels of fit dataset only; shared permutation' if condition.endswith('_shuffled') else 'none',
        exported_training_target='fit dataset: labels actually used; validation/test: real labels for alignment only, not used in fitting',
        temperature=1.0, metadata_temperature=0.5 if kind.startswith('soft') else None, artifacts=artifacts))
    return 'completed'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS, action='append')
    parser.add_argument('--scope', choices=['split', 'full'], action='append')
    parser.add_argument('--rat', choices=RATS, action='append')
    parser.add_argument('--condition', choices=CONDITIONS, action='append')
    args = parser.parse_args()
    required = [ROOT / 'data' / 'PREPARED.json', ROOT / 'data' / 'PREFLIGHT.json', Path(__file__),
                Path(__file__).with_name('cebra_trial_safe.py'), Path(__file__).with_name('soft_trial_safe.py')]
    if any(not p.is_file() for p in required): raise RuntimeError('Missing prepared inputs or preflight')
    preflight = json.loads(required[1].read_text(encoding='utf-8'))
    _check_prepared(ROOT, json.loads(required[0].read_text(encoding='utf-8')))
    if (preflight.get('status') != 'pass' or preflight.get('prepared_manifest_sha256') != sha256(required[0])
        or preflight.get('cebra_adapter_sha256') != sha256(required[3])
        or preflight.get('soft_adapter_sha256') != sha256(required[4])):
        raise RuntimeError('Preflight is stale or did not pass')
    signature = hashlib.sha256(''.join(sha256(p) for p in required).encode()).hexdigest()
    combinations = [(s,r,m,c) for s in (args.scope or ['split','full']) for r in (args.rat or RATS)
                    for m in (args.model or MODELS) for c in (args.condition or CONDITIONS)]
    state = dict(signature=signature, total=len(combinations), done=0, current=None, started=datetime.now(timezone.utc).isoformat())
    for scope, rat, model, condition in combinations:
        state['current'] = dict(scope=scope, rat=rat, model=model, condition=condition)
        write_json(ROOT / 'CAMPAIGN_STATE.json', state)
        print(f"START {scope} {rat} {model} {condition}", flush=True)
        try:
            result = run_fit(scope, rat, model, condition, signature)
        except Exception:
            state['error'] = traceback.format_exc(); write_json(ROOT / 'CAMPAIGN_STATE.json', state)
            raise
        state['done'] += 1
        print(f"{result.upper()} {scope} {rat} {model} {condition}", flush=True)
    state['current'] = None; state['finished'] = datetime.now(timezone.utc).isoformat()
    write_json(ROOT / 'CAMPAIGN_STATE.json', state)


if __name__ == '__main__': main()
