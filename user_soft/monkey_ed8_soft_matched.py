"""ED8 Soft campaign. Default: smoke only. Training requires --execute.

The official plotting notebook does not specify fit dimensions, splits, loss
weights or shuffle algorithm for panel e. The explicit choices below are Soft
study conventions, not claims about the unpublished original training code.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import torch

from . import monkey_data as md, rat_runner as rr, transformer_runner as tr
from .data import config_hash, eligible_centers, sha256
from .full_recording_runner import atomic_json, make_encoder, save_checkpoint, source_record, write_history
from .loss import evaluate_soft, fit_distance_scales

ROOT = rr.ROOT
OUT = ROOT / 'results/monkey_ed8_soft_matched_v1'
DATA = ROOT / 'data/monkey_reaching_preload_smth_40'
PROVENANCE = ROOT / 'results/monkey_figure3_v1/stage00_data/DATA_AND_SPLIT_PROVENANCE.json'
REFERENCE = ROOT / 'cebra_replica/monkey/official_reference/ExtendedDataFigure8.py'
FIELDS = ('position', 'direction', 'ap')
GEOMETRY = {'position': 'euclidean', 'direction': 'categorical', 'ap': 'categorical', 'direction_ap': 'categorical', 'time_id': 'temporal'}
ARCHITECTURES = ('cnn', 'transformer')
DIMENSIONS = (3, 4, 8, 16, 32)
SEEDS = (42, 43, 44, 45, 46)


def roster():
    jobs = []
    for target, panel in [('direction', 'a'), ('position', 'b'), ('time_id', 'c')]:
        for arch in ARCHITECTURES:
            jobs.append(dict(panel=panel, scope='active_full', fields=[target], shuffle=None,
                             dimension=2, seed=42, architecture=arch))
    # The notebook has seven observed and three single-label controls for all;
    # active has position, direction, their pair and two single controls.
    for scope, fields in [('all_full', FIELDS), ('active_full', FIELDS[:2])]:
        for n in range(1, len(fields) + 1):
            for combination in itertools.combinations(fields, n):
                for arch in ARCHITECTURES:
                    jobs.append(dict(panel='e', scope=scope, fields=list(combination), shuffle=None,
                                     dimension=32, seed=42, architecture=arch))
        for field in fields:
            for arch in ARCHITECTURES:
                jobs.append(dict(panel='e', scope=scope, fields=[field], shuffle=field,
                                 dimension=32, seed=42, architecture=arch))
    for fields, dim, seed, arch in itertools.product(
            [('position', 'ap'), ('direction', 'ap')], DIMENSIONS, SEEDS, ARCHITECTURES):
        jobs.append(dict(panel='f', scope='all_train', fields=list(fields), shuffle=None,
                         dimension=dim, seed=seed, architecture=arch))
    return jobs


def specs(job):
    if job['panel'] == 'f' and set(job['fields']) == {'direction', 'ap'}:
        # Match the local CEBRA encoder's joint 16-class label.
        return [dict(key='direction_ap', geometry='categorical', weight=1.)]
    return [dict(key=key, geometry=GEOMETRY[key], weight=1. / len(job['fields']))
            for key in job['fields']]


def config(job):
    factory = rr.frozen_config if job['architecture'] == 'cnn' else tr.frozen_config
    cfg = factory('monkey_' + job['scope'], 0, job['seed'], 'soft_v1_stable', 10000)
    cfg.update(output_dimension=job['dimension'], metadata_specs=specs(job),
               embedding_temperature=1., metadata_temperature=.5, batch_size=512,
               learning_rate=3e-4, max_updates=10000, time_bin_ms=1,
               input_policy='legacy_concat', offset_left=5, offset_right=5,
               early_stopping=False, checkpoint_interval=500,
               scope=job['scope'], shuffle=job['shuffle'],
               shuffle_seed=job['seed'] + 10000 if job['shuffle'] else None)
    return cfg


def load_full(population):
    source = DATA / (population + '_all.jl')
    provenance = json.loads(PROVENANCE.read_text(encoding='utf8'))
    if sha256(source) != provenance['source_file_hashes'][source.name]['sha256']:
        raise ValueError('Source hash mismatch: ' + str(source))
    data = joblib.load(source)
    n, length = int(data['num_trials']), int(data['trial_len'])
    neural = np.asarray(data['spikes'], dtype=np.float32)
    position = np.asarray(data['pos'], dtype=np.float32)
    if length != 600 or neural.shape != (n * length, 65) or position.shape != (n * length, 2):
        raise ValueError('Unexpected source dimensions')
    if not np.isfinite(neural).all() or not np.isfinite(position).all():
        raise ValueError('Nonfinite source data')
    labels = {}
    for field, key, maximum in [('direction', 'movement_dir', 7), ('ap', 'passive', 1),
                                ('direction_ap', 'movement_dir_actpas', 15)]:
        x = np.asarray(data[key])
        if x.shape != (n,) or not np.isfinite(x).all() or not np.equal(x, np.floor(x)).all() or np.any((x < 0) | (x > maximum)):
            raise ValueError('Invalid label ' + key)
        labels[field] = np.repeat(x.astype(np.int64), length)
    if population == 'active' and np.any(labels['ap']):
        raise ValueError('Expected active trials')
    ids = np.arange(len(neural), dtype=np.int64)
    return SimpleNamespace(neural=neural, position=position, time_id=ids,
        trial_id=np.repeat(np.arange(n), length), source_id=ids,
        session_id=np.zeros(len(ids), dtype=np.int64), **labels), str(source), sha256(source)


def split_rows(split):
    r = md._load_split(ROOT, 'all', split)
    return SimpleNamespace(neural=r.neural, position=r.position_xy, direction=r.direction,
        ap=r.passive, direction_ap=r.direction_active_passive, time_id=r.time_id,
        trial_id=r.trial_id, source_id=r.source_id, session_id=r.session_id), r.provenance


def values_for(rows, device, shuffle=None, seed=42):
    values = {k: torch.as_tensor(getattr(rows, k), device=device) for k in
              ('neural', 'position', 'direction', 'ap', 'direction_ap', 'time_id', 'trial_id', 'source_id', 'session_id')}
    values['neural'] = values['neural'].to(dtype=torch.float32)
    if shuffle:
        permutation = np.random.default_rng(seed + 10000).permutation(len(rows.neural))
        values[shuffle] = values[shuffle][torch.as_tensor(permutation, device=device)]
    return values


def prepare(job, rows, device):
    values = values_for(rows, device, job['shuffle'], job['seed'])
    objective_specs = specs(job)
    scales = fit_distance_scales({s['key']: values[s['key']] for s in objective_specs}, objective_specs,
        split='train' if job['scope'] == 'all_train' else 'full_recording', source_ids=rows.source_id)
    return values, objective_specs, scales


def loss(model, values, indices, job, objective_specs, scales):
    z, batch = rr.encode_batch(model, values, indices)
    result = evaluate_soft(z, batch, objective_specs, version='soft_v1_stable', scales=scales,
        temperature=1., metadata_temperature=.5, allow_full_recording_scales=job['scope'] != 'all_train')
    if not torch.isfinite(result.loss).item():
        raise FloatingPointError('Nonfinite loss')
    return z, result


def fit_key(job):
    # Panel is deliberately omitted: exact fit identity controls reuse.
    fields = '+'.join(sorted(job['fields']))
    return '/'.join([job['scope'], fields, 'shuffle_' + str(job['shuffle']), job['architecture'],
                     'dim_' + str(job['dimension']), 'seed_' + str(job['seed'])])


def strict_mask(rows):
    proxy = SimpleNamespace(**vars(rows))
    if len(rows.time_id) and rows.time_id.max() >= 600:
        proxy.time_id = np.tile(np.arange(600), len(rows.neural) // 600)
    mask = np.zeros(len(rows.neural), dtype=bool)
    mask[eligible_centers(proxy, left=5, right=5)] = True
    return mask


def export(model, rows, path, device, job):
    values = values_for(rows, device)
    mask = strict_mask(rows)
    # Descriptive panels include the entire recording, as in the local CEBRA
    # exports. Held-out train/test embeddings use only strict trial interiors.
    indices = np.flatnonzero(mask) if job['scope'] == 'all_train' else np.arange(len(mask))
    parts = []
    model.eval()
    with torch.no_grad():
        for chunk in np.array_split(indices, max(1, (len(indices) + 511) // 512)):
            z, _ = rr.encode_batch(model, values, torch.as_tensor(chunk))
            parts.append(z.cpu().numpy())
    embedding = np.concatenate(parts)
    if embedding.shape != (len(indices), job['dimension']) or not np.isfinite(embedding).all():
        raise ValueError('Invalid strict-mask embedding')
    np.savez_compressed(path, embedding=embedding, valid_mask=mask, row_index=indices,
        position_xy=rows.position[indices], direction=rows.direction[indices],
        direction_active_passive=rows.direction_ap[indices], passive=rows.ap[indices],
        source_id=rows.source_id[indices], trial_id=rows.trial_id[indices], time_id=rows.time_id[indices])


def score(folder, job):
    from cebra_replica.monkey.score_plot_figure3h_soft_matched import knn
    from sklearn.metrics import accuracy_score, r2_score
    target = 'position_xy' if 'position' in job['fields'] else 'direction_active_passive'
    with np.load(folder / 'embedding_train.npz') as train, np.load(folder / 'embedding_test.npz') as test:
        if np.intersect1d(train['trial_id'], test['trial_id']).size:
            raise ValueError('Decoder train/test overlap')
        prediction = knn(train['embedding'], test['embedding'], train[target],
                         torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        value = (r2_score(test[target], prediction) if target == 'position_xy'
                 else accuracy_score(test[target], prediction))
        record = dict(score_percent=100 * float(value), metric='R2' if target == 'position_xy' else 'accuracy',
            target=target, train_rows=len(train['embedding']), test_rows=len(test['embedding']),
            decoder='cebra.KNNDecoder', k=36, distance='cosine', fit_split='train', score_split='test',
            note='K=36 follows local CEBRA decoding demo; original paper K not verified')
    atomic_json(folder / 'score.json', record)


def fit(job, device):
    if job['scope'] == 'all_train':
        rows, provenance = split_rows('train')
        source = provenance['source_path']; digest = provenance['source_sha256']
    else:
        rows, source, digest = load_full(job['scope'].split('_')[0])
        provenance = None
    cfg = config(job)
    values, objective_specs, scales = prepare(job, rows, device)
    sources = source_record(job['architecture'])
    signature = config_hash(dict(config=cfg, source_sha256=digest, sources=sources,
                                 script_sha256=sha256(Path(__file__)), scales=scales))
    folder = OUT / 'fits' / fit_key(job)
    checkpoint, manifest = folder / 'checkpoint.pt', folder / 'manifest.json'
    if manifest.exists():
        record = json.loads(manifest.read_text(encoding='utf8'))
        if record['signature'] != signature or any(sha256(folder / name) != digest for name, digest in record['artifacts'].items()):
            raise RuntimeError('Completed artifact mismatch: ' + str(folder))
        print('REUSE', fit_key(job), flush=True)
        return
    if folder.exists() and any(folder.iterdir()) and not checkpoint.exists():
        raise RuntimeError('Partial output without checkpoint: ' + str(folder))
    folder.mkdir(parents=True, exist_ok=True)
    random.seed(job['seed']); np.random.seed(job['seed']); torch.manual_seed(job['seed'])
    if device.type == 'cuda': torch.cuda.manual_seed_all(job['seed'])
    model = make_encoder(job['architecture'], 65, cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], betas=tuple(cfg['betas']),
                                 eps=cfg['eps'], weight_decay=cfg['weight_decay'])
    generator = torch.Generator().manual_seed(job['seed'])
    step, history = 0, []
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state['signature'] != signature or state['device_type'] != device.type:
            raise RuntimeError('Resume mismatch')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['generator'].cpu()); torch.set_rng_state(state['torch_rng'].cpu())
        if device.type == 'cuda': torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        step, history = state['step'], state['history']
    model.train()
    while step < cfg['max_updates']:
        indices = torch.randint(len(rows.neural), (cfg['batch_size'],), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        _, result = loss(model, values, indices, job, objective_specs, scales)
        result.loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite gradients')
        optimizer.step(); step += 1
        history.append(dict(update=step, train_loss=float(result.loss.detach())))
        if step % cfg['checkpoint_interval'] == 0 or step == cfg['max_updates']:
            save_checkpoint(checkpoint, dict(signature=signature, model=model.state_dict(), optimizer=optimizer.state_dict(),
                generator=generator.get_state(), torch_rng=torch.get_rng_state(), device_type=device.type,
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None, step=step, history=history))
            write_history(folder / 'training_history.csv', history)
            print('UPDATE', fit_key(job), step, flush=True)
    write_history(folder / 'training_history.csv', history)
    artifacts = ['checkpoint.pt', 'training_history.csv']
    if job['panel'] in ('a', 'b', 'c'):
        export(model, rows, folder / 'embedding_full.npz', device, job)
        artifacts.append('embedding_full.npz')
    if job['scope'] == 'all_train':
        test, test_provenance = split_rows('test')
        if np.intersect1d(rows.trial_id, test.trial_id).size:
            raise ValueError('Train/test overlap')
        for split, split_data in [('train', rows), ('test', test)]:
            name = 'embedding_' + split + '.npz'
            export(model, split_data, folder / name, device, job); artifacts.append(name)
        score(folder, job)
        artifacts.append('score.json')
    else:
        test_provenance = None
    atomic_json(manifest, dict(signature=signature, config=cfg, job=job, source=source, source_sha256=digest,
        source_provenance=provenance, test_provenance=test_provenance, sources=sources, scales=scales,
        script_sha256=sha256(Path(__file__)), actual_updates=step,
        artifacts={name: sha256(folder / name) for name in artifacts},
        training_window_policy='legacy_concat including cross-trial boundaries',
        export_policy=('strict trial interior [-5,+4] only' if job['scope'] == 'all_train'
                       else 'all rows; valid_mask marks strict trial interior [-5,+4]'),
        shuffle_policy='single metadata field permuted by row within fit scope; XY kept together',
        held_out=job['scope'] == 'all_train'))
    print('COMPLETE', fit_key(job), flush=True)


def smoke(device):
    jobs = roster()
    assert len(jobs) == 136 and len({fit_key(j) for j in jobs}) == len(jobs)
    full = {p: load_full(p) for p in ('active', 'all')}
    train, train_provenance = split_rows('train')
    test, test_provenance = split_rows('test')
    assert not np.intersect1d(train.trial_id, test.trial_id).size
    assert strict_mask(train).sum() == len(train.neural) // 600 * 591
    assert strict_mask(test).sum() == len(test.neural) // 600 * 591
    reports = []
    # Every distinct loss geometry/shuffle and architecture is exercised; all
    # requested configurations are independently recorded in the campaign plan.
    checked = set()
    for job in jobs:
        identity = (job['scope'], tuple(job['fields']), job['shuffle'], job['architecture'], job['dimension'])
        if identity in checked: continue
        checked.add(identity)
        rows = train if job['scope'] == 'all_train' else full[job['scope'].split('_')[0]][0]
        values, objective_specs, scales = prepare(job, rows, device)
        torch.manual_seed(job['seed'])
        model = make_encoder(job['architecture'], 65, config(job), device).eval()
        indices = torch.randint(len(rows.neural), (512,), generator=torch.Generator().manual_seed(42))
        with torch.no_grad(): z, result = loss(model, values, indices, job, objective_specs, scales)
        assert z.shape == (512, job['dimension'])
        reports.append(dict(job=job, loss=float(result.loss), shape=list(z.shape)))
        print('SMOKE', fit_key(job), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / 'SMOKE.json', dict(optimizer_updates=0, device=str(device), checks=reports,
        script_sha256=sha256(Path(__file__)), train=train_provenance, test=test_provenance,
        strict_train_rows=int(strict_mask(train).sum()), strict_test_rows=int(strict_mask(test).sum())))
    atomic_json(OUT / 'PLAN.json', dict(jobs=jobs, planned=len(jobs),
        reference=str(REFERENCE), reference_sha256=sha256(REFERENCE),
        choices=['panel e dimension 32 and full-recording scopes are explicit conventions, not specified by plotting notebook',
                 'equal weights per metadata field; row-wise single-label shuffle, seed+10000',
                 'panel f AP+direction uses separate binary AP and 8-class direction distances; local CEBRA uses one joint 16-class label',
                 'Soft objective is not CEBRA InfoNCE or InfoMSE',
                 'strict exports omit invalid trial edges; legacy training windows can cross trial boundaries',
                 'no reuse between full-recording and train-only fits or different dimensions']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--smoke', action='store_true')
    action.add_argument('--execute', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--panel', choices=('a', 'b', 'c', 'e', 'f'))
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available(): parser.error('CUDA unavailable')
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if not args.execute:
        if args.panel: parser.error('--panel requires --execute')
        smoke(device)
    else:
        for job in roster():
            if args.panel is None or job['panel'] == args.panel: fit(job, device)


if __name__ == '__main__':
    main()
