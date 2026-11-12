"""Descriptive full-recording Soft Fig. 3 fits; review before --execute.

Default --smoke performs forward/loss checks only, with zero optimizer updates.
Run as ``python -m user_soft.monkey_fig3_descriptive_matched --smoke``.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import torch

from . import rat_runner as rr, transformer_runner as tr
from .data import config_hash, eligible_centers, sha256
from .full_recording_runner import atomic_json, make_encoder, save_checkpoint, source_record, write_history
from .loss import evaluate_soft, fit_distance_scales

ROOT = rr.ROOT
OUT = ROOT / 'results/monkey_fig3_soft_descriptive_matched_v1'
DATA = ROOT / 'data/monkey_reaching_preload_smth_40'
PROVENANCE = ROOT / 'results/monkey_figure3_v1/stage00_data/DATA_AND_SPLIT_PROVENANCE.json'
CONDITIONS = {
    'b_time_active_4d': ('active', 'time', 4),
    'c_position_active_3d': ('active', 'position', 3),
    'd_time_active_3d': ('active', 'time', 3),
    'e_direction_passive_4d': ('passive', 'direction', 4),
    'f_direction_ap_all_4d': ('all', 'direction_ap', 4),
    'g_position_all_4d': ('all', 'position', 4),
}
ARCHITECTURES = ('cnn', 'transformer')


def prepare(name):
    population, target, dimension = CONDITIONS[name]
    source = DATA / f'{population}_all.jl'
    provenance = json.loads(PROVENANCE.read_text(encoding='utf8'))
    digest = sha256(source)
    if digest != provenance['source_file_hashes'][source.name]['sha256']:
        raise ValueError(f'Source hash mismatch: {source}')
    data = joblib.load(source)
    n, length = int(data['num_trials']), int(data['trial_len'])
    x = np.asarray(data['spikes'], dtype=np.float32)
    position = np.asarray(data['pos'], dtype=np.float32)
    if length != 600 or x.shape != (n * length, 65) or position.shape != (len(x), 2):
        raise ValueError(f'Unexpected source dimensions: {source}')
    if not np.isfinite(x).all() or not np.isfinite(position).all():
        raise ValueError('Nonfinite source values')
    labels = {}
    for key, max_value in (('movement_dir', 7), ('movement_dir_actpas', 15), ('passive', 1)):
        raw = np.asarray(data[key])
        if raw.shape != (n,) or not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all() or np.any((raw < 0) | (raw > max_value)):
            raise ValueError(f'Invalid {key}')
        labels[key] = np.repeat(raw.astype(np.int64), length)
    if population != 'all' and not np.all(labels['passive'] == (population == 'passive')):
        raise ValueError('Population label mismatch')
    if target == 'direction_ap' and set(np.unique(labels['movement_dir_actpas'])) != set(range(16)):
        raise ValueError('Expected all 16 direction by active/passive categories')
    behavior = (position if target == 'position' else
                labels['movement_dir' if target == 'direction' else 'movement_dir_actpas'][:, None])
    ids = np.arange(len(x), dtype=np.int64)
    # The time target is the source's concatenated recording index, not trial-relative time.
    rows = SimpleNamespace(neural=x, behavior=behavior, source_id=ids,
                           trial_id=np.repeat(np.arange(n), length), time_id=ids,
                           session_id=np.zeros(len(x), dtype=np.int64))
    specs = [dict(key='time_id', geometry='temporal', weight=1.)] if target == 'time' else [
        dict(key='behavior', geometry='euclidean' if target == 'position' else 'categorical', weight=1.)]
    cpu = rr.tensors(rows, torch.device('cpu'))
    scales = fit_distance_scales({s['key']: cpu[s['key']] for s in specs}, specs,
                                 split='full_recording', source_ids=ids)
    return rows, labels, position, specs, scales, source, digest


def configuration(name, architecture, specs):
    population, target, dimension = CONDITIONS[name]
    factory = rr.frozen_config if architecture == 'cnn' else tr.frozen_config
    cfg = factory('monkey_' + population, 0, 42, 'soft_v1_stable', 10000)
    cfg.update(condition=name, target=target, fit_scope=population + '_all_full_recording',
               output_dimension=dimension, metadata_specs=specs, embedding_temperature=1.,
               metadata_temperature=.5, batch_size=512, learning_rate=3e-4,
               max_updates=10000, time_bin_ms=1, input_policy='legacy_concat',
               offset_left=5, offset_right=5, early_stopping=False, checkpoint_interval=500,
               time_coordinate='concatenated_source_row_index', held_out=False)
    return cfg


def objective(model, values, indices, specs, scales):
    z, batch = rr.encode_batch(model, values, indices)
    result = evaluate_soft(z, batch, specs, version='soft_v1_stable', scales=scales,
                           temperature=1., metadata_temperature=.5, allow_full_recording_scales=True)
    if not torch.isfinite(result.loss):
        raise FloatingPointError('Nonfinite Soft loss')
    return z, result


def smoke(device):
    report = []
    for name in CONDITIONS:
        rows, labels, position, specs, scales, source, digest = prepare(name)
        values = rr.tensors(rows, device)
        for architecture in ARCHITECTURES:
            cfg = configuration(name, architecture, specs)
            torch.manual_seed(42)
            model = make_encoder(architecture, 65, cfg, device)
            model.eval()
            indices = torch.randint(len(rows.neural), (512,), generator=torch.Generator().manual_seed(42))
            with torch.no_grad():
                z, result = objective(model, values, indices, specs, scales)
            if z.shape != (512, cfg['output_dimension']):
                raise ValueError('Unexpected encoder output dimensions')
            report.append(dict(condition=name, architecture=architecture, config=cfg,
                               source=str(source), source_sha256=digest, rows=len(rows.neural),
                               trials=len(rows.neural)//600, loss=float(result.loss),
                               embedding_shape=list(z.shape), scales=scales))
            print('SMOKE', name, architecture, 'loss', float(result.loss), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / 'SMOKE.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        optimizer_updates=0, device=str(device), script_sha256=sha256(Path(__file__)),
        source_provenance_sha256=sha256(PROVENANCE), checks=report,
        note='Forward/loss only; no training or optimizer updates. Full-recording descriptive fits are not held-out.'))


def fit(name, architecture, device):
    rows, labels, position, specs, scales, source, digest = prepare(name)
    cfg = configuration(name, architecture, specs)
    sources = source_record(architecture)
    signature = config_hash(dict(config=cfg, source_sha256=digest, sources=sources,
                                 script_sha256=sha256(Path(__file__)), scales=scales))
    folder = OUT / name / architecture / 'seed_42'
    checkpoint, manifest, embedding = (folder / s for s in ('checkpoint.pt', 'manifest.json', 'embedding.npz'))
    if manifest.exists():
        record = json.loads(manifest.read_text(encoding='utf8'))
        if record['signature'] != signature or any(sha256(folder / f) != record[k] for f, k in
             (('checkpoint.pt', 'checkpoint_sha256'), ('embedding.npz', 'embedding_sha256'),
              ('training_history.csv', 'history_sha256'))):
            raise RuntimeError('Completed output mismatch')
        print('REUSE', name, architecture, flush=True)
        return
    if folder.exists() and any(folder.iterdir()) and not checkpoint.exists():
        raise RuntimeError('Partial output without checkpoint; inspect')
    folder.mkdir(parents=True, exist_ok=True)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(42)
    model = make_encoder(architecture, 65, cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], betas=tuple(cfg['betas']),
                           eps=cfg['eps'], weight_decay=cfg['weight_decay'])
    gen = torch.Generator().manual_seed(42)
    step, history = 0, []
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state['signature'] != signature or state['device_type'] != device.type:
            raise RuntimeError('Resume signature or device mismatch')
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        gen.set_state(state['generator'].cpu()); torch.set_rng_state(state['torch_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
        step, history = state['step'], state['history']
    values = rr.tensors(rows, device)
    started = time.perf_counter()
    model.train()
    while step < cfg['max_updates']:
        indices = torch.randint(len(rows.neural), (cfg['batch_size'],), generator=gen)
        opt.zero_grad(set_to_none=True)
        _, result = objective(model, values, indices, specs, scales)
        result.loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite gradients')
        opt.step(); step += 1
        history.append(dict(update=step, train_loss=float(result.loss.detach())))
        if step % cfg['checkpoint_interval'] == 0 or step == cfg['max_updates']:
            save_checkpoint(checkpoint, dict(signature=signature, model=model.state_dict(), optimizer=opt.state_dict(),
                generator=gen.get_state(), torch_rng=torch.get_rng_state(), device_type=device.type,
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                step=step, history=history, config=cfg, scales=scales, sources=sources))
            write_history(folder / 'training_history.csv', history)
            print('UPDATE', name, architecture, step, flush=True)
    # Rebuild history if interrupted between checkpoint and CSV write.
    write_history(folder / 'training_history.csv', history)
    model.eval(); parts = []
    with torch.no_grad():
        for start in range(0, len(rows.neural), 512):
            z, _ = rr.encode_batch(model, values, torch.arange(start, min(start + 512, len(rows.neural))))
            parts.append(z.cpu().numpy())
    emb = np.concatenate(parts)
    if emb.shape != (len(rows.neural), cfg['output_dimension']) or not np.isfinite(emb).all():
        raise ValueError('Invalid embedding')
    # eligible_centers expects trial-relative time; training/export use legacy concatenated windows.
    trial_time = np.tile(np.arange(600, dtype=np.int64), len(rows.neural)//600)
    support = SimpleNamespace(**vars(rows)); support.time_id = trial_time
    valid = np.zeros(len(rows.neural), dtype=bool); valid[eligible_centers(support)] = True
    np.savez_compressed(embedding, embedding=emb, position_xy=position, direction=labels['movement_dir'],
        direction_active_passive=labels['movement_dir_actpas'], passive=labels['passive'],
        source_id=rows.source_id, trial_id=rows.trial_id, time_id=trial_time,
        recording_time_id=rows.time_id, valid_mask=valid)
    atomic_json(manifest, dict(signature=signature, config=cfg, sources=sources, source=str(source),
        source_sha256=digest, source_provenance_sha256=sha256(PROVENANCE), scales=scales,
        script_sha256=sha256(Path(__file__)), architecture=architecture, actual_updates=step,
        last_attempt_seconds=time.perf_counter()-started, device=str(device), training_rows=len(rows.neural),
        checkpoint_sha256=sha256(checkpoint), embedding_sha256=sha256(embedding),
        history_sha256=sha256(folder/'training_history.csv'), held_out=False,
        note='Full descriptive recording; legacy concatenated windows may cross trial boundaries; valid_mask marks strict trial interiors. Soft temporal distance is not CEBRA time-offset sampling. Panel g uses XY only.'))
    print('COMPLETE', name, architecture, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--smoke', action='store_true')
    action.add_argument('--execute', action='store_true', help='Start training only after review')
    parser.add_argument('--condition', choices=CONDITIONS)
    parser.add_argument('--architecture', choices=ARCHITECTURES)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.smoke and (args.condition or args.architecture):
        parser.error('Smoke always checks all conditions and architectures')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if args.smoke:
        smoke(device)
    else:
        for name in ([args.condition] if args.condition else CONDITIONS):
            for architecture in ([args.architecture] if args.architecture else ARCHITECTURES):
                fit(name, architecture, device)


if __name__ == '__main__':
    main()
