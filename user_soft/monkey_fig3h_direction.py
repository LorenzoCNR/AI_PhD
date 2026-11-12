"""Held-out active monkey Direction Soft fits. Run with --execute; --smoke never fits."""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch

from . import monkey_data as md, rat_runner as rr, transformer_runner as tr
from .data import assert_disjoint, config_hash, sha256
from .full_recording_runner import make_encoder, save_checkpoint
from .loss import evaluate_soft, fit_distance_scales

ROOT = rr.ROOT
OUTPUT = ROOT / 'results/monkey_fig3h_soft_direction_v1'
ARCHITECTURES = ('cnn', 'transformer')
DIMENSIONS = (3, 4, 8, 16, 32)
SEEDS = (42, 43, 44, 45, 46)
SPECS = [dict(key='behavior', geometry='categorical', weight=1.)]


def config_for(architecture, dimension, seed):
    factory = rr.frozen_config if architecture == 'cnn' else tr.frozen_config
    config = factory('monkey_active', 0, seed, 'soft_v1_stable', 10000)
    config.update(output_dimension=dimension, condition='direction', population='active',
                  metadata_specs=SPECS, embedding_temperature=1., metadata_temperature=.5,
                  batch_size=512, learning_rate=3e-4, max_updates=10000,
                  input_policy='legacy_concat', offset_left=5, offset_right=5,
                  time_bin_ms=1, early_stopping=False, checkpoint_interval=500,
                  diagnostic_interval=500)
    return config


def values_for(rows, device):
    return rr.tensors(rows.as_trial_rows(rows.direction[:, None]), device)


def objective(model, values, indices, config, scales):
    z, batch = rr.encode_batch(model, values, indices)
    return evaluate_soft(z, batch, SPECS, version='soft_v1_stable', scales=scales,
                         temperature=1., metadata_temperature=.5)


def fit_cell(architecture, config, train, valid, scales, dependencies, device):
    folder = OUTPUT / 'stage01_fit' / architecture / f"dim_{config['output_dimension']}" / f"seed_{config['seed']}"
    checkpoint, manifest = folder / 'checkpoint.pt', folder / 'manifest.json'
    signature = config_hash(dict(config=config, dependencies=dependencies, scales=scales))
    if manifest.exists():
        record = json.loads(manifest.read_text(encoding='utf8'))
        if record['signature'] != signature or sha256(checkpoint) != record['checkpoint_sha256']:
            raise RuntimeError(f'Fit provenance mismatch: {folder}')
        return record
    folder.mkdir(parents=True, exist_ok=True)
    seed = config['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
    model = make_encoder(architecture, train.neural.shape[1], config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                 betas=tuple(config['betas']), eps=config['eps'], weight_decay=config['weight_decay'])
    generator = torch.Generator().manual_seed(seed)
    step, history, diagnostics = 0, [], []
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state['signature'] != signature or state['device_type'] != device.type:
            raise RuntimeError(f'Checkpoint dependencies or device changed: {checkpoint}')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['generator'].cpu()); torch.set_rng_state(state['torch_rng'].cpu())
        if device.type == 'cuda': torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        step, history, diagnostics = state['step'], state['history'], state['diagnostics']
    train_values, valid_values = values_for(train, device), values_for(valid, device)
    train_ids = torch.arange(len(train.neural))
    valid_ids = torch.from_numpy(np.flatnonzero(valid.valid_mask))
    eval_gen = torch.Generator().manual_seed(seed + 20000)
    eval_ids = valid_ids[torch.randint(len(valid_ids), (512,), generator=eval_gen)]

    def commit():
        save_checkpoint(checkpoint, dict(signature=signature, device_type=device.type,
            model=model.state_dict(), optimizer=optimizer.state_dict(), generator=generator.get_state(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
            step=step, history=history, diagnostics=diagnostics, config=config))

    if not checkpoint.exists(): commit()
    model.train()
    try:
        while step < config['max_updates']:
            ids = train_ids[torch.randint(len(train_ids), (512,), generator=generator)]
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model, train_values, ids, config, scales).loss
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite loss or gradients')
            optimizer.step(); step += 1
            history.append(float(loss.detach()))
            if step % config['diagnostic_interval'] == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = objective(model, valid_values, eval_ids, config, scales).loss
                diagnostics.append(dict(update=step, validation_loss=float(val_loss)))
                model.train()
            if step % config['checkpoint_interval'] == 0: commit()
    except BaseException:
        # Resume from the last fully completed checkpoint, never a partial update.
        raise
    commit()
    record = dict(signature=signature, config=config, dependencies=dependencies, scales=scales,
                  actual_updates=step, checkpoint_sha256=sha256(checkpoint),
                  validation_diagnostics=diagnostics, validation_role='diagnostic_only',
                  train_valid_centers=int(len(train_ids)), device=str(device))
    rr.atomic_json(manifest, record)
    return record


def export_cell(architecture, config, fit, splits, device):
    relative = Path(architecture) / f"dim_{config['output_dimension']}" / f"seed_{config['seed']}"
    folder = OUTPUT / 'stage02_embedding' / relative
    checkpoint = OUTPUT / 'stage01_fit' / relative / 'checkpoint.pt'
    signature = config_hash(dict(fit=fit['checkpoint_sha256'], splits={k: v.provenance for k, v in splits.items()},
                                 script=sha256(__file__)))
    manifest = folder / 'manifest.json'
    if manifest.exists():
        record = json.loads(manifest.read_text(encoding='utf8'))
        if record['signature'] != signature or any(sha256(folder / f'{s}_embedding.npz') != h for s, h in record['files'].items()):
            raise RuntimeError(f'Embedding provenance mismatch: {folder}')
        return
    folder.mkdir(parents=True, exist_ok=True)
    model = make_encoder(architecture, splits['train'].neural.shape[1], config, device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)['model']); model.eval()
    files, counts = {}, {}
    for split, rows in splits.items():
        ids = np.flatnonzero(rows.valid_mask)
        values = values_for(rows, device)
        parts = []
        with torch.no_grad():
            for start in range(0, len(ids), 512):
                z, _ = rr.encode_batch(model, values, torch.from_numpy(ids[start:start + 512]))
                parts.append(z.cpu().numpy())
        embedding = np.concatenate(parts)
        if embedding.shape != (len(ids), config['output_dimension']) or not np.isfinite(embedding).all():
            raise ValueError('Invalid embedding')
        path = folder / f'{split}_embedding.npz'
        temporary = folder / f'{split}_embedding.tmp.npz'
        np.savez_compressed(temporary, embedding_raw=embedding, valid_mask=np.ones(len(ids), dtype=bool),
            position_xy=rows.position_xy[ids], direction=rows.direction[ids], passive=rows.passive[ids],
            trial_id=rows.trial_id[ids], source_id=rows.source_id[ids], time_id=rows.time_id[ids])
        temporary.replace(path)
        files[split] = sha256(path); counts[split] = len(ids)
    rr.atomic_json(manifest, dict(signature=signature, checkpoint_sha256=fit['checkpoint_sha256'],
        files=files, valid_counts=counts, split_provenance={k: v.provenance for k, v in splits.items()},
        support='only strict valid centers, within trial offsets [-5,+4]; valid_mask all true',
        embedding_raw='native encoder output'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--execute', action='store_true')
    group.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows = md.load_train_validation(ROOT, 'active')
    train, valid = rows['train'], rows['valid']
    ids = np.flatnonzero(train.valid_mask)
    scales = fit_distance_scales({'behavior': torch.as_tensor(train.direction[ids, None])}, SPECS,
                                 split='train', source_ids=train.source_id[ids])
    dependencies = dict(script=sha256(__file__), adapter=sha256(md.__file__),
        cnn_sources=rr.sources(), transformer_sources=tr.sources(),
        train=train.provenance, validation=valid.provenance)
    if not args.execute:
        for architecture in ARCHITECTURES:
            cfg = config_for(architecture, 3, 42)
            model = make_encoder(architecture, 65, cfg, torch.device('cpu'))
            model.eval()
            with torch.no_grad():
                result = objective(model, values_for(train, torch.device('cpu')),
                                   torch.from_numpy(ids[:16]), cfg, scales)
            if not torch.isfinite(result.loss): raise ValueError('Smoke loss not finite')
        print(json.dumps(dict(status='smoke_ok', configurations=50, optimizer_updates=0,
                              train_valid=int(len(ids)), validation_valid=int(valid.valid_mask.sum()), output=str(OUTPUT))))
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test = None
    for architecture, dimension, seed in itertools.product(ARCHITECTURES, DIMENSIONS, SEEDS):
        config = config_for(architecture, dimension, seed)
        fit = fit_cell(architecture, config, train, valid, scales, dependencies, device)
        if test is None:
            test = md.load_test_for_export(ROOT, 'active')
            assert_disjoint(dict(train=train, valid=valid, test=test))
        export_cell(architecture, config, fit, dict(train=train, test=test), device)
        print(f'complete {architecture} dim={dimension} seed={seed}', flush=True)


if __name__ == '__main__':
    main()
