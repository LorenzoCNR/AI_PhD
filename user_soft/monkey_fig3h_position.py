"""Held-out active-monkey Position XY Soft fits. Run with --execute to fit."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from . import monkey_data as md, monkey_runner as mr, rat_runner as rr
from .data import assert_disjoint, config_hash, eligible_centers, sha256
from .loss import fit_distance_scales

ROOT = rr.ROOT
OUTPUT = ROOT / 'results/monkey_fig3h_soft_position_v1'
ARCHITECTURES = ('cnn', 'transformer')
DIMENSIONS = (3, 4, 8, 16, 32)
SEEDS = (42, 43, 44, 45, 46)


def config_for(architecture, dimension, seed):
    config = mr.frozen_config(architecture, 'behavior_only', dimension, seed)
    config.update(embedding_temperature=1.0, metadata_temperature=0.5,
                  trial_id=f'{architecture}/dim_{dimension}/seed_{seed}',
                  input_policy='legacy_concat', early_stopping=False)
    return config


def export_split(path, rows, model, config, device):
    """Export only trial-interior rows, with IDs aligned to the window center."""
    trial_rows = rows.as_trial_rows(rows.position_xy)
    ids = eligible_centers(trial_rows)
    values = rr.tensors(trial_rows, device)
    parts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ids), config['batch_size']):
            z, _ = rr.encode_batch(model, values, torch.as_tensor(ids[start:start + config['batch_size']]))
            parts.append(z.cpu().numpy())
    embedding = np.concatenate(parts)
    if embedding.shape != (len(ids), config['output_dimension']) or not np.isfinite(embedding).all():
        raise ValueError('Invalid strict-support embedding')
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, embedding_raw=embedding, valid_mask=np.ones(len(ids), dtype=bool),
                            position_xy=rows.position_xy[ids], direction=rows.direction[ids],
                            passive=rows.passive[ids], trial_id=rows.trial_id[ids],
                            source_id=rows.source_id[ids], time_id=rows.time_id[ids])
    tmp.replace(path)
    return dict(sha256=sha256(path), rows=len(ids), source_id_sha256=config_hash(rows.source_id[ids].tolist()))


def export_cell(config, fit, fit_dir, rows, device):
    folder = OUTPUT / 'stage02_embedding' / config['trial_id']
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / 'manifest.json'
    signature = config_hash(dict(config=config, checkpoint=fit['checkpoint_sha256'],
                                 data={s: rows[s].provenance for s in ('train', 'test')},
                                 script=sha256(__file__)))
    if marker.exists():
        saved = json.loads(marker.read_text(encoding='utf-8'))
        if saved['signature'] != signature or any(sha256(folder / f'{s}_embedding.npz') != saved['splits'][s]['sha256'] for s in ('train', 'test')):
            raise RuntimeError('Existing export does not match its provenance')
        return saved
    with mr.context_bindings(config):
        model = rr.make_encoder(rows['train'].neural.shape[1], config, device)
    model.load_state_dict(torch.load(fit_dir / 'checkpoint.pt', map_location=device, weights_only=False)['model'])
    splits = {s: export_split(folder / f'{s}_embedding.npz', rows[s], model, config, device) for s in ('train', 'test')}
    saved = dict(signature=signature, config=config, checkpoint_sha256=fit['checkpoint_sha256'],
                 support='strict trial interior: offsets [-5,+4]; arrays contain valid centers only',
                 embedding_raw='native encoder output', splits=splits)
    rr.atomic_json(marker, saved)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Run the 50 fits and exports')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    configs = [config_for(a, d, s) for a, d, s in itertools.product(ARCHITECTURES, DIMENSIONS, SEEDS)]
    sources = dict(script=sha256(__file__), monkey_adapter=sha256(md.__file__),
                   monkey_runner=sha256(mr.__file__), rat_runner=sha256(rr.__file__),
                   cnn=rr.sources(), transformer=mr.tr.sources())
    split_file = ROOT / 'results/monkey_figure3_v1/stage00_data/DATA_AND_SPLIT_PROVENANCE.json'
    plan = dict(configs=configs, sources=sources, split_provenance_sha256=sha256(split_file),
                objective='position XY Euclidean distance, TRAIN bounding-box scale',
                training_support='legacy concatenated split windows, including trial boundaries',
                export_support='strict trial interior only', validation='diagnostic only, terminal checkpoint',
                status='planned', optimizer_updates=0)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUTPUT / 'manifest.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding='utf-8')) != plan:
            raise RuntimeError('Campaign manifest differs; do not overwrite')
    else:
        rr.atomic_json(plan_path, plan)
    if not args.execute:
        print(json.dumps(dict(output=str(OUTPUT), planned=len(configs), optimizer_updates=0)))
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device = torch.device(args.device)
    torch.set_num_threads(2)
    train_valid = md.load_train_validation(ROOT, 'active')
    train = train_valid['train'].as_trial_rows(train_valid['train'].position_xy)
    valid = train_valid['valid'].as_trial_rows(train_valid['valid'].position_xy)
    specs = mr.specs_for('behavior_only')
    scales = fit_distance_scales(train.metadata(), specs, split='train', source_ids=train.source_id)
    parent = config_hash(dict(provenance={s: r.provenance for s, r in train_valid.items()}, scales=scales))
    test = None
    for config in configs:
        fit_dir = OUTPUT / 'stage01_fit' / config['trial_id']
        with mr.context_bindings(config):
            fit, _ = rr.fit_cell(fit_dir, train, valid, config, scales, parent, sources, device)
        if test is None:
            test = md.load_test_for_export(ROOT, 'active')
            assert_disjoint({'train': train, 'valid': valid, 'test': test.as_trial_rows(test.position_xy)})
        export_cell(config, fit, fit_dir, {'train': train_valid['train'], 'test': test}, device)
        print(f"COMPLETE {config['trial_id']}", flush=True)


if __name__ == '__main__':
    main()
