"""Held-out monkey Figure 3h AP: Soft XY fits and frozen embedding exports.

Run `python -m user_soft.monkey_fig3h_ap --smoke` for a no-training check,
or `python -m user_soft.monkey_fig3h_ap --execute` for the 50-fit campaign.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import torch

from . import monkey_data as md, monkey_runner as mr, rat_runner as rr, transformer_runner as tr
from .data import TrialRows, assert_disjoint, config_hash, eligible_centers, sha256
from .loss import evaluate_soft, fit_distance_scales

ROOT = rr.ROOT
OUT = ROOT / 'results/monkey_fig3h_soft_ap_v1'
ARCHITECTURES = ('cnn', 'transformer')
DIMENSIONS = (3, 4, 8, 16, 32)
SEEDS = (42, 43, 44, 45, 46)
SPECS = [dict(key='behavior', geometry='euclidean', weight=1.)]


def config(architecture, dimension, seed):
    factory = rr.frozen_config if architecture == 'cnn' else tr.frozen_config
    c = factory('monkey_all', 0, seed, 'soft_v1_stable', 10000)
    c.update(output_dimension=dimension, condition='behavior_only', metadata_specs=SPECS,
             population='all', behavior_target='position_xy_joint', time_bin_ms=1,
             embedding_temperature=1., metadata_temperature=.5,
             input_policy='legacy_concat', offset_left=5, offset_right=5)
    return c


def sources():
    return dict(script=sha256(__file__), monkey_adapter=sha256(md.__file__),
                monkey_runner=sha256(mr.__file__), cnn=rr.sources(), transformer=tr.sources())


def path_for(stage, architecture, dimension, seed):
    return OUT / stage / architecture / f'dim_{dimension}' / f'seed_{seed}'


def export_one(architecture, dimension, seed, c, fit, rows, device):
    folder = path_for('stage02_embedding', architecture, dimension, seed)
    marker = folder / 'manifest.json'
    signature = config_hash(dict(checkpoint=fit['checkpoint_sha256'], config=c,
                                 sources=sources(), data={k: v.provenance for k, v in rows.items()}))
    if marker.exists():
        old = json.loads(marker.read_text(encoding='utf-8'))
        if old['signature'] != signature or any(sha256(folder / name) != h for name, h in old['artifacts'].items()):
            raise RuntimeError(f'Embedding provenance mismatch: {folder}')
        return
    folder.mkdir(parents=True, exist_ok=True)
    with mr.context_bindings(c):
        model = rr.make_encoder(65, c, device)
    checkpoint = path_for('stage01_fit', architecture, dimension, seed) / 'checkpoint.pt'
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)['model'])
    model.eval()
    artifacts = {}
    for split in ('train', 'test'):
        source = rows[split]
        trial_rows = source.as_trial_rows(source.position_xy)
        labels = {k: getattr(source, k) for k in ('position_xy', 'direction', 'passive')}
        payload = mr.exported_arrays(trial_rows, labels, c, model, device)
        if not np.array_equal(payload['valid_mask'], source.valid_mask):
            raise ValueError('Strict valid support mismatch')
        if not np.array_equal(payload['source_id'], payload['trial_id'] * 600 + payload['time_id']):
            raise ValueError('Source identity mismatch')
        target = folder / f'{split}_embedding.npz'
        temporary = folder / f'{split}_embedding.tmp.npz'
        np.savez_compressed(temporary, **payload)
        temporary.replace(target)
        artifacts[target.name] = sha256(target)
    rr.atomic_json(marker, dict(signature=signature, config=c, artifacts=artifacts,
        checkpoint_sha256=fit['checkpoint_sha256'], splits={k: rows[k].provenance for k in rows},
        valid_support='within trial offsets [-5,+4]; use valid_mask',
        input_policy='legacy concatenation; invalid edge rows must be masked',
        labels='passive and direction exported only; XY alone supervises encoder'))


def execute(args):
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; choose --device cpu explicitly if intended')
    device = torch.device(args.device)
    torch.set_num_threads(2)
    train_valid = md.load_train_validation(ROOT, 'all')
    train = train_valid['train'].as_trial_rows(train_valid['train'].position_xy)
    valid = train_valid['valid'].as_trial_rows(train_valid['valid'].position_xy)
    scales = fit_distance_scales(train.metadata(), SPECS, split='train', source_ids=train.source_id)
    source_hashes = sources()
    parent = config_hash(dict(train=train_valid['train'].provenance,
                              valid=train_valid['valid'].provenance, scales=scales))
    plans = [(a, d, s) for a, d, s in itertools.product(ARCHITECTURES, DIMENSIONS, SEEDS)
             if (args.architecture is None or args.architecture == a)
             and (args.dimension is None or args.dimension == d)
             and (args.seed is None or args.seed == s)]
    rr.atomic_json(OUT / 'CAMPAIGN_MANIFEST.json', dict(task='figure3h_ap', population='all',
        target='position_xy only', planned_total=50, dimensions=DIMENSIONS, seeds=SEEDS,
        architectures=ARCHITECTURES, fit_split='train', validation='diagnostics only',
        test='export only after each completed fit', train_scales=scales,
        train_validation={k: v.provenance for k, v in train_valid.items()}, sources=source_hashes))
    test = None
    for architecture, dimension, seed in plans:
        c = config(architecture, dimension, seed)
        folder = path_for('stage01_fit', architecture, dimension, seed)
        with mr.context_bindings(c):
            fit, _ = rr.fit_cell(folder, train, valid, c, scales, parent, source_hashes, device)
        if test is None:
            test = md.load_test_for_export(ROOT, 'all')
            assert_disjoint({'train': train, 'valid': valid, 'test': test.as_trial_rows(test.position_xy)})
        export_one(architecture, dimension, seed, c, fit,
                   {'train': train_valid['train'], 'test': test}, device)
        print(f'complete {architecture} dim={dimension} seed={seed}', flush=True)


def smoke():
    # Synthetic forward/loss only: no optimizer, checkpoints, or production data writes.
    n = 40
    rng = np.random.default_rng(7)
    rows = TrialRows(rng.normal(size=(n, 65)).astype('float32'),
        rng.normal(size=(n, 2)).astype('float32'), np.arange(n), np.repeat([0, 1], 20),
        np.tile(np.arange(20), 2), np.zeros(n, dtype='int64'), 'train')
    scales = fit_distance_scales(rows.metadata(), SPECS, split='train', source_ids=rows.source_id)
    centers = eligible_centers(rows)
    for architecture in ARCHITECTURES:
        c = config(architecture, 3, 42)
        with mr.context_bindings(c):
            model = rr.make_encoder(65, c, torch.device('cpu'))
            model.eval()
            with torch.no_grad():
                z, batch = rr.encode_batch(model, rr.tensors(rows, torch.device('cpu')),
                                           torch.as_tensor(centers[:8]))
                result = evaluate_soft(z, batch, SPECS, version=c['version'], scales=scales,
                                       temperature=1., metadata_temperature=.5)
        assert z.shape == (8, 3) and torch.isfinite(result.loss)
    assert len(centers) == 22 and len(list(itertools.product(ARCHITECTURES, DIMENSIONS, SEEDS))) == 50
    print('Smoke OK: CNN + Transformer forward/loss, strict trial mask, 50 configurations; optimizer updates=0')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--architecture', choices=ARCHITECTURES)
    parser.add_argument('--dimension', type=int, choices=DIMENSIONS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if args.smoke:
        smoke()
    else:
        execute(args)


if __name__ == '__main__':
    main()
