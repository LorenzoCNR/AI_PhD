"""Rebuild rat train-consistency and held-out decoding from saved embeddings."""
from __future__ import annotations
import csv
import hashlib
import json
from collections import defaultdict
from itertools import product
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from user_soft.data import sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rat_rebuild_20261027/Figure_1E_ED4'
CAMP = ROOT / 'results/rat_paper_fig1e_ed4_6seed_v1'
BASE = ROOT / 'results/rat_manifold_baselines_v1'
RATS = ('achilles', 'buddy', 'cicero', 'gatsby')
SEEDS = tuple(range(42, 48))
FOLDS = range(3)
KS = (1, 4, 9, 16, 25)
METHODS = [('CEBRA behavior (position + direction)', 'cebra', 'behavior'),
           ('CEBRA time', 'cebra', 'time')]
for cond in ('time_direction_position', 'time_only', 'direction_position'):
    for arch in ('cnn', 'transformer'):
        METHODS.append((f'Soft {arch} {cond}', 'soft', (cond, arch)))
METHODS += [('conv-piVAE q(z|x)', 'pivae', 'z_mean_raw'),
            ('conv-piVAE q(z|x,u)', 'pivae', 'post_mean_behavior_informed_raw'),
            ('UMAP', 'base', 'umap'), ('t-SNE', 'base', 'opentsne'),
            ('PCA', 'base', 'pca')]


def resolve(kind, variant, rat, fold, seed, split):
    if kind == 'cebra':
        folder = CAMP / 'cebra' / rat / f'fold_{fold}' / variant / f'seed_{seed}'
        record = json.loads((folder / 'manifest.json').read_text())
        p = folder / 'embeddings' / f'{split}.npz'
        expected = record['embeddings'][split]['sha256']
        key, ykey = 'embedding', 'behavior'
    elif kind == 'soft':
        condition, arch = variant
        folder = OUT / 'embeddings/soft' / condition / arch / rat / f'fold_{fold}' / 'soft_v1_stable' / f'seed_{seed}'
        p = folder / f'{split}.npz'
        if not p.is_file():
            raise RuntimeError(f'Missing Soft staged embedding: {p}')
        expected = None
        key, ykey = 'embedding', 'behavior'
    elif kind == 'pivae':
        folders = [p for p in (CAMP / 'pivae/runs' / rat / f'fold_{fold}' / f'seed_{seed}').glob('*/run.json')
                   if json.loads(p.read_text()).get('status') == 'complete']
        if len(folders) != 1:
            raise RuntimeError(f'Expected one piVAE manifest: {rat}/{fold}/{seed}, got {len(folders)}')
        record = json.loads(folders[0].read_text())
        p = folders[0].parent / f'embedding_{split}.npz'
        expected = record['split_embeddings'][split]['embedding_sha256']
        key, ykey = variant, 'u'
    else:
        folder = BASE / variant / rat / f'fold_{fold}' / ('seed_deterministic' if variant == 'pca' else f'seed_{seed}')
        record = json.loads((folder / 'result.json').read_text())
        p = folder / f'{split}_embedding.npz'
        expected = record['embeddings'][split]['sha256']
        key, ykey = 'embedding_raw', 'behavior'
    digest = sha256(p)
    if expected and expected != digest:
        raise RuntimeError(f'Embedding hash mismatch: {p}')
    with np.load(p, allow_pickle=False) as a:
        z = np.asarray(a[key], dtype=np.float32)
        y = np.asarray(a[ykey], dtype=np.float32)
        ids = np.asarray(a['source_id'], dtype=np.int64)
        mask = np.asarray(a['valid_mask'], dtype=bool) if kind == 'base' else np.ones(len(ids), dtype=bool)
        if kind == 'soft':
            condition, arch = variant
            original = CAMP / 'soft/stage01_fit' / condition / arch / rat / f'fold_{fold}' / 'soft_v1_stable' / f'seed_{seed}'
            side = json.loads((original / 'CELL_MANIFEST.json').read_text())
            cache = CAMP / 'soft/stage00_data' / rat / f'fold_{fold}' / f'{split}.npz'
            if str(a['checkpoint_sha256'].item()) != side['checkpoint_sha256'] or sha256(original / 'checkpoint.pt') != side['checkpoint_sha256']:
                raise RuntimeError(f'Soft checkpoint provenance mismatch: {p}')
            if str(a['source_npz_sha256'].item()) != sha256(cache):
                raise RuntimeError(f'Soft data provenance mismatch: {p}')
        if len(np.unique(ids)) != len(ids) or z.shape[0] != len(ids) or y.shape != (len(ids), 3):
            raise RuntimeError(f'Invalid row identity/shape: {p}')
        z, y, ids = z[mask], y[mask], ids[mask]
        if not np.isfinite(z).all() or not np.isfinite(y).all() or len(ids) < 26:
            raise RuntimeError(f'Nonfinite or insufficient valid rows: {p}')
        return z, y, ids, str(p.relative_to(ROOT)), digest


def r2(x, y):
    return float(LinearRegression().fit(x, y).score(x, y))


def decode(train, valid, test, device):
    x, y = train[:2]
    if not np.allclose(y[:, 1:].sum(axis=1), 1):
        raise RuntimeError('Direction labels not one-hot')
    train_x = torch.nn.functional.normalize(torch.as_tensor(x, device=device), dim=1)
    labels = y[:, 1:].argmax(axis=1)
    outputs = []
    for data in (valid, test):
        q = torch.nn.functional.normalize(torch.as_tensor(data[0], device=device), dim=1)
        neighbors = []
        with torch.inference_mode():
            for offset in range(0, len(q), 512):
                neighbors.append(torch.topk(q[offset:offset + 512] @ train_x.T, max(KS), dim=1).indices.cpu().numpy())
        ids = np.concatenate(neighbors)
        predictions = {}
        for k in KS:
            subset = ids[:, :k]
            pos = y[subset, 0].mean(axis=1)
            votes = (labels[subset] == 1).sum(axis=1)
            direction = (votes > k / 2).astype(int)
            onehot = np.eye(2, dtype=np.float32)[direction]
            predictions[k] = np.column_stack((pos, onehot))
        outputs.append(predictions)
    scores = {k: float(r2_score(valid[1], outputs[0][k])) for k in KS}
    k = max(KS, key=lambda n: scores[n])
    pred = outputs[1][k]
    return k, scores[k], float(r2_score(test[1], pred)), float(np.median(np.abs(test[1][:, 0] - pred[:, 0])))


def bins(z, y, normalized):
    if np.any(y[:, 0] < 0) or np.any(y[:, 0] > 1.60001):
        raise RuntimeError('position outside 0..1.6 m')
    if not np.allclose(y[:, 1:].sum(axis=1), 1):
        raise RuntimeError('direction labels not one-hot')
    indexes = np.minimum((y[:, 0] / 1.6 * 100).astype(int), 99) + y[:, 1:].argmax(axis=1) * 100
    counts = np.bincount(indexes, minlength=200)
    sums = np.zeros((200, z.shape[1]), dtype=np.float64)
    np.add.at(sums, indexes, z)
    means = np.divide(sums, counts[:, None], out=np.full_like(sums, np.nan), where=counts[:, None] > 0)
    # Only use immediate observed neighboring bins; do not invent extrapolation.
    for i in np.flatnonzero(counts == 0):
        if i % 100 == 0 or i % 100 == 99 or counts[i - 1] == 0 or counts[i + 1] == 0:
            return None, int((counts == 0).sum())
        means[i] = (sums[i - 1] + sums[i + 1]) / (counts[i - 1] + counts[i + 1])
    if normalized:
        norms = np.linalg.norm(means, axis=1)
        if np.any(norms == 0):
            return None, int((counts == 0).sum())
        means /= norms[:, None]
    return means, int((counts == 0).sum())


def write_csv(path, rows):
    if path.exists():
        raise RuntimeError(f'Refusing to overwrite existing output: {path}')
    with path.open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main():
    torch.set_num_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    OUT.mkdir(parents=True, exist_ok=True)
    inventory, decoding, run_rows, cross_rows, blockers = [], [], [], [], []
    pieces = OUT / 'metric_parts'
    pieces.mkdir(exist_ok=True)
    for method_index, (label, kind, variant) in enumerate(METHODS):
        print(f'METHOD {label}', flush=True)
        part = pieces / f'method_{method_index:02d}.json'
        if part.exists():
            payload = json.loads(part.read_text(encoding='utf-8'))
            if payload['method'] != label or payload['script_sha256'] != sha256(Path(__file__)):
                raise RuntimeError(f'Existing metric part has incompatible provenance: {part}')
            if any(sha256(ROOT / row['path']) != row['sha256'] for row in payload['inventory']):
                raise RuntimeError(f'Input changed since metric part was computed: {part}')
            inventory.extend(payload['inventory']); decoding.extend(payload['decoding'])
            run_rows.extend(payload['run_rows']); cross_rows.extend(payload['cross_rows']); blockers.extend(payload['blockers'])
            print(f'REUSED {label}', flush=True)
            continue
        starts = tuple(map(len, (inventory, decoding, run_rows, cross_rows, blockers)))
        for fold in FOLDS:
            trains = {}
            for rat in RATS:
                for seed in (SEEDS[:1] if variant == 'pca' else SEEDS):
                    cell = {}
                    for split in ('train', 'valid', 'test'):
                        item = resolve(kind, variant, rat, fold, seed, split)
                        cell[split] = item
                        inventory.append(dict(method=label, rat=rat, fold=fold, seed=seed, split=split, rows=len(item[2]), path=item[3], sha256=item[4]))
                    for first, second in (('train', 'valid'), ('train', 'test'), ('valid', 'test')):
                        if np.intersect1d(cell[first][2], cell[second][2]).size:
                            raise RuntimeError(f'Split source ID overlap: {label}/{rat}/{fold}/{seed}/{first}/{second}')
                    trains[(rat, seed)] = cell['train']
                    if kind == 'pivae' and variant == 'post_mean_behavior_informed_raw':
                        continue
                    k, v, total, error = decode(cell['train'], cell['valid'], cell['test'], device)
                    decoding.append(dict(method=label, rat=rat, fold=fold, seed=seed, selected_k=k,
                        validation_total_r2=v, test_total_r2=total, test_position_error_m=error,
                        train_rows=len(cell['train'][2]), valid_rows=len(cell['valid'][2]), test_rows=len(cell['test'][2])))
            if variant != 'pca':
                for rat in RATS:
                    for seed in SEEDS:
                        a = trains[(rat, seed)]
                        for peer in SEEDS:
                            if seed == peer: continue
                            b = trains[(rat, peer)]
                            if not np.array_equal(a[2], b[2]):
                                raise RuntimeError(f'Across-run train source ID mismatch {label}/{rat}/{fold}/{seed}/{peer}')
                            run_rows.append(dict(method=label, rat=rat, fold=fold, seed=seed, peer_seed=peer,
                                                 train_rows=len(a[2]), train_run_r2=r2(a[0], b[0])))
            seed_grid = SEEDS[:1] if variant == 'pca' else SEEDS
            for seed in seed_grid:
                binned = {}
                for rat in RATS:
                    z, y = trains[(rat, seed)][:2]
                    binned[rat], missing = bins(z, y, kind in ('cebra', 'soft'))
                    if binned[rat] is None:
                        blockers.append(dict(method=label, rat=rat, fold=fold, seed=seed,
                            issue='Missing position-direction bins lack immediate observed neighbors; no extrapolation', missing_bins=missing))
                for rat in RATS:
                    for peer in RATS:
                        if rat != peer and binned[rat] is not None and binned[peer] is not None:
                            cross_rows.append(dict(method=label, rat=rat, peer_rat=peer, fold=fold, seed=seed,
                                                   train_cross_subject_r2=r2(binned[rat], binned[peer])))
        payload = dict(method=label, script_sha256=sha256(Path(__file__)),
            inventory=inventory[starts[0]:], decoding=decoding[starts[1]:],
            run_rows=run_rows[starts[2]:], cross_rows=cross_rows[starts[3]:], blockers=blockers[starts[4]:])
        temporary = part.with_suffix('.json.tmp')
        with temporary.open('x', encoding='utf-8') as f: json.dump(payload, f)
        temporary.rename(part)
    write_csv(OUT / 'embedding_inventory.csv', inventory)
    write_csv(OUT / 'heldout_decoding.csv', decoding)
    write_csv(OUT / 'train_run_consistency.csv', run_rows)
    if cross_rows: write_csv(OUT / 'train_cross_subject_consistency.csv', cross_rows)
    if blockers: write_csv(OUT / 'cross_subject_blockers.csv', blockers)
    manifest = dict(status='partial' if blockers else 'complete', training_performed=False,
        paper_methods='https://www.nature.com/articles/s41586-023-06031-6',
        paper_ed4_plot='https://cebra.ai/docs/cebra-figures/figures/ExtendedDataFigure4.html',
        embedding_count=len(inventory), decoding_cells=len(decoding), train_run_directed_pairs=len(run_rows),
        train_cross_subject_directed_pairs=len(cross_rows), cross_subject_blockers=len(blockers),
        train_consistency='Linear regression with intercept, fit and score on train embedding. Runs: 30 directed seed pairs per rat/fold.',
        across_subject='100 bins per direction, 0..1.6m, fill only from both immediately adjacent observed bins; normalize CEBRA/Soft bin means; same seed and fold across subjects.',
        heldout_decoding='cosine kNN; position regression and direction classification; k=[1,4,9,16,25] selected by validation R2 of position plus two one-hot direction outputs; test is held out.',
        notes=['Six seeds rather than ten. Baseline train data use valid_mask; CEBRA/Soft/piVAE use all recorded train rows.',
               't-SNE reused baseline perplexity 30, not the Figure 1e rat benchmark perplexity 10.',
               'piVAE q(z|x,u) uses behavior during inference and is excluded from decoding.',
               'Paper plotting notebook does not specify all aggregation and interpolation details; this is an explicit implementation, not claimed bitwise reproduction.',
               'PCA is deterministic and has no independent across-run metric.'],
        source_script_sha256=sha256(Path(__file__)), exporter_script_sha256=sha256(ROOT/'cebra_replica/rat_rebuild_1e_ed4_export_soft.py'))
    p = OUT / 'manifest.json'
    with p.open('x', encoding='utf-8') as f: json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
