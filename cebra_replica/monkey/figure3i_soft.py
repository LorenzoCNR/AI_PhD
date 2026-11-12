"""Figure 3i Soft from the two frozen 3h position exports; never train an encoder."""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/monkey_fig3h_soft_position_v1/stage02_embedding'
OUTPUT = ROOT / 'results/figura_3_2026-10-28/i'
ARCHS = ('cnn', 'transformer')
K = 36


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load_cell(base, arch):
    folder = base / arch / 'dim_16/seed_42'
    marker = folder / 'manifest.json'
    if not marker.is_file():
        return None
    manifest = json.loads(marker.read_text(encoding='utf8'))
    if manifest['config']['output_dimension'] != 16 or manifest['config']['seed'] != 42:
        raise ValueError(f'Incorrect cell: {marker}')
    data, hashes = {}, {}
    for split in ('train', 'test'):
        path = folder / f'{split}_embedding.npz'
        hashes[split] = sha(path)
        if hashes[split] != manifest['splits'][split]['sha256']:
            raise ValueError(f'Embedding hash mismatch: {path}')
        with np.load(path, allow_pickle=False) as f:
            d = {k: f[k] for k in ('embedding_raw', 'valid_mask', 'position_xy', 'trial_id', 'source_id', 'time_id')}
        n = len(d['source_id'])
        if d['embedding_raw'].shape != (n, 16) or d['position_xy'].shape != (n, 2) or any(len(v) != n for v in d.values()):
            raise ValueError(f'Invalid shapes: {path}')
        if not np.isfinite(d['embedding_raw']).all() or not np.isfinite(d['position_xy']).all() or len(np.unique(d['source_id'])) != n:
            raise ValueError(f'Invalid values or IDs: {path}')
        mask = d.pop('valid_mask').astype(bool)
        data[split] = {k: v[mask] for k, v in d.items()}
    if len(data['train']['source_id']) < K or not len(data['test']['source_id']):
        raise ValueError('Insufficient valid rows')
    for key in ('source_id', 'trial_id'):
        if np.intersect1d(data['train'][key], data['test'][key]).size:
            raise ValueError(f'TRAIN/TEST overlap: {key}')
    return data, hashes, sha(marker)


def generate(base, output):
    cells = {arch: load_cell(base, arch) for arch in ARCHS}
    if any(cell is None for cell in cells.values()):
        return False
    a, b = (cells[arch][0]['test'] for arch in ARCHS)
    order_a, order_b = np.argsort(a['source_id']), np.argsort(b['source_id'])
    for key in ('source_id', 'trial_id', 'time_id', 'position_xy'):
        if not np.array_equal(a[key][order_a], b[key][order_b]):
            raise ValueError(f'TEST support differs between architectures: {key}')
    trial = int(np.min(a['trial_id']))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    records = {}
    for ax, arch in zip(axes, ARCHS):
        data, hashes, manifest_hash = cells[arch]
        train, test = data['train'], data['test']
        ids = np.flatnonzero(test['trial_id'] == trial)
        ids = ids[np.argsort(test['time_id'][ids])]
        if len(np.unique(test['time_id'][ids])) != len(ids):
            raise ValueError('Duplicate time IDs in trial')
        x = torch.nn.functional.normalize(torch.as_tensor(train['embedding_raw'], dtype=torch.float32), dim=1)
        q = torch.nn.functional.normalize(torch.as_tensor(test['embedding_raw'][ids], dtype=torch.float32), dim=1)
        with torch.no_grad():
            neighbors = (q @ x.T).topk(K, dim=1, largest=True, sorted=True).indices
            predicted = torch.as_tensor(train['position_xy'], dtype=torch.float32)[neighbors].mean(dim=1).numpy()
        actual = test['position_xy'][ids]
        ax.plot(actual[:, 0], actual[:, 1], color='#555555', lw=1.7, label='Posizione reale')
        ax.plot(predicted[:, 0], predicted[:, 1], color='#d62728', lw=1.5, label='Posizione prevista')
        ax.set(title=arch.upper(), xlabel='Posizione X (cm)', ylabel='Posizione Y (cm)')
        ax.set_aspect('equal', adjustable='box'); ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False)
        records[arch] = {'embedding_sha256': hashes, 'manifest_sha256': manifest_hash,
                         'train_rows': len(train['source_id']), 'trial_rows': len(ids),
                         'source_ids': test['source_id'][ids].tolist(), 'time_ids': test['time_id'][ids].tolist(),
                         'actual_xy': actual.tolist(), 'predicted_xy': predicted.tolist()}
    fig.suptitle(f'Figura 3i Soft — posizione TEST, trial {trial} | 16D, seed 42')
    output.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'figura_3i_Soft_posizione_trial_test_CNN_Transformer.{ext}', dpi=240, bbox_inches='tight')
    plt.close(fig)
    provenance = {'created_utc': datetime.now(timezone.utc).isoformat(), 'trial_id': trial,
                  'selection': 'minimum TEST trial ID, fixed independently of predictions and scores',
                  'dimension': 16, 'seed': 42, 'decoder': 'uniform kNN, cosine similarity, k=36',
                  'decoder_reference': 'TRAIN embeddings only', 'evaluation': 'TEST, strict trial-interior support',
                  'source_root': str(base), 'script_sha256': sha(Path(__file__)), 'architectures': records}
    (output / 'figura_3i_Soft_tracciabilita.json').write_text(json.dumps(provenance, indent=2), encoding='utf8')
    return True


def position_process():
    matches = []
    for p in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmd = p.info['cmdline'] or []
            if '-m' in cmd and 'user_soft.monkey_fig3h_position' in cmd and '--execute' in cmd:
                matches.append(p.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return matches


def watch(interval):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    state = OUTPUT / 'figura_3i_Soft_watcher.json'
    while True:
        ready = {a: (SOURCE / a / 'dim_16/seed_42/manifest.json').is_file() for a in ARCHS}
        pids = position_process()
        status = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'cells_ready': ready,
                  'position_process_pids': pids, 'watcher_pid': os.getpid()}
        if all(ready.values()):
            try:
                generate(SOURCE, OUTPUT)
                status['status'] = 'complete'
            except Exception as exc:
                status.update(status='validation_or_render_failed', error=repr(exc))
            state.write_text(json.dumps(status, indent=2), encoding='utf8')
            return 0 if status['status'] == 'complete' else 1
        status['status'] = 'waiting' if pids else 'position_process_not_detected'
        state.write_text(json.dumps(status, indent=2), encoding='utf8')
        if not pids:
            return 1
        time.sleep(interval)


def self_test():
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as tmp:
        base, out = Path(tmp) / 'input', Path(tmp) / 'output'
        for arch in ARCHS:
            folder = base / arch / 'dim_16/seed_42'; folder.mkdir(parents=True)
            manifest = {'config': {'output_dimension': 16, 'seed': 42}, 'splits': {}}
            for split, n, trial in [('train', 50, 1), ('test', 8, 9)]:
                path = folder / f'{split}_embedding.npz'
                # TEST coordinates and IDs are shared; embeddings need not be.
                np.savez_compressed(path, embedding_raw=rng.normal(size=(n, 16)).astype('float32'),
                                    valid_mask=np.ones(n, bool), position_xy=np.column_stack((np.arange(n), np.arange(n) * 2)),
                                    trial_id=np.full(n, trial), source_id=trial * 600 + np.arange(n), time_id=np.arange(n))
                manifest['splits'][split] = {'sha256': sha(path)}
            (folder / 'manifest.json').write_text(json.dumps(manifest), encoding='utf8')
            if arch == 'cnn':
                assert not generate(base, out) and not out.exists()
        assert generate(base, out)
        record = json.loads((out / 'figura_3i_Soft_tracciabilita.json').read_text(encoding='utf8'))
        assert record['trial_id'] == 9 and all(record['architectures'][a]['trial_rows'] == 8 for a in ARCHS)
        assert len(list(out.glob('*.png'))) == 1 and len(list(out.glob('*.pdf'))) == 1
    print('Synthetic test passed: incomplete pair blocked, complete pair rendered and traced.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.watch:
        if args.interval < 1:
            parser.error('--interval must be positive')
        sys.exit(watch(args.interval))
    else:
        sys.exit(0 if generate(SOURCE, OUTPUT) else 2)
