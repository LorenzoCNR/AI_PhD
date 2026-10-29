"""Render existing Soft Direction embeddings as Figure 3b mean trajectories."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/user_soft_monkey_figure3b_direction_4d_v1'
OUT = ROOT / 'results/figura_3_2026-10-28/b'

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    names = ['FIGURE3B_SOFT_DIRECTION_CNN_TRANSFORMER.png',
             'FIGURE3B_SOFT_DIRECTION_CNN_TRANSFORMER.pdf',
             'FIGURE3B_SOFT_DIRECTION_CNN_TRANSFORMER_PROVENANCE.json']
    records = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, arch in zip(axes, ('cnn', 'transformer')):
        folder = SOURCE / arch / 'seed_42'
        path = folder / 'embedding.npz'
        manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        digest = sha256(path)
        if digest != manifest['embedding_sha256']:
            raise ValueError(f'Embedding hash mismatch: {path}')
        with np.load(path, allow_pickle=False) as data:
            z, direction = data['embedding'], data['direction']
            trial, time, valid = data['trial_id'], data['time_id'], data['valid_mask']
            source_id = data['source_id']
        if z.shape != (115800, 4) or not np.isfinite(z).all():
            raise ValueError('Unexpected embedding')
        if not (np.array_equal(trial, np.repeat(np.arange(193), 600)) and
                np.array_equal(time, np.tile(np.arange(600), 193)) and
                np.array_equal(source_id, np.arange(115800))):
            raise ValueError('Unexpected trial, time, or source alignment')
        labels = direction.reshape(193, 600)
        if not np.all(labels == labels[:, :1]) or set(labels[:, 0]) != set(range(8)):
            raise ValueError('Unexpected direction labels')
        trials = z.reshape(193, 600, 4)
        counts = {}
        for label in range(8):
            selected = trials[labels[:, 0] == label]
            mean = selected.mean(axis=0)
            ax.plot(mean[:, 2], mean[:, 0], color=plt.cm.hsv(label / 8), lw=2.3)
            counts[str(label)] = len(selected)
        ax.set_title(f'Soft Direction — {arch.upper() if arch == "cnn" else "Transformer"}', fontsize=15)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        records[arch] = {'embedding': str(path.relative_to(ROOT)), 'embedding_sha256': digest,
            'manifest': str((folder / 'manifest.json').relative_to(ROOT)),
            'training_rows': manifest['training_rows'], 'fit_scope': manifest['config']['fit_scope'],
            'plotted_rows': len(z), 'valid_mask_true': int(valid.sum()),
            'valid_mask_applied': False, 'trials_per_direction_0_to_7': counts}
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names[:2]:
        fig.savefig(OUT / name, dpi=300, bbox_inches='tight')
    plt.close(fig)
    record = {'origin': 'Existing Soft Direction 4D embeddings; no retraining',
        'aggregation': 'Mean across trials of the same direction at each trial time bin',
        'coordinates_zero_based': [2, 0], 'palette': 'matplotlib hsv(direction / 8), directions 0..7',
        'support': 'all 193 active trials, 600 time bins each; full-sample descriptive',
        'mask': 'valid_mask retained in source but not applied; complete trials as in local Figure 3b',
        'models': records, 'output_sha256': {name: sha256(OUT / name) for name in names[:2]}}
    (OUT / names[2]).write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')

if __name__ == '__main__':
    main()
