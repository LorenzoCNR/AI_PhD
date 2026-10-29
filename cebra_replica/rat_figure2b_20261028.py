"""Render Figure 2b from existing rat embeddings without fitting models."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import rat_rebuild_fig2bc as source

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'figura_2_2026-10-28' / 'b'


def read(rat, condition):
    folder = source.CEBRA / rat / condition / 'seed_42'
    marker = folder / 'manifest.json'
    record = json.loads(marker.read_text(encoding='utf-8'))
    path = folder / 'all_embedding.npz'
    provenance = source.verified(path, record['artifact_sha256'][path.name])
    if record['seed'] != 42 or record['condition'] != condition:
        raise ValueError(f'Unexpected fit: {marker}')
    with np.load(path, allow_pickle=False) as a:
        z, behavior, ids = (a[k].copy() for k in ('embedding', 'behavior', 'source_id'))
    return z, behavior, ids, provenance


def read_mixed(rat):
    folder = source.HYBRID / rat / 'hybrid_position_time_5d' / 'seed_42'
    marker = folder / 'result.json'
    record = json.loads(marker.read_text(encoding='utf-8'))
    path = folder / 'all_embedding.npz'
    provenance = source.verified(path, record['artifacts'][path.name])
    with np.load(path, allow_pickle=False) as a:
        z, behavior, ids = (a[k].copy() for k in ('embedding', 'behavior', 'source_id'))
    if z.shape[1] != 5:
        raise ValueError(f'Unexpected embedding size: {path}')
    return z, behavior, ids, provenance


def main():
    for rat in source.RATS:
        time = read(rat, 'time')
        position = read(rat, 'position')
        mixed = read_mixed(rat)
        shuffled = read(rat, 'position_shuffled')
        for item in (position, mixed, shuffled):
            if not np.array_equal(time[2], item[2]) or not np.array_equal(time[1], item[1]):
                raise ValueError(f'Incompatible rows: {rat}')
        panels = [(None, None, 'CEBRA — Position\n(input: dati comportamentali)'),
                  (time, [0, 1, 2], 'CEBRA — Time'),
                  (position, [0, 1, 2], 'CEBRA — Position'),
                  (mixed, [1, 2, 0], 'CEBRA — Time + Position'),
                  (shuffled, [0, 1, 2], 'CEBRA — Position shuffled')]
        norm = Normalize(0, 1.6)
        fig = plt.figure(figsize=(20, 5))
        for i, (item, dims, title) in enumerate(panels, 1):
            ax = fig.add_subplot(1, 5, i, projection='3d')
            ax.set_title(title, fontsize=11)
            ax.set_axis_off()
            if item is None:
                ax.text2D(.5, .5, 'Fit non disponibile', ha='center', transform=ax.transAxes)
                continue
            z, y = item[0][:, dims], item[1]
            for col, cmap in ((2, 'cool'), (1, 'viridis')):
                mask = y[:, col] == 1
                ax.scatter(*z[mask].T, c=y[mask, 0], cmap=cmap, norm=norm, s=.7, linewidths=0, rasterized=True)
        for x, cmap, label in ((.91, 'cool', 'Left'), (.95, 'viridis', 'Right')):
            bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([x, .25, .012, .5]))
            bar.set_ticks([0, .5, 1, 1.5])
            fig.text(x+.006, .20, label, ha='center', fontsize=9)
        fig.suptitle(rat.capitalize(), fontsize=14)
        fig.subplots_adjust(left=.01, right=.89, top=.86, bottom=.12, wspace=.05)
        dest = OUT / rat.capitalize()
        dest.mkdir(parents=True, exist_ok=True)
        stem = dest / f'Figura_2b_{rat.capitalize()}_CEBRA'
        for extension in ('.png', '.pdf'):
            output = stem.with_suffix(extension)
            temporary = output.with_name(output.stem + '_temp' + extension)
            fig.savefig(temporary, dpi=220, bbox_inches='tight')
            temporary.replace(output)
        plt.close(fig)
        metadata = {'rat': rat, 'seed': 42, 'training_performed': False,
                    'input_first_panel': 'behavioral data; position label; fit unavailable; exact input columns unverified',
                    'plotted_rows': 'all transformed recording rows',
                    'fit_support': 'first 80% of recording; see source manifests',
                    'panels': [{'title': title, 'coordinates_zero_based': dims,
                                'source': None if item is None else item[3],
                                'fit_dimension': None if item is None else int(item[0].shape[1])}
                               for item, dims, title in panels],
                    'shuffled': 'position labels permuted for training; plotted colors use true labels',
                    'reference': 'https://cebra.ai/docs/cebra-figures/figures/Figure2.html'}
        (dest / 'RIFERIMENTI.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(rat, 'ok', flush=True)
    (OUT.parent / 'RIFERIMENTI.md').write_text('# Figura 2b\n\nCodice: `cebra_replica/rat_figure2b_20261028.py`. Ogni ratto ha un file RIFERIMENTI.json con sorgenti e hash. Il primo pannello richiede un fit su dati comportamentali non disponibile; non è sostituito da un fit neurale. Nessun nuovo training.\n', encoding='utf-8')


if __name__ == '__main__':
    main()
