"""Make 2D views of the frozen Figure 2b embeddings; no fit or transform."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import rat_rebuild_fig2bc as bc
import rat_rebuild_fig2d as downstream


def bars(fig, norm, height, bottom, x1, x2):
    for x, cmap, label in ((x1, 'cool', 'Left'), (x2, 'viridis', 'Right')):
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                          cax=fig.add_axes([x, bottom, .012, height]), orientation='vertical')
        cb.set_ticks([])
        fig.text(x + .006, bottom - .02, label, ha='center', va='top', fontsize=8, rotation=90)


def scatter(ax, z, behavior, norm, title):
    for column, cmap in ((2, 'cool'), (1, 'viridis')):
        mask = behavior[:, column] == 1
        ax.scatter(z[mask, 0], z[mask, 1], c=behavior[mask, 0], cmap=cmap,
                   norm=norm, s=1.5, alpha=.8, linewidths=0, rasterized=True)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)


def render_cebra(rat):
    parents, panels, common = [], [], None
    for condition in bc.CONDS:
        folder = bc.CEBRA / rat / condition / 'seed_42'
        marker = folder / 'manifest.json'
        record = json.loads(marker.read_text())
        path = folder / 'all_embedding.npz'
        parents.extend([{'path': str(marker.relative_to(bc.ROOT)), 'sha256': bc.digest(marker)},
                        bc.verified(path, record['artifact_sha256'][path.name])])
        if record['seed'] != 42 or record['condition'] != condition or record['config']['parameters']['output_dimension'] != 32:
            raise ValueError(f'Unexpected fit: {marker}')
        with np.load(path, allow_pickle=False) as a:
            z, behavior, ids = a['embedding'].copy(), a['behavior'].copy(), a['source_id'].copy()
        if z.shape != (len(ids), 32) or behavior.shape != (len(ids), 3) or len(np.unique(ids)) != len(ids) or not np.isfinite(z).all():
            raise ValueError(f'Invalid embedding: {path}')
        if common is None:
            common = (ids, behavior)
        elif not (np.array_equal(ids, common[0]) and np.array_equal(behavior, common[1])):
            raise ValueError(f'Inconsistent support: {path}')
        panels.append((f'CEBRA: {bc.TITLES[condition]}', z[:, :2], behavior))
    folder = bc.HYBRID / rat / 'hybrid_position_time_5d' / 'seed_42'
    marker = folder / 'result.json'
    record = json.loads(marker.read_text())
    path = folder / 'all_embedding.npz'
    parents.extend([{'path': str(marker.relative_to(bc.ROOT)), 'sha256': bc.digest(marker)},
                    bc.verified(path, record['artifacts'][path.name])])
    with np.load(path, allow_pickle=False) as a:
        z, behavior, ids = a['embedding'].copy(), a['behavior'].copy(), a['source_id'].copy()
    if z.shape != (len(ids), 5) or not np.array_equal(ids, common[0]) or not np.array_equal(behavior, common[1]) or not np.isfinite(z).all():
        raise ValueError(f'Invalid hybrid support: {path}')
    # First two coordinates of the existing 3D hybrid view [1, 2, 0].
    panels.append(('CEBRA: Time + position (Hybrid)', z[:, [1, 2]], behavior))
    norm = Normalize(float(common[1][:, 0].min()), float(common[1][:, 0].max()))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (title, z, behavior) in zip(axes.flat, panels):
        scatter(ax, z, behavior, norm, title)
    fig.suptitle(rat.title(), fontsize=13)
    fig.subplots_adjust(top=.91, bottom=.05, right=.89, wspace=.06, hspace=.14)
    bars(fig, norm, .32, .34, .92, .94)
    bc.save(fig, bc.OUT / '2B' / f'{rat}_cebra_2d_vertical',
            {'rat': rat, 'seed': 42, 'fit': 'contiguous first 80%; not trial split',
             'display': 'all transformed rows; includes training and test',
             'coordinates_zero_based': {'32D': [0, 1], '5D_hybrid': [1, 2]},
             'projection': 'coordinates taken directly from existing embeddings; no fit or transformation',
             'parents': parents})


def render_soft(rat, architecture):
    data = downstream.cache(rat, 0)
    mask = data['test']['valid_mask']
    behavior = data['test']['behavior'][mask]
    norm = Normalize(float(behavior[:, 0].min()), float(behavior[:, 0].max()))
    parents, panels = [], []
    for condition, title in downstream.SC.items():
        result = downstream.load_soft(rat, 0, 42, condition, architecture, data)
        if result is None:
            raise FileNotFoundError(f'Missing shared embeddings {rat}/{architecture}/{condition}')
        arrays, paths = result
        z = arrays['test'][mask]
        if z.shape != (len(behavior), 3) or not np.isfinite(z).all():
            raise ValueError(f'Invalid Soft embedding {rat}/{architecture}/{condition}')
        panels.append((f'Soft {architecture.upper()}: {title}', z[:, :2], behavior))
        parents.extend(paths)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, (title, z, y) in zip(axes, panels):
        scatter(ax, z, y, norm, title)
    fig.suptitle(rat.title(), fontsize=12)
    fig.subplots_adjust(bottom=.08, top=.86, right=.87, wspace=.08)
    bars(fig, norm, .42, .29, .90, .92)
    bc.save(fig, bc.OUT / '2B' / f'{rat}_soft_{architecture}_2d_vertical',
            {'rat': rat, 'architecture': architecture, 'seed': 42, 'fold': 0,
             'embedding_dimension': 3, 'coordinates_zero_based': [0, 1],
             'fit_split': 'training trials', 'display_split': 'held-out test trials, valid context only',
             'projection': 'coordinates taken directly from existing embeddings; no fit or transformation',
             'comparison_warning': 'CEBRA gallery displays all transformed rows and uses a different fit/split protocol.',
             'parents': parents})


if __name__ == '__main__':
    for rat in bc.RATS:
        render_cebra(rat)
        for architecture in ('cnn', 'transformer'):
            render_soft(rat, architecture)
        print(f'{rat}: CEBRA, Soft CNN, Soft Transformer 2D complete', flush=True)
