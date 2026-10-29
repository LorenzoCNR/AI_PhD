"""Render only rat Figure 2B from frozen embeddings; no encoder fitting.

Writes three model-specific figures per rat into the existing dated staging
branch. Soft figures show 2D and 3D coordinate views of the same 32D fit.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

from rat_revision_common import (
    ROOT, STAGE, RATS, embedding, fit_manifest, save, sha, write_csv,
)

OUT = STAGE / 'Figure_2' / '2B'
SOFT_CONDITIONS = ('behavior_only', 'time_only', 'joint')
SOFT_TITLES = {
    'behavior_only': 'Behavior (Position + direction)',
    'time_only': 'Time',
    'joint': 'Time + Behavior (Position + direction)',
}
CEBRA_PANELS = (
    ('position', 'CEBRA-Behavior\n(neural input; position hypothesis)', [0, 1, 2]),
    ('position_shuffled', 'Position shuffle\n(neural input)', [0, 1, 2]),
    ('time', 'CEBRA-Time\n(neural input)', [0, 1, 2]),
    ('hybrid', 'CEBRA-Hybrid\n(Time + Behavior)', [1, 2, 0]),
)


def manifest_dimensions(value):
    """Return every declared output_dimension nested in a manifest."""
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'output_dimension':
                found.append(int(item))
            else:
                found.extend(manifest_dimensions(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(manifest_dimensions(item))
    return found


def read_cebra_embedding(path, marker, expected_hash, artifact):
    if sha(path) != expected_hash:
        raise RuntimeError(f'Frozen CEBRA embedding hash mismatch: {path}')
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in ('embedding', 'behavior', 'source_id')}
    if data['embedding'].ndim != 2 or data['behavior'].shape != (len(data['source_id']), 3):
        raise RuntimeError(f'Invalid CEBRA embedding/behavior shape: {path}')
    if any(len(value) != len(data['source_id']) for value in data.values()):
        raise RuntimeError(f'CEBRA sample count mismatch: {path}')
    return data, [
        dict(path=path.relative_to(ROOT).as_posix(), sha256=expected_hash, artifact=artifact),
        dict(path=marker.relative_to(ROOT).as_posix(), sha256=sha(marker), artifact='fit manifest'),
    ]


def cebra_data_for_rat(rat):
    panels = {}
    parents = []
    for condition in ('position', 'position_shuffled'):
        folder = ROOT / 'results' / 'rat_figure2c_3d_v1' / rat / condition / 'seed_42'
        marker = folder / 'result.json'
        record = json.loads(marker.read_text(encoding='utf-8'))
        artifact = 'all_embedding.npz'
        data, lineage = read_cebra_embedding(
            folder / artifact, marker, record['artifacts'][artifact], artifact)
        if int(data['embedding'].shape[1]) != int(record['output_dimension']):
            raise RuntimeError(f'CEBRA fit/output dimension mismatch: {rat}/{condition}')
        panels[condition] = (data, record, lineage)
        parents.extend(lineage)

    folder = ROOT / 'results' / 'rat_trial_benchmark_v1' / rat / 'fold_0' / 'time' / 'seed_42'
    marker = folder / 'result.json'
    record = json.loads(marker.read_text(encoding='utf-8'))
    time_parts = []
    time_lineage = [dict(path=marker.relative_to(ROOT).as_posix(), sha256=sha(marker), artifact='fit manifest')]
    for split in ('train', 'valid', 'test'):
        artifact = f'{split}_embedding.npz'
        data, lineage = read_cebra_embedding(
            folder / artifact, marker, record['embeddings'][split]['embedding_sha256'], artifact)
        time_parts.append(data)
        time_lineage.extend(lineage[:1])
    time_data = {key: np.concatenate([part[key] for part in time_parts]) for key in time_parts[0]}
    if len(np.unique(time_data['source_id'])) != len(time_data['source_id']):
        raise RuntimeError(f'Repeated CEBRA-Time sample IDs: {rat}')
    if int(time_data['embedding'].shape[1]) != int(record['parameters']['output_dimension']):
        raise RuntimeError(f'CEBRA-Time fit/output dimension mismatch: {rat}')
    panels['time'] = (time_data, record, time_lineage)
    parents.extend(time_lineage)

    folder = ROOT / 'results' / 'rat_figure_gap_closure_v1' / 'figure2b_fits' / rat / 'hybrid_position_time_5d' / 'seed_42'
    marker = folder / 'result.json'
    record = json.loads(marker.read_text(encoding='utf-8'))
    artifact = 'all_embedding.npz'
    data, lineage = read_cebra_embedding(
        folder / artifact, marker, record['artifacts'][artifact], artifact)
    if int(data['embedding'].shape[1]) != int(record['output_dimension']):
        raise RuntimeError(f'CEBRA-Hybrid fit/output dimension mismatch: {rat}')
    panels['hybrid'] = (data, record, lineage)
    parents.extend(lineage)

    reference_ids = None
    reference_behavior = None
    for key, (data, _, _) in panels.items():
        order = np.argsort(data['source_id'])
        data['embedding'] = data['embedding'][order]
        data['behavior'] = data['behavior'][order]
        data['source_id'] = data['source_id'][order]
        if len(np.unique(data['source_id'])) != len(data['source_id']):
            raise RuntimeError(f'Repeated CEBRA source IDs: {rat}/{key}')
        if reference_ids is None:
            reference_ids = data['source_id']
            reference_behavior = data['behavior']
        elif not np.array_equal(data['source_id'], reference_ids) or not np.array_equal(data['behavior'], reference_behavior):
            raise RuntimeError(f'CEBRA all-row support mismatch: {rat}/{key}')
    return panels, reference_behavior, parents


def draw_3d(ax, points, behavior, norm, title, note, title_color='black'):
    if points.shape != (len(behavior), 3):
        raise ValueError('3D panel needs aligned three-coordinate points')
    for column, cmap in ((2, 'cool'), (1, 'viridis')):
        mask = behavior[:, column] == 1
        ax.scatter(*points[mask].T, c=behavior[mask, 0], cmap=cmap, norm=norm,
                   s=2.0, linewidths=0, alpha=.86, rasterized=True)
    ax.set_axis_off()
    ax.set_title(title, color=title_color, fontsize=11, pad=2)
    ax.text2D(.5, -.025, note, ha='center', va='top', transform=ax.transAxes,
              fontsize=7.5, linespacing=1.25)


def draw_2d(ax, points, behavior, norm, title, note):
    if points.shape != (len(behavior), 2):
        raise ValueError('2D panel needs aligned two-coordinate points')
    for column, cmap in ((2, 'cool'), (1, 'viridis')):
        mask = behavior[:, column] == 1
        ax.scatter(points[mask, 0], points[mask, 1], c=behavior[mask, 0],
                   cmap=cmap, norm=norm, s=1.8, linewidths=0,
                   alpha=.86, rasterized=True)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_axis_off()
    ax.set_title(title, color='#9d1c1c', fontsize=10.5, pad=2)
    ax.text(.5, -.025, note, ha='center', va='top', transform=ax.transAxes,
            fontsize=7.5, linespacing=1.25)


def add_vertical_position_bars(fig, norm, left, right, bottom=.25, height=.52):
    for x, cmap, label, side in (
        (left, 'cool', 'Left | position', 'left'),
        (right, 'viridis', 'Right | position', 'right'),
    ):
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                           cax=fig.add_axes([x, bottom, .012, height]),
                           orientation='vertical')
        bar.set_ticks([])
        bar.set_label(label, fontsize=8, labelpad=4)
        bar.ax.yaxis.set_label_position(side)
        bar.outline.set_visible(False)


def render_cebra(rat):
    panels, reference_behavior, parents = cebra_data_for_rat(rat)
    norm = Normalize(float(reference_behavior[:, 0].min()),
                     float(reference_behavior[:, 0].max()))
    fig = plt.figure(figsize=(18, 6.5), facecolor='white')
    for index, (key, title, dims) in enumerate(CEBRA_PANELS, start=1):
        data, record, _ = panels[key]
        fit_dimension = int(data['embedding'].shape[1])
        plotted = data['embedding'][:, dims]
        if plotted.shape[1] != 3:
            raise RuntimeError(f'Invalid CEBRA 3D view: {rat}/{key}')
        if key in ('position', 'position_shuffled'):
            fit_steps = int(record['max_iterations'])
            sample_note = 'all rows transformed'
        elif key == 'time':
            fit_steps = int(record['max_iterations'])
            sample_note = 'train + valid + test rows'
        else:
            fit_steps = int(record['max_iterations'])
            sample_note = 'all rows transformed'
        note = f'Neural input | fit {fit_dimension}D, {fit_steps:,} updates\n'
        note += f'plot coords {",".join(str(d + 1) for d in dims)} | {sample_note} (n={len(data["source_id"]):,})'
        if key == 'position_shuffled':
            note += '\nposition labels shuffled for fit'
        if key == 'hybrid':
            note += '\nfit: first 3 behavior+time; last 2 time only'
        ax = fig.add_subplot(1, 4, index, projection='3d')
        draw_3d(ax, plotted, data['behavior'], norm, title, note)

    fig.suptitle(f'{rat.capitalize()} | CEBRA frozen neural-input fits | all rows shown',
                 y=.98, fontsize=13)
    fig.text(.5, .115,
             '3D displays use the stated latent coordinates directly; no plot-time fitting or renormalization. '
             'Not a held-out-only comparison.', ha='center', fontsize=8.5)
    fig.text(.5, .073,
             'The paper caption also lists a separate CEBRA fit on behavioral data; that fit is not in this local archive.',
             ha='center', fontsize=8, color='#555555')
    fig.subplots_adjust(top=.84, bottom=.24, left=.045, right=.94, wspace=.04)
    add_vertical_position_bars(fig, norm, .018, .969, bottom=.28, height=.47)
    save(fig, OUT / f'FIGURE2B_CEBRA_{rat.upper()}', '2B_CEBRA', parents,
         dict(rat=rat, seed=42, split_display='all rows transformed; not held-out-only',
              methods=[key for key, _, _ in CEBRA_PANELS],
              fit_dimensions={key: int(panels[key][0]['embedding'].shape[1]) for key, _, _ in CEBRA_PANELS},
              displayed_coordinates={key: [d + 1 for d in dims] for key, _, dims in CEBRA_PANELS},
              hybrid_subspaces='first 3 dimensions behavior+time; last 2 time only',
              omitted_paper_panel='CEBRA fit on behavioral data is absent from local saved fit archive',
              encoder_retrained=False))
    return [dict(rat=rat, model='CEBRA', panel=key,
                 fit_dimension=int(panels[key][0]['embedding'].shape[1]),
                 plotted_coordinates=','.join(str(d + 1) for d in dims),
                 displayed_rows=len(panels[key][0]['source_id']),
                 split_display='all transformed rows; not held-out-only',
                 artifact_path=panels[key][2][0]['path'])
            for key, _, dims in CEBRA_PANELS]


def render_soft_architecture(rat, architecture):
    items = {}
    parents = []
    reference = None
    for condition in SOFT_CONDITIONS:
        arrays, embedding_parent = embedding(architecture, condition, rat,
                                             seed=42, fold=0, split='test')
        record, fit_parent = fit_manifest(architecture, condition, rat,
                                          seed=42, fold=0)
        valid = arrays['valid_mask']
        support = {key: arrays[key][valid]
                   for key in ('source_id', 'trial_id', 'time_id', 'behavior')}
        if reference is None:
            reference = support
        elif any(not np.array_equal(support[key], reference[key]) for key in reference):
            raise RuntimeError(f'Soft conditions do not share held-out support: {rat}/{architecture}/{condition}')
        points = arrays['embedding_raw'][valid]
        fit_dims = set(manifest_dimensions(record))
        if points.ndim != 2 or points.shape[1] != 32 or (fit_dims and fit_dims != {32}):
            raise RuntimeError(f'Expected the existing Soft 32D fit: {rat}/{architecture}/{condition}')
        items[condition] = points
        parents.extend([embedding_parent, fit_parent])

    norm = Normalize(float(reference['behavior'][:, 0].min()),
                     float(reference['behavior'][:, 0].max()))
    fig = plt.figure(figsize=(17, 9), facecolor='white')
    for row, dimensions in enumerate((2, 3)):
        for column, condition in enumerate(SOFT_CONDITIONS):
            points = items[condition]
            title = SOFT_TITLES[condition] + (' | 2D' if dimensions == 2 else ' | 3D')
            note = f'fit 32D | latent coords 1–{dimensions}\n'
            note += f'held-out TEST only (n={len(reference["source_id"]):,})'
            ax = fig.add_subplot(2, 3, row * 3 + column + 1,
                                 projection='3d' if dimensions == 3 else None)
            if dimensions == 2:
                draw_2d(ax, points[:, :2], reference['behavior'], norm,
                        title, note)
            else:
                draw_3d(ax, points[:, :3], reference['behavior'], norm,
                        title, note, title_color='#9d1c1c')

    fig.suptitle(f'{rat.capitalize()} | Soft {architecture} | seed 42, fold 0 | held-out TEST',
                 y=.98, fontsize=13, color='#9d1c1c')
    fig.text(.5, .925,
             'Neural input | existing 32D fit | panels are direct coordinate views, not new fits',
             ha='center', fontsize=9)
    fig.text(.5, .065,
             '2D = latent 1–2; 3D = latent 1–3. No PCA/t-SNE/UMAP, no projection fitting, '
             'no extra normalization.', ha='center', fontsize=8.5)
    fig.subplots_adjust(top=.86, bottom=.17, left=.055, right=.94,
                        hspace=.31, wspace=.035)
    add_vertical_position_bars(fig, norm, .018, .969, bottom=.29, height=.44)
    save(fig, OUT / f'FIGURE2B_SOFT_{architecture.upper()}_{rat.upper()}',
         f'2B_SOFT_{architecture.upper()}', parents,
         dict(rat=rat, architecture=architecture, seed=42, fold=0,
              training_input='neural activity', fit_dimension=32,
              displayed_split='test only, after valid_mask',
              displayed_samples=len(reference['source_id']),
              conditions={
                  'behavior_only': 'raw position + right/left direction metadata; no time term',
                  'time_only': 'time metadata only; behavior used only for display colors',
                  'joint': 'time and behavior metadata, equal 0.5/0.5 weighting',
              },
              views={'2D': [1, 2], '3D': [1, 2, 3]},
              plot_time_fitting=False, plot_time_normalization=False,
              encoder_retrained=False))
    rows = []
    for condition in SOFT_CONDITIONS:
        for dimensions in (2, 3):
            rows.append(dict(rat=rat, model=f'Soft {architecture}',
                             condition=condition, title=SOFT_TITLES[condition],
                             seed=42, fold=0, fitted_dimension=32,
                             plotted_coordinates=','.join(str(i) for i in range(1, dimensions + 1)),
                             split='held_out_test_only', samples=len(reference['source_id']),
                             representation='embedding_raw',
                             plot_time_fitting=False, plot_time_normalization=False))
    return rows


def main():
    plt.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'text.color': 'black',
        'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black',
    })
    cebra_rows, soft_rows = [], []
    for rat in RATS:
        cebra_rows.extend(render_cebra(rat))
        for architecture in ('CNN', 'Transformer'):
            soft_rows.extend(render_soft_architecture(rat, architecture))
    write_csv(OUT / 'FIGURE2B_CEBRA_PANEL_INDEX.csv', cebra_rows)
    write_csv(OUT / 'FIGURE2B_SOFT_PANEL_INDEX.csv', soft_rows)
    print(json.dumps({
        'status': 'complete', 'rats': list(RATS),
        'figures': {'CEBRA': len(RATS), 'Soft_CNN': len(RATS), 'Soft_Transformer': len(RATS)},
        'encoder_fits': 0, 'output': str(OUT),
    }, indent=2))


if __name__ == '__main__':
    main()
