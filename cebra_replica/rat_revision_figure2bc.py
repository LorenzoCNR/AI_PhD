"""Re-render rat Figure 2B/2C from frozen arrays/histories; never fit models.

Writes only the dated RAT_FIGURES staging branch. CEBRA descriptive references
are separate files, not image composites with the held-out Soft gallery.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from rat_revision_common import ROOT, STAGE, RATS, DISPLAY, SOFT_KEYS, embedding, fit_manifest, save, sha, csv_rows, write_csv

BASE_LOSS = ROOT/'results/rat_figure_gap_closure_v1/figure2c_loss/FIGURE2C_LOSS_LONG.csv'
OUT = STAGE/'Figure_2'
CONDITIONS = ('behavior_only', 'joint', 'time_only')


def scatter(ax, points, behavior, norm, title, soft=False, note=''):
    if points.ndim != 2 or points.shape[1] != 3 or behavior.shape != (len(points), 3):
        raise ValueError('Expected aligned 3-coordinate projection and position/right/left metadata')
    for column, cmap in ((2, 'cool'), (1, 'viridis')):
        mask = behavior[:, column] == 1
        ax.scatter(*points[mask].T, c=behavior[mask, 0], cmap=cmap, norm=norm,
                   s=2.5, linewidths=0, alpha=.85, rasterized=True)
    ax.set_axis_off()
    ax.set_title(title, color='red' if soft else 'black', fontsize=12, pad=0)
    if note:
        ax.text2D(.5, -.01, note, ha='center', va='top', transform=ax.transAxes, fontsize=8)


def bars(fig, norm):
    for bounds, cmap, title in (([.28, .055, .18, .014], 'cool', 'Left'),
                                ([.55, .055, .18, .014], 'viridis', 'Right')):
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes(bounds), orientation='horizontal')
        bar.set_label(title, fontsize=10)
        bar.set_ticks([])
        bar.outline.set_visible(False)


def render_soft2b():
    index = []
    for rat in RATS:
        items, parents = {}, []
        reference = None
        for architecture, condition in SOFT_KEYS:
            arrays, parent = embedding(architecture, condition, rat)
            valid = arrays['valid_mask']
            support = {key: arrays[key][valid] for key in ('source_id', 'trial_id', 'time_id', 'behavior')}
            if reference is None:
                reference = support
            elif any(not np.array_equal(support[key], reference[key]) for key in reference):
                raise RuntimeError(f'Figure 2B common held-out support mismatch: {rat}/{architecture}/{condition}')
            points = arrays['embedding_raw'][valid]
            if points.shape[1] != 32:
                raise RuntimeError('Expected frozen 32D Soft fit; no projection fitting allowed')
            items[architecture, condition] = points[:, :3]
            parents.append(parent)
        norm = Normalize(float(reference['behavior'][:, 0].min()), float(reference['behavior'][:, 0].max()))
        fig = plt.figure(figsize=(13, 8.8), facecolor='white')
        for ri, architecture in enumerate(('CNN', 'Transformer')):
            for ci, condition in enumerate(CONDITIONS):
                ax = fig.add_subplot(2, 3, ri*3+ci+1, projection='3d')
                scatter(ax, items[architecture, condition], reference['behavior'], norm,
                    DISPLAY[architecture, condition], soft=True)
                index.append(dict(rat=rat, architecture=architecture, condition=condition,
                    seed=42, fold=0, split='held_out_test', valid_samples=len(reference['source_id']),
                    fit_dimension=32, plotted_coordinates='0,1,2', representation='embedding_raw'))
        fig.suptitle(f'{rat.capitalize()} | held-out trial split | seed 42, split 0', y=.98, fontsize=13)
        fig.text(.5,.105,'32D fits; first 3 coordinates shown (no 3D renormalization)',ha='center',fontsize=9)
        fig.subplots_adjust(top=.90, bottom=.12, left=.01, right=.99, hspace=.14, wspace=.01)
        bars(fig, norm)
        save(fig, OUT/'2B'/f'FIGURE2B_{rat.upper()}', '2B', parents,
            dict(rat=rat, branch='Soft held_out_test only', seed=42, fold=0,
                same_support_verified=['source_id', 'trial_id', 'time_id', 'behavior'],
                fit_dimension=32, displayed_coordinates=[0,1,2], projection_fitted=False,
                representation='raw native L2-normalized output; no 3D renormalization',
                reference_gallery='separate CEBRA_REFERENCE file, no mixed-support image composite'))
    write_csv(OUT/'2B'/'FIGURE2B_PANEL_INDEX.csv', index)


def read_reference(path, artifact, expected):
    if sha(path) != expected:
        raise RuntimeError(f'Frozen reference embedding hash mismatch: {path}')
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key].copy() for key in ('embedding', 'behavior', 'source_id')}
    if any(len(v) != len(result['source_id']) for v in result.values()):
        raise RuntimeError(f'Reference metadata cardinality mismatch: {path}')
    return result, dict(path=path.relative_to(ROOT).as_posix(), sha256=expected, artifact=artifact)


def render_reference2b():
    """Existing four-hypothesis descriptive support; never mixed with Soft test."""
    index = []
    for rat in RATS:
        panels, parents = [], []
        for condition, title in (('position', 'CEBRA-Position'), ('position_shuffled', 'Position shuffle')):
            folder = ROOT/'results/rat_figure2c_3d_v1'/rat/condition/'seed_42'
            marker = folder/'result.json'
            record = json.loads(marker.read_text())
            data, parent = read_reference(folder/'all_embedding.npz', condition, record['artifacts']['all_embedding.npz'])
            parents.extend([parent, dict(path=marker.relative_to(ROOT).as_posix(), sha256=sha(marker))])
            panels.append((title, data, '3D fit; all rows transformed', [0,1,2]))
        folder = ROOT/'results/rat_trial_benchmark_v1'/rat/'fold_0/time/seed_42'
        marker = folder/'result.json'
        record = json.loads(marker.read_text())
        parts = []
        for split in ('train', 'valid', 'test'):
            data, parent = read_reference(folder/f'{split}_embedding.npz', split,
                record['embeddings'][split]['embedding_sha256'])
            parts.append(data); parents.append(parent)
        time_data = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
        panels.append(('CEBRA-Time', time_data, '32D fit; first 3 coordinates', [0,1,2]))
        parents.append(dict(path=marker.relative_to(ROOT).as_posix(), sha256=sha(marker)))
        folder = ROOT/'results/rat_figure_gap_closure_v1/figure2b_fits'/rat/'hybrid_position_time_5d/seed_42'
        marker = folder/'result.json'
        record = json.loads(marker.read_text())
        data, parent = read_reference(folder/'all_embedding.npz', 'hybrid', record['artifacts']['all_embedding.npz'])
        panels.append(('CEBRA-Hybrid', data, '5D fit; coordinates 1, 2, 0', [1,2,0]))
        parents.extend([parent, dict(path=marker.relative_to(ROOT).as_posix(), sha256=sha(marker))])
        reference = None
        for title, data, note, dims in panels:
            order = np.argsort(data['source_id'])
            for key in data:
                data[key] = data[key][order]
            if len(np.unique(data['source_id'])) != len(data['source_id']):
                raise RuntimeError('Repeated reference source rows')
            if reference is None:
                reference = data
            elif not np.array_equal(reference['source_id'], data['source_id']) or not np.array_equal(reference['behavior'], data['behavior']):
                raise RuntimeError('Reference gallery support mismatch')
            index.append(dict(rat=rat, method=title, branch='descriptive_all_rows_NOT_held_out',
                samples=len(data['source_id']), fit_dimension=data['embedding'].shape[1], coordinates=str(dims),
                fit_protocol='frozen distinct existing fits; not a controlled training-budget comparison'))
        norm = Normalize(float(reference['behavior'][:,0].min()), float(reference['behavior'][:,0].max()))
        fig = plt.figure(figsize=(17, 5.1), facecolor='white')
        for i, (title, data, note, dims) in enumerate(panels):
            scatter(fig.add_subplot(1,4,i+1,projection='3d'), data['embedding'][:,dims],
                data['behavior'], norm, title, note=note)
        fig.suptitle(f'{rat.capitalize()} | CEBRA descriptive reference | all rows, not held-out evidence', y=.97, fontsize=12)
        fig.subplots_adjust(top=.86, bottom=.16, left=.01, right=.99, wspace=.02)
        bars(fig,norm)
        save(fig, OUT/'2B'/f'FIGURE2B_{rat.upper()}_CEBRA_REFERENCE', '2B_REFERENCE', parents,
            dict(rat=rat, branch='descriptive all rows transformed by distinct existing fits',
                held_out_claim=False, training_budget_equated=False, support_verified=True,
                original_axis_order_preserved=True, no_image_composite=True))
    write_csv(OUT/'2B'/'FIGURE2B_REFERENCE_PANEL_INDEX.csv',index)


def history(architecture, condition, rat, shuffle=False):
    record, parent = fit_manifest(architecture, condition, rat, shuffle=shuffle)
    values = record.get('history', [])
    updates = np.asarray([row['update'] for row in values], dtype=np.int64)
    losses = np.asarray([row['train_loss'] for row in values], dtype=float)
    if len(updates) != 10000 or not np.array_equal(updates,np.arange(1,10001)) or not np.isfinite(losses).all():
        raise RuntimeError(f'Incomplete frozen training history: {rat}/{architecture}/{condition}/shuffle={shuffle}')
    return updates, losses, parent


def render_loss2c():
    reference_rows = csv_rows(BASE_LOSS)
    required = ('position','direction','position_direction','position_shuffled','direction_shuffled','position_direction_shuffled')
    names = {'position':'Position','direction':'Direction','position_direction':'Position + direction',
        'position_shuffled':'Position shuffle','direction_shuffled':'Direction shuffle',
        'position_direction_shuffled':'Position + direction shuffle'}
    colors = {'position':'#4477AA','direction':'#EE6677','position_direction':'#228833',
        'position_shuffled':'#4477AA','direction_shuffled':'#EE6677','position_direction_shuffled':'#228833'}
    index = []
    for rat in RATS:
        parents = [dict(path=BASE_LOSS.relative_to(ROOT).as_posix(),sha256=sha(BASE_LOSS))]
        fig, axes = plt.subplots(2,3,figsize=(16,8),facecolor='white')
        for condition in required:
            rows = sorted([r for r in reference_rows if r['rat']==rat and r['condition']==condition],key=lambda r:int(r['optimizer_update']))
            if not rows:raise RuntimeError(f'Missing CEBRA reference curve: {rat}/{condition}')
            x=np.asarray([int(r['optimizer_update']) for r in rows]);y=np.asarray([float(r['training_objective']) for r in rows])
            if len(x)!=10000 or not np.array_equal(x,np.arange(1,10001)) or not np.isfinite(y).all():
                raise RuntimeError('CEBRA curve update/finiteness invariant')
            axes[0,0].plot(x,y,label=names[condition],color=colors[condition],
                ls='--' if 'shuffled' in condition else '-',lw=.65,alpha=.8)
            index.append(dict(rat=rat,method='CEBRA InfoNCE reference',condition=condition,seed=42,
                split='training',updates=len(x),first_loss=y[0],last_loss=y[-1],source=parents[0]['path'],
                source_sha256=parents[0]['sha256'],loss_not_comparable_across_objectives=True))
        axes[0,0].set_title('CEBRA InfoNCE reference',fontsize=12)
        panels = [(axes[0,1],'CNN','behavior_only'),(axes[0,2],'Transformer','behavior_only'),
                  (axes[1,0],'CNN','joint'),(axes[1,1],'Transformer','joint')]
        for ax,architecture,condition in panels:
            for shuffled in (False,True):
                x,y,parent=history(architecture,condition,rat,shuffled);parents.append(parent)
                ax.plot(x,y,label='Behavior shuffle (TRAIN)' if shuffled else 'Observed',
                    color='#888888' if shuffled else '#6930C3',ls='--' if shuffled else '-',lw=.8,alpha=.85)
                index.append(dict(rat=rat,method=f'Soft {architecture}',condition=condition,shuffle=shuffled,
                    seed=42,fold=0,split='training',updates=len(x),first_loss=y[0],last_loss=y[-1],
                    source=parent['path'],source_sha256=parent['sha256'],loss_not_comparable_across_objectives=True))
            ax.set_title(DISPLAY[architecture,condition].replace('\n',' | '),color='red',fontsize=12)
        for architecture,color in [('CNN','#6930C3'),('Transformer','#197278')]:
            x,y,parent=history(architecture,'time_only',rat);parents.append(parent)
            axes[1,2].plot(x,y,label=f'Soft {architecture}',color=color,lw=.8)
            index.append(dict(rat=rat,method=f'Soft {architecture}',condition='time_only',shuffle=False,
                seed=42,fold=0,split='training',updates=len(x),first_loss=y[0],last_loss=y[-1],
                source=parent['path'],source_sha256=parent['sha256'],behavior_shuffle_applicable=False))
        axes[1,2].set_title('Soft | Time only\nBehavior shuffle: not applicable',color='red',fontsize=11)
        for ax in axes.flat:
            ax.set_xlabel('Optimizer update');ax.set_ylabel('Training objective')
            ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.16)
            ax.legend(frameon=True,facecolor='white',edgecolor='none',framealpha=.95,fontsize=8,loc='best');ax.set_xlim(0,10000)
        fig.suptitle(f'{rat.capitalize()} | frozen training histories | seed 42 (Soft: split 0)',fontsize=13,y=.99)
        fig.text(.5,.012,'Separate axes: CEBRA InfoNCE and Soft objective scales are not directly comparable. All updates shown; no smoothing.',
            ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.045,1,.96),h_pad=2,w_pad=2)
        save(fig,OUT/'2C'/f'FIGURE2C_{rat.upper()}','2C',parents,
            dict(rat=rat,branch='training histories only; no held-out loss claim',seed=42,fold=0,
                reference_curve_conditions=list(required),soft_observed_series=6,soft_shuffle_series=4,
                time_only_behavior_shuffle='NOT_APPLICABLE',smoothing=False,raw_updates=10000,
                objective_scales_comparable=False,observed_scientific_metrics_modified=False))
    write_csv(OUT/'2C'/'FIGURE2C_HISTORY_INDEX.csv',index)


def main():
    plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
        'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black'})
    render_soft2b();render_reference2b();render_loss2c()
    print(json.dumps(dict(status='complete',rats=list(RATS),soft_2b=4,reference_2b=4,loss_2c=4,
        encoder_fits=0,output=str(OUT)),indent=2))


if __name__=='__main__':main()
