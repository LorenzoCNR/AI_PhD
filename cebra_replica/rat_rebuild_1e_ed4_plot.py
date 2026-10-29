"""Plot audited train consistency and held-out decoding from staged tables."""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from user_soft.data import sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rat_rebuild_20261027/Figure_1E_ED4'
RATS = ('achilles', 'buddy', 'cicero', 'gatsby')


def read(name):
    path = OUT / name
    if not path.exists(): return []
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def average(values):
    return float(np.mean(values)) if values else np.nan


def display_name(name):
    return name.replace('time_direction_position', 'time + direction + position').replace(
        'direction_position', 'direction + position').replace('time_only', 'time')


def save(fig, stem):
    paths = [OUT / f'{stem}.{ext}' for ext in ('png', 'pdf')]
    if any(path.exists() for path in paths):
        raise RuntimeError(f'Refusing to overwrite plot: {stem}')
    for path in paths:
        fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return paths


def main():
    audit = json.loads((OUT / 'support_audit.json').read_text())
    if audit['status'] != 'pass':
        raise RuntimeError('Support identity audit failed; refusing figures')
    cross = read('train_cross_subject_consistency.csv')
    run = read('train_run_consistency.csv')
    decode = read('heldout_decoding.csv')
    methods = list(dict.fromkeys(row['method'] for row in decode + cross + run))
    if not methods:
        raise RuntimeError('No metric rows')
    files = []
    fig, axes = plt.subplots(4, 4, figsize=(15, 14))
    image = None
    for ax, method in zip(axes.flat, methods):
        mat = np.full((4, 4), np.nan)
        for i, rat in enumerate(RATS):
            for j, peer in enumerate(RATS):
                values = [float(row['train_cross_subject_r2']) for row in cross
                          if row['method'] == method and row['rat'] == rat and row['peer_rat'] == peer]
                if values: mat[i, j] = np.mean(values) * 100
        image = ax.imshow(mat, vmin=0, vmax=100, cmap='viridis')
        for i in range(4):
            for j in range(4):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f'{mat[i,j]:.1f}', ha='center', va='center', fontsize=8, color='white')
        ax.set_xticks(range(4), RATS, rotation=45, ha='right', fontsize=6)
        ax.set_yticks(range(4), RATS, fontsize=7)
        ax.set_title(display_name(method), fontsize=8)
    for ax in axes.flat[len(methods):]: ax.set_visible(False)
    fig.subplots_adjust(left=.06, right=.90, bottom=.06, top=.92, wspace=.55, hspace=.65)
    if image is not None:
        color_ax = fig.add_axes([.93, .30, .015, .35])
        fig.colorbar(image, cax=color_ax, label='Train consistency R² (%)')
    fig.suptitle('Figure 1e: train cross-subject consistency; 3 folds, 6 seeds (PCA deterministic)', fontsize=13)
    files += save(fig, 'Figure_1e_train_consistency_v3')
    metric_specs = [
        (cross, 'train_cross_subject_r2', 'Train subject consistency R² (%)', 100),
        (run, 'train_run_r2', 'Train run consistency R² (%)', 100),
        (decode, 'test_total_r2', 'Held-out test position + direction R² (%)', 100),
        (decode, 'test_position_error_m', 'Held-out test median position error (cm)', 100),
    ]
    summary = []
    for rat in RATS:
        fig, axes = plt.subplots(1, 4, figsize=(17, max(6, len(methods)*.38)), sharey=True)
        for ax, (rows, metric, title, scale) in zip(axes, metric_specs):
            for i, method in enumerate(methods):
                per_fold_seed = defaultdict(list)
                for row in rows:
                    if row['rat'] == rat and row['method'] == method:
                        per_fold_seed[(int(row['seed']), int(row['fold']))].append(float(row[metric]))
                per_seed = defaultdict(list)
                for (seed, fold), values in per_fold_seed.items():
                    per_seed[seed].append(average(values))
                seed_points = [(seed, average(folds)*scale) for seed, folds in sorted(per_seed.items()) if len(folds) == 3]
                points = [value for seed, value in seed_points]
                for seed, value in seed_points:
                    summary.append(dict(rat=rat, method=method, metric=metric, seed=seed, value=value,
                                        unit='cm' if metric == 'test_position_error_m' else 'percent',
                                        aggregation='directed peer mean within fold, then mean over 3 folds per seed'))
                if points:
                    ax.scatter(points, np.full(len(points), i), s=14, facecolors='none', edgecolors='black')
                    ax.scatter([np.median(points)], [i], marker='|', s=125, color='darkorange', zorder=3)
                else:
                    ax.text(.01, i, 'n/a', transform=ax.get_yaxis_transform(), fontsize=7, va='center')
            ax.set_title(title, fontsize=9)
            ax.grid(axis='x', alpha=.15)
            ax.set_yticks(range(len(methods)), [display_name(m) for m in methods], fontsize=8)
        axes[0].invert_yaxis()
        fig.suptitle(f'ED4 — {rat}: individual seeds; fold mean; train consistency / held-out decoding')
        fig.tight_layout()
        files += save(fig, f'ED4_{rat}_v3')
    summary_path = OUT / 'ed4_plot_values_v3.csv'
    with summary_path.open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    manifest = {'status':'complete', 'support_audit_sha256':sha256(OUT/'support_audit.json'),
                'plot_script_sha256':sha256(Path(__file__)),
                'input_tables':{name:sha256(OUT/name) for name in
                    ('train_cross_subject_consistency.csv','train_run_consistency.csv','heldout_decoding.csv') if (OUT/name).exists()},
                'outputs':{str(p.relative_to(OUT)):sha256(p) for p in files+[summary_path]},
                'warning':'Six seeds; see manifest.json and cross_subject_blockers.csv for limitations.'}
    with (OUT/'plot_manifest_v3.json').open('x', encoding='utf-8') as f: json.dump(manifest,f,indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
