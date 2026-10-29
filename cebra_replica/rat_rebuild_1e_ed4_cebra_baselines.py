"""Plot rat Fig 1e for CEBRA and baselines from audited train metrics only."""
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
SOURCE = ROOT / 'results/rat_rebuild_20261027/Figure_1E_ED4'
OUT = ROOT / 'results/rat_rebuild_20261027/Figure_1/1E'
RATS = ('achilles', 'buddy', 'cicero', 'gatsby')
METHODS = (
    ('CEBRA behavior (position + direction)', 'CEBRA — Position + Direction',
     'Position + Direction', 'neural data only'),
    ('CEBRA time', 'CEBRA — Time', 'Time', 'neural data only'),
    ('conv-piVAE q(z|x)', 'conv-piVAE\nfit: Position + Direction\ninference: neural x only',
     'Position + Direction', 'neural data only, q(z|x)'),
    ('conv-piVAE q(z|x,u)', 'conv-piVAE\nfit: Position + Direction\ninference: neural x + Position + Direction',
     'Position + Direction', 'neural data and Position + Direction, q(z|x,u)'),
    ('UMAP', 'UMAP', 'neural data only', 'neural data only'),
    ('t-SNE', 't-SNE', 'neural data only', 'neural data only'),
    ('PCA', 'PCA', 'neural data only', 'neural data only'),
)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = [OUT / 'Figure_1E_CEBRA_baselines.png', OUT / 'Figure_1E_CEBRA_baselines.pdf',
               OUT / 'matrix_values.csv', OUT / 'manifest.json']
    existing = [str(p) for p in outputs if p.exists()]
    if existing:
        raise RuntimeError(f'Refusing to overwrite existing outputs: {existing}')
    audit = json.loads((SOURCE / 'support_audit.json').read_text(encoding='utf-8'))
    if audit.get('status') != 'pass':
        raise RuntimeError('Source support audit did not pass')
    earlier_manifest = json.loads((SOURCE / 'plot_manifest_v3.json').read_text(encoding='utf-8'))
    source_csv = SOURCE / 'train_cross_subject_consistency.csv'
    if sha256(source_csv) != earlier_manifest['input_tables']['train_cross_subject_consistency.csv']:
        raise RuntimeError('Source metric CSV hash mismatch')
    if sha256(SOURCE / 'support_audit.json') != earlier_manifest['support_audit_sha256']:
        raise RuntimeError('Source support audit hash mismatch')
    with source_csv.open(newline='', encoding='utf-8') as f:
        data = list(csv.DictReader(f))
    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    fig.subplots_adjust(left=.06, right=.90, bottom=.11, top=.86, wspace=.55, hspace=.75)
    records = []
    for ax, (original, title, fit, inference) in zip(axes.flat, METHODS):
        grouped = defaultdict(list)
        identities = defaultdict(set)
        for row in data:
            if row['method'] != original:
                continue
            pair = (row['rat'], row['peer_rat'])
            identity = (int(row['fold']), int(row['seed']))
            if pair[0] not in RATS or pair[1] not in RATS or pair[0] == pair[1] or identity in identities[pair]:
                raise RuntimeError(f'Invalid or duplicate comparison identity: {original}/{pair}/{identity}')
            value = float(row['train_cross_subject_r2'])
            if not np.isfinite(value):
                raise RuntimeError(f'Nonfinite consistency: {original}/{pair}/{identity}')
            grouped[pair].append(value)
            identities[pair].add(identity)
        matrix = np.full((4, 4), np.nan)
        for i, rat in enumerate(RATS):
            for j, peer in enumerate(RATS):
                if i == j:
                    continue
                expected = {(fold, seed) for fold in range(3)
                            for seed in ((42,) if original == 'PCA' else range(42, 48))}
                if identities[(rat, peer)] != expected:
                    raise RuntimeError(f'Incomplete comparison grid: {original}/{rat}/{peer}')
                val = float(np.mean(grouped[(rat, peer)]))
                matrix[i, j] = 100 * val
                records.append(dict(method=title.replace('\n', '; '), method_source=original,
                    fit_labels=fit, inference_inputs=inference, source_rat=rat, target_rat=peer,
                    split='train', r2_mean=val, comparisons=len(expected),
                    seed_count=1 if original == 'PCA' else 6, fold_count=3))
        im = ax.imshow(matrix, vmin=0, vmax=100, cmap='viridis')
        for i in range(4):
            for j in range(4):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f'{matrix[i, j]:.1f}', ha='center', va='center', color='white', fontsize=9)
        ax.set_xticks(range(4), RATS, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(4), RATS, fontsize=8)
        ax.set_title(title, fontsize=9, linespacing=1.25)
    axes.flat[-1].set_visible(False)
    fig.suptitle('Figure 1E — CEBRA and baselines: cross-subject consistency on train', fontsize=15)
    fig.text(.47, .04, 'Column: target rat; row: source rat. Mean over 3 folds and 6 seeds (PCA deterministic).', ha='center', fontsize=9)
    color_ax = fig.add_axes([.93, .30, .015, .40])
    fig.colorbar(im, cax=color_ax, label='Train consistency R² (%)')
    fig.savefig(outputs[0], dpi=220, bbox_inches='tight')
    fig.savefig(outputs[1], bbox_inches='tight')
    plt.close(fig)
    with outputs[2].open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    manifest = {
        'status': 'staged_review_required',
        'scope': 'Figure 1E only: CEBRA and baselines; no Soft',
        'training_performed': False,
        'metric': 'directed linear-regression R2, train-to-train cross-subject, averaged over 3 folds and 6 seeds; PCA deterministic',
        'alignment': 'position 0..1.6m in 100 bins per direction; average embeddings within bins; empty bins use samples in immediately adjacent bins; CEBRA bin means renormalized',
        'support': 'CEBRA and piVAE: all train rows; UMAP, t-SNE and PCA: train rows with valid-window mask. Source IDs, trial IDs and behavioral labels were checked against CEBRA split reference in source support audit.',
        'source_metric_path': str(source_csv.relative_to(ROOT)),
        'source_metric_sha256': sha256(source_csv),
        'source_metric_manifest_sha256': sha256(SOURCE / 'manifest.json'),
        'source_support_audit_sha256': sha256(SOURCE / 'support_audit.json'),
        'source_support_audit': audit,
        'script_sha256': sha256(Path(__file__)),
        'selected_methods': [dict(source=original, display=title, fit=fit, inference=inference)
                             for original, title, fit, inference in METHODS],
        'limitations': [
            'Six seeds rather than the paper ten; PCA uses one deterministic fit per fold.',
            'Reused t-SNE is perplexity 30, not the paper rat benchmark perplexity 10; early exaggeration of the reused fit is not explicitly recorded.',
            'Reused UMAP and t-SNE were fit on training windows only, unlike the paper Methods description of unsupervised full-dataset fitting.',
            'Aligned bin averaging and same-seed/fold comparisons implement the described method, but exact experimental source-code equivalence is not verified.',
            'piVAE q(z|x,u) uses Position + Direction during inference; it is not a behavior-blind representation.',
            'Different methods have different valid train supports; see source support audit.',
            'These results are not declared an exact or final replication of the paper.',
        ],
        'outputs': {p.name: sha256(p) for p in outputs[:3]},
    }
    with outputs[3].open('x', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({'status': manifest['status'], 'outputs': [str(p) for p in outputs], 'comparisons': len(records)}, indent=2))


if __name__ == '__main__':
    main()
