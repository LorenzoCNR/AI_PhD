"""Render verified seed-42 full-recording rat embeddings; never fit models."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'rat_rebuild_20261027' / 'Figure_1D_ED3'
ED3 = ROOT / 'results' / 'rat_paper_replication_2023_v1' / 'ED3'
SOFT = ROOT / 'results' / 'user_soft_rat_fullrecording_t1_3conditions_seed42_v1'
RATS = ('achilles', 'buddy', 'cicero', 'gatsby')
PAPER = (
    ('CEBRA\nPosition + Direction', 'cebra', 'behavior', 'embedding', 3),
    ('conv-piVAE\nfit: Position + Direction; output: mean q(z|x,u)', 'pivae', '', 'post_mean_behavior_informed_raw', 2),
    ('conv-piVAE\nfit: Position + Direction; output: mean q(z|x)', 'pivae', '', 'z_mean_raw', 2),
    ('CEBRA\nTime', 'cebra', 'time', 'embedding', 3),
    ('t-SNE', 'manifold', 'tsne', 'embedding', 2),
    ('UMAP', 'manifold', 'umap', 'embedding', 2),
)
SOFT_SPECS = tuple((f'Soft {name}\n{label}', 'soft', (condition, arch),
                    'embedding_unit', 3)
                   for condition, label in (('time_direction_position', 'Time + Position + Direction'),
                                            ('time_only', 'Time'),
                                            ('direction_position', 'Position + Direction'))
                   for arch, name in (('cnn', 'CNN'), ('transformer', 'Transformer')))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def read_panel(rat, raw, source_hash, spec, campaign):
    title, family, method, key, dimension = spec
    if family == 'cebra':
        base = ED3 / 'cebra' / rat / method / 'seed_42'
        ep = base / 'embedding.npz'
    elif family == 'pivae':
        base = ED3 / 'pivae_full_recording' / rat / 'seed_42'
        ep = base / 'embedding_full.npz'
    elif family == 'manifold':
        base = ED3 / 'manifold' / method / rat / 'window10_flat' / 'seed_42'
        ep = base / 'embedding.npz'
    else:
        condition, arch = method
        base = SOFT / rat / condition / arch / 'seed_42'
        ep = base / 'embedding_full.npz'
    mp = base / 'manifest.json'
    if not mp.is_file() or not ep.is_file():
        raise FileNotFoundError('missing ' + ', '.join(relative(p) for p in (mp, ep) if not p.is_file()))
    m = json.loads(mp.read_text(encoding='utf-8'))
    cfg = m['config']
    if cfg.get('rat') != rat or cfg.get('seed') != 42:
        raise ValueError('manifest rat/seed mismatch')
    if m.get('status', 'complete') != 'complete':
        raise ValueError('manifest status is not complete')
    expected = m.get('embedding_sha256') or m.get('artifact_hashes', {}).get(ep.name)
    if not expected or sha(ep) != expected:
        raise ValueError('embedding SHA-256 missing or mismatched')
    if family == 'cebra':
        if cfg.get('condition') != method or cfg['params']['output_dimension'] != dimension or cfg.get('input_scope') != 'full recording' or cfg.get('data_sha256') != source_hash:
            raise ValueError('CEBRA config or source mismatch')
        checkpoint, chash = base / 'checkpoint.pt', m.get('checkpoint_sha256')
        support = 'Fit full recording; all saved rows plotted.'
        normalization = 'CEBRA output, L2 unit norm'
    elif family == 'pivae':
        if cfg['latent_dim'] != dimension or cfg.get('source_data_sha256') != source_hash or 'all rows' not in cfg.get('input_scope', ''):
            raise ValueError('piVAE config or source mismatch')
        checkpoint, chash = base / 'terminal.weights.h5', m.get('artifact_hashes', {}).get('terminal.weights.h5')
        support = 'Fit full recording; only saved valid-window rows plotted.'
        normalization = 'raw posterior mean'
    elif family == 'manifold':
        if cfg.get('method') != method or cfg['params']['n_components'] != dimension or cfg.get('source_sha256') != source_hash or cfg.get('input_representation') != 'window10_flat':
            raise ValueError('manifold config or source mismatch')
        checkpoint, chash = None, None
        support = 'Fit and plot full-recording complete 10-bin windows (T-9); no held-out split.'
        normalization = 'raw manifold coordinates'
    else:
        rec = campaign.get('records', {}).get(f'{rat}/{condition}/{arch}/seed_42', {})
        if (rec.get('status') != 'complete' or rec.get('embedding_sha256') != expected
                or m.get('condition') != condition or m.get('architecture') != arch
                or cfg.get('version') != 'soft_v1_stable' or cfg.get('output_dimension') != dimension
                or cfg.get('fit_scope') != 'full_recording' or m.get('actual_updates') != 10000
                or cfg.get('embedding_temperature') != 1.0 or cfg.get('metadata_temperature') != 0.5):
            raise ValueError('Soft campaign status/config mismatch')
        checkpoint, chash = base / 'checkpoint.pt', m.get('checkpoint_sha256')
        support = 'Fit full recording; only saved valid-window rows plotted.'
        normalization = 'saved embedding_unit, L2 unit norm'
    if checkpoint is not None and (not checkpoint.is_file() or not chash or sha(checkpoint) != chash):
        raise ValueError('checkpoint SHA-256 missing or mismatched')
    with np.load(ep, allow_pickle=False) as data:
        z, behavior, ids = data[key].copy(), data['behavior'].copy(), data['source_id'].copy()
        mask = data['valid_mask'].astype(bool) if 'valid_mask' in data else np.ones(len(ids), dtype=bool)
    if z.ndim != 2 or z.shape != (len(ids), dimension) or behavior.shape != (len(ids), 3) or mask.shape != (len(ids),):
        raise ValueError('inconsistent array dimensions')
    if ids.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(ids >= len(raw)) or len(np.unique(ids)) != len(ids):
        raise ValueError('invalid or duplicate source IDs')
    if family != 'manifold' and not np.array_equal(ids, np.arange(len(raw))):
        raise ValueError('full-recording source IDs do not cover every row')
    if family == 'manifold' and len(ids) != len(raw) - 9:
        raise ValueError('manifold does not contain T-9 full windows')
    if not np.isfinite(z).all() or not np.isfinite(behavior).all() or not np.allclose(behavior, raw[ids], rtol=0, atol=1e-6):
        raise ValueError('non-finite values or behavior/source ID mismatch')
    if family in ('cebra', 'soft') and not np.allclose(np.linalg.norm(z, axis=1), 1, atol=2e-5):
        raise ValueError('expected unit-norm embedding')
    left, right = behavior[:, 2] == 1, behavior[:, 1] == 1
    if np.any(left & right):
        raise ValueError('ambiguous direction labels')
    plotted = mask & (left | right)
    if not plotted.any():
        raise ValueError('no rows with a valid direction to plot')
    record = dict(title=title.replace('\n', ' '), family=family, rat=rat, seed=42,
                  fit_dimension=dimension, plotted_dimensions=dimension,
                  coordinates=list(range(1, dimension + 1)), fit_scope=support,
                  stored_rows=len(ids), valid_mask_rows=int(mask.sum()), plotted_rows=int(plotted.sum()),
                  excluded_without_direction=int((mask & ~(left | right)).sum()),
                  plotted_source_ids_sha256=hashlib.sha256(np.ascontiguousarray(ids[plotted]).tobytes()).hexdigest(),
                  normalization=normalization, embedding=relative(ep), embedding_sha256=expected,
                  manifest=relative(mp), manifest_sha256=sha(mp), checkpoint_sha256=chash,
                  config=cfg, source_data_sha256=source_hash)
    return dict(title=title, z=z, behavior=behavior, mask=plotted, record=record)


def draw(panels, rat, heading, target, raw):
    fig = plt.figure(figsize=(16, 8), facecolor='white')
    grid = fig.add_gridspec(2, 3, left=.035, right=.895, top=.88, bottom=.08, wspace=.12, hspace=.25)
    norm = Normalize(float(raw[:, 0].min()), float(raw[:, 0].max()))
    for i, panel in enumerate(panels):
        d = panel['dimension']
        ax = fig.add_subplot(grid[i // 3, i % 3], projection='3d' if d == 3 else None)
        if 'error' in panel:
            ax.set_axis_off()
            ax.text2D(.5, .5, 'NOT AVAILABLE', transform=ax.transAxes, ha='center', va='center', color='#aa3030') if d == 3 else ax.text(.5, .5, 'NOT AVAILABLE', transform=ax.transAxes, ha='center', va='center', color='#aa3030')
        else:
            z, b, mask = panel['z'], panel['behavior'], panel['mask']
            for col, cmap in ((2, 'cool'), (1, 'viridis')):
                select = mask & (b[:, col] == 1)
                coords = [z[select, j] for j in range(d)]
                kw = dict(c=b[select, 0], cmap=cmap, norm=norm, s=.7, linewidths=0, rasterized=True)
                if d == 3:
                    kw['depthshade'] = False
                ax.scatter(*coords, **kw)
            if d == 3:
                ax.view_init(elev=22, azim=38)
                ax.set(xlabel='Latent 1', ylabel='Latent 2', zlabel='Latent 3')
                ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
                ax.set_box_aspect((1, 1, 1))
            else:
                ax.set_axis_off()
        ax.set_title(panel['title'] + f'\n{d}D native', fontsize=9)
    fig.suptitle(f'{rat.capitalize()} — {heading}\nFull-recording fit · seed 42', fontsize=14)
    for x, cmap, label in ((.915, 'cool', 'Left — position'), (.96, 'viridis', 'Right — position')):
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([x, .25, .012, .50]))
        cb.set_label(label, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.text(.47, .025, 'Colour: position; Left / Right: direction. Missing panels are not estimated.', ha='center', fontsize=8)
    target.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(target.with_suffix('.' + suffix), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    campaign = json.loads((SOFT / 'CAMPAIGN_STATE.json').read_text(encoding='utf-8'))
    all_records, report = [], ['# Figure 1D / Extended Data 3 — seed 42', '',
        'All fits use full recording; these are descriptive embeddings, not held-out test plots.',
        'CEBRA and Soft are native 3D; piVAE, UMAP and t-SNE native 2D. Soft is an extension, not a paper method.',
        'Position is continuous; direction uses right/left indicators. Left uses cool, Right viridis.',
        'Soft and piVAE show saved valid-window rows. CEBRA shows all saved rows; manifold shows T-9 complete windows.',
        'Soft configuration: stable, 10,000 updates, batch 512, embedding temperature 1, metadata temperature 0.5.',
        'Missing panels are explicit; no fit or transform is performed.', '',
        '| Rat | Method | Status | Fit / plot D | Stored / plotted rows | Source or issue |', '|---|---|---|---|---|---|']
    for rat in RATS:
        source = ROOT / 'data' / 'rat_hippocampus' / f'{rat}.jl'
        raw = np.asarray(joblib.load(source)['position'], dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
            raise ValueError(f'invalid raw behavior for {rat}')
        source_hash = sha(source)
        groups = []
        for specs in (PAPER, SOFT_SPECS):
            panels = []
            for spec in specs:
                try:
                    panel = read_panel(rat, raw, source_hash, spec, campaign)
                    rec = dict(status='verified', **panel['record'])
                    evidence = rec['embedding']
                    counts = f"{rec['stored_rows']} / {rec['plotted_rows']}"
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    evidence = str(exc).replace('|', '/')
                    panel = dict(title=spec[0], error=evidence)
                    rec = dict(status='unavailable', rat=rat, seed=42, title=spec[0].replace('\n', ' '), reason=evidence)
                    counts = '—'
                panel['dimension'] = spec[4]
                panels.append(panel); all_records.append(rec)
                report.append(f"| {rat} | {spec[0].replace(chr(10), ' ')} | {rec['status']} | {spec[4]}D / {spec[4]}D | {counts} | {evidence} |")
            groups.append(panels)
        draw(groups[0], rat, 'Extended Data Figure 3 — partial if indicated', OUT / 'Extended_Data_3' / rat / 'ED3_baselines', raw)
        draw(groups[1], rat, 'Soft Stable — three objectives', OUT / 'Extended_Data_3' / rat / 'ED3_soft_stable', raw)
        if rat == 'achilles':
            draw(groups[0], rat, 'Figure 1D — partial if indicated', OUT / 'Figure_1D' / 'achilles' / 'Figure_1D_baselines', raw)
            draw(groups[1], rat, 'Figure 1D — Soft Stable extension', OUT / 'Figure_1D' / 'achilles' / 'Figure_1D_soft_stable', raw)
    report.extend(['', 'For piVAE, with labels is q(z|x,u) and uses Position + Direction at inference; without labels is q(z|x).',
                   'Source manifests contain the exact fit settings. The JSON inventory records paths, hashes, dimensions, masks and row counts.',
                   'Window flattening for manifold inputs is a documented local choice; its exact implementation is not established by the published figure repository.'])
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    files = {relative(p): sha(p) for p in OUT.rglob('*') if p.is_file()}
    (OUT / 'PROVENANCE.json').write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
        script=relative(Path(__file__)), script_sha256=sha(Path(__file__)), panels=all_records, output_sha256=files), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'verified': sum(r['status'] == 'verified' for r in all_records),
                      'unavailable': sum(r['status'] == 'unavailable' for r in all_records),
                      'output': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()
