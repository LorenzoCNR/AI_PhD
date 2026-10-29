"""Add a common position scale and interactive CEBRA views without fitting."""
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import plotly
import plotly.graph_objects as go
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_hex
import matplotlib

import rat_rebuild_1d_baseline_20261027 as baseline
import rat_rebuild_1d_ed3_20261027 as base


OUT = baseline.OUT / 'etichette_pivae'
TICKS = [0, 0.5, 1.0, 1.5]
TICK_TEXT = ['0', '0.5', '1.0', '1.5']
SPECS = list(baseline.SPECS)

# The two piVAE inference views were both fitted with Position + Direction.
SPECS[1] = ('conv-piVAE — Position + Direction (q(z|x,u))', *SPECS[1][1:])
SPECS[2] = ('conv-piVAE — Position + Direction (q(z|x))', *SPECS[2][1:])
SPECS[1] = ('conv-piVAE — Fit: Position + Direction\nInference: with labels', *SPECS[1][1:])
SPECS[2] = ('conv-piVAE — Fit: Position + Direction\nInference: without labels', *SPECS[2][1:])


def static_figure(panels, rat, target):
    fig = base.plt.figure(figsize=(16, 9), facecolor='white')
    grid = fig.add_gridspec(2, 3, left=.035, right=.895, top=.78, bottom=.08, wspace=.12, hspace=.30)
    norm = Normalize(vmin=0, vmax=1.6)
    for i, panel in enumerate(panels):
        d = panel['dimension']
        ax = fig.add_subplot(grid[i // 3, i % 3], projection='3d' if d == 3 else None)
        if 'error' in panel:
            ax.set_axis_off()
        else:
            z, behavior, mask = panel['z'], panel['behavior'], panel['mask']
            for col, cmap in ((2, 'cool'), (1, 'viridis')):
                sel = mask & (behavior[:, col] == 1)
                kwargs = dict(c=behavior[sel, 0], cmap=cmap, norm=norm, s=.7, linewidths=0, rasterized=True)
                if d == 3:
                    kwargs['depthshade'] = False
                ax.scatter(*(z[sel, j] for j in range(d)), **kwargs)
            if d == 3:
                ax.view_init(elev=22, azim=38)
                ax.set(xlabel='Latent 1', ylabel='Latent 2', zlabel='Latent 3')
                ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
                ax.set_box_aspect((1, 1, 1))
            else:
                ax.set_axis_off()
        if 'error' not in panel:
            ax.set_title(panel['title'], fontsize=9)
    fig.suptitle(f'{rat.capitalize()} — Figure 1D baseline; partial if indicated\nFull-recording fit · seed 42', y=.98, fontsize=13)
    bars = []
    for x, cmap, label, side in ((.922, 'cool', 'Left — position (m)', 'left'),
                                  (.943, 'viridis', 'Right — position (m)', 'right')):
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([x, .25, .012, .48]), ticks=TICKS)
        cb.ax.set_yticklabels(TICK_TEXT)
        cb.set_label('')
        cb.ax.yaxis.set_label_position(side)
        cb.ax.yaxis.set_ticks_position(side)
        cb.ax.tick_params(labelsize=7, pad=2)
        bars.append(cb)
        fig.text(x + .006, .22, 'Left' if cmap == 'cool' else 'Right',
                 ha='center', va='top', fontsize=8)
    assert all(list(cb.get_ticks()) == TICKS and cb.norm.vmin == 0 and cb.norm.vmax == 1.6 for cb in bars)
    fig.text(.47, .025, 'Position scale: 0–1.6 m; Left / Right: direction. Missing panels are not estimated.', ha='center', fontsize=8)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.texts[0].set_text(f'{rat.capitalize()} — Figure 1D')
    fig.texts[-1].set_text('')
    for suffix in ('png', 'pdf'):
        path = target.with_suffix('.' + suffix)
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}')
        fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    base.plt.close(fig)


def interactive(panel, rat, target, overwrite=False):
    z, behavior, mask = panel['z'], panel['behavior'], panel['mask']
    fig = go.Figure()
    counts = {}
    for name, column, cmap, x in (('Left', 2, 'cool', 1.02), ('Right', 1, 'viridis', 1.14)):
        sel = mask & (behavior[:, column] == 1)
        counts[name] = int(sel.sum())
        colormap = matplotlib.colormaps[cmap]
        colorscale = [[j / 255, to_hex(colormap(j / 255))] for j in range(256)]
        fig.add_trace(go.Scatter3d(
            x=z[sel, 0], y=z[sel, 1], z=z[sel, 2], mode='markers', name=name,
            marker=dict(size=2, color=behavior[sel, 0], colorscale=colorscale, cmin=0, cmax=1.6,
                        opacity=1, showscale=True,
                        colorbar=dict(title=dict(text=name, side='bottom'),
                                      x=x, len=.65, thickness=12, tickmode='array',
                                      tickvals=TICKS, ticktext=TICK_TEXT)),
            hovertemplate=f'{name}<br>Position: %{{marker.color:.3f}} m<br>Latent 1: %{{x:.3f}}<br>Latent 2: %{{y:.3f}}<br>Latent 3: %{{z:.3f}}<extra></extra>'))
    fig.update_layout(title=f"{rat.capitalize()} — {panel['title']}<br><sup>Full-recording fit, seed 42 · native 3D · position scale 0–1.6 m</sup>",
                      scene=dict(xaxis_title='Latent 1', yaxis_title='Latent 2', zaxis_title='Latent 3', aspectmode='cube'),
                      width=1100, height=800, margin=dict(l=0, r=190, t=90, b=0),
                      showlegend=True, legend=dict(x=.02, y=.99))
    for trace in fig.data:
        assert trace.marker.cmin == 0 and trace.marker.cmax == 1.6
        assert list(trace.marker.colorbar.tickvals) == TICKS
        assert list(trace.marker.colorbar.ticktext) == TICK_TEXT
    assert sum(counts.values()) == int(mask.sum())
    fig.update_layout(title=f"{rat.capitalize()} — {panel['title'].replace(chr(10), ' — ')}")
    if target.exists() and not overwrite:
        raise FileExistsError(f'Refusing to overwrite {target}')
    fig.write_html(str(target), full_html=True, include_plotlyjs=True,
                   config={'responsive': True, 'displaylogo': False})
    if target.stat().st_size < 100000 or 'Plotly.newPlot' not in target.read_text(encoding='utf-8'):
        raise ValueError(f'Invalid Plotly HTML: {target}')
    return counts


def main():
    if not OUT.is_dir():
        raise FileNotFoundError(f'Previous versioned folder is missing: {OUT}')
    outputs, records = [], []
    parent_manifest = OUT / 'PROVENANCE.json'
    for rat in base.RATS:
        source = base.ROOT / 'data' / 'rat_hippocampus' / f'{rat}.jl'
        raw = np.asarray(joblib.load(source)['position'], dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
            raise ValueError(f'invalid position data for {rat}')
        if float(raw[:, 0].min()) < -1e-6 or float(raw[:, 0].max()) > 1.6 + 1e-6:
            raise ValueError(f'{rat} has positions outside requested range 0–1.6')
        panels = []
        for spec in SPECS:
            try:
                panel = base.read_panel(rat, raw, base.sha(source), spec, {})
                record = dict(status='verified', **panel['record'])
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                panel = dict(title=spec[0], error=str(exc))
                record = dict(status='unavailable', rat=rat, title=spec[0].replace('\n', ' '),
                              expected_dimension=spec[4], reason=str(exc))
            panel['dimension'] = spec[4]
            records.append(record)
            panels.append(panel)
        target = OUT / rat / f'Figure_1D_{rat}_baseline_scala_0_1_6m_v2'
        static_figure(panels, rat, target)
        outputs.extend([target.with_suffix('.png'), target.with_suffix('.pdf')])
        for panel, slug in ((panels[0], 'CEBRA_position_direction'), (panels[3], 'CEBRA_time')):
            if 'error' in panel:
                records.append(dict(status='interactive_unavailable', rat=rat, method=slug, reason=panel['error']))
                continue
            path = OUT / rat / f'Figure_1D_{rat}_{slug}_3D_scala_v2.html'
            direction_counts = interactive(panel, rat, path)
            outputs.append(path)
            records.append(dict(status='interactive_verified', rat=rat, method=slug,
                                plotted_dimension=3, position_unit='m', counts=direction_counts,
                                parent_embedding_sha256=panel['record']['embedding_sha256'],
                                parent_source_ids_sha256=panel['record']['plotted_source_ids_sha256'],
                                html=base.relative(path)))
    report = OUT / 'SCALA_POSIZIONE_E_3D_INTERATTIVI_v2.md'
    if report.exists():
        raise FileExistsError(report)
    report.write_text('# Figura 1D: scala posizione e 3D interattivi\n\n'
        'La posizione usa metri: il notebook ufficiale CEBRA indica Position [m] sullo stesso dataset; '
        'il paper descrive un percorso lineare di 1,6 m.\n\n'
        'Tutte le barre usano 0–1,6 m e tacche 0, 0.5, 1.0, 1.5. Left usa cool; Right usa viridis. '
        'Le figure HTML contengono due tracce e Plotly incorporato, ruotabili offline. '
        'I punti e gli embedding sono gli stessi verificati nei manifest. t-SNE resta 2D e i pannelli mancanti sono espliciti.\n\n'
        'Fonti: https://cebra.ai/docs/demo_notebooks/Demo_hippocampus.html e '
        'https://www.nature.com/articles/s41586-023-06031-6.pdf\n', encoding='utf-8')
    outputs.append(report)
    manifest = OUT / 'MANIFEST_SCALA_0_1_6M_v2.json'
    if manifest.exists():
        raise FileExistsError(manifest)
    manifest.write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
        script=base.relative(Path(__file__).resolve()), script_sha256=base.sha(Path(__file__).resolve()),
        validator=base.relative(Path(base.__file__).resolve()), validator_sha256=base.sha(Path(base.__file__).resolve()),
        parent_provenance=base.relative(parent_manifest), parent_provenance_sha256=base.sha(parent_manifest),
        position_unit='m', position_unit_source='https://cebra.ai/docs/demo_notebooks/Demo_hippocampus.html',
        scale=[0, 1.6], ticks=TICKS, tick_labels=TICK_TEXT, colormaps={'Left': 'cool', 'Right': 'viridis'},
        matplotlib_version=matplotlib.__version__, plotly_version=plotly.__version__,
        records=records, output_sha256={base.relative(p): base.sha(p) for p in outputs}),
        indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'static_figures': 4, 'interactive_html': sum(r.get('status') == 'interactive_verified' for r in records),
                      'missing_panels': sum(r.get('status') == 'unavailable' for r in records), 'manifest': str(manifest)}, indent=2))


if __name__ == '__main__':
    main()
