"""Regenerate the verified panels with a corrected layout in a new directory."""
import json
from pathlib import Path

import rat_rebuild_1d_ed3_20261027 as base
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def draw(panels, rat, heading, target, raw):
    fig = base.plt.figure(figsize=(16, 9), facecolor='white')
    grid = fig.add_gridspec(2, 3, left=.035, right=.895, top=.78, bottom=.08,
                            wspace=.12, hspace=.30)
    norm = Normalize(float(raw[:, 0].min()), float(raw[:, 0].max()))
    for i, panel in enumerate(panels):
        d = panel['dimension']
        ax = fig.add_subplot(grid[i // 3, i % 3], projection='3d' if d == 3 else None)
        if 'error' in panel:
            ax.set_axis_off()
            if d == 3:
                ax.text2D(.5, .5, 'NOT AVAILABLE', transform=ax.transAxes, ha='center', va='center', color='#aa3030')
            else:
                ax.text(.5, .5, 'NOT AVAILABLE', transform=ax.transAxes, ha='center', va='center', color='#aa3030')
        else:
            z, behavior, mask = panel['z'], panel['behavior'], panel['mask']
            for col, cmap in ((2, 'cool'), (1, 'viridis')):
                select = mask & (behavior[:, col] == 1)
                coords = [z[select, j] for j in range(d)]
                kwargs = dict(c=behavior[select, 0], cmap=cmap, norm=norm, s=.7, linewidths=0, rasterized=True)
                if d == 3:
                    kwargs['depthshade'] = False
                ax.scatter(*coords, **kwargs)
            if d == 3:
                ax.view_init(elev=22, azim=38)
                ax.set(xlabel='Latent 1', ylabel='Latent 2', zlabel='Latent 3')
                ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
                ax.set_box_aspect((1, 1, 1))
            else:
                ax.set_axis_off()
        ax.set_title(panel['title'] + f'\n{d}D native', fontsize=9)
    fig.suptitle(f'{rat.capitalize()} — {heading}\nFull-recording fit · seed 42', y=.98, fontsize=13)
    for x, cmap, label, side in ((.922, 'cool', 'Left — position', 'left'),
                                  (.943, 'viridis', 'Right — position', 'right')):
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([x, .25, .012, .48]))
        cb.set_label(label, fontsize=8, labelpad=5)
        cb.ax.yaxis.set_label_position(side)
        cb.ax.yaxis.set_ticks_position(side)
        cb.ax.tick_params(labelsize=7, pad=2)
    fig.text(.47, .025, 'Colour: position; Left / Right: direction. Missing panels are not estimated.', ha='center', fontsize=8)
    target.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(target.with_suffix('.' + suffix), dpi=200, bbox_inches='tight', facecolor='white')
    base.plt.close(fig)


if __name__ == '__main__':
    base.OUT = base.OUT / 'layout_corretto'
    base.draw = draw
    base.main()
    provenance = base.OUT / 'PROVENANCE.json'
    data = json.loads(provenance.read_text(encoding='utf-8'))
    data['layout_script'] = base.relative(Path(__file__).resolve())
    data['layout_script_sha256'] = base.sha(Path(__file__).resolve())
    provenance.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
