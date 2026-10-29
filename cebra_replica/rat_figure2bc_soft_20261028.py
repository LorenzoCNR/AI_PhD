"""Render Figure 2b Soft and Figure 2c from existing verified artifacts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import rat_rebuild_fig2bc as source
import rat_rebuild_1e_ed4_metrics as metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'figura_2_2026-10-28'
SOFT = [('time_only', 'Time'), ('direction_position', 'Position + Direction'),
        ('time_direction_position', 'Time + Position + Direction')]
CEBRA = [('position', 'Position'), ('direction', 'Direction'),
         ('position_direction', 'Position + Direction'),
         ('position_shuffled', 'Position shuffled'),
         ('direction_shuffled', 'Direction shuffled'),
         ('position_direction_shuffled', 'Position + Direction shuffled')]


def plot(rat, family, panel, entries, histories, refs):
    n = len(entries)
    fig = plt.figure(figsize=(16, 9 if n > 3 else 5.5))
    rows = 2 if n > 3 else 1
    norm = Normalize(0, 1.6)
    for i, (title, z, y) in enumerate(entries, 1):
        ax = fig.add_subplot(rows, 3, i, projection='3d')
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()
        for col, cmap in ((2, 'cool'), (1, 'viridis')):
            mask = y[:, col] == 1
            ax.scatter(*z[mask, :3].T, c=y[mask, 0], cmap=cmap, norm=norm,
                       s=.7, linewidths=0, rasterized=True)
    for x, cmap, label in ((.91, 'cool', 'Left'), (.95, 'viridis', 'Right')):
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([x,.25,.012,.5]))
        bar.set_ticks([0,.5,1,1.5]); fig.text(x+.006,.20,label,ha='center',fontsize=9)
    fig.suptitle(rat.capitalize(), fontsize=14)
    fig.subplots_adjust(left=.02,right=.89,top=.91,bottom=.07,wspace=.08,hspace=.12)
    dest = OUT / panel / rat.capitalize()
    dest.mkdir(parents=True, exist_ok=True)
    stem = dest / f'Figura_2{panel}_{rat.capitalize()}_{family}'
    for ext in ('.png', '.pdf'):
        path = stem.with_suffix(ext)
        if path.exists(): raise FileExistsError(path)
        fig.savefig(path,dpi=220,bbox_inches='tight')
    plt.close(fig)
    if histories:
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, updates, loss in histories:
            ax.plot(updates, loss, label=name, linewidth=.8, linestyle='--' if 'shuffled' in name else '-')
        ax.set_xlabel('Update'); ax.set_ylabel('Training loss'); ax.legend(fontsize=8); ax.grid(alpha=.2)
        fig.tight_layout()
        for ext in ('.png', '.pdf'):
            path = stem.parent / (stem.name + '_loss' + ext)
            if path.exists(): raise FileExistsError(path)
            fig.savefig(path,dpi=220,bbox_inches='tight')
        plt.close(fig)
    (dest / f'RIFERIMENTI_{family}.json').write_text(json.dumps(refs,indent=2,ensure_ascii=False),encoding='utf-8')


def main():
    for rat in source.RATS:
        entries, histories, refs = [], [], []
        for condition, title in CEBRA:
            folder = source.CEBRA / rat / condition / 'seed_42'
            manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
            emb = folder/'all_embedding.npz'; loss = folder/'loss_history.csv'
            refs.extend([source.verified(p,manifest['artifact_sha256'][p.name]) for p in (emb,loss)])
            with np.load(emb,allow_pickle=False) as a:
                z, y = a['embedding'].copy(), a['behavior'].copy()
            entries.append((f'CEBRA — {title}',z,y))
            history = np.loadtxt(loss,delimiter=',',skiprows=1)
            histories.append((title,history[:,0],history[:,1]))
        plot(rat,'CEBRA','c',entries,histories,{'seed':42,'sources':refs,'fit':'32D, first three coordinates; first 80% recording','plot':'all transformed rows','training_performed':False})
        for arch in ('cnn','transformer'):
            entries, histories, refs = [], [], []
            for condition,title in SOFT:
                arrays = [metrics.resolve('soft',(condition,arch),rat,0,42,s) for s in ('train','valid','test')]
                ids = np.concatenate([a[2] for a in arrays]); order = np.argsort(ids)
                if len(np.unique(ids)) != len(ids): raise ValueError('Overlapping split IDs')
                z = np.concatenate([a[0] for a in arrays])[order]
                y = np.concatenate([a[1] for a in arrays])[order]
                refs.extend({'path':a[3],'sha256':a[4]} for a in arrays)
                folder = metrics.CAMP/'soft/stage01_fit'/condition/arch/rat/'fold_0'/'soft_v1_stable'/'seed_42'
                marker = folder/'manifest.json'; cell = json.loads((folder/'CELL_MANIFEST.json').read_text(encoding='utf-8'))
                refs.append(source.verified(marker,cell['fit_manifest_sha256']))
                history = json.loads(marker.read_text(encoding='utf-8'))['history']
                histories.append((title,[h['update'] for h in history],[h['train_loss'] for h in history]))
                entries.append((f'Soft {arch.upper()} — {title}',z,y))
            info = {'seed':42,'fold':0,'sources':refs,'fit':'3D, train trials','plot':'train + validation + test transformed','conditions_unavailable':'Position only, Direction only, and shuffled fits not in this campaign','training_performed':False}
            plot(rat,'SOFT_'+arch.upper(),'b',entries,[],info)
            plot(rat,'SOFT_'+arch.upper(),'c',entries,histories,info)
        print(rat,'complete',flush=True)


if __name__ == '__main__': main()
