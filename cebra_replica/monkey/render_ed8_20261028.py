"""Render ED8 from saved arrays and scores only; never fit or modify inputs."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/figura_ext_8_2026-10-28'
DATA = ROOT / 'results/monkey_extended8_paper_v1'
SCORES = ROOT / 'results/PAPER_FIGURE_DELIVERY/Extended_Data/ED8/ED8_AP_DECODING_SEEDS.csv'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(fig, panel, name):
    folder = OUT / panel
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(folder / (name + '.' + ext), dpi=200, bbox_inches='tight')
    plt.close(fig)

def main():
    inputs = []
    for mode, panel in [('target', 'a'), ('position', 'b'), ('time', 'c')]:
        path = DATA / 'stage02_embedding_mse' / mode / 'seed_42/all_trials.npz'
        manifest = json.loads((path.parent / 'EMBEDDING.json').read_text())
        assert digest(path) == manifest['artifact_sha256']
        inputs.append(path)
        with np.load(path) as z:
            emb = z['embedding_raw'].copy()
            pos = z['position_xy'].copy()
            direction = z['direction'].copy()
            assert emb.shape == (193 * 600, 2) and z['valid_mask'].all()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        if mode == 'target':
            axes[0].scatter(emb[:, 0], emb[:, 1], c=direction, cmap='hsv', s=.05, rasterized=True)
            axes[0].set_title('193 trial attivi: tutti i campioni')
            trials = emb.reshape(193, 600, 2)
            labels = direction.reshape(193, 600)
            assert np.all(labels == labels[:, :1])
            for d in range(8):
                avg = trials[labels[:, 0] == d].mean(axis=0)
                axes[1].scatter(avg[:, 0], avg[:, 1], s=2, color=plt.cm.hsv(d / 8), label=str(d))
            axes[1].set_title('Media per direzione')
            axes[1].legend(title='Direzione', fontsize=7, ncol=2)
            fig.suptitle('CEBRA: etichetta direzione, InfoMSE — seed 42')
        else:
            for i, ax in enumerate(axes):
                sc = ax.scatter(emb[:, 0], emb[:, 1], c=pos[:, i], cmap='seismic', vmin=-15, vmax=15, s=.05, rasterized=True)
                ax.set_title('Posizione ' + 'xy'[i])
            fig.colorbar(sc, ax=axes, label='Posizione (cm)', shrink=.8)
            fig.suptitle(('CEBRA: posizione x,y' if mode == 'position' else 'CEBRA-Time: senza etichette esterne') + ', InfoMSE — seed 42')
        for ax in axes: ax.axis('off')
        save(fig, panel, 'ED8_' + panel.upper() + '_locale')

    path = DATA / 'stage04_direction_timecourse/DIRECTION_ACCURACY_BY_TIME.csv'
    inputs.append(path)
    with path.open(newline='') as stream: rows = list(csv.DictReader(stream))
    fig, ax = plt.subplots(figsize=(8, 4))
    for key, label in [('CEBRA-Behavior', 'CEBRA: direzione (media 5 seed)'), ('conv-piVAE', 'conv-piVAE (seed 42)'), ('Ridge classifier', 'Classificatore Ridge')]:
        groups = {}
        for row in rows:
            if row['method'] == key: groups.setdefault(row['training_seed'], []).append(row)
        series = [sorted(v, key=lambda x: int(x['time_bin'])) for v in groups.values()]
        assert series and all(len(s) == 600 for s in series)
        x = np.array([int(r['time_from_onset_ms']) for r in series[0]])
        y = np.mean([[float(r['accuracy_percent']) for r in s] for s in series], axis=0)
        ax.plot(x, y, label=label)
    ax.axhline(12.5, color='gray', ls='--', label='Caso: 1/8')
    ax.set(xlabel='Tempo dall’inizio del movimento (ms)', ylabel='Accuratezza direzione (%)', title='Trial attivi: decoding sul test')
    ax.legend(fontsize=8)
    save(fig, 'd', 'ED8_D_locale')

    fig, ax = plt.subplots(figsize=(8, 4))
    for mode, label in [('target', 'Direzione'), ('position', 'Posizione x,y'), ('time', 'Tempo, senza etichette esterne')]:
        path = DATA / 'stage01_fit_mse' / mode / 'seed_42/training_loss.npy'
        manifest = json.loads((path.parent / 'FIT.json').read_text())
        assert digest(path) == manifest['history_sha256']
        inputs.append(path)
        loss = np.load(path)
        assert len(loss) == 5000
        ax.plot(np.arange(1, len(loss)+1), loss, label=label, lw=1)
    ax.set(xlabel='Iterazione', ylabel='Loss di training', title='InfoMSE locale: trial attivi, 2D, seed 42')
    ax.legend()
    save(fig, 'e', 'ED8_E_loss_locale_3_condizioni')

    inputs.append(SCORES)
    with SCORES.open(newline='', encoding='utf-8-sig') as stream: rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, task, title, ylabel in [(axes[0], 'ap_pos', 'Posizione; fit: attivo/passivo + x,y', 'R² (%)'), (axes[1], 'ap_direction', 'Classe congiunta attivo/passivo + direzione', 'Accuratezza (%)')]:
        values_all = []
        for i, dim in enumerate([3, 4, 8, 16, 32]):
            subset = [r for r in rows if r['task'] == task and int(r['dimension']) == dim]
            assert sorted(int(r['seed']) for r in subset) == [42,43,44,45,46]
            assert all(r['split'] == 'test' for r in subset)
            vals = [float(r['score_percent']) for r in subset]
            values_all.extend(vals)
            ax.scatter([i]*5, vals, facecolors='none', edgecolors='black')
            ax.hlines(np.median(vals), i-.22, i+.22, colors='orange', lw=2)
        if task == 'ap_direction': ax.axhline(100/16, color='gray', ls='--', label='Caso congiunto 1/16')
        ax.set_xticks(range(5), [3,4,8,16,32])
        ax.set(xlabel='Dimensioni', ylabel=ylabel, title=title)
        ax.set_ylim(min(min(values_all)-5, 0), max(max(values_all)+5, 20))
        if task == 'ap_direction': ax.legend(fontsize=8)
    fig.suptitle('Decoder kNN: fit train, punteggi test; punti: seed 42–46')
    fig.tight_layout()
    save(fig, 'f', 'ED8_F_locale')
    (OUT / 'INPUT_SHA256.json').write_text(json.dumps({str(p.relative_to(ROOT)): digest(p) for p in inputs}, indent=2), encoding='utf-8')

if __name__ == '__main__': main()
