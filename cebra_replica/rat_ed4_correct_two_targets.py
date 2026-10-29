"""Recalculate ED4 decoding from saved embeddings; do not train encoders."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score

import rat_rebuild_1e_ed4_metrics as source
import rat_render_extended_data_3_4_20261028 as renderer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'figura_ext_4_2026-10-28_corretto'
TABLES = OUT / 'calcoli'


def decode(train, valid, test, device):
    x, y = train[:2]
    if not np.allclose(y[:, 1:].sum(axis=1), 1):
        raise ValueError('Expected complementary direction labels')
    tx = torch.nn.functional.normalize(torch.as_tensor(x, device=device), dim=1)
    predictions = []
    for data in (valid, test):
        q = torch.nn.functional.normalize(torch.as_tensor(data[0], device=device), dim=1)
        neighbors = []
        with torch.inference_mode():
            for start in range(0, len(q), 512):
                neighbors.append(torch.topk(q[start:start+512] @ tx.T, max(source.KS), dim=1).indices.cpu().numpy())
        indices = np.concatenate(neighbors)
        by_k = {}
        for k in source.KS:
            selected = indices[:, :k]
            pos = y[selected, 0].mean(axis=1)
            right = (y[selected, 1].sum(axis=1) > k / 2).astype(np.float32)
            by_k[k] = np.column_stack((pos, right))
        predictions.append(by_k)
    scores = {k: float(r2_score(valid[1][:, :2], predictions[0][k])) for k in source.KS}
    k = max(source.KS, key=lambda n: scores[n])
    pred = predictions[1][k]
    test_scores = r2_score(test[1][:, :2], pred, multioutput='raw_values')
    return k, scores[k], float(np.mean(test_scores)), float(np.median(abs(test[1][:, 0]-pred[:, 0]))), *map(float, test_scores)


def main():
    torch.set_num_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TABLES.mkdir(parents=True, exist_ok=True)
    old = source.OUT / 'heldout_decoding.csv'
    with old.open(newline='', encoding='utf-8') as f:
        cells = list(csv.DictReader(f))
    mapping = {label: (kind, variant) for label, kind, variant in source.METHODS}
    parts = TABLES / 'celle'
    parts.mkdir(exist_ok=True)
    rows = []
    for index, cell in enumerate(cells):
        method, rat = cell['method'], cell['rat']
        fold, seed = int(cell['fold']), int(cell['seed'])
        part = parts / f'{index:04d}.json'
        if part.exists():
            row = json.loads(part.read_text(encoding='utf-8'))
            if (row['method'], row['rat'], row['fold'], row['seed']) != (method, rat, fold, seed):
                raise RuntimeError(f'Incompatible cached cell {part}')
        else:
            kind, variant = mapping[method]
            data = [source.resolve(kind, variant, rat, fold, seed, s) for s in ('train', 'valid', 'test')]
            for i, j in ((0, 1), (0, 2), (1, 2)):
                if np.intersect1d(data[i][2], data[j][2]).size:
                    raise RuntimeError('Split overlap')
            k, val, total, error, pos_r2, dir_r2 = decode(*data, device)
            row = dict(method=method, rat=rat, fold=fold, seed=seed, selected_k=k,
                       validation_total_r2=val, test_total_r2=total, test_position_error_m=error,
                       test_position_r2=pos_r2, test_direction_r2=dir_r2,
                       input_paths=[d[3] for d in data], input_sha256=[d[4] for d in data])
            part.write_text(json.dumps(row, ensure_ascii=False), encoding='utf-8')
        rows.append(row)
        if index % 20 == 0 or index+1 == len(cells):
            print(f'{index+1}/{len(cells)} {method} {rat} fold={fold} seed={seed}', flush=True)
    with (TABLES / 'heldout_decoding_due_variabili.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (source.OUT / 'ed4_plot_values_v3.csv').open(newline='', encoding='utf-8') as f:
        summary = [r for r in csv.DictReader(f) if r['metric'] not in ('test_total_r2', 'test_position_error_m')]
    groups = defaultdict(list)
    for row in rows:
        groups[(row['rat'], row['method'], row['seed'])].append(row)
    for (rat, method, seed), group in groups.items():
        if len(group) != 3 or {r['fold'] for r in group} != {0, 1, 2}:
            raise ValueError(f'Incomplete folds: {rat}, {method}, {seed}')
        for key in ('test_total_r2', 'test_position_error_m'):
            summary.append(dict(rat=rat, method=method, metric=key, seed=seed,
                                value=float(np.mean([r[key] for r in group])) * 100,
                                unit='cm' if key.endswith('_m') else 'percent',
                                aggregation='mean of three folds per seed'))
    with (TABLES / 'ed4_plot_values_due_variabili.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    for letter, rat in renderer.RATS:
        dest = OUT / letter / rat.capitalize()
        for name, methods in (('Paper', renderer.PAPER), ('Soft', renderer.SOFT)):
            renderer.render_ed4(rat, methods, summary, dest / f'ED4_{rat.upper()}_{name.upper()}')
    (OUT / 'RIFERIMENTI.md').write_text('''# ED4 - riferimenti e metodo\n\nDecoder su embedding salvati, senza training. kNN cosine: posizione regressione, direzione classificazione; k=1,4,9,16,25 scelto sulla validation con media dei due R², test held-out. Il cambio di k modifica anche l'errore mediano di posizione.\n\nSei seed, tre fold; punti = media dei fold. Le consistenze train sono riutilizzate senza modifiche. q(z|x,u) non viene valutato per decoding. L'implementazione a due variabili segue gli esempi CEBRA disponibili; l'esatta procedura dietro i CSV ufficiali ED4 non è stata confermata.\n\nCodice: `cebra_replica/rat_ed4_correct_two_targets.py`; input e hash per ogni cella: `calcoli/heldout_decoding_due_variabili.csv`. Consistenza: `results/rat_rebuild_20261027/Figure_1E_ED4/ed4_plot_values_v3.csv`.\n''', encoding='utf-8')
    print('COMPLETED', len(rows), OUT, flush=True)


if __name__ == '__main__':
    main()
