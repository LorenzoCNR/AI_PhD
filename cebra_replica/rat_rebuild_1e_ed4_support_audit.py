"""Audit staged metric input identities against official-split CEBRA arrays."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rat_rebuild_20261027/Figure_1E_ED4'
CAMP = ROOT / 'results/rat_paper_fig1e_ed4_6seed_v1/cebra'


def main():
    with (OUT / 'embedding_inventory.csv').open(newline='', encoding='utf-8') as f:
        inventory = list(csv.DictReader(f))
    refs = {}
    checked = 0
    failures = []
    subsets = 0
    for item in inventory:
        key = (item['rat'], int(item['fold']), item['split'])
        if key not in refs:
            path = CAMP / key[0] / f'fold_{key[1]}' / 'behavior/seed_42/embeddings' / f'{key[2]}.npz'
            with np.load(path, allow_pickle=False) as a:
                refs[key] = (a['source_id'].copy(), a['trial_id'].copy(), a['behavior'].copy())
        ids_ref, trials_ref, y_ref = refs[key]
        path = ROOT / item['path']
        with np.load(path, allow_pickle=False) as a:
            ids = a['source_id']
            y = a['u'] if 'u' in a.files else a['behavior']
            trials = a['trial_id'] if 'trial_id' in a.files else None
            if item['method'] in ('UMAP', 't-SNE', 'PCA'):
                mask = a['valid_mask'].astype(bool)
                ids, y = ids[mask], y[mask]
                trials = trials[mask] if trials is not None else None
            order = np.argsort(ids_ref)
            ix = np.searchsorted(ids_ref[order], ids)
            good = (ix < len(order))
            if good.all():
                ix = order[ix]
                good = np.array_equal(ids_ref[ix], ids)
            if not bool(np.all(good)):
                failures.append(dict(path=item['path'], reason='source IDs outside reference split'))
            elif not np.allclose(y_ref[ix], y, atol=1e-6, rtol=1e-6):
                failures.append(dict(path=item['path'], reason='behavior differs from reference for same source IDs'))
            elif trials is not None and not np.array_equal(trials_ref[ix], trials):
                failures.append(dict(path=item['path'], reason='trial IDs differ for same source IDs'))
            elif len(ids) < len(ids_ref):
                subsets += 1
            checked += 1
    report = dict(status='pass' if not failures else 'failed', checked_records=checked,
                  strict_subsets=subsets, failures=failures,
                  reference='CEBRA behavior seed 42 embedding source_id, trial_id and behavior in same rat/fold/split',
                  note='Baseline window-valid masks produce strict subsets; rows are not silently equalized across methods.')
    with (OUT / 'support_audit.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
