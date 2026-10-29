"""Independent spot checks of staged results against sklearn and CEBRA metrics."""
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from cebra.integrations.sklearn.metrics import consistency_score
from cebra_replica.rat_rebuild_1e_ed4_metrics import ROOT, OUT, METHODS, KS, resolve, r2


def main():
    with (OUT / 'heldout_decoding.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    results = []
    indices = (0, 2, 3, 8, 10, 11)
    for idx in indices:
        label, kind, variant = METHODS[idx]
        a, b, c = [resolve(kind, variant, 'achilles', 0, 42, split) for split in ('train', 'valid', 'test')]
        predicted = {}
        scores = {}
        for k in KS:
            reg = KNeighborsRegressor(n_neighbors=k, metric='cosine', algorithm='brute', n_jobs=2)
            clf = KNeighborsClassifier(n_neighbors=k, metric='cosine', algorithm='brute', n_jobs=2)
            reg.fit(a[0], a[1][:, 0]); clf.fit(a[0], a[1][:, 1:].argmax(axis=1))
            direction = clf.predict(b[0])
            pred = np.column_stack((reg.predict(b[0]), np.eye(2)[direction]))
            scores[k] = float(r2_score(b[1], pred))
            predicted[k] = (reg, clf)
        best = max(KS, key=lambda k: scores[k])
        reg, clf = predicted[best]
        pred = np.column_stack((reg.predict(c[0]), np.eye(2)[clf.predict(c[0])]))
        actual = next(r for r in rows if r['method'] == label and r['rat'] == 'achilles' and r['fold'] == '0' and r['seed'] == '42')
        test_r2 = float(r2_score(c[1], pred))
        error = float(np.median(np.abs(c[1][:, 0] - pred[:, 0])))
        results.append(dict(method=label, staged_k=int(actual['selected_k']), sklearn_k=best,
            validation_r2_delta=abs(scores[best] - float(actual['validation_total_r2'])),
            test_r2_delta=abs(test_r2 - float(actual['test_total_r2'])),
            test_error_m_delta=abs(error - float(actual['test_position_error_m']))))
    label, kind, variant = METHODS[0]
    e1 = resolve(kind, variant, 'achilles', 0, 42, 'train')
    e2 = resolve(kind, variant, 'achilles', 0, 43, 'train')
    if not np.array_equal(e1[2], e2[2]):
        raise RuntimeError('QA run comparison IDs do not match')
    scores, pairs, _ = consistency_score([e1[0], e2[0]], between='runs')
    pairs = np.asarray(pairs).reshape(-1, 2)
    scores = np.asarray(scores).reshape(-1)
    official = float(scores[np.flatnonzero((pairs[:, 0] == 0) & (pairs[:, 1] == 1))[0]])
    local = r2(e1[0], e2[0])
    tolerance = 1e-4
    passed = all(r['staged_k'] == r['sklearn_k'] and r['validation_r2_delta'] < tolerance and
                 r['test_r2_delta'] < tolerance and r['test_error_m_delta'] < tolerance for r in results)
    report = dict(status='pass' if passed and abs(official-local) < tolerance else 'needs_review',
        decoder_comparisons=results, consistency_cebra_score=official, consistency_local_score=local,
        note='Six Achilles/fold0/seed42 decoder spot checks; train/validation/test kept separate. One train run-consistency pair checked against CEBRA.')
    with (OUT/'numerical_qa.json').open('x', encoding='utf-8') as f: json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if report['status'] != 'pass': raise SystemExit(1)


if __name__ == '__main__':
    main()
