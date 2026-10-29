"""Render only Figure 1D baseline panels from existing verified full recordings."""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

import rat_rebuild_1d_ed3_20261027 as base
import rat_rebuild_1d_ed3_layout_20261027 as layout


OUT = base.ROOT / 'results' / 'rat_rebuild_20261027' / 'Figure_1' / '1D'
SPECS = (('CEBRA Position + Direction', 'cebra', 'behavior', 'embedding', 3),) + base.PAPER[1:]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if any(OUT.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing output: {OUT}')
    rows = []
    report = [
        '# Figure 1D — baseline full-recording seed 42', '',
        'The fit uses each entire recording; these are descriptive embeddings, not held-out plots.',
        'CEBRA is native 3D; piVAE, UMAP and t-SNE are native 2D.',
        'Colour encodes position; Left and Right indicate direction. piVAE plots valid-window rows; CEBRA plots all saved rows; manifold fits use complete 10-bin windows.',
        'The two piVAE panels show q(z|x,u), with Position + Direction at inference, and q(z|x), without labels at inference.',
        'UMAP/t-SNE window flattening is a local implementation choice, not an asserted reproduction of unpublished original code.',
        'Unavailable artifacts are identified explicitly; no substitute fit is used.', '',
        '| Rat | Model | Status | Native D | Saved / plotted rows | Source or reason |',
        '|---|---|---|---|---|---|',
    ]
    for rat in base.RATS:
        source = base.ROOT / 'data' / 'rat_hippocampus' / f'{rat}.jl'
        raw = np.asarray(joblib.load(source)['position'], dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
            raise ValueError(f'invalid behavior data: {rat}')
        data_hash = base.sha(source)
        panels = []
        for spec in SPECS:
            try:
                panel = base.read_panel(rat, raw, data_hash, spec, {})
                record = dict(status='verified', **panel['record'])
                evidence = record['embedding']
                counts = f"{record['stored_rows']} / {record['plotted_rows']}"
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                evidence = str(exc).replace('|', '/')
                panel = dict(title=spec[0], error=evidence)
                record = dict(status='unavailable', rat=rat, seed=42,
                              title=spec[0].replace('\n', ' '), reason=evidence,
                              expected_dimension=spec[4])
                if spec[1] == 'manifold':
                    marker = base.ED3 / 'manifold' / spec[2] / rat / 'window10_flat' / 'seed_42' / 'RUNNING.json'
                    if marker.is_file():
                        record['marker'] = base.relative(marker)
                        record['marker_sha256'] = base.sha(marker)
                        record['marker_contents'] = json.loads(marker.read_text(encoding='utf-8'))
                counts = '—'
            panel['dimension'] = spec[4]
            rows.append(record)
            panels.append(panel)
            report.append(f"| {rat} | {spec[0].replace(chr(10), ' ')} | {record['status']} | {spec[4]}D | {counts} | {evidence} |")
        layout.draw(panels, rat, 'Figure 1D — baseline; partial if indicated',
                    OUT / rat / f'Figure_1D_{rat}_baseline', raw)
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    output_hashes = {base.relative(p): base.sha(p) for p in OUT.rglob('*') if p.is_file()}
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                      script=base.relative(Path(__file__).resolve()), script_sha256=base.sha(Path(__file__).resolve()),
                      validator_script=base.relative(Path(base.__file__).resolve()), validator_sha256=base.sha(Path(base.__file__).resolve()),
                      layout_script=base.relative(Path(layout.__file__).resolve()), layout_sha256=base.sha(Path(layout.__file__).resolve()),
                      panels=rows, output_sha256=output_hashes)
    (OUT / 'PROVENANCE.json').write_text(json.dumps(provenance, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'verified': sum(r['status'] == 'verified' for r in rows),
                      'unavailable': sum(r['status'] == 'unavailable' for r in rows),
                      'output': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()
