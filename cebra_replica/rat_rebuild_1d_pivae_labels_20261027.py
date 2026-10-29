"""Render the baseline 1D figures with explicit piVAE fit/inference labels."""
import json
from pathlib import Path

import rat_rebuild_1d_baseline_20261027 as baseline


if __name__ == '__main__':
    baseline.OUT = baseline.OUT / 'etichette_pivae'
    specs = list(baseline.SPECS)
    specs[1] = ('conv-piVAE — Fit: Position + Direction\nInference: with labels', *specs[1][1:])
    specs[2] = ('conv-piVAE — Fit: Position + Direction\nInference: without labels', *specs[2][1:])
    baseline.SPECS = tuple(specs)
    baseline.main()
    provenance = baseline.OUT / 'PROVENANCE.json'
    data = json.loads(provenance.read_text(encoding='utf-8'))
    this_file = Path(__file__).resolve()
    data['label_revision_script'] = baseline.base.relative(this_file)
    data['label_revision_script_sha256'] = baseline.base.sha(this_file)
    provenance.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
