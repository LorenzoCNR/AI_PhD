"""Verify the four scaled 1D figures and the two requested Achilles HTML views."""
from datetime import datetime, timezone
import json
from pathlib import Path

import rat_rebuild_1d_ed3_20261027 as base


OUT = base.ROOT / 'results' / 'rat_rebuild_20261027' / 'Figure_1' / '1D' / 'etichette_pivae'
PARENT = OUT / 'MANIFEST_SCALA_0_1_6M_v2.json'


def main():
    parent = json.loads(PARENT.read_text(encoding='utf-8'))
    if parent['scale'] != [0, 1.6] or parent['ticks'] != [0, 0.5, 1.0, 1.5] or parent['position_unit'] != 'm':
        raise ValueError('Position scale does not match the requested specification')
    selected = []
    for rat in base.RATS:
        for suffix in ('png', 'pdf'):
            selected.append(OUT / rat / f'Figure_1D_{rat}_baseline_scala_0_1_6m_v2.{suffix}')
    for method in ('CEBRA_position_direction', 'CEBRA_time'):
        selected.append(OUT / 'achilles' / f'Figure_1D_achilles_{method}_3D_scala_v2.html')
    for path in selected:
        relative = base.relative(path)
        if not path.is_file() or base.sha(path) != parent['output_sha256'].get(relative):
            raise ValueError(f'Missing or hash mismatch: {path}')
    records = [r for r in parent['records'] if r['status'] in ('verified', 'unavailable')
               or (r['status'] == 'interactive_verified' and r['rat'] == 'achilles')]
    if sum(r['status'] == 'interactive_verified' for r in records) != 2:
        raise ValueError('Expected exactly two verified Achilles interactive views')
    report = OUT / 'FINAL_REPORT_1D_ACHILLES_3D.md'
    manifest = OUT / 'FINAL_MANIFEST_1D_ACHILLES_3D.json'
    if report.exists() or manifest.exists():
        raise FileExistsError('Refusing to overwrite final report or manifest')
    report.write_text('# Figura 1D — risultati finali\n\n'
        'Quattro figure statiche, una per ratto, con scala posizione comune 0–1.6 m '
        'e tacche 0, 0.5, 1.0, 1.5. Sono finali soltanto i due HTML 3D CEBRA di Achilles '
        '(Position + Direction e Time).\n\n'
        'Fit full-recording seed 42; nessun nuovo training. t-SNE è previsto in 2D e '
        'rimane non disponibile per i quattro ratti; UMAP è disponibile solo per Achilles. '
        'Per i dettagli su supporti, sorgenti e hash si veda il manifest allegato.\n', encoding='utf-8')
    manifest.write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
        script=base.relative(Path(__file__).resolve()), script_sha256=base.sha(Path(__file__).resolve()),
        parent_manifest=base.relative(PARENT), parent_manifest_sha256=base.sha(PARENT),
        position_unit='m', position_scale=[0, 1.6], position_ticks=[0, 0.5, 1.0, 1.5],
        colormaps={'Left': 'cool', 'Right': 'viridis'},
        final_outputs={base.relative(p): base.sha(p) for p in selected},
        report=base.relative(report), report_sha256=base.sha(report), records=records),
        indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'verified_static_figures': 4, 'verified_achilles_html': 2,
                      'missing_panels': sum(r['status'] == 'unavailable' for r in records),
                      'manifest': str(manifest)}, indent=2))


if __name__ == '__main__':
    main()
