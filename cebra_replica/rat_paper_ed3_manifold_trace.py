"""Diagnostic bootstrap for the unchanged resumable ED3 UMAP/t-SNE runner."""
from __future__ import annotations

import faulthandler
import hashlib
import json
import runpy
import sys
from pathlib import Path


runner = Path(__file__).with_name("rat_paper_ed3_manifold_resumable.py")
faulthandler.dump_traceback_later(20, repeat=True)
runner_args = sys.argv[1:]
is_tsne = "--method" in runner_args and runner_args[runner_args.index("--method") + 1] == "tsne"
if is_tsne:
    import openTSNE
    import numpy as np

    _original_fit = openTSNE.TSNE.fit

    def _fit_barnes_hut(self, *args, **kwargs):
        self.negative_gradient_method = "bh"
        embedding = _original_fit(self, *args, **kwargs)
        values = np.asarray(embedding)
        finite = np.isfinite(values)
        print(
            f"TSNE_DIAGNOSTIC shape={values.shape} finite={int(finite.sum())}/{values.size} "
            f"min={np.nanmin(values)} max={np.nanmax(values)}",
            flush=True,
        )
        return embedding

    openTSNE.TSNE.fit = _fit_barnes_hut

sys.argv = [str(runner), *runner_args]
runpy.run_path(str(runner), run_name="__main__")

if is_tsne:
    rat = runner_args[runner_args.index("--rat") + 1]
    seed = runner_args[runner_args.index("--seed") + 1]
    output = runner.parent.parent / "results" / "rat_paper_replication_2023_v1" / "ED3" / "manifold"
    manifest_path = output / "tsne" / rat / "window10_flat" / f"seed_{seed}" / "manifest.json"
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved = record.get("resolved_parameters", {})
    if resolved.get("negative_gradient_method") != "bh":
        raise RuntimeError("The t-SNE manifest does not confirm the Barnes-Hut fallback.")
    wrapper_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record["runtime_override"] = {
        "negative_gradient_method": "bh",
        "theta": 0.5,
        "reason": "The default FFT backend crashed on Windows with access violation -1073741819 in kl_divergence_fft.",
        "other_requested_parameters_preserved": [
            "n_components=2", "metric=cosine", "perplexity=29", "initialization=pca",
            "random_state=42", "n_iter=1000", "learning_rate=auto", "early_exaggeration=12",
            "early_exaggeration_iter=250", "n_jobs=1",
        ],
        "wrapper_sha256": wrapper_hash,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
