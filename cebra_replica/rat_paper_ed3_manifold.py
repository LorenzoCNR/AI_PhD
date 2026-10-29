"""Fit ED3 UMAP/t-SNE on full rat recordings in native 2D."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "rat_paper_replication_2023_v1" / "ED3" / "manifold"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rat", required=True, choices=("achilles", "buddy", "cicero", "gatsby"))
    p.add_argument("--method", required=True, choices=("umap", "tsne"))
    p.add_argument(
        "--input-representation",
        choices=("window10_flat", "single_bin"),
        default="window10_flat",
        help=("window10_flat is the Methods-based paper replication: ten 25-ms bins "
              "(250 ms) flattened to one feature vector per segment. single_bin is "
              "a separate 120-neuron-per-bin comparison, not the published setup."),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    source = ROOT / "data" / "rat_hippocampus" / f"{a.rat}.jl"
    data = joblib.load(source)
    spikes = np.asarray(data["spikes"], dtype=np.float32)
    labels = np.asarray(data["position"], dtype=np.float32)
    ids = np.arange(5, len(spikes) - 4, dtype=np.int64)
    n_neurons = int(spikes.shape[1])
    if a.input_representation == "window10_flat":
        input_description = (
            "full recording; 250-ms segments from 10 consecutive 25-ms bins; "
            "225-ms overlap (stride 1 bin); each 10 x n_neurons segment flattened "
            "in CEBRA channel-major order to 10*n_neurons features"
        )
        input_features = 10 * n_neurons
    else:
        input_description = (
            "full recording; one n_neurons spike-count vector per 25-ms bin; "
            "single-bin alternative, not the Methods-described 250-ms setup"
        )
        input_features = n_neurons
    config = dict(figure="ED3", method=a.method, rat=a.rat, seed=a.seed,
                  input_representation=a.input_representation,
                  input=input_description, n_neurons=n_neurons,
                  n_input_features=input_features,
                  window_bins=10 if a.input_representation == "window10_flat" else 1,
                  bin_ms=25,
                  overlap_ms=225 if a.input_representation == "window10_flat" else 0,
                  source_basis=(
                      "Schneider et al. 2023 Methods: rat continuous 25-ms spiking-rate "
                      "segments of 250 ms with 225 ms overlap for CEBRA, UMAP, t-SNE, pi-VAE; "
                      "exact UMAP reshape is not published in the figure-reproduction repository"
                      if a.input_representation == "window10_flat" else
                      "Explicit single-bin alternative; not the Methods-described paper input"
                  ),
                  source_sha256=sha(source), script_sha256=sha(Path(__file__)),
                  params=(dict(n_components=2, metric="cosine", n_neighbors=31, min_dist=0.99)
                          if a.method == "umap" else dict(n_components=2, metric="cosine", perplexity=29)))
    dest = OUT / a.method / a.rat / a.input_representation / f"seed_{a.seed}"
    print(json.dumps(dict(config=config, rows=len(ids), output=str(dest)), indent=2), flush=True)
    if a.dry_run:
        return
    if dest.exists():
        manifest = dest / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text())["config"] == config:
            if (dest / "embedding.npz").exists():
                print("REUSE", dest, flush=True)
                return
        raise RuntimeError(f"Existing output requires inspection: {dest}")
    dest.mkdir(parents=True)
    # Both representations are explicit and isolated in distinct output paths.
    # UMAP/openTSNE take a 2D (samples, features) matrix. The Methods-based
    # representation forms one 10-bin segment per center and flattens the
    # CEBRA-shaped (samples, neurons, time) tensor to a feature vector.
    if a.input_representation == "window10_flat":
        offsets = np.arange(-5, 5, dtype=np.int64)
        x = spikes[ids[:, None] + offsets[None, :]]
        x = x.transpose(0, 2, 1).reshape(len(ids), -1)
    else:
        x = spikes[ids]
    if x.shape != (len(ids), input_features):
        raise AssertionError(f"Unexpected UMAP/t-SNE input shape: {x.shape}")
    started = time.perf_counter()
    if a.method == "umap":
        import umap
        model = umap.UMAP(**config["params"], random_state=a.seed, transform_seed=a.seed,
                          n_jobs=1, verbose=True)
        z = model.fit_transform(x)
        version = umap.__version__
    else:
        import openTSNE
        from openTSNE import TSNE
        model = TSNE(**config["params"], initialization="pca", random_state=a.seed,
                     n_iter=1000, learning_rate="auto", n_jobs=1, verbose=False)
        z = model.fit(x)
        version = openTSNE.__version__
    z = np.asarray(z, dtype=np.float32)
    if z.shape != (len(ids), 2) or not np.isfinite(z).all():
        raise RuntimeError(f"Invalid embedding: {z.shape}")
    path = dest / "embedding.npz"
    np.savez_compressed(path, embedding=z, behavior=labels[ids], source_id=ids)
    (dest / "manifest.json").write_text(json.dumps(dict(config=config, fit_seconds=time.perf_counter()-started,
        embedding_sha256=sha(path), library_version=version), indent=2) + "\n")
    print("COMPLETE", dest, flush=True)


if __name__ == "__main__":
    main()
