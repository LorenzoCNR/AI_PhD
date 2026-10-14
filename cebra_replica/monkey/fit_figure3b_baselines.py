"""Local Figure 3b UMAP/openTSNE fits on the 193 active monkey trials.

These full-sample, unsupervised maps are descriptive only. Fit, embedding export,
and plotting have separate artifacts. The reference HDF5 is never an input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "data/monkey_reaching_preload_smth_40/active_all.jl"
ROOT = PROJECT / "results/monkey_figure3_paper_v1"
SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path: Path, value: dict) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise FileExistsError(f"Immutable artifact differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def configuration(method: str) -> dict:
    common = {"source": "official active_all.jl", "rows": 115800, "channels": 65,
              "split": "full_sample_descriptive", "seed": SEED,
              "input": "65-dimensional Gaussian-smoothed spikes, no newly estimated normalization",
              "output_dimensions": 2}
    if method == "umap":
        return {**common, "method": "UMAP", "metric": "cosine", "n_neighbors": 24,
                "min_dist": 0.0001, "random_state": SEED, "transform_seed": SEED,
                "n_jobs": 1, "parameter_origin": "rat manifold baseline configuration; monkey paper does not specify these values"}
    if method == "opentsne":
        return {**common, "method": "openTSNE", "metric": "cosine", "perplexity": 30,
                "n_iter": 1000, "initialization": "pca", "learning_rate": "auto",
                "random_state": SEED, "n_jobs": 1,
                "parameter_origin": "rat manifold baseline configuration; monkey paper does not specify these values"}
    raise ValueError(method)


def fit(method: str) -> None:
    source_hash = sha256(SOURCE)
    config = configuration(method)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    directory = ROOT / "stage01_fit_baselines" / method / f"seed_{SEED}"
    manifest = directory / "FIT.json"
    if manifest.exists():
        prior = json.loads(manifest.read_text(encoding="utf-8"))
        if prior["source_sha256"] != source_hash or prior["config_sha256"] != config_hash:
            raise ValueError("Existing baseline fit provenance differs")
        if all(sha256(directory / name) == digest for name, digest in prior["artifact_sha256"].items()):
            print(f"REUSE {method}: fit hashes verified", flush=True)
            return
        raise ValueError("Existing baseline fit artifact damaged; no overwrite")
    directory.mkdir(parents=True, exist_ok=True)
    source = joblib.load(SOURCE)
    x = np.asarray(source["spikes"], dtype=np.float32)
    if x.shape != (115800, 65) or not np.isfinite(x).all():
        raise ValueError(f"Source invariant failed: {x.shape}")
    started = time.perf_counter()
    if method == "umap":
        import umap
        model = umap.UMAP(n_components=2, metric="cosine", n_neighbors=24,
                          min_dist=0.0001, random_state=SEED, transform_seed=SEED,
                          n_jobs=1)
        embedding = model.fit_transform(x)
    else:
        from openTSNE import TSNE
        model = TSNE(n_components=2, metric="cosine", perplexity=30,
                     n_iter=1000, initialization="pca", learning_rate="auto",
                     random_state=SEED, n_jobs=1, verbose=False).fit(x)
        embedding = np.asarray(model)
    elapsed = time.perf_counter() - started
    embedding = np.asarray(embedding, dtype=np.float32)
    if embedding.shape != (len(x), 2) or not np.isfinite(embedding).all():
        raise ValueError(f"Embedding invariant failed: {embedding.shape}")
    data_path = directory / "fitted_embedding.npy"
    if data_path.exists():
        raise FileExistsError("Partial fit output exists; inspect before retry")
    np.save(data_path, embedding)
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": source_hash, "script_sha256": sha256(Path(__file__)),
              "config": config, "config_sha256": config_hash, "fit_seconds": elapsed,
              "artifact_sha256": {"fitted_embedding.npy": sha256(data_path)},
              "mapper_checkpoint_saved": False,
              "mapper_checkpoint_note": "openTSNE mapper serialization failed on Windows during attempt 1; the nonparametric full embedding is retained as the reproducible Figure 3b input",
              "reference_hdf5_used": False, "label_supervision_used": False,
              "held_out_claim": False}
    write_json_new(manifest, result)
    print(f"COMPLETE {method}: {elapsed:.1f}s -> {directory}", flush=True)


def export(method: str) -> None:
    fit_directory = ROOT / "stage01_fit_baselines" / method / f"seed_{SEED}"
    fit_manifest = fit_directory / "FIT.json"
    fit_record = json.loads(fit_manifest.read_text(encoding="utf-8"))
    if sha256(SOURCE) != fit_record["source_sha256"]:
        raise ValueError("Source hash changed")
    fitted = fit_directory / "fitted_embedding.npy"
    if sha256(fitted) != fit_record["artifact_sha256"]["fitted_embedding.npy"]:
        raise ValueError("Fitted embedding hash changed")
    directory = ROOT / "stage02_embedding_baselines" / method / f"seed_{SEED}"
    manifest = directory / "EMBEDDING.json"
    if manifest.exists():
        prior = json.loads(manifest.read_text(encoding="utf-8"))
        if prior["fit_manifest_sha256"] != sha256(fit_manifest) or sha256(directory / "all_trials.npz") != prior["artifact_sha256"]:
            raise ValueError("Existing baseline embedding provenance differs")
        print(f"REUSE {method}: embedding hash verified", flush=True)
        return
    source = joblib.load(SOURCE)
    raw = np.load(fitted, allow_pickle=False)
    n_trials = int(source["num_trials"])
    length = int(source["trial_len"])
    if raw.shape != (n_trials * length, 2):
        raise ValueError("Fitted embedding and trial metadata cardinalities differ")
    direction = np.repeat(np.asarray(source["movement_dir"], dtype=np.int16), length)
    trial_id = np.repeat(np.arange(n_trials, dtype=np.int32), length)
    time_id = np.tile(np.arange(length, dtype=np.int16), n_trials)
    norm = np.linalg.norm(raw, axis=1, keepdims=True)
    unit = raw / np.maximum(norm, 1e-12)
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / "all_trials.npz"
    if artifact.exists():
        raise FileExistsError(f"Partial baseline export exists: {artifact}")
    np.savez_compressed(artifact, embedding_raw=raw, embedding_unit=unit,
                        neural=np.asarray(source["spikes"], dtype=np.float32),
                        position_xy=np.asarray(source["pos"], dtype=np.float32),
                        direction=direction, trial_id=trial_id, time_id=time_id,
                        valid_mask=np.ones(len(raw), dtype=bool),
                        split=np.asarray("full_sample_descriptive"))
    write_json_new(manifest, {"created_utc": datetime.now(timezone.utc).isoformat(),
                             "fit_manifest_sha256": sha256(fit_manifest),
                             "source_sha256": fit_record["source_sha256"],
                             "script_sha256": sha256(Path(__file__)),
                             "artifact_sha256": sha256(artifact),
                             "rows": len(raw), "trial_count": n_trials,
                             "embedding_shape": list(raw.shape)})
    print(f"EXPORTED {method}: {artifact}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("fit", "export"))
    parser.add_argument("method", choices=("umap", "opentsne"))
    arguments = parser.parse_args()
    (fit if arguments.phase == "fit" else export)(arguments.method)
