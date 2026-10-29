"""Resumable ED3 UMAP/t-SNE fits on full rat recordings (paper-native 2D)."""
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "rat_paper_replication_2023_v1" / "ED3" / "manifold"
RATS = ("achilles", "buddy", "cicero", "gatsby")
COMPATIBLE_KEYS = (
    "figure", "method", "rat", "seed", "input_representation", "input",
    "n_neurons", "n_input_features", "window_bins", "bin_ms", "overlap_ms",
    "source_basis", "source_sha256", "params",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, record: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def compatible(old: dict, new: dict) -> bool:
    return all(old.get(key) == new.get(key) for key in COMPATIBLE_KEYS)


def pid_exists(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return ctypes.get_last_error() == 5
        kernel.CloseHandle(handle)
        return True
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def verify_embedding(path: Path, n_rows: int) -> bool:
    try:
        with np.load(path, allow_pickle=False) as stored:
            embedding = stored["embedding"]
            source_id = stored["source_id"]
            behavior = stored["behavior"]
        return (embedding.shape == (n_rows, 2) and behavior.shape == (n_rows, 3)
                and source_id.shape == (n_rows,) and np.isfinite(embedding).all()
                and np.isfinite(behavior).all())
    except (OSError, ValueError, KeyError):
        return False


def recover_manifest(path: Path, manifest: Path, config: dict, n_rows: int) -> bool:
    """Recover the sidecar if the atomic embedding commit succeeded first."""
    if not path.is_file() or not verify_embedding(path, n_rows):
        return False
    try:
        with np.load(path, allow_pickle=False) as stored:
            stored_config = json.loads(str(stored["run_config_json"].item()))
            resolved = json.loads(str(stored["resolved_params_json"].item()))
            version = str(stored["library_version"].item())
            fit_seconds = float(stored["fit_seconds"].item())
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if not compatible(stored_config, config):
        return False
    atomic_json(manifest, dict(status="complete", config=stored_config,
                               resolved_parameters=resolved, library_version=version,
                               fit_seconds=fit_seconds, embedding_sha256=sha(path),
                               recovered_manifest=True))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rat", required=True, choices=RATS)
    parser.add_argument("--method", required=True, choices=("umap", "tsne"))
    parser.add_argument("--input-representation", choices=("window10_flat", "single_bin"),
                        default="window10_flat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    a = parser.parse_args()

    source = ROOT / "data" / "rat_hippocampus" / f"{a.rat}.jl"
    raw = joblib.load(source)
    spikes = np.asarray(raw["spikes"], dtype=np.float32)
    labels = np.asarray(raw["position"], dtype=np.float32)
    if spikes.ndim != 2 or labels.shape != (len(spikes), 3):
        raise ValueError(f"Unexpected source shapes for {a.rat}: spikes={spikes.shape}, labels={labels.shape}")
    ids = np.arange(5, len(spikes) - 4, dtype=np.int64) if a.input_representation == "window10_flat" else np.arange(len(spikes), dtype=np.int64)
    n_neurons = int(spikes.shape[1])
    window_bins = 10 if a.input_representation == "window10_flat" else 1
    overlap_ms = 225 if window_bins == 10 else 0
    input_description = (
        "full recording; 250-ms segments from 10 consecutive 25-ms bins; "
        "225-ms overlap (stride 1 bin); each 10 x n_neurons segment flattened "
        "in CEBRA channel-major order to 10*n_neurons features"
        if window_bins == 10 else
        "full recording; one n_neurons spike-count vector per 25-ms bin; "
        "single-bin alternative, not the Methods-described 250-ms setup"
    )
    paper_basis = (
        "Schneider et al. 2023 Methods: rat continuous 25-ms spiking-rate segments of 250 ms "
        "with 225 ms overlap for CEBRA, UMAP, t-SNE, pi-VAE; exact UMAP reshape is not published "
        "in the figure-reproduction repository"
        if window_bins == 10 else
        "Explicit single-bin alternative; not the Methods-described paper input"
    )

    if a.method == "umap":
        params = dict(n_components=2, metric="cosine", n_neighbors=31, min_dist=0.99)
        fixed = dict(random_state=a.seed, transform_seed=a.seed, n_jobs=1, verbose=True)
    else:
        params = dict(n_components=2, metric="cosine", perplexity=29)
        fixed = dict(initialization="pca", random_state=a.seed, n_iter=1000,
                     learning_rate="auto", early_exaggeration=12,
                     early_exaggeration_iter=250, n_jobs=1, verbose=False)

    config = dict(
        figure="ED3", method=a.method, rat=a.rat, seed=a.seed,
        input_representation=a.input_representation, input=input_description,
        n_neurons=n_neurons, n_input_features=window_bins * n_neurons,
        window_bins=window_bins, bin_ms=25, overlap_ms=overlap_ms,
        source_basis=paper_basis, source_sha256=sha(source),
        script_sha256=sha(Path(__file__)), params=params,
    )
    dest = OUT / a.method / a.rat / a.input_representation / f"seed_{a.seed}"
    embedding_path = dest / "embedding.npz"
    manifest_path = dest / "manifest.json"
    print(json.dumps(dict(config=config, fixed_parameters=fixed, rows=len(ids),
                          output=str(dest)), indent=2), flush=True)
    if a.dry_run:
        return

    if dest.exists():
        if manifest_path.is_file():
            record = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not compatible(record.get("config", {}), config):
                raise RuntimeError(f"Existing result has incompatible configuration: {dest}")
            if (not embedding_path.is_file()
                    or record.get("embedding_sha256") != sha(embedding_path)
                    or not verify_embedding(embedding_path, len(ids))):
                raise RuntimeError(f"Existing result failed hash/shape validation: {dest}")
            print("REUSE", dest, flush=True)
            return
        if recover_manifest(embedding_path, manifest_path, config, len(ids)):
            print("RECOVERED_MANIFEST", dest, flush=True)
            return
        running_path = dest / "RUNNING.json"
        if running_path.is_file():
            state = json.loads(running_path.read_text(encoding="utf-8"))
            if pid_exists(int(state.get("pid", -1))):
                raise RuntimeError(f"Fit is already running (pid {state.get('pid')}): {dest}")
        allowed = {"RUNNING.json", "embedding.partial.npz"}
        unexpected = [item.name for item in dest.iterdir()
                      if item.name not in allowed and not item.name.startswith("failure_")]
        if unexpected:
            raise RuntimeError(f"Incomplete destination contains unrecognized files; inspect manually: {dest}: {unexpected}")
        partial = dest / "embedding.partial.npz"
        if partial.exists():
            partial.unlink()
    else:
        dest.mkdir(parents=True)

    running_path = dest / "RUNNING.json"
    atomic_json(running_path, dict(status="running", pid=os.getpid(),
                                   started_utc=datetime.now(timezone.utc).isoformat(), config=config))

    try:
        if window_bins == 10:
            offsets = np.arange(-5, 5, dtype=np.int64)
            x = spikes[ids[:, None] + offsets[None, :]]
            x = x.transpose(0, 2, 1).reshape(len(ids), -1)
        else:
            x = spikes[ids]
        if x.shape != (len(ids), window_bins * n_neurons):
            raise AssertionError(f"Unexpected UMAP/t-SNE input shape: {x.shape}")

        started = time.perf_counter()
        if a.method == "umap":
            import umap
            model = umap.UMAP(**params, **fixed)
            z = model.fit_transform(x)
            version = umap.__version__
        else:
            import openTSNE
            from openTSNE import TSNE
            model = TSNE(**params, **fixed)
            z = model.fit(x)
            version = openTSNE.__version__
        fit_seconds = time.perf_counter() - started
        z = np.asarray(z, dtype=np.float32)
        if z.shape != (len(ids), 2) or not np.isfinite(z).all():
            raise RuntimeError(f"Invalid embedding: {z.shape}")

        resolved = dict(model.get_params())
        if a.method == "tsne":
            resolved["effective_early_exaggeration"] = model.early_exaggeration
        elif getattr(model, "n_epochs", None) is not None:
            resolved["effective_n_epochs"] = model.n_epochs

        partial = dest / "embedding.partial.npz"
        np.savez_compressed(
            partial, embedding=z, behavior=labels[ids], source_id=ids,
            run_config_json=np.asarray(json.dumps(config, sort_keys=True)),
            resolved_params_json=np.asarray(json.dumps(resolved, sort_keys=True, default=str)),
            library_version=np.asarray(version), fit_seconds=np.asarray(fit_seconds),
        )
        final_hash = sha(partial)
        partial.replace(embedding_path)
        record = dict(status="complete", config=config, resolved_parameters=resolved,
                      fit_seconds=fit_seconds, embedding_sha256=final_hash,
                      library_version=version)
        atomic_json(manifest_path, record)
        atomic_json(running_path, dict(status="complete", pid=os.getpid(),
                                       completed_utc=datetime.now(timezone.utc).isoformat(),
                                       manifest=str(manifest_path), embedding_sha256=final_hash))
        print("COMPLETE", dest, flush=True)
    except Exception as error:
        failure = dict(status="failed", pid=os.getpid(),
                       failed_utc=datetime.now(timezone.utc).isoformat(),
                       error_type=type(error).__name__, error=str(error), config=config)
        atomic_json(dest / f"failure_{int(time.time())}.json", failure)
        atomic_json(running_path, failure)
        raise


if __name__ == "__main__":
    main()
