"""Trial-wise rat dimensionality-reduction baselines with immutable outputs.

This runner fits PCA, UMAP, and openTSNE on train-only 250 ms spike-count
windows and applies each learned map to validation/test samples. It deliberately
does not generate figures or inspect held-out labels during representation fit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "rat_hippocampus"
OUT_ROOT = ROOT / "results" / "rat_manifold_baselines_v1"
RATS = ("achilles", "buddy", "cicero", "gatsby")
FOLDS = (0, 1, 2)
SEEDS = tuple(range(42, 52))
METHODS = ("pca", "umap", "opentsne")
WINDOW_OFFSETS = np.arange(-5, 5, dtype=np.int64)  # 10 bins = 250 ms
SPLITTER_SOURCE = ROOT / "CEBRA" / "cebra" / "datasets" / "hippocampus.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def file_id(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def trial_bounds(position: np.ndarray) -> np.ndarray:
    """Replicate SingleRatTrialSplitDataset's documented trial segmentation."""
    direction_change = np.flatnonzero(position[1:, 1] != position[:-1, 1])
    bounds = np.append(np.insert(direction_change[1::2], 0, 0), len(position)).astype(np.int64)
    if bounds[0] != 0 or bounds[-1] != len(position) or np.any(np.diff(bounds) <= 0):
        raise AssertionError("Invalid trial boundaries")
    return bounds


def trial_split(bounds: np.ndarray, fold: int) -> dict[str, np.ndarray]:
    """Replicate the official non-shuffled, nested 3-fold rat split exactly."""
    if fold not in FOLDS:
        raise ValueError(f"fold must be in {FOLDS}")
    outer_folds = np.array_split(np.arange(len(bounds) - 1), 3)
    train_trials: list[int] = []
    valid_trials: list[int] = []
    test_trials: list[int] = []
    splitter = KFold(n_splits=3, shuffle=False)
    for outer in outer_folds:
        train_index, valid_test_index = list(splitter.split(outer))[fold]
        test_index, valid_index = np.array_split(valid_test_index, 2)
        train_trials.extend(np.asarray(outer)[train_index].tolist())
        valid_trials.extend(np.asarray(outer)[valid_index].tolist())
        test_trials.extend(np.asarray(outer)[test_index].tolist())
    return {
        "train": np.asarray(train_trials, dtype=np.int64),
        "valid": np.asarray(valid_trials, dtype=np.int64),
        "test": np.asarray(test_trials, dtype=np.int64),
    }


def row_ids(bounds: np.ndarray, trial_ids: np.ndarray) -> np.ndarray:
    return np.concatenate([np.arange(bounds[t], bounds[t + 1], dtype=np.int64)
                           for t in trial_ids])


def verify_official_split(rat: str, fold: int, splits: dict[str, np.ndarray]) -> None:
    """Check this splitter's exact row order against immutable CEBRA embeddings."""
    reference = (ROOT / "results" / "rat_trial_benchmark_v1" / rat / f"fold_{fold}"
                 / "position_direction" / "seed_42")
    for split, ids in splits.items():
        path = reference / f"{split}_embedding.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Cannot verify split against existing CEBRA artifact: {path}")
        with np.load(path, allow_pickle=False) as artifact:
            observed = artifact["source_id"]
        if not np.array_equal(ids, observed):
            raise AssertionError(f"Official split mismatch for {rat}/fold_{fold}/{split}")


def build_split_arrays(spikes: np.ndarray, position: np.ndarray,
                       bounds: np.ndarray, ids: np.ndarray) -> dict[str, np.ndarray]:
    trial_id = np.searchsorted(bounds[1:], ids, side="right").astype(np.int64)
    local_time = (ids - bounds[trial_id]).astype(np.int64)
    trial_length = (bounds[trial_id + 1] - bounds[trial_id]).astype(np.int64)
    valid = (local_time >= 5) & (local_time < trial_length - 4)
    valid_ids = ids[valid]
    valid_trial = trial_id[valid]
    valid_time = local_time[valid]
    window_rows = valid_ids[:, None] + WINDOW_OFFSETS[None, :]
    window_trials = np.searchsorted(bounds[1:], window_rows, side="right")
    if not np.all(window_trials == valid_trial[:, None]):
        raise AssertionError("A window crossed a trial boundary")
    expected_rows = valid_ids[:, None] + WINDOW_OFFSETS[None, :]
    if not np.array_equal(window_rows, expected_rows):
        raise AssertionError("Window row support mismatch")
    windows = spikes[window_rows].transpose(0, 2, 1).reshape(len(valid_ids), -1)
    windows = np.asarray(windows, dtype=np.float32)
    if not np.isfinite(windows).all():
        raise AssertionError("Non-finite neural window")
    behavior = np.asarray(position[ids], dtype=np.float32)
    return {
        "windows": windows,
        "source_id": ids.astype(np.int64),
        "trial_id": trial_id,
        "time_id": local_time,
        "behavior": behavior,
        "valid_mask": valid,
        "valid_source_id": valid_ids.astype(np.int64),
        "valid_trial_id": valid_trial,
        "valid_time_id": valid_time,
    }


def config_for(method: str, seed: int | None) -> dict[str, Any]:
    if method == "pca":
        return {"n_components": 2, "svd_solver": "covariance_eigh", "seed": None,
                "input": "raw 10-bin x 120-neuron window, flattened neuron-major"}
    if method == "umap":
        return {"n_components": 2, "metric": "cosine", "n_neighbors": 24,
                "min_dist": 0.0001, "random_state": seed, "transform_seed": seed,
                "n_jobs": 1, "input": "raw 10-bin x 120-neuron window, flattened neuron-major"}
    if method == "opentsne":
        return {"n_components": 2, "metric": "cosine", "perplexity": 30,
                "n_iter": 1000, "initialization": "pca", "learning_rate": "auto",
                "random_state": seed, "n_jobs": 1,
                "out_of_sample": "openTSNE transform fitted from training embedding",
                "input": "raw 10-bin x 120-neuron window, flattened neuron-major"}
    raise ValueError(method)


def require_method(method: str) -> None:
    module = {"pca": None, "umap": "umap", "opentsne": "openTSNE"}[method]
    if module is not None and importlib.util.find_spec(module) is None:
        raise RuntimeError(f"Missing dependency {module}; use the isolated rat_manifold environment")


def fit_map(method: str, seed: int | None, train_x: np.ndarray):
    if method == "pca":
        from sklearn.decomposition import PCA
        model = PCA(n_components=2, svd_solver="covariance_eigh")
        train_embedding = model.fit_transform(train_x)
    elif method == "umap":
        import umap
        model = umap.UMAP(n_components=2, metric="cosine", n_neighbors=24,
                          min_dist=0.0001, random_state=seed, transform_seed=seed,
                          n_jobs=1)
        train_embedding = model.fit_transform(train_x)
    elif method == "opentsne":
        from openTSNE import TSNE
        model = TSNE(n_components=2, metric="cosine", perplexity=30,
                     n_iter=1000, initialization="pca", learning_rate="auto",
                     random_state=seed, n_jobs=1, verbose=False).fit(train_x)
        train_embedding = np.asarray(model)
    else:
        raise ValueError(method)
    return model, np.asarray(train_embedding, dtype=np.float32)


def transform_map(method: str, model, x: np.ndarray) -> np.ndarray:
    return np.asarray(model.transform(x), dtype=np.float32)


def materialize_embedding(split: dict[str, np.ndarray], valid_embedding: np.ndarray,
                           split_name: str, embedding_parent: str) -> dict[str, np.ndarray]:
    mask = split["valid_mask"]
    if valid_embedding.shape != (int(mask.sum()), 2):
        raise AssertionError(f"Embedding shape does not match valid support: {valid_embedding.shape}")
    full = np.full((len(mask), 2), np.nan, dtype=np.float32)
    full[mask] = valid_embedding
    unit = np.full_like(full, np.nan)
    unit[mask] = normalize(valid_embedding, norm="l2", axis=1)
    if not np.isfinite(full[mask]).all() or not np.isfinite(unit[mask]).all():
        raise AssertionError("Non-finite embedding on valid support")
    return {
        "embedding_raw": full,
        "embedding_unit": unit,
        "behavior": split["behavior"],
        "source_id": split["source_id"],
        "trial_id": split["trial_id"],
        "time_id": split["time_id"],
        "valid_mask": mask,
        "split": np.asarray(split_name),
        "embedding_parent": np.asarray(embedding_parent),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def expected_artifact(path: Path, result: dict[str, Any]) -> bool:
    try:
        if json.loads(path.read_text(encoding="utf-8")) != result:
            return False
        for split, entry in result["embeddings"].items():
            artifact = path.parent / f"{split}_embedding.npz"
            if sha256_file(artifact) != entry["sha256"]:
                return False
        if not result.get("model_saved", True):
            return True
        model = path.parent / "model.joblib"
        return model.exists() and sha256_file(model) == result["model_sha256"]
    except (OSError, ValueError, KeyError):
        return False


def metadata_matches(existing: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if key == "software" and "software" in existing:
            # Older pilots recorded only Python and NumPy; validate those values
            # and permit missing package fields without refitting valid embeddings.
            if any(existing["software"].get(pkg) != ver
                   for pkg, ver in value.items() if pkg in existing["software"]):
                return False
        elif key == "software" and key not in existing:
            continue
        elif existing.get(key) != value:
            return False
    return True


def software_versions() -> dict[str, str]:
    packages = {"python": os.sys.version.split()[0], "numpy": np.__version__}
    for distribution in ("scipy", "scikit-learn", "joblib", "umap-learn", "openTSNE"):
        try:
            packages[distribution] = version(distribution)
        except Exception:
            pass
    return packages


def recover_opentsne_embeddings(destination: Path, expected: dict[str, Any],
                                split_arrays: dict[str, dict[str, np.ndarray]]) -> dict[str, Any] | None:
    """Recover finished frozen embeddings after Windows blocks openTSNE pickling."""
    failure_path = destination / "failure.json"
    if expected["method"] != "opentsne" or not failure_path.is_file():
        return None
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    if any(failure.get(key) != value for key, value in expected.items() if key != "software"):
        raise RuntimeError(f"Partial openTSNE artifacts do not match current provenance: {destination}")
    embeddings: dict[str, dict[str, Any]] = {}
    for name, split in split_arrays.items():
        artifact_path = destination / f"{name}_embedding.npz"
        if not artifact_path.is_file():
            return None
        with np.load(artifact_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        n = len(split["source_id"])
        valid = split["valid_mask"]
        checks = {"source_id": split["source_id"], "trial_id": split["trial_id"],
                  "time_id": split["time_id"], "valid_mask": valid,
                  "behavior": split["behavior"]}
        if any(key not in arrays or not np.array_equal(arrays[key], value)
               for key, value in checks.items()):
            raise AssertionError(f"Partial openTSNE metadata mismatch: {artifact_path}")
        raw = arrays.get("embedding_raw")
        unit = arrays.get("embedding_unit")
        if (raw is None or unit is None or raw.shape != (n, 2) or unit.shape != (n, 2)
                or not np.isfinite(raw[valid]).all() or not np.isfinite(unit[valid]).all()
                or not np.isnan(raw[~valid]).all() or not np.isnan(unit[~valid]).all()):
            raise AssertionError(f"Partial openTSNE embedding shape/support mismatch: {artifact_path}")
        embeddings[name] = {"sha256": sha256_file(artifact_path),
                            "shape": list(raw.shape), "valid_rows": int(valid.sum())}
    model_path = destination / "model.joblib"
    recovered = {**expected,
                 "created_utc": failure.get("created_utc"),
                 "recovered_utc": datetime.now(timezone.utc).isoformat(),
                 "fit_seconds": None,
                 "model_saved": False,
                 "model_save_error": failure.get("error"),
                 "model_attempt_sha256": sha256_file(model_path) if model_path.exists() else None,
                 "fit_script_hash_at_fit": None,
                 "recovery_script_sha256": sha256_file(Path(__file__)),
                 "failure_record": "failure.json",
                 "failure_record_sha256": sha256_file(failure_path),
                 "embeddings": embeddings,
                 "status": "complete_embeddings_only",
                 "recovery": "fit and embeddings validated; no refit; incomplete model pickle retained"}
    atomic_json(destination / "result.json", recovered)
    print(f"RECOVERED_EMBEDDINGS_ONLY {expected['run_id']} (no refit)", flush=True)
    return recovered


def run_fit(method: str, rat: str, fold: int, seed: int | None, dry_run: bool = False) -> dict[str, Any]:
    require_method(method)
    if rat not in RATS or fold not in FOLDS:
        raise ValueError("Unknown rat/fold")
    data_path = DATA_DIR / f"{rat}.jl"
    print(f"LOAD {file_id(data_path)}", flush=True)
    raw = joblib.load(data_path)
    spikes = np.asarray(raw["spikes"], dtype=np.float32)
    position = np.asarray(raw["position"], dtype=np.float32)
    bounds = trial_bounds(position)
    trial_sets = trial_split(bounds, fold)
    split_ids = {name: row_ids(bounds, trials) for name, trials in trial_sets.items()}
    verify_official_split(rat, fold, split_ids)
    split_arrays = {name: build_split_arrays(spikes, position, bounds, ids)
                    for name, ids in split_ids.items()}
    # `windows` already contains only the valid center rows; `valid_mask` is
    # retained at original split cardinality for row-aligned embedding export.
    train_x = split_arrays["train"]["windows"]
    if len(train_x) <= 3 * 30:
        raise AssertionError("Insufficient valid training windows for requested reducer")
    seed_value = None if method == "pca" else int(seed)
    config = config_for(method, seed_value)
    split_hash = hashlib.sha256("".join(sha256_array(split_ids[k]) for k in ("train", "valid", "test")).encode()).hexdigest()
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    run_id = f"{method}-{rat}-fold{fold}-seed{seed_value if seed_value is not None else 'deterministic'}-{config_hash[:12]}"
    destination = OUT_ROOT / method / rat / f"fold_{fold}" / f"seed_{seed_value if seed_value is not None else 'deterministic'}"
    manifest_path = destination / "result.json"
    expected = {"run_id": run_id, "method": method, "rat": rat, "fold": fold,
                "seed": seed_value, "config": config, "config_sha256": config_hash,
                "data_path": file_id(data_path), "data_sha256": sha256_file(data_path),
                "splitter_source": file_id(SPLITTER_SOURCE),
                "splitter_source_sha256": sha256_file(SPLITTER_SOURCE),
                "split_sha256": split_hash,
                "rows": {name: int(len(ids)) for name, ids in split_ids.items()},
                "valid_window_rows": {name: int(split_arrays[name]["valid_mask"].sum())
                                      for name in split_ids},
                "input_shape": [int(train_x.shape[1])],
                "window_bins": 10, "sampling_ms": 25, "window_duration_ms": 250,
                "trial_boundary_policy": "centered offsets [-5,+4], invalid boundary centers excluded",
                "representation_fit": "training valid windows only; behavior labels are not used",
                "valid_test_transform": "transform only; no valid/test behavior labels used",
                "embedding_dim": 2,
                "software": software_versions()}
    print(f"PREPARED {run_id} | train_valid={len(train_x)} | feature_dim={train_x.shape[1]}", flush=True)
    if dry_run:
        print(json.dumps(expected, indent=2), flush=True)
        return {"status": "dry_run", **expected}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if expected_artifact(manifest_path, existing) and metadata_matches(existing, expected):
            print(f"REUSE {run_id}", flush=True)
            return {"status": "reused", "artifact_status": existing.get("status"), **expected}
        raise RuntimeError(f"Existing artifact has mismatch or failed hash: {manifest_path}")
    if destination.exists() and any(destination.iterdir()):
        recovered = recover_opentsne_embeddings(destination, expected, split_arrays)
        if recovered is not None:
            return recovered
        raise RuntimeError(f"Refusing to overwrite partial artifact: {destination}")

    # An empty directory can be left by a dry-run or a pre-fit interruption.
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        print(f"FIT_START {run_id}", flush=True)
        model, train_latent = fit_map(method, seed_value, train_x)
        print(f"FIT_DONE {run_id} | shape={train_latent.shape}", flush=True)
        train_valid_mask = split_arrays["train"]["valid_mask"]
        embeddings: dict[str, dict[str, str]] = {}
        for name in ("train", "valid", "test"):
            split = split_arrays[name]
            mask = split["valid_mask"]
            if name == "train":
                latent = train_latent
            else:
                latent = transform_map(method, model, split["windows"])
            parent_id = "model.joblib" if method != "opentsne" else "frozen_train_embedding"
            arrays = materialize_embedding(split, latent, name, parent_id)
            artifact_path = destination / f"{name}_embedding.npz"
            np.savez_compressed(artifact_path, **arrays)
            print(f"EMBED_SAVED {run_id} | {name} | n={len(mask)} valid={int(mask.sum())}", flush=True)
            embeddings[name] = {"sha256": sha256_file(artifact_path),
                                "shape": list(arrays["embedding_raw"].shape),
                                "valid_rows": int(mask.sum())}
        model_path = destination / "model.joblib"
        model_saved = method != "opentsne"
        if model_saved:
            joblib.dump(model, model_path, compress=3)
        result = {**expected, "created_utc": datetime.now(timezone.utc).isoformat(),
                  "fit_seconds": time.perf_counter() - started,
                  "fit_script_sha256": sha256_file(Path(__file__)),
                  "model_saved": model_saved,
                  "model_sha256": sha256_file(model_path) if model_saved else None,
                  "embeddings": embeddings,
                  "status": "complete" if model_saved else "complete_embeddings_only"}
        atomic_json(manifest_path, result)
        return {"status": "complete", **result}
    except Exception as error:
        failure_path = destination / "failure.json"
        atomic_json(failure_path, {**expected, "created_utc": datetime.now(timezone.utc).isoformat(),
                                   "error_type": type(error).__name__, "error": str(error)})
        raise


def specs(methods: tuple[str, ...]) -> list[tuple[str, str, int, int | None]]:
    requested = []
    for method in methods:
        for rat in RATS:
            for fold in FOLDS:
                if method == "pca":
                    requested.append((method, rat, fold, None))
                else:
                    requested.extend((method, rat, fold, seed) for seed in SEEDS)
    return requested


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--rat", choices=RATS)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--campaign", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.campaign:
        if any(value is not None for value in (args.method, args.rat, args.fold, args.seed)):
            parser.error("--campaign cannot be combined with one-fit selectors")
        requested = specs(METHODS)
    else:
        if not all(value is not None for value in (args.method, args.rat, args.fold)):
            parser.error("One-fit mode requires --method, --rat, and --fold")
        seed = None if args.method == "pca" else args.seed
        if args.method != "pca" and seed is None:
            parser.error("UMAP/openTSNE require --seed")
        requested = [(args.method, args.rat, args.fold, seed)]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    state_path = OUT_ROOT / "campaign_state.json" if args.campaign else None
    state = {"created_utc": datetime.now(timezone.utc).isoformat(),
             "status": "running", "planned": len(requested), "completed": [], "reused": [], "failed": []}
    if state_path is not None and state_path.exists() and not args.dry_run:
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("protocol") == "rat-manifold-baselines-v1":
            if previous.get("planned") == len(requested):
                state = previous
                state["planned"] = len(requested)
                state["status"] = "running"
                state["previous_state_sha256"] = sha256_file(state_path)
            else:
                state["historical_attempts"] = previous.get("historical_attempts", []) + [{
                    "prior_state_sha256": sha256_file(state_path),
                    "planned": previous.get("planned"),
                    "completed": previous.get("completed", []),
                    "failed": previous.get("failed", []),
                    "finished_utc": previous.get("finished_utc"),
                }]
    state["protocol"] = "rat-manifold-baselines-v1"
    for i, (method, rat, fold, seed) in enumerate(requested, start=1):
        print(f"RUN {i}/{len(requested)} | {method} | {rat} | fold={fold} | seed={seed}", flush=True)
        try:
            result = run_fit(method, rat, fold, seed, dry_run=args.dry_run)
            if result["status"] in ("complete", "complete_embeddings_only", "reused"):
                if result["run_id"] not in state["completed"]:
                    state["completed"].append(result["run_id"])
                if result["status"] == "reused" and result["run_id"] not in state["reused"]:
                    state["reused"].append(result["run_id"])
                if result["status"] == "complete_embeddings_only":
                    recoveries = state.setdefault("recoveries", [])
                    if result["run_id"] not in [item["run_id"] for item in recoveries]:
                        recoveries.append({"run_id": result["run_id"],
                                           "model_saved": False,
                                           "failure_record": "preserved in run directory"})
                if result.get("artifact_status") == "complete_embeddings_only":
                    recoveries = state.setdefault("recoveries", [])
                    if result["run_id"] not in [item["run_id"] for item in recoveries]:
                        recoveries.append({"run_id": result["run_id"],
                                           "reused": True, "model_saved": False,
                                           "failure_record": "preserved in run directory"})
        except Exception as error:
            state["failed"].append({"method": method, "rat": rat, "fold": fold,
                                    "seed": seed, "error": f"{type(error).__name__}: {error}"})
            print(f"FAILED {method}/{rat}/fold_{fold}/seed_{seed}: {error}", flush=True)
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        if not args.dry_run and state_path is not None:
            atomic_json(state_path, state)
    state["status"] = "dry_run_complete" if args.dry_run else (
        "complete" if len(set(state["completed"])) == len(requested) else "incomplete_or_failed")
    state["finished_utc"] = datetime.now(timezone.utc).isoformat()
    if not args.dry_run and state_path is not None:
        atomic_json(state_path, state)
    print(json.dumps({"planned": len(requested), "completed": len(set(state["completed"])),
                      "reused": len(set(state["reused"])), "failed": len(state["failed"]),
                      "state": state["status"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
