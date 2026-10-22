"""Validation-selected rat probes from frozen embeddings; never fit encoders."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Recovered without alteration: Neuro_Bridge/tools/final_thesis_core_metrics.py:70.
RIDGE_ALPHA_GRID = (0.1, 1.0, 10.0)
KNN_NEIGHBORS = 36


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _LazyArchive(Mapping):
    """Read only requested NPZ fields, keeping test outcomes unopened at selection."""
    def __init__(self, path: Path):
        self.path = path

    def __getitem__(self, key):
        with np.load(self.path, allow_pickle=False) as archive:
            return archive[key].copy()

    def __iter__(self):
        with np.load(self.path, allow_pickle=False) as archive:
            return iter(tuple(archive.files))

    def __len__(self):
        with np.load(self.path, allow_pickle=False) as archive:
            return len(archive.files)

    def __contains__(self, key):
        with np.load(self.path, allow_pickle=False) as archive:
            return key in archive.files


def _metadata(data: Mapping) -> np.ndarray:
    ids = np.asarray(data["source_id"])
    mask = np.asarray(data["valid_mask"])
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError("source_id must be a unique one-dimensional row identifier")
    if mask.shape != ids.shape or mask.dtype != np.bool_:
        raise ValueError("Explicit Boolean valid_mask must match source_id cardinality")
    for field in ("trial_id", "time_id"):
        if np.asarray(data[field]).shape != ids.shape:
            raise ValueError(f"{field} cardinality differs from source_id")
    if not mask.any():
        raise ValueError("No valid evaluation rows")
    return ids


def _rows(data: Mapping, representation: str) -> tuple[np.ndarray, np.ndarray]:
    raw_key = "embedding_raw" if "embedding_raw" in data else "embedding"
    raw = np.asarray(data[raw_key])
    mask = np.asarray(data["valid_mask"])
    behavior = np.asarray(data["behavior"])
    n = len(mask)
    if raw.ndim != 2 or len(raw) != n or raw.shape[1] < 1:
        raise ValueError("Embedding must be N x D with metadata cardinality")
    if behavior.ndim != 2 or len(behavior) != n or behavior.shape[1] < 2:
        raise ValueError("behavior must be N x >=2: position in meters, right indicator")
    x = raw[mask].astype(np.float64, copy=False)
    y = behavior[mask].astype(np.float64, copy=False)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite embedding or behavior on valid support")
    if not np.isin(y[:, 1], (0, 1)).all():
        raise ValueError("Direction must be the binary right indicator")
    if representation == "embedding_unit":
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise ValueError("Unit normalization is undefined for zero embeddings")
        expected = x / norms
        if "embedding_unit" in data:
            stored = np.asarray(data["embedding_unit"])
            if stored.shape != raw.shape:
                raise ValueError("Stored unit embedding shape differs from raw")
            x = stored[mask].astype(np.float64, copy=False)
            if not np.isfinite(x).all() or not np.allclose(x, expected, rtol=1e-5, atol=1e-6):
                raise ValueError("Stored unit embedding differs from row-normalized raw")
        else:
            x = expected
    return x, y


def score_arrays(arrays: Mapping, *, ridge_alphas=RIDGE_ALPHA_GRID) -> list[dict]:
    """Fit probes on train; choose Ridge alpha on valid; then evaluate test.

    Explicit valid_mask identifies pre-established common trial-interior support.
    This function never estimates support, changes embeddings or fits encoders.
    Training, validation and test outcomes are separate; test outcomes are accessed
    only after alpha selection for both raw and unit representations is complete.
    """
    alphas = tuple(float(a) for a in ridge_alphas)
    if not alphas or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("Ridge alpha grid must contain finite positive values")
    ids = {split: _metadata(arrays[split]) for split in ("train", "valid", "test")}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if np.intersect1d(ids[left], ids[right]).size:
            raise ValueError("Train/validation/test source rows overlap")
        # A trial cannot belong to multiple splits in this complete-trial design.
        if np.intersect1d(arrays[left]["trial_id"], arrays[right]["trial_id"]).size:
            raise ValueError("Train/validation/test trial identities overlap")

    fitted = {}
    for representation in ("embedding_raw", "embedding_unit"):
        x_train, y_train = _rows(arrays["train"], representation)
        x_valid, y_valid = _rows(arrays["valid"], representation)
        if len(x_train) < KNN_NEIGHBORS:
            raise ValueError("Frozen kNN36 requires at least 36 valid training rows")
        if len(x_valid) < 2:
            raise ValueError("Validation R2 requires at least two valid rows")
        position = KNeighborsRegressor(n_neighbors=KNN_NEIGHBORS, metric="cosine")
        direction = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, metric="cosine")
        position.fit(x_train, y_train[:, 0])
        direction.fit(x_train, y_train[:, 1].astype(np.int64))
        best_score, best_alpha, best_ridge = -np.inf, None, None
        candidates = []
        for alpha in alphas:
            probe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            probe.fit(x_train, y_train[:, 0])
            score = float(r2_score(y_valid[:, 0], probe.predict(x_valid)))
            if not np.isfinite(score):
                raise ValueError("Nonfinite validation R2 prevents Ridge selection")
            candidates.append({"alpha": alpha, "validation_position_r2": score})
            if score > best_score:  # Exact ties retain first entry in fixed grid order.
                best_score, best_alpha, best_ridge = score, alpha, probe
        fitted[representation] = (position, direction, best_ridge, best_alpha, candidates)

    records = []
    for representation, (position, direction, ridge, alpha, candidates) in fitted.items():
        for split in ("valid", "test"):
            x, y = _rows(arrays[split], representation)
            if len(x) < 2:
                raise ValueError("Position R2 requires at least two evaluation rows")
            d_pred = direction.predict(x)
            for method, p_pred in (("knn", position.predict(x)), ("ridge", ridge.predict(x))):
                record = {
                    "representation": representation, "split": split, "decoder": method, "method": method,
                    "n_samples": len(x),
                    "n_trials": len(np.unique(np.asarray(arrays[split]["trial_id"])[arrays[split]["valid_mask"]])),
                    "position_median_absolute_error_cm": float(np.median(np.abs(p_pred-y[:, 0]))*100),
                    "position_r2": float(r2_score(y[:, 0], p_pred)),
                    "direction_balanced_accuracy": float(balanced_accuracy_score(y[:, 1], d_pred)) if method == "knn" else None,
                    "direction_accuracy": float(accuracy_score(y[:, 1], d_pred)) if method == "knn" else None,
                    "selected_alpha": alpha if method == "ridge" else None,
                    "ridge_validation_candidates": candidates if method == "ridge" else None,
                    "knn_neighbors": KNN_NEIGHBORS if method == "knn" else None,
                    "scope": "validation" if split == "valid" else "trial_held_out_test",
                }
                records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing output: {output}")
    if output == run or run in output.parents:
        raise RuntimeError("Downstream output must be separate from frozen run")
    paths = {split: run / f"{split}_embedding.npz" for split in ("train", "valid", "test")}
    parents = {split: {"path": str(path), "sha256": sha256(path)} for split, path in paths.items()}
    manifest_path = next((run / name for name in ("result.json", "manifest.json") if (run / name).exists()), None)
    if manifest_path is None:
        raise ValueError("Frozen parent run requires result.json or manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for split, parent in parents.items():
        declared = manifest.get("embeddings", {}).get(split, {}).get("embedding_sha256")
        if declared is not None and declared != parent["sha256"]:
            raise ValueError(f"Frozen parent embedding hash mismatch: {split}")
    records = score_arrays({split: _LazyArchive(path) for split, path in paths.items()})
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_embeddings": parents,
        "parent_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "script_sha256": sha256(Path(__file__)), "ridge_alpha_grid": RIDGE_ALPHA_GRID,
        "ridge_selection": "validation position R2; exact ties first fixed grid entry; train-only StandardScaler and fit",
        "knn": {"neighbors": KNN_NEIGHBORS, "metric": "cosine"},
        "valid_support": "explicit supplied common trial-interior valid_mask",
        "representations": "raw=native encoder output; unit=row normalized raw; official normalized CNN may make them equal",
        "training_seed": manifest.get("seed"), "rat": manifest.get("rat"), "fold": manifest.get("fold"),
        "test_outcomes_access": "after both raw/unit alpha selections; no encoder training",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2)+"\n", encoding="utf-8")
    (output / "RAT_DECODING_RESULTS.json").write_text(json.dumps(records, indent=2)+"\n", encoding="utf-8")
    with (output / "RAT_DECODING_RESULTS.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [key for key in records[0] if key != "ridge_validation_candidates"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
