"""Reconstruct explicit trial/time metadata beside immutable rat embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parent
FITS = ROOT / "results" / "rat_trial_benchmark_v1"
OUT = ROOT / "results" / "rat_embedding_metadata_v1"
DATASET_SOURCE = ROOT / "CEBRA" / "cebra" / "datasets" / "hippocampus.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(result_path: Path) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rat, fold, condition, seed = (result[k] for k in ("rat", "fold", "condition", "seed"))
    source = ROOT / "data" / "rat_hippocampus" / f"{rat}.jl"
    if sha256(source) != result["source_sha256"]:
        raise AssertionError(f"Raw data hash mismatch: {result_path}")
    fit_dir = result_path.parent
    checkpoint = fit_dir / "checkpoint.pt"
    if sha256(checkpoint) != result["checkpoint_sha256"]:
        raise AssertionError(f"Checkpoint hash mismatch: {result_path}")
    target = OUT / rat / f"fold_{fold}" / condition / f"seed_{seed}"
    existing_manifest = target / "PROVENANCE.json"
    if existing_manifest.exists():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if (existing.get("fit_manifest_sha256") != sha256(result_path) or
            existing.get("source_sha256") != sha256(source)):
            raise RuntimeError(f"Existing metadata provenance mismatch: {target}")
        for split, info in existing["splits"].items():
            if sha256(target / f"{split}_metadata.npz") != info["metadata_sha256"]:
                raise RuntimeError(f"Existing metadata hash mismatch: {target}, {split}")
        print(f"REUSE METADATA {target}", flush=True)
        return
    if target.exists() and any(target.iterdir()):
        raise RuntimeError(f"Partial metadata requires inspection: {target}")

    raw = joblib.load(source)
    position = np.asarray(raw["position"])
    direction_change = np.where(position[1:, 1] != position[:-1, 1])[0]
    trial_change = np.append(np.insert(direction_change[1::2], 0, 0), len(position))
    trial_slices = tuple(slice(int(trial_change[i]), int(trial_change[i + 1]))
                         for i in range(len(trial_change) - 1))
    split_trials = {name: [] for name in ("train", "valid", "test")}
    for outer in np.array_split(np.arange(len(trial_slices)), 3):
        train_idx, val_test_idx = list(KFold(n_splits=3, shuffle=False).split(outer))[fold]
        test_idx, valid_idx = np.array_split(val_test_idx, 2)
        for name, selected in (("train", train_idx), ("valid", valid_idx), ("test", test_idx)):
            split_trials[name].extend(outer[selected].tolist())
    original_rows = {split: np.concatenate([np.arange(trial_slices[i].start,
                                                      trial_slices[i].stop, dtype=np.int64)
                                            for i in trials])
                     for split, trials in split_trials.items()}
    if any(np.intersect1d(original_rows[a], original_rows[b]).size
           for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise AssertionError("Overlapping split rows")

    parent_hashes = {}
    for split in split_trials:
        parent = fit_dir / f"{split}_embedding.npz"
        parent_hash = sha256(parent)
        if parent_hash != result["embeddings"][split]["embedding_sha256"]:
            raise AssertionError(f"Embedding hash mismatch: {parent}")
        with np.load(parent, allow_pickle=False) as stored:
            source_id = stored["source_id"]
            embedding = stored["embedding"]
            behavior = stored["behavior"]
        if not np.array_equal(source_id, original_rows[split]):
            raise AssertionError(f"Source row order mismatch: {parent}")
        if embedding.ndim != 2 or embedding.shape[0] != len(source_id):
            raise AssertionError(f"Embedding shape mismatch: {parent}")
        expected_behavior = position[source_id].astype(np.float32)
        if behavior.shape != expected_behavior.shape or not np.array_equal(
                behavior, expected_behavior):
            raise AssertionError(f"Behavior mismatch: {parent}")

        trial_id = np.searchsorted([s.stop for s in trial_slices], source_id, side="right").astype(np.int64)
        starts = np.array([s.start for s in trial_slices], dtype=np.int64)
        time_id = source_id - starts[trial_id]
        for trial in np.unique(trial_id):
            local = time_id[trial_id == trial]
            if not np.array_equal(local, np.arange(len(local), dtype=np.int64)):
                raise AssertionError(f"Nonconsecutive time IDs: {split}, trial {trial}")
        valid_mask = np.ones(len(source_id), dtype=bool)
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target / f"{split}_metadata.npz", source_id=source_id,
                            trial_id=trial_id, time_id=time_id, valid_mask=valid_mask,
                            split=np.array(split), embedding_parent_sha256=np.array(parent_hash))
        parent_hashes[split] = {"embedding_path": str(parent),
                                "embedding_sha256": parent_hash,
                                "metadata_sha256": sha256(target / f"{split}_metadata.npz"),
                                "rows": int(len(source_id)), "trials": int(len(np.unique(trial_id)))}

    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "rat": rat, "fold": fold, "condition": condition, "seed": seed,
                "source_sha256": sha256(source), "fit_manifest_sha256": sha256(result_path),
                "dataset_source_sha256": sha256(DATASET_SOURCE),
                "script_sha256": sha256(Path(__file__)), "splits": parent_hashes,
                "meaning": "trial_id is zero-based ordinal of complete source trial; "
                           "time_id is zero-based 25-ms sample index within trial; "
                           "valid_mask is true for all exported split observations."}
    (target / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"METADATA {target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-existing", action="store_true")
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.all_existing == bool(args.result):
        parser.error("Specify exactly one of --all-existing or --result")
    results = sorted(FITS.rglob("result.json")) if args.all_existing else [args.result]
    for path in results:
        build(path)


if __name__ == "__main__":
    main()
