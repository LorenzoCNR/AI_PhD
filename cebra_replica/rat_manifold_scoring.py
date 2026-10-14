"""Held-out decoding from frozen rat embeddings; no model/metric selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


ROOT = Path(__file__).resolve().parents[1]
REPLICA = ROOT / "cebra_replica"
sys.path.insert(0, str(REPLICA))
from rat_manifold_benchmark import (  # noqa: E402
    DATA_DIR,
    FOLDS,
    OUT_ROOT as BASELINE_ROOT,
    RATS,
    SEEDS,
    SPLITTER_SOURCE,
    row_ids,
    sha256_file,
    trial_bounds,
    trial_split,
)


CEBRA_ROOT = ROOT / "results" / "rat_trial_benchmark_v1"
OUT_ROOT = ROOT / "results" / "rat_manifold_decoding_v1"
K = 36
CEBRA_CONDITIONS = (("CEBRA-Time-32D", "time"), ("CEBRA-Behavior-32D", "position_direction"))
METRICS = ("position_median_absolute_error_cm", "position_r2", "direction_balanced_accuracy")


def atomic_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def support_for(rat: str, fold: int) -> dict[str, dict[str, np.ndarray]]:
    raw = joblib.load(DATA_DIR / f"{rat}.jl")
    position = np.asarray(raw["position"], dtype=np.float32)
    bounds = trial_bounds(position)
    split_trials = trial_split(bounds, fold)
    result = {}
    for split, trials in split_trials.items():
        ids = row_ids(bounds, trials)
        trial_id = np.searchsorted(bounds[1:], ids, side="right").astype(np.int64)
        time_id = (ids - bounds[trial_id]).astype(np.int64)
        length = (bounds[trial_id + 1] - bounds[trial_id]).astype(np.int64)
        valid = (time_id >= 5) & (time_id < length - 4)
        result[split] = {"source_id": ids, "trial_id": trial_id, "time_id": time_id,
                         "valid_mask": valid, "behavior": position[ids].astype(np.float32)}
    return result


def load_npz(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in keys}


def verify_parent_embeddings(manifest_path: Path, manifest: dict, baseline: bool,
                             support: dict[str, dict[str, np.ndarray]]) -> dict[str, dict]:
    arrays = {}
    for split in ("train", "valid", "test"):
        path = manifest_path.parent / f"{split}_embedding.npz"
        entry = manifest["embeddings"][split]
        expected_hash = entry["sha256"] if baseline else entry["embedding_sha256"]
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"Embedding hash mismatch: {path}")
        if baseline:
            data = load_npz(path, ("embedding_raw", "embedding_unit", "behavior", "source_id",
                                   "trial_id", "time_id", "valid_mask"))
            embedding = data["embedding_raw"]
            for key in ("source_id", "trial_id", "time_id", "behavior", "valid_mask"):
                if not np.array_equal(data[key], support[split][key]):
                    raise RuntimeError(f"Metadata/support mismatch: {path}, {key}")
            valid = data["valid_mask"]
            if embedding.ndim != 2 or embedding.shape != (len(valid), 2):
                raise RuntimeError(f"Invalid baseline embedding cardinality: {path}")
            if (not np.isfinite(embedding[valid]).all()
                    or not np.isnan(embedding[~valid]).all()
                    or data["embedding_unit"].shape != embedding.shape):
                raise RuntimeError(f"Invalid baseline embedding values/support: {path}")
        else:
            data = load_npz(path, ("embedding", "behavior", "source_id"))
            if not np.array_equal(data["source_id"], support[split]["source_id"]):
                raise RuntimeError(f"CEBRA source-row mismatch: {path}")
            if not np.array_equal(data["behavior"], support[split]["behavior"]):
                raise RuntimeError(f"CEBRA behavior mismatch: {path}")
            embedding = data["embedding"]
            valid = support[split]["valid_mask"]
            if embedding.ndim != 2 or embedding.shape[0] != len(valid):
                raise RuntimeError(f"Invalid CEBRA embedding cardinality: {path}")
            if not np.isfinite(embedding).all():
                raise RuntimeError(f"Non-finite CEBRA embedding: {path}")
        data["valid_mask"] = support[split]["valid_mask"]
        arrays[split] = data
    return arrays


def score_fit(manifest_path: Path, method: str, condition: str | None,
              baseline: bool, support: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if baseline:
        if manifest.get("status") not in ("complete", "complete_embeddings_only"):
            return []
        rat, fold, seed = manifest["rat"], int(manifest["fold"]), manifest["seed"]
        dim = int(manifest["embedding_dim"])
    else:
        rat, fold, seed = manifest["rat"], int(manifest["fold"]), int(manifest["seed"])
        condition = manifest["condition"]
        dim = int(manifest["parameters"]["output_dimension"])
    arrays = verify_parent_embeddings(manifest_path, manifest, baseline, support)
    train = arrays["train"]
    train_mask = train["valid_mask"]
    train_x = (train["embedding_raw"] if baseline else train["embedding"])[train_mask]
    train_behavior = train["behavior"][train_mask]
    if len(train_x) < K:
        raise RuntimeError(f"Too few train observations for k={K}: {manifest_path}")
    position_decoder = KNeighborsRegressor(n_neighbors=K, metric="cosine")
    direction_decoder = KNeighborsClassifier(n_neighbors=K, metric="cosine")
    position_decoder.fit(train_x, train_behavior[:, 0])
    direction_decoder.fit(train_x, train_behavior[:, 1].astype(np.int64))
    rows = []
    for split in ("valid", "test"):
        sample = arrays[split]
        mask = sample["valid_mask"]
        x = (sample["embedding_raw"] if baseline else sample["embedding"])[mask]
        behavior = sample["behavior"][mask]
        position_prediction = position_decoder.predict(x)
        direction_prediction = direction_decoder.predict(x)
        rows.append({
            "method": method, "condition": condition or "unsupervised",
            "embedding_dim": dim, "embedding_variant": "raw",
            "rat": rat, "fold": fold,
            "seed": "deterministic" if seed is None else seed,
            "split": split, "n_samples_scored": int(mask.sum()),
            "k_neighbors": K, "distance": "cosine",
            "position_median_absolute_error_cm": float(np.median(np.abs(
                position_prediction - behavior[:, 0])) * 100),
            "position_r2": float(r2_score(behavior[:, 0], position_prediction)),
            "direction_balanced_accuracy": float(balanced_accuracy_score(
                behavior[:, 1].astype(np.int64), direction_prediction)),
            "parent_result_path": manifest_path.relative_to(ROOT).as_posix(),
            "parent_result_sha256": sha256_file(manifest_path),
            "parent_embedding_sha256": manifest["embeddings"][split][
                "sha256" if baseline else "embedding_sha256"],
            "support_policy": "same within-trial 250 ms centers; exclude five left/four right boundary centers",
        })
    return rows


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    seed_fold: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["method"], row["condition"], row["embedding_dim"], row["rat"],
               row["seed"], row["split"])
        seed_fold[key].append(row)
    run_rows = []
    for key, group in seed_fold.items():
        for metric in METRICS:
            run_rows.append({"method": key[0], "condition": key[1], "embedding_dim": key[2],
                             "rat": key[3], "seed": key[4], "split": key[5],
                             "metric": metric, "fold_mean": mean(float(item[metric]) for item in group),
                             "n_folds": len(group)})
    by_seed: dict[tuple, list[float]] = defaultdict(list)
    for row in run_rows:
        key = (row["method"], row["condition"], row["embedding_dim"], row["rat"],
               row["split"], row["metric"])
        by_seed[key].append(float(row["fold_mean"]))
    summary = []
    for key, values in by_seed.items():
        summary.append({"method": key[0], "condition": key[1], "embedding_dim": key[2],
                        "rat": key[3], "split": key[4], "metric": key[5],
                        "n_seed_means": len(values), "mean": mean(values),
                        "median": median(values),
                        "sd_across_seed_means": stdev(values) if len(values) > 1 else "",
                        "min": min(values), "max": max(values),
                        "aggregation": "mean across 3 folds per seed; then summarize seed means"})
    return run_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--available-only", action="store_true",
                        help="Score only currently complete baseline and CEBRA artifacts")
    parser.add_argument("--include-cebra", action="store_true",
                        help="Also rescore frozen CEBRA-Time and CEBRA-Behavior on common support")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Write a versioned result set here; defaults to the legacy v1 path")
    args = parser.parse_args()
    output_root = args.output_dir if args.output_dir is not None else OUT_ROOT
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    if output_root.exists() and any(output_root.iterdir()):
        parser.error(f"Refusing to overwrite non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    long_rows = []
    parents = []
    support_cache = {}

    def support(rat: str, fold: int):
        key = (rat, fold)
        if key not in support_cache:
            support_cache[key] = support_for(rat, fold)
        return support_cache[key]

    baseline_manifests = sorted(BASELINE_ROOT.rglob("result.json"))
    for manifest_path in baseline_manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") not in ("complete", "complete_embeddings_only"):
            continue
        long_rows.extend(score_fit(manifest_path, manifest["method"].upper(), None, True,
                                   support(manifest["rat"], int(manifest["fold"]))))
        parents.append({"path": manifest_path.relative_to(ROOT).as_posix(),
                        "sha256": sha256_file(manifest_path)})

    if args.include_cebra:
        for rat in RATS:
            for fold in FOLDS:
                for _, condition in CEBRA_CONDITIONS:
                    for seed in SEEDS:
                        manifest_path = (CEBRA_ROOT / rat / f"fold_{fold}" / condition
                                         / f"seed_{seed}" / "result.json")
                        if not manifest_path.is_file():
                            continue
                        for display, selected_condition in CEBRA_CONDITIONS:
                            if condition != selected_condition:
                                continue
                            long_rows.extend(score_fit(manifest_path, display, condition, False,
                                                       support(rat, fold)))
                            parents.append({"path": manifest_path.relative_to(ROOT).as_posix(),
                                            "sha256": sha256_file(manifest_path)})

    run_rows, summary_rows = aggregate(long_rows)
    fields_long = list(long_rows[0].keys()) if long_rows else []
    fields_aggregate = list(run_rows[0].keys()) if run_rows else []
    fields_summary = list(summary_rows[0].keys()) if summary_rows else []
    atomic_csv(output_root / "RAT_METHOD_DECODING_LONG.csv", long_rows, fields_long)
    atomic_csv(output_root / "RAT_METHOD_DECODING_SEED_LEVEL.csv", run_rows, fields_aggregate)
    atomic_csv(output_root / "RAT_METHOD_DECODING_SUMMARY.csv", summary_rows, fields_summary)
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "scorer_sha256": sha256_file(Path(__file__)),
                  "splitter_source_sha256": sha256_file(SPLITTER_SOURCE),
                  "data_hashes": {rat: sha256_file(DATA_DIR / f"{rat}.jl") for rat in RATS},
                  "parent_manifests": parents,
                  "decoder": {"type": "scikit-learn k-nearest-neighbors", "k": K,
                              "distance": "cosine", "fit_split": "train only"},
                  "scored_splits": ["valid", "test"],
                  "metric_use": "descriptive held-out evaluation only; no model selection",
                  "common_support": "valid within-trial 250 ms centers; invalid boundary centers excluded",
                  "primary_table": "RAT_METHOD_DECODING_LONG.csv",
                  "summary_aggregation": "average three folds within seed; summarize seed means",
                  "output_dir": str(output_root),
                  "figure_generation": "not performed"}
    atomic_json(output_root / "RAT_METHOD_DECODING_PROVENANCE.json", provenance)
    print(json.dumps({"scored_rows": len(long_rows), "seed_level_rows": len(run_rows),
                      "summary_rows": len(summary_rows), "cebra_included": args.include_cebra,
                      "output": str(output_root)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
