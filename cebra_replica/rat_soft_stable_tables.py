"""Read-only tables for the frozen rat soft-v1-stable embeddings.

This module never fits an encoder. It calls the same CEBRA consistency_score
function and arguments as rat_figure_gap_closure_analysis.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
DEFAULT_OUT = RESULTS / "RAT_FIGURES" / "Tables"
RATS = ("achilles", "buddy", "cicero", "gatsby")
FOLDS = (0, 1, 2)
SEEDS = range(42, 52)
REPRESENTATIONS = ("embedding_raw", "embedding_unit")
FAMILIES = {
    "CNN-joint": ("user_soft_rat_v1", ""),
    "CNN-time": ("user_soft_rat_time_only_v1", ""),
    "Transformer-joint": ("user_soft_rat_transformer_5seed_v1", "joint"),
    "Transformer-time": ("user_soft_rat_transformer_5seed_v1", "time_only"),
}


def display_method(family: str) -> str:
    return "Soft CNN" if family.startswith("CNN") else "Soft Transformer"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rel(path: Path) -> str:
    return path.relative_to(PROJECT).as_posix()


def locate(family: str, rat: str, fold: int, seed: int) -> tuple[Path, Path] | None:
    root_name, branch = FAMILIES[family]
    base = RESULTS / root_name
    if branch:
        base /= branch
    suffix = Path(rat) / f"fold_{fold}" / "soft_v1_stable" / f"seed_{seed}"
    embedding = base / "stage02_embedding" / suffix / "test_embedding.npz"
    metrics = base / "stage03_metrics" / suffix / "metrics.json"
    if embedding.is_file() and metrics.is_file():
        return embedding, metrics
    # First Transformer fits were completed in the immutable initial branch.
    if family.startswith("Transformer"):
        initial = RESULTS / "user_soft_rat_transformer_v1"
        if branch == "time_only":
            initial /= "time_only"
        emb2 = initial / "stage02_embedding" / suffix / "test_embedding.npz"
        met2 = initial / "stage03_metrics" / suffix / "metrics.json"
        if emb2.is_file() and met2.is_file():
            return emb2, met2
    return None


def verify_pair(embedding: Path, metrics: Path) -> None:
    manifest_path = embedding.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metric_payload = json.loads(metrics.read_text(encoding="utf-8"))
    if metric_payload["parent_embedding_manifest_sha256"] != digest(manifest_path):
        raise ValueError(f"Metric parent manifest changed: {metrics}")
    if manifest["embeddings"]["test"]["embedding_sha256"] != digest(embedding):
        raise ValueError(f"Embedding hash changed: {embedding}")
    if manifest["config"]["version"] != "soft_v1_stable":
        raise ValueError(f"Wrong loss version: {manifest_path}")


def load_valid(path: Path, representation: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        keys = (representation, "behavior", "source_id", "trial_id", "time_id", "valid_mask")
        data = {key: z[key] for key in keys}
    n = len(data[representation])
    if data[representation].shape != (n, 32) or any(len(data[k]) != n for k in keys[1:]):
        raise ValueError(f"Invalid embedding and metadata cardinality: {path}")
    mask = data["valid_mask"].astype(bool)
    if not np.isfinite(data[representation][mask]).all():
        raise ValueError(f"Nonfinite valid embedding: {path}")
    return {"embedding": data[representation][mask], "position": data["behavior"][mask, 0],
            "source_id": data["source_id"][mask]}


def complete_sets() -> tuple[dict, list[dict]]:
    selected = {}
    coverage = []
    for family in FAMILIES:
        for seed in SEEDS:
            found = {(rat, fold): locate(family, rat, fold, seed)
                     for rat in RATS for fold in FOLDS}
            complete = all(value is not None for value in found.values())
            coverage.append({"method": family, "seed": seed,
                             "completed_cells": sum(value is not None for value in found.values()),
                             "expected_cells": len(found), "included": complete})
            if complete:
                for (rat, fold), paths in found.items():
                    verify_pair(*paths)
                    selected[(family, seed, rat, fold)] = paths
    return selected, coverage


def decode_rows(selected: dict) -> list[dict]:
    rows = []
    for (family, seed, rat, fold), (_, metrics_path) in selected.items():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload["config"]["version"] != "soft_v1_stable":
            raise ValueError(f"Wrong loss version: {metrics_path}")
        for item in payload["metrics"]:
            if item["representation"] not in REPRESENTATIONS:
                continue
            rows.append({"method": display_method(family), "condition": "joint" if "joint" in family else "time_only",
                         "architecture": "CNN1D" if family.startswith("CNN") else "Transformer",
                         "version": "soft_v1_stable", "rat": rat, "fold": fold, "seed": seed,
                         "split": item["split"], "decoder": item["decoder"],
                         "representation": item["representation"], "n_samples": item["n_samples"],
                         "n_trials": item["n_trials"],
                         "position_median_absolute_error_cm": item["position_median_absolute_error_cm"],
                         "position_r2": item["position_r2"],
                         "direction_balanced_accuracy": item["direction_balanced_accuracy"],
                         "direction_accuracy": item["direction_accuracy"],
                         "selected_alpha": item["selected_alpha"],
                         "parent_metrics_path": rel(metrics_path),
                         "parent_metrics_sha256": digest(metrics_path)})
    return rows


def consistency_rows(selected: dict) -> tuple[list[dict], list[dict]]:
    from cebra.integrations.sklearn.metrics import consistency_score

    cross, across = [], []
    cache = {}

    def get(family: str, seed: int, rat: str, fold: int, rep: str) -> dict:
        key = (family, seed, rat, fold, rep)
        if key not in cache:
            emb_path, _ = selected[(family, seed, rat, fold)]
            data = load_valid(emb_path, rep)
            data["path"] = rel(emb_path)
            data["sha256"] = digest(emb_path)
            cache[key] = data
        return cache[key]

    for family in FAMILIES:
        seeds = [seed for seed in SEEDS if (family, seed, RATS[0], 0) in selected]
        for rep in REPRESENTATIONS:
            for fold in FOLDS:
                for seed in seeds:
                    records = [get(family, seed, rat, fold, rep) for rat in RATS]
                    scores, pairs, _ = consistency_score(
                        [r["embedding"] for r in records], between="datasets",
                        labels=[r["position"] for r in records], dataset_ids=list(RATS),
                        num_discretization_bins=100)
                    for score, pair in zip(scores, pairs):
                        source, target = map(str, pair)
                        a, b = records[RATS.index(source)], records[RATS.index(target)]
                        cross.append({"method": display_method(family), "condition": "joint" if "joint" in family else "time_only",
                                      "architecture": "CNN1D" if family.startswith("CNN") else "Transformer",
                                      "representation": rep, "fold": fold, "seed": seed,
                                      "source_rat": source, "target_rat": target,
                                      "held_out_behavior_aligned_r2": float(score),
                                      "n_source_rows": len(a["position"]), "n_target_rows": len(b["position"]),
                                      "source_embedding_path": a["path"], "source_embedding_sha256": a["sha256"],
                                      "target_embedding_path": b["path"], "target_embedding_sha256": b["sha256"]})
                if len(seeds) < 2:
                    continue
                for rat in RATS:
                    records = [get(family, seed, rat, fold, rep) for seed in seeds]
                    if not all(np.array_equal(records[0]["source_id"], rec["source_id"])
                               for rec in records[1:]):
                        raise ValueError(f"Across-run source support mismatch: {family}/{rat}/fold{fold}/{rep}")
                    scores, pairs, _ = consistency_score([r["embedding"] for r in records], between="runs")
                    for score, pair in zip(scores, pairs):
                        a, b = (int(i) for i in pair)
                        across.append({"method": display_method(family), "condition": "joint" if "joint" in family else "time_only",
                                       "architecture": "CNN1D" if family.startswith("CNN") else "Transformer",
                                       "representation": rep, "rat": rat, "fold": fold,
                                       "source_seed": seeds[a], "target_seed": seeds[b],
                                       "held_out_across_run_r2": float(score),
                                       "n_rows": len(records[a]["source_id"]),
                                       "source_embedding_path": records[a]["path"],
                                       "source_embedding_sha256": records[a]["sha256"],
                                       "target_embedding_path": records[b]["path"],
                                       "target_embedding_sha256": records[b]["sha256"]})
    return cross, across


def fold_seed_summary(rows: list[dict], keys: tuple[str, ...], value: str,
                      output_value: str) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(float(row[value]))
    return [{**dict(zip(keys, key)), "n_folds": len(values), output_value: mean(values)}
            for key, values in grouped.items()]


def decoding_summary(rows: list[dict]) -> list[dict]:
    metrics = ("position_median_absolute_error_cm", "position_r2",
               "direction_balanced_accuracy", "direction_accuracy")
    grouped = defaultdict(list)
    for row in rows:
        for metric in metrics:
            if row[metric] is not None:
                key = tuple(row[k] for k in ("method", "condition", "architecture", "representation",
                                              "rat", "seed", "split", "decoder")) + (metric,)
                grouped[key].append(float(row[metric]))
    keys = ("method", "condition", "architecture", "representation", "rat", "seed",
            "split", "decoder", "metric")
    return [{**dict(zip(keys, key)), "n_folds": len(values), "mean_across_folds": mean(values)}
            for key, values in grouped.items()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Immutable output directory already populated: {out}")
    selected, coverage = complete_sets()
    decoding = decode_rows(selected)
    cross, across = consistency_rows(selected)
    cross_seed = fold_seed_summary(cross, ("method", "condition", "architecture", "representation",
                                               "seed", "source_rat", "target_rat"),
                                   "held_out_behavior_aligned_r2", "mean_across_folds")
    across_seed = fold_seed_summary(across, ("method", "condition", "architecture", "representation",
                                                 "rat", "source_seed"),
                                    "held_out_across_run_r2", "mean_across_folds")
    outputs = {
        "SOFT_COVERAGE.csv": coverage,
        "SOFT_DECODING_LONG.csv": decoding,
        "SOFT_DECODING_BY_SEED.csv": decoding_summary(decoding),
        "SOFT_CROSS_RAT_CONSISTENCY_LONG.csv": cross,
        "SOFT_CROSS_RAT_CONSISTENCY_BY_SEED.csv": cross_seed,
        "SOFT_ACROSS_RUN_CONSISTENCY_LONG.csv": across,
        "SOFT_ACROSS_RUN_CONSISTENCY_BY_SEED.csv": across_seed,
    }
    for name, rows in outputs.items():
        if not rows:
            raise ValueError(f"No rows for {name}")
        write_csv(out / name, rows, list(rows[0]))
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": digest(Path(__file__)),
        "reference_consistency_source": "cebra_replica/rat_figure_gap_closure_analysis.py::consistency_outputs",
        "reference_consistency_sha256": digest(PROJECT / "cebra_replica/rat_figure_gap_closure_analysis.py"),
        "consistency_definition": "CEBRA consistency_score: between=datasets, held-out continuous position labels, 100 discretization bins, directed R2; between=runs, source IDs aligned",
        "selection": "Only complete rat x 3-fold seed groups existing at snapshot; no test-based model selection",
        "method_labels": {"Soft CNN": "CNN1D trained with internal soft_v1_stable",
                          "Soft Transformer": "TemporalTransformerEncoder trained with internal soft_v1_stable"},
        "representation": "raw and unit reported separately; no post-hoc choice",
        "baseline_references": {
            "cross_rat": "results/RAT_FIGURES/Reference_components/Figure_1/Figure_1E/CROSS_RAT_CONSISTENCY_BY_SEED.csv",
            "decoding": "results/RAT_FIGURES/Tables/PAPER_REFERENCE_METHOD_DECODING_ALL_METHODS_LONG.csv",
            "ed4": "results/RAT_FIGURES/Extended_Data/ED4/REFERENCE_METRICS_BY_SEED.csv",
        },
        "output_rows": {name: len(rows) for name, rows in outputs.items()},
        "outputs_sha256": {name: digest(out / name) for name in outputs},
        "included_cells": [{"method": family, "rat": rat, "fold": fold, "seed": seed,
                            "embedding_path": rel(paths[0]), "embedding_sha256": digest(paths[0]),
                            "metrics_path": rel(paths[1]), "metrics_sha256": digest(paths[1])}
                           for (family, seed, rat, fold), paths in selected.items()],
    }
    (out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance["output_rows"], indent=2))


if __name__ == "__main__":
    main()
