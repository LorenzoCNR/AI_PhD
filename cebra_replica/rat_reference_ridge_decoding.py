"""Score position Ridge probes on frozen non-CEBRA reference embeddings.

No encoder is trained. The probe follows the local Soft/CEBRA Ridge protocol:
train-only StandardScaler + Ridge, alpha selected by validation position R2,
then evaluated once on held-out test rows. Results are an added comparison,
not a literal component of paper Figure 2d (whose method comparison is kNN).
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rat_manifold_scoring import RATS, support_for
from user_soft.downstream import RIDGE_ALPHA_GRID


SOURCE = ROOT / "results/RAT_FIGURES/Tables/PAPER_REFERENCE_METHOD_DECODING_ALL_METHODS_LONG.csv"
OUT = ROOT / "results/RAT_FIGURES/Tables/FIGURE2D_REFERENCE_RIDGE_POSITION_TEST_BY_FOLD.csv"
METHODS = {
    "Conv-piVAE-z_mean": "conv-piVAE",
    "PCA": "PCA",
    "UMAP": "UMAP",
    "OPENTSNE": "tSNE",
}
ALPHAS = tuple(float(value) for value in RIDGE_ALPHA_GRID)
SEEDS = tuple(range(42, 48))
SUPPORT_POLICY = "same within-trial 250 ms centers; exclude five left/four right boundary centers"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_rows() -> list[dict]:
    with SOURCE.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = []
    for row in rows:
        if row["method"] not in METHODS or row["split"] != "test":
            continue
        if row["method"] == "PCA":
            if row["seed"] != "deterministic":
                continue
        elif row["seed"] not in {str(seed) for seed in SEEDS}:
            continue
        selected.append(row)
    keys = [(r["method"], r["rat"], r["fold"], r["seed"]) for r in selected]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate reference fit in source decoding table")
    expected = 4 * 3 * (1 + 3 * 6)
    if len(selected) != expected:
        raise RuntimeError(f"Expected {expected} test fits, found {len(selected)}")
    return selected


def split_archive(parent: Path, method: str, split: str) -> tuple[Path, str]:
    if method == "Conv-piVAE-z_mean":
        path = parent / f"embedding_{split}.npz"
        key = "z_mean_raw"
    else:
        path = parent / f"{split}_embedding.npz"
        key = "embedding_raw"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, key


def score_fit(source_row: dict, support_cache: dict) -> dict:
    method = source_row["method"]
    rat = source_row["rat"]
    fold = int(source_row["fold"])
    seed = source_row["seed"]
    parent = ROOT / source_row["parent_result_path"]
    if not parent.is_file() or sha256(parent) != source_row["parent_result_sha256"]:
        raise RuntimeError(f"Reference fit provenance mismatch: {parent}")
    manifest = json.loads(parent.read_text(encoding="utf-8"))
    if method == "Conv-piVAE-z_mean":
        identity = (manifest["config"]["rat"], int(manifest["config"]["fold"]), str(manifest["config"]["seed"]))
        expected_identity = (rat, fold, seed)
        if identity != expected_identity or manifest.get("status") != "complete":
            raise RuntimeError(f"Conv-piVAE run identity/status mismatch: {parent}")
    else:
        manifest_seed = "deterministic" if manifest.get("seed") is None else str(manifest["seed"])
        identity = (manifest["rat"], int(manifest["fold"]), manifest_seed)
        expected_identity = (rat, fold, seed)
        if identity != expected_identity or manifest.get("status") not in ("complete", "complete_embeddings_only"):
            raise RuntimeError(f"Baseline run identity/status mismatch: {parent}")

    support_key = (rat, fold)
    if support_key not in support_cache:
        support_cache[support_key] = support_for(rat, fold)
    support = support_cache[support_key]
    x_by_split, y_by_split, counts, embedding_parents = {}, {}, {}, {}
    for split in ("train", "valid", "test"):
        expected = support[split]
        path, embedding_key = split_archive(parent.parent, method, split)
        actual_sha = sha256(path)
        if method == "Conv-piVAE-z_mean":
            expected_sha = manifest["artifact_hashes"][f"embedding_{split}_sha256"]
            if actual_sha != expected_sha:
                raise RuntimeError(f"Conv-piVAE embedding hash mismatch: {path}")
            with np.load(path, allow_pickle=False) as archive:
                x = archive[embedding_key].copy()
                source_ids = archive["source_id"].copy()
                boundary = archive["crosses_trial_boundary"].astype(bool, copy=True)
            if not np.array_equal(source_ids, expected["source_id"]):
                raise RuntimeError(f"Conv-piVAE source-row mismatch: {path}")
            if np.any(boundary & expected["valid_mask"]):
                raise RuntimeError(f"Conv-piVAE marks a scored center as boundary-crossing: {path}")
        else:
            expected_sha = manifest["embeddings"][split]["sha256"]
            if actual_sha != expected_sha:
                raise RuntimeError(f"Baseline embedding hash mismatch: {path}")
            with np.load(path, allow_pickle=False) as archive:
                x = archive[embedding_key].copy()
                for name in ("source_id", "trial_id", "time_id", "behavior", "valid_mask"):
                    if not np.array_equal(archive[name], expected[name]):
                        raise RuntimeError(f"Baseline {name} mismatch: {path}")
        if split == "test" and actual_sha != source_row["parent_embedding_sha256"]:
            raise RuntimeError(f"Test embedding hash differs from published score parent: {path}")
        mask = expected["valid_mask"]
        if x.ndim != 2 or len(x) != len(mask) or x.shape[1] < 1:
            raise RuntimeError(f"Invalid embedding shape: {path}: {x.shape}")
        x_valid = x[mask].astype(np.float64, copy=False)
        y_valid = expected["behavior"][mask].astype(np.float64, copy=False)
        if not np.isfinite(x_valid).all() or not np.isfinite(y_valid).all():
            raise RuntimeError(f"Non-finite valid values: {path}")
        x_by_split[split], y_by_split[split] = x_valid, y_valid[:, 0]
        counts[split] = len(x_valid)
        embedding_parents[split] = {"path": path.relative_to(ROOT).as_posix(), "sha256": actual_sha}
    if counts["test"] != int(source_row["n_samples_scored"]):
        raise RuntimeError(
            f"Test support count differs from reference score: {parent} "
            f"({counts['test']} != {source_row['n_samples_scored']})"
        )

    candidates = []
    for alpha in ALPHAS:
        probe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        probe.fit(x_by_split["train"], y_by_split["train"])
        val_r2 = float(r2_score(y_by_split["valid"], probe.predict(x_by_split["valid"])))
        candidates.append((alpha, val_r2, probe))
    best_alpha, best_valid_r2, best_probe = max(candidates, key=lambda item: item[1])
    prediction = best_probe.predict(x_by_split["test"])
    mae_cm = float(np.median(np.abs(prediction - y_by_split["test"])) * 100)
    test_r2 = float(r2_score(y_by_split["test"], prediction))
    row = {
        "rat": rat,
        "method": METHODS[method],
        "source_method": method,
        "seed": seed,
        "fold": fold,
        "decoder": "ridge",
        "split": "test",
        "representation": "embedding_raw",
        "embedding_dimension": int(x_by_split["train"].shape[1]),
        "position_median_absolute_error_cm": mae_cm,
        "position_r2": test_r2,
        "selected_alpha": best_alpha,
        "validation_position_r2": best_valid_r2,
        **{f"validation_r2_alpha_{str(alpha).replace('.', '_')}": value for alpha, value, _ in candidates},
        "n_train": counts["train"],
        "n_valid": counts["valid"],
        "n_test": counts["test"],
        "parent_metrics_path": parent.relative_to(ROOT).as_posix(),
        "parent_metrics_sha256": sha256(parent),
        "parent_embeddings": json.dumps(embedding_parents, sort_keys=True),
        "support_policy": SUPPORT_POLICY,
    }
    return row


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {OUT}")
    rows = reference_rows()
    support_cache = {}
    scored = [score_fit(row, support_cache) for row in rows]
    expected_groups = {(method, rat) for method in METHODS for rat in RATS}
    groups = {(r["source_method"], r["rat"]) for r in scored}
    if groups != expected_groups:
        raise RuntimeError(f"Incomplete method/rat coverage: {groups ^ expected_groups}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scored[0]))
        writer.writeheader()
        writer.writerows(scored)
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scorer": Path(__file__).relative_to(ROOT).as_posix(),
        "scorer_sha256": sha256(Path(__file__)),
        "source_reference_table": {"path": SOURCE.relative_to(ROOT).as_posix(), "sha256": sha256(SOURCE)},
        "parent_fit_count": len(scored),
        "method_counts": {method: sum(row["source_method"] == method for row in scored) for method in METHODS},
        "rats": list(RATS),
        "seeds": list(SEEDS),
        "folds": [0, 1, 2],
        "decoder": "sklearn StandardScaler(train only) + Ridge(alpha); fit on train valid-centers",
        "alpha_selection": {"grid": list(ALPHAS), "criterion": "maximum validation position R2", "tie_break": "first alpha in fixed grid"},
        "test_evaluation": "position median absolute error in cm and position R2, after alpha selection",
        "embeddings": "frozen reference outputs only; Conv-piVAE z_mean q(z|x); no encoder retraining",
        "support": SUPPORT_POLICY,
        "scope_note": "Additional local decoder comparison, not part of original paper Figure 2d method comparison (kNN position error).",
        "output": {"path": OUT.relative_to(ROOT).as_posix(), "sha256": sha256(OUT)},
    }
    sidecar = OUT.with_name(OUT.stem + "_PROVENANCE.json")
    sidecar.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"fit_rows": len(scored), "method_rat_coverage": len(groups), "output": str(OUT), "sha256": sha256(OUT)}, indent=2))


if __name__ == "__main__":
    main()
