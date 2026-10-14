"""Fixed held-out kNN probes from frozen rat embeddings; no encoder fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


ROOT = Path(__file__).resolve().parent
K = 36


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_embedding(path: Path, expected_hash: str) -> dict[str, np.ndarray]:
    if sha256(path) != expected_hash:
        raise RuntimeError(f"Parent embedding hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in ("embedding", "behavior", "source_id")}
    n = len(result["embedding"])
    if len(result["behavior"]) != n or len(result["source_id"]) != n:
        raise RuntimeError(f"Cardinality mismatch: {path}")
    return result


def longest_contiguous_run(ids: np.ndarray) -> slice:
    bounds = np.r_[0, np.flatnonzero(np.diff(ids) != 1) + 1, len(ids)]
    j = int(np.argmax(np.diff(bounds)))
    return slice(int(bounds[j]), int(bounds[j + 1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    parent = run / "result.json"
    fit = json.loads(parent.read_text(encoding="utf-8"))
    arrays = {split: load_embedding(run / f"{split}_embedding.npz",
                                    fit["embeddings"][split]["embedding_sha256"])
              for split in ("train", "valid", "test")}
    if any(np.intersect1d(arrays[a]["source_id"], arrays[b]["source_id"]).size
           for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise RuntimeError("Split source rows overlap")

    train = arrays["train"]
    position_probe = KNeighborsRegressor(n_neighbors=K, metric="cosine")
    direction_probe = KNeighborsClassifier(n_neighbors=K, metric="cosine")
    position_probe.fit(train["embedding"], train["behavior"][:, 0])
    direction_probe.fit(train["embedding"], train["behavior"][:, 1].astype(np.int64))

    destination = ROOT / "results" / "rat_trial_scoring_v1" / fit["rat"] / f"fold_{fit['fold']}" / fit["condition"] / f"seed_{fit['seed']}"
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing scoring output: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    metrics = []
    predictions = {}
    for split in ("valid", "test"):
        data = arrays[split]
        p_true = data["behavior"][:, 0]
        d_true = data["behavior"][:, 1].astype(np.int64)
        p_pred = position_probe.predict(data["embedding"])
        d_pred = direction_probe.predict(data["embedding"])
        row = {"rat": fit["rat"], "fold": fit["fold"], "condition": fit["condition"],
               "seed": fit["seed"], "split": split, "n_samples": len(p_true),
               "position_median_absolute_error_cm": float(np.median(np.abs(p_pred - p_true)) * 100),
               "position_r2": float(r2_score(p_true, p_pred)),
               "direction_balanced_accuracy": float(balanced_accuracy_score(d_true, d_pred))}
        metrics.append(row)
        predictions[split] = (data["source_id"], p_true, p_pred, d_true, d_pred)
        np.savez_compressed(destination / f"{split}_predictions.npz", source_id=data["source_id"],
                            position_true=p_true, position_pred=p_pred,
                            direction_true=d_true, direction_pred=d_pred)
    test_ids, true, predicted, _, _ = predictions["test"]
    selection = longest_contiguous_run(test_ids)
    t = np.arange(selection.stop - selection.start) * 0.025
    fig, axis = plt.subplots(figsize=(8, 3.6), dpi=180)
    axis.plot(t, true[selection] * 100, label="Measured position", color="black", lw=1.8)
    axis.plot(t, predicted[selection] * 100, label="Decoded position", color="#6b42c8", lw=1.3)
    axis.set(xlabel="Time within held-out trial (s)", ylabel="Position (cm)",
             title=f"{fit['rat']} | {fit['condition']} | fold {fit['fold']} | seed {fit['seed']}")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(destination / "held_out_position_trajectory.png", dpi=220)
    fig.savefig(destination / "held_out_position_trajectory.svg")
    plt.close(fig)
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "source_fit_manifest": str(parent), "source_fit_manifest_sha256": sha256(parent),
                  "scorer_sha256": sha256(Path(__file__)), "decoder": "scikit-learn cosine kNN",
                  "n_neighbors": K, "position_target": "meters; median absolute error reported in cm",
                  "direction_target": "binary right indicator, integer cast for classification",
                  "metric_scope": "valid and trial-held-out test; no model selection",
                  "metrics": metrics}
    (destination / "scores.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
