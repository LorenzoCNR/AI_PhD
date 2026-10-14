"""Held-out Figure 3h kNN scores from frozen embeddings; never refit encoders.

The published Fig. 3 specifies kNN but not its K in the plotting source.
K=36/cosine follows the authors' official decoding demo and is declared here.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cebra
import numpy as np
from sklearn.metrics import accuracy_score, r2_score

from export_figure3h import ROOT as EMBEDDING_ROOT
from fit_figure3h import DIMENSIONS, PROJECT, SEEDS, TASKS, sha256


ROOT = PROJECT / "results" / "monkey_figure3_paper_v1" / "stage03_decoding_heldout"
EXTENDED8_EMBED_ROOT = PROJECT / "results" / "monkey_extended8_paper_v1" / "stage02_embedding_heldout"
EXTENDED8_SCORE_ROOT = PROJECT / "results" / "monkey_extended8_paper_v1" / "stage03_decoding_heldout"
K = 36


def score_one(task: str, dimension: int, seed: int) -> str:
    extended8 = task in ("ap_pos", "ap_direction")
    embedding_root = EXTENDED8_EMBED_ROOT if extended8 else EMBEDDING_ROOT
    score_root = EXTENDED8_SCORE_ROOT if extended8 else ROOT
    source_dir = embedding_root / task / f"dim_{dimension}" / f"seed_{seed}"
    train = source_dir / "train" / "embedding_and_trials.npz"
    test = source_dir / "test" / "embedding_and_trials.npz"
    for path in (train, test):
        provenance = path.parent / "EMBEDDING_PROVENANCE.json"
        if not provenance.exists() or json.loads(provenance.read_text(encoding="utf-8"))["embedding_sha256"] != sha256(path):
            return f"pending or invalid embedding {task} d={dimension} seed={seed} {path.parent.name}"
    dest = score_root / task / f"dim_{dimension}" / f"seed_{seed}"
    target = dest / "DECODING_RESULT.json"
    parents = {"train": sha256(train), "test": sha256(test)}
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        if old["parent_embedding_sha256"] != parents or old["k"] != K:
            raise ValueError(f"Immutable decoding conflict: {target}")
        return f"reused {task} d={dimension} seed={seed}: {old['score']:.4f}"
    with np.load(train) as archive:
        x_train = np.asarray(archive["embedding_raw"], dtype=np.float32)
        y_train = _target(archive, task)
        train_trials = np.unique(archive["trial_id"])
    with np.load(test) as archive:
        x_test = np.asarray(archive["embedding_raw"], dtype=np.float32)
        y_test = _target(archive, task)
        test_trials = np.unique(archive["trial_id"])
    if np.intersect1d(train_trials, test_trials).size:
        raise ValueError("Train/test trial leakage")
    decoder = cebra.KNNDecoder(n_neighbors=K, metric="cosine")
    decoder.fit(x_train, y_train)
    predictions = decoder.predict(x_test)
    if task in ("position", "ap_pos"):
        metric = "R2_percent"
        score = 100 * r2_score(y_test, predictions)
    else:
        metric = "accuracy_percent"
        score = 100 * accuracy_score(y_test, predictions)
    dest.mkdir(parents=True, exist_ok=False)
    target.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": task, "dimension": dimension, "seed": seed,
        "split_fit": "train", "split_score": "test", "metric": metric,
        "score": float(score), "n_train_rows": len(x_train), "n_test_rows": len(x_test),
        "n_train_trials": len(train_trials), "n_test_trials": len(test_trials),
        "decoder": "cebra.KNNDecoder", "k": K, "distance": "cosine",
        "representation": "embedding_raw", "parent_embedding_sha256": parents,
        "script_sha256": sha256(Path(__file__)), "encoder_retrained": False,
        "protocol_note": "k=36 is official CEBRA decoding-demo alignment; exact Figure 3h paper K was not verified",
    }, indent=2), encoding="utf-8")
    return f"scored {task} d={dimension} seed={seed}: {score:.4f} {metric}"


def _target(archive: np.lib.npyio.NpzFile, task: str) -> np.ndarray:
    if task == "position":
        return np.asarray(archive["position_xy"], dtype=np.float32)
    if task == "ap_pos":
        return np.asarray(archive["position_xy"], dtype=np.float32)
    if task == "direction":
        return np.asarray(archive["direction"], dtype=np.int32)
    if task == "ap_direction":
        return np.asarray(archive["direction_active_passive"], dtype=np.int32)
    return np.asarray(archive["passive"], dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--task", choices=list(TASKS))
    parser.add_argument("--dimension", type=int, choices=DIMENSIONS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--all-for-task", action="store_true")
    args = parser.parse_args()
    if args.all:
        if any(value is not None for value in (args.task, args.dimension, args.seed)):
            parser.error("--all cannot be combined with selectors")
        plan = ((task, dimension, seed) for task in TASKS for dimension in DIMENSIONS for seed in SEEDS)
    elif args.all_for_task:
        if args.task is None or args.dimension is not None or args.seed is not None:
            parser.error("--all-for-task requires only --task")
        plan = ((args.task, dimension, seed) for dimension in DIMENSIONS for seed in SEEDS)
    elif None not in (args.task, args.dimension, args.seed):
        plan = ((args.task, args.dimension, args.seed),)
    else:
        parser.error("Use --all, --all-for-task with --task, or all three single-score selectors")
    for task, dimension, seed in plan:
        print(score_one(task, dimension, seed), flush=True)


if __name__ == "__main__":
    main()
