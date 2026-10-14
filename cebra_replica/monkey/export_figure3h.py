"""Export immutable train/valid/test embeddings from completed Figure 3h fits."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from cebra import CEBRA

from fit_figure3h import DATA, DIMENSIONS, PROJECT, ROOT as FIT_ROOT, SEEDS, SPLITS, TASKS, sha256


ROOT = PROJECT / "results" / "monkey_figure3_paper_v1" / "stage02_embedding_heldout"
EXTENDED8_FIT_ROOT = PROJECT / "results" / "monkey_extended8_paper_v1" / "stage01_fit_heldout"
EXTENDED8_EMBED_ROOT = PROJECT / "results" / "monkey_extended8_paper_v1" / "stage02_embedding_heldout"


def export_one(task: str, dimension: int, seed: int, split: str) -> str:
    session, _ = TASKS[task]
    extended8 = task in ("ap_pos", "ap_direction")
    fit_root = EXTENDED8_FIT_ROOT if extended8 else FIT_ROOT
    embedding_root = EXTENDED8_EMBED_ROOT if extended8 else ROOT
    fit_dir = fit_root / task / f"dim_{dimension}" / f"seed_{seed}"
    manifest = fit_dir / "FIT_PROVENANCE.json"
    if not manifest.exists():
        return f"pending {task} d={dimension} seed={seed}"
    fit_record = json.loads(manifest.read_text(encoding="utf-8"))
    checkpoint = fit_dir / "model.pt"
    if fit_record["checkpoint_sha256"] != sha256(checkpoint):
        raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
    source = DATA / f"{session}_{split}.jl"
    stage00 = json.loads(SPLITS.read_text(encoding="utf-8"))
    source_hash = stage00["source_file_hashes"][source.name]["sha256"]
    if sha256(source) != source_hash:
        raise ValueError(f"Source hash mismatch: {source}")
    if session == "all":
        ids = np.asarray(stage00["all_trial_indices_by_split"][split], dtype=np.int32)
    else:
        ids = np.asarray(stage00["session_trial_indices_by_split"][session][split], dtype=np.int32)
    dest = embedding_root / task / f"dim_{dimension}" / f"seed_{seed}" / split
    target = dest / "embedding_and_trials.npz"
    provenance = dest / "EMBEDDING_PROVENANCE.json"
    if provenance.exists():
        old = json.loads(provenance.read_text(encoding="utf-8"))
        if old["checkpoint_sha256"] != fit_record["checkpoint_sha256"] or old["embedding_sha256"] != sha256(target):
            raise ValueError(f"Immutable embedding conflict: {target}")
        return f"reused {task} d={dimension} seed={seed} split={split}"
    if target.exists():
        raise FileExistsError(f"Embedding without provenance: {target}")
    data = joblib.load(source)
    trial_len = int(data["trial_len"])
    n_trials = int(data["num_trials"])
    neural = np.asarray(data["spikes"], dtype=np.float32)
    if trial_len != 600 or len(ids) != n_trials or len(neural) != n_trials * trial_len:
        raise ValueError(f"Trial or neural cardinality mismatch: {source}")
    model = CEBRA.load(str(checkpoint))
    raw = np.asarray(model.transform(neural), dtype=np.float32)
    if raw.shape != (len(neural), dimension) or not np.isfinite(raw).all():
        raise ValueError(f"Invalid embedding: {task} {dimension} {seed} {split}")
    unit = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), np.finfo(np.float32).eps)
    dest.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        target, embedding_raw=raw, embedding_unit=unit, neural=neural,
        trial_id=np.repeat(ids, trial_len),
        time_id=np.tile(np.arange(trial_len, dtype=np.int16), n_trials),
        valid_mask=np.ones(len(neural), dtype=bool),
        position_xy=np.asarray(data["pos"], dtype=np.float32),
        velocity_xy=np.asarray(data["vel"], dtype=np.float32),
        direction=np.repeat(np.asarray(data["movement_dir"], dtype=np.int8), trial_len),
        direction_active_passive=np.repeat(np.asarray(data["movement_dir_actpas"], dtype=np.int8), trial_len),
        passive=np.repeat(np.asarray(data["passive"], dtype=bool), trial_len),
        split=np.asarray(split), checkpoint_sha256=np.asarray(fit_record["checkpoint_sha256"]),
    )
    with np.load(target) as saved:
        n = len(saved["embedding_raw"])
        for key in ("embedding_unit", "neural", "trial_id", "time_id", "valid_mask", "position_xy", "direction", "passive"):
            if len(saved[key]) != n:
                raise ValueError(f"Cardinality mismatch: {target}/{key}")
        if len(np.unique(saved["trial_id"])) != n_trials:
            raise ValueError(f"Trial ID mismatch: {target}")
    provenance.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": task, "dimension": dimension, "seed": seed, "split": split,
        "source_sha256": source_hash, "split_provenance_sha256": sha256(SPLITS),
        "fit_manifest_sha256": sha256(manifest),
        "checkpoint_sha256": fit_record["checkpoint_sha256"],
        "script_sha256": sha256(Path(__file__)), "embedding_sha256": sha256(target),
        "embedding_shape": list(raw.shape), "trial_count": n_trials,
        "training_repeated": False,
    }, indent=2), encoding="utf-8")
    return f"exported {task} d={dimension} seed={seed} split={split} shape={raw.shape}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--task", choices=list(TASKS))
    parser.add_argument("--dimension", type=int, choices=DIMENSIONS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--split", choices=("train", "valid", "test"))
    parser.add_argument("--all-for-task", action="store_true")
    args = parser.parse_args()
    if args.all:
        if any(value is not None for value in (args.task, args.dimension, args.seed, args.split)):
            parser.error("--all cannot be combined with selectors")
        for task in TASKS:
            for dimension in DIMENSIONS:
                for seed in SEEDS:
                    for split in ("train", "valid", "test"):
                        print(export_one(task, dimension, seed, split), flush=True)
    elif args.all_for_task:
        if args.task is None or any(value is not None for value in (args.dimension, args.seed, args.split)):
            parser.error("--all-for-task requires only --task")
        for dimension in DIMENSIONS:
            for seed in SEEDS:
                for split in ("train", "valid", "test"):
                    print(export_one(args.task, dimension, seed, split), flush=True)
    elif None not in (args.task, args.dimension, args.seed, args.split):
        print(export_one(args.task, args.dimension, args.seed, args.split), flush=True)
    else:
        parser.error("Use --all, --all-for-task with --task, or all four single-export selectors")


if __name__ == "__main__":
    main()
