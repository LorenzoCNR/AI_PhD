"""Resumable held-out Figure 3h CEBRA fits; no decoder or test access here."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from cebra import CEBRA


PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data" / "monkey_reaching_preload_smth_40"
SPLITS = PROJECT / "results" / "monkey_figure3_v1" / "stage00_data" / "DATA_AND_SPLIT_PROVENANCE.json"
ROOT = PROJECT / "results" / "monkey_figure3_paper_v1" / "stage01_fit_heldout"
EXTENDED8_FIT_ROOT = PROJECT / "results" / "monkey_extended8_paper_v1" / "stage01_fit_heldout"
TASKS = {
    "position": ("active", "position"),
    "direction": ("active", "target"),
    "ap": ("all", "position"),
    "ap_pos": ("all", "position_ap"),
    # Extended Data 8: active/passive mixed 16-class direction label.
    "ap_direction": ("all", "target_ap"),
}
DIMENSIONS = (3, 4, 8, 16, 32)
SEEDS = (42, 43, 44, 45, 46)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_one(task: str, dimension: int, seed: int) -> str:
    session, label_mode = TASKS[task]
    source = DATA / f"{session}_train.jl"
    expected = json.loads(SPLITS.read_text(encoding="utf-8"))["source_file_hashes"][source.name]["sha256"]
    if sha256(source) != expected:
        raise ValueError(f"Frozen source hash mismatch: {source}")
    task_root = EXTENDED8_FIT_ROOT if task in ("ap_pos", "ap_direction") else ROOT
    dest = task_root / task / f"dim_{dimension}" / f"seed_{seed}"
    checkpoint = dest / "model.pt"
    manifest = dest / "FIT_PROVENANCE.json"
    config = {
        "branch": f"extended8_{task}_heldout_train_only" if task in ("ap_pos", "ap_direction") else "figure3h_heldout_train_only",
        "task": task, "session": session, "label_mode": label_mode,
        "dimension": dimension, "seed": seed, "source_split": "train",
        "model_architecture": "offset10-model", "time_offsets": 10,
        "batch_size": 512, "learning_rate": 0.0003, "temperature": 1.0,
        "max_iterations": 10000, "output_dimension": dimension,
        "distance": "cosine", "conditional": "time_delta", "device": "cuda_if_available",
    }
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old["config"] != config or old["source_sha256"] != expected or old["checkpoint_sha256"] != sha256(checkpoint):
            raise ValueError(f"Immutable fit conflict: {dest}")
        return f"reused {task} d={dimension} seed={seed}"
    if checkpoint.exists() or dest.exists():
        raise FileExistsError(f"Incomplete or unproven fit; inspect before resuming: {dest}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    data = joblib.load(source)
    x = np.asarray(data["spikes"], dtype=np.float32)
    trial_len = int(data["trial_len"])
    if trial_len != 600 or x.shape != (int(data["num_trials"]) * trial_len, 65) or not np.isfinite(x).all():
        raise ValueError(f"Invalid train observations: {source} {x.shape}")
    if label_mode == "position":
        labels = np.asarray(data["pos"], dtype=np.float32)
    elif label_mode == "position_ap":
        labels = (np.asarray(data["pos"], dtype=np.float32),
                  np.repeat(np.asarray(data["passive"], dtype=np.int32), trial_len))
    elif label_mode == "target":
        labels = np.repeat(np.asarray(data["movement_dir"], dtype=np.int32), trial_len)
    else:
        labels = np.repeat(np.asarray(data["movement_dir_actpas"], dtype=np.int32), trial_len)
    label_arrays = labels if isinstance(labels, tuple) else (labels,)
    if any(len(label) != len(x) or not np.isfinite(label).all() for label in label_arrays):
        raise ValueError("Invalid train labels")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = CEBRA(**{key: config[key] for key in (
        "model_architecture", "time_offsets", "batch_size", "learning_rate",
        "temperature", "max_iterations", "output_dimension", "distance",
        "conditional", "device",
    )})
    started = time.perf_counter()
    model.fit(x, *label_arrays)
    elapsed = time.perf_counter() - started
    dest.mkdir(parents=True, exist_ok=False)
    model.save(str(checkpoint))
    np.save(dest / "training_loss.npy", np.asarray(model.state_dict_["loss"], dtype=np.float32))
    manifest.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "source_sha256": expected, "split_provenance_sha256": sha256(SPLITS),
        "script_sha256": sha256(Path(__file__)), "checkpoint_sha256": sha256(checkpoint),
        "loss_sha256": sha256(dest / "training_loss.npy"),
        "input_shape": list(x.shape), "label_shape": [list(label.shape) for label in label_arrays],
        "optimizer_updates": len(model.state_dict_["loss"]), "training_seconds": elapsed,
        "device_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
        "test_accessed": False,
    }, indent=2, default=str), encoding="utf-8")
    return f"completed {task} d={dimension} seed={seed} updates=10000 seconds={elapsed:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(TASKS))
    parser.add_argument("--dimension", type=int, choices=DIMENSIONS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--all-for-task", action="store_true")
    args = parser.parse_args()
    if args.all and args.all_for_task:
        parser.error("Choose only one of --all and --all-for-task")
    if args.all:
        if any(value is not None for value in (args.task, args.dimension, args.seed)):
            parser.error("--all cannot be combined with one-fit selectors")
        plan = ((task, dimension, seed) for task in TASKS for dimension in DIMENSIONS for seed in SEEDS)
    elif args.all_for_task:
        if args.task is None or args.dimension is not None or args.seed is not None:
            parser.error("--all-for-task requires only --task")
        plan = ((args.task, dimension, seed) for dimension in DIMENSIONS for seed in SEEDS)
    elif None not in (args.task, args.dimension, args.seed):
        plan = ((args.task, args.dimension, args.seed),)
    else:
        parser.error("Use --all, --all-for-task with --task, or one task/dimension/seed")
    for task, dimension, seed in plan:
        print(fit_one(task, dimension, seed), flush=True)


if __name__ == "__main__":
    main()
