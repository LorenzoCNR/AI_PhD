"""Figure 3 CEBRA descriptive fits with paper Methods settings.

These models use all available trials for paper-style descriptive panels.
They must never be scored as held-out decoding models. Each condition is an
independent immutable checkpoint with a resume manifest.
"""

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
ROOT = PROJECT / "results" / "monkey_figure3_paper_v1" / "stage01_fit"
SEED = 42
CONFIGS = (
    ("figure3b_e_active_target_4d", "active", "target", 4),
    ("figure3b_active_time_4d", "active", "time", 4),
    ("figure3c_active_position_3d", "active", "position", 3),
    ("figure3d_active_time_3d", "active", "time", 3),
    ("figure3e_passive_target_4d", "passive", "target", 4),
    ("figure3f_all_target_ap_4d", "all", "target_ap", 4),
    ("figure3g_all_position_ap_4d", "all", "position_ap", 4),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fit_one(item: tuple[str, str, str, int]) -> str:
    name, session, labels, dimension = item
    dest = ROOT / name / f"seed_{SEED}"
    checkpoint = dest / "model.pt"
    manifest = dest / "FIT_PROVENANCE.json"
    source = DATA / f"{session}_all.jl"
    stage00 = json.loads(SPLITS.read_text(encoding="utf-8"))
    if sha256(source) != stage00["source_file_hashes"][source.name]["sha256"]:
        raise ValueError(f"Source changed: {source}")
    config = {
        "source": "Schneider, Lee & Mathis 2023, Fig. 3 and Methods; current CEBRA 0.6.1 default LR/iteration semantics",
        "branch": "paper_descriptive_full_sample_not_held_out",
        "name": name,
        "session": session,
        "split": "all",
        "label_mode": labels,
        "seed": SEED,
        "model_architecture": "offset10-model",
        "time_offsets": 10,
        "batch_size": 512,
        "learning_rate": 0.0003,
        "temperature": 1.0,
        "max_iterations": 10000,
        "output_dimension": dimension,
        "distance": "cosine",
        "conditional": "time" if labels == "time" else "time_delta",
        "device": "cuda_if_available",
    }
    if manifest.exists():
        prior = json.loads(manifest.read_text(encoding="utf-8"))
        if prior["config"] != config or prior["source_sha256"] != sha256(source) or prior["checkpoint_sha256"] != sha256(checkpoint):
            raise ValueError(f"Existing fit provenance conflict: {name}")
        return f"reused {name}"
    if checkpoint.exists():
        raise FileExistsError(f"Checkpoint without manifest: {checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    data = joblib.load(source)
    x = np.asarray(data["spikes"], dtype=np.float32)
    if x.shape != (int(data["num_trials"]) * 600, 65) or not np.isfinite(x).all():
        raise ValueError(f"Invalid neural shape/values: {name} {x.shape}")
    trial_len = int(data["trial_len"])
    if trial_len != 600:
        raise ValueError(f"Unexpected trial length: {trial_len}")
    if labels == "time":
        args = ()
    elif labels == "target":
        args = (np.repeat(np.asarray(data["movement_dir"], dtype=np.int32), trial_len),)
    elif labels == "position":
        args = (np.asarray(data["pos"], dtype=np.float32),)
    elif labels == "target_ap":
        args = (np.repeat(np.asarray(data["movement_dir_actpas"], dtype=np.int32), trial_len),)
    elif labels == "position_ap":
        args = (
            np.asarray(data["pos"], dtype=np.float32),
            np.repeat(np.asarray(data["passive"], dtype=np.int32), trial_len),
        )
    else:
        raise ValueError(labels)
    for arg in args:
        if len(arg) != len(x) or not np.isfinite(arg).all():
            raise ValueError(f"Invalid label cardinality/values: {name}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    kwargs = {key: config[key] for key in (
        "model_architecture", "time_offsets", "batch_size", "learning_rate",
        "temperature", "max_iterations", "output_dimension", "distance",
        "conditional", "device",
    )}
    model = CEBRA(**kwargs)
    start = time.perf_counter()
    model.fit(x, *args)
    elapsed = time.perf_counter() - start
    dest.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint))
    history = np.asarray(model.state_dict_["loss"], dtype=np.float32)
    np.save(dest / "training_loss.npy", history)
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "script_sha256": sha256(Path(__file__)),
        "stage00_sha256": sha256(SPLITS),
        "source_sha256": sha256(source),
        "checkpoint_sha256": sha256(checkpoint),
        "training_loss_sha256": sha256(dest / "training_loss.npy"),
        "input_shape": list(x.shape),
        "label_shapes": [list(arg.shape) for arg in args],
        "optimizer_updates": len(history),
        "training_seconds": elapsed,
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "held_out_claim": False,
    }
    manifest.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return f"complete {name} updates={len(history)} seconds={elapsed:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=[item[0] for item in CONFIGS])
    parser.add_argument("--all", action="store_true")
    arguments = parser.parse_args()
    if arguments.all == bool(arguments.name):
        parser.error("Specify exactly one of --name or --all")
    selected = CONFIGS if arguments.all else tuple(item for item in CONFIGS if item[0] == arguments.name)
    for item in selected:
        print(fit_one(item), flush=True)


if __name__ == "__main__":
    main()
