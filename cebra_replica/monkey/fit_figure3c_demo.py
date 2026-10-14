"""Train-only, resumable CEBRA position fit for a Figure 3c-style reproduction.

Configuration is from the official primate-reaching demo, not inferred from
the precomputed paper figure. This script never loads validation or test data.
"""

from __future__ import annotations

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
DATA = PROJECT / "data" / "monkey_reaching_preload_smth_40" / "active_train.jl"
SPLITS = PROJECT / "results" / "monkey_figure3_v1" / "stage00_data" / "DATA_AND_SPLIT_PROVENANCE.json"
OUTPUT = PROJECT / "results" / "monkey_figure3_v1" / "stage01_fit" / "figure3c_position_3d_demo" / "seed_42"
CONFIG = {
    "source": "https://cebra.ai/docs/demo_notebooks/Demo_primate_reaching.html",
    "status": "official-demo-aligned, train-only reproduction; not asserted identical to paper fit",
    "session": "active",
    "split": "train",
    "seed": 42,
    "model_architecture": "offset10-model",
    "batch_size": 512,
    "learning_rate": 0.0001,
    "temperature": 1.0,
    "output_dimension": 3,
    "max_iterations": 5000,
    "distance": "cosine",
    "conditional": "time_delta",
    "device": "cuda_if_available",
    "time_offsets": 10,
    "label": "hand_position_xy",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    source = json.loads(SPLITS.read_text(encoding="utf-8"))
    if sha256(DATA) != source["source_file_hashes"][DATA.name]["sha256"]:
        raise ValueError("Training source hash differs from verified stage00 bundle")
    if not torch.cuda.is_available():
        raise RuntimeError("Expected CUDA GPU is unavailable; no CPU fallback for this fit")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    checkpoint = OUTPUT / "model.pt"
    manifest = OUTPUT / "FIT_PROVENANCE.json"
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old["config"] != CONFIG or old["train_source_sha256"] != sha256(DATA) or old["checkpoint_sha256"] != sha256(checkpoint):
            raise ValueError("Existing checkpoint provenance conflict")
        print(json.dumps({"status": "reused", "checkpoint": str(checkpoint)}))
        return
    if checkpoint.exists():
        raise FileExistsError("Checkpoint exists without provenance; refusing to overwrite")

    train = joblib.load(DATA)
    x = np.asarray(train["spikes"], dtype=np.float32)
    y = np.asarray(train["pos"], dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 65 or y.shape != (len(x), 2):
        raise ValueError(f"Unexpected training shape: {x.shape}, {y.shape}")
    if len(x) != int(train["num_trials"]) * int(train["trial_len"]):
        raise ValueError("Training rows/trial metadata mismatch")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Non-finite training input")

    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])
    model_kwargs = {key: value for key, value in CONFIG.items() if key in {
        "model_architecture", "batch_size", "learning_rate", "temperature",
        "output_dimension", "max_iterations", "distance", "conditional",
        "device", "time_offsets",
    }}
    model = CEBRA(**model_kwargs)
    start = time.perf_counter()
    model.fit(x, y)
    elapsed = time.perf_counter() - start
    model.save(str(checkpoint))
    history = np.asarray(model.state_dict_["loss"], dtype=float)
    np.save(OUTPUT / "training_loss.npy", history)
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": CONFIG,
        "config_sha256": hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest(),
        "script_sha256": sha256(Path(__file__)),
        "train_source_sha256": sha256(DATA),
        "stage00_manifest_sha256": sha256(SPLITS),
        "checkpoint_sha256": sha256(checkpoint),
        "training_loss_sha256": sha256(OUTPUT / "training_loss.npy"),
        "input_shape": list(x.shape),
        "label_shape": list(y.shape),
        "train_trials": int(train["num_trials"]),
        "optimizer_updates": len(history),
        "training_seconds": elapsed,
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "validation_or_test_loaded": False,
    }
    manifest.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": "complete", "checkpoint": str(checkpoint), "optimizer_updates": len(history), "seconds": elapsed}))


if __name__ == "__main__":
    main()
