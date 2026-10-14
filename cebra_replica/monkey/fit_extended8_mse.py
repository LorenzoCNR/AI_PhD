"""Extended Data 8 MSE-loss fits from the authors' S1 demo settings."""

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
SOURCE = PROJECT / "data/monkey_reaching_preload_smth_40/active_all.jl"
ROOT = PROJECT / "results/monkey_extended8_paper_v1"
SEED = 42
MODES = ("target", "position", "time")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit(mode: str) -> None:
    dest = ROOT / "stage01_fit_mse" / mode / f"seed_{SEED}"
    manifest = dest / "FIT.json"
    checkpoint = dest / "model.pt"
    config = {"source": "CEBRA Demo_primate_reaching_mse_loss, ED8",
              "mode": mode, "split": "active_all_full_sample_descriptive", "seed": SEED,
              "model_architecture": "offset10-model-mse", "batch_size": 512,
              "learning_rate": 0.00005, "temperature": 0.01,
              "output_dimension": 2, "max_iterations": 5000,
              "distance": "euclidean", "conditional": "time" if mode == "time" else "time_delta",
              "device": "cuda_if_available", "time_offsets": 10}
    source_hash = sha256(SOURCE)
    if manifest.exists():
        prior = json.loads(manifest.read_text(encoding="utf-8"))
        if prior["config"] != config or prior["source_sha256"] != source_hash or sha256(checkpoint) != prior["checkpoint_sha256"]:
            raise ValueError(f"Existing ED8 MSE fit differs: {mode}")
        print(f"REUSE ED8 MSE {mode}", flush=True)
        return
    if checkpoint.exists():
        raise FileExistsError(f"Checkpoint without immutable manifest: {checkpoint}")
    data = joblib.load(SOURCE)
    x = np.asarray(data["spikes"], dtype=np.float32)
    if x.shape != (193 * 600, 65):
        raise ValueError(f"Unexpected active monkey source: {x.shape}")
    if mode == "target":
        labels = (np.repeat(np.asarray(data["movement_dir"], dtype=np.int32), 600),)
    elif mode == "position":
        labels = (np.asarray(data["pos"], dtype=np.float32),)
    else:
        labels = ()
    if not torch.cuda.is_available():
        raise RuntimeError("ED8 MSE fit requires CUDA")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    kwargs = {key: config[key] for key in ("model_architecture", "batch_size", "learning_rate",
             "temperature", "output_dimension", "max_iterations", "distance", "conditional",
             "device", "time_offsets")}
    model = CEBRA(**kwargs)
    started = time.perf_counter()
    model.fit(x, *labels)
    elapsed = time.perf_counter() - started
    history = np.asarray(model.state_dict_["loss"], dtype=np.float32)
    if len(history) != 5000:
        raise ValueError(f"ED8 MSE did not complete 5000 updates: {len(history)}")
    dest.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint))
    np.save(dest / "training_loss.npy", history)
    manifest.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(),
                                    "config": config, "source_sha256": source_hash,
                                    "script_sha256": sha256(Path(__file__)),
                                    "checkpoint_sha256": sha256(checkpoint),
                                    "history_sha256": sha256(dest / "training_loss.npy"),
                                    "training_seconds": elapsed,
                                    "train_updates": len(history),
                                    "reference_hdf5_used": False,
                                    "held_out_claim": False}, indent=2) + "\n", encoding="utf-8")
    print(f"COMPLETE ED8 MSE {mode}: {elapsed:.1f}s", flush=True)


def export(mode: str) -> None:
    fit_dir = ROOT / "stage01_fit_mse" / mode / f"seed_{SEED}"
    fit_manifest = fit_dir / "FIT.json"
    fit_record = json.loads(fit_manifest.read_text(encoding="utf-8"))
    checkpoint = fit_dir / "model.pt"
    if sha256(checkpoint) != fit_record["checkpoint_sha256"] or sha256(SOURCE) != fit_record["source_sha256"]:
        raise ValueError("ED8 MSE parent hash changed")
    dest = ROOT / "stage02_embedding_mse" / mode / f"seed_{SEED}"
    target = dest / "all_trials.npz"
    manifest = dest / "EMBEDDING.json"
    if manifest.exists():
        prior = json.loads(manifest.read_text(encoding="utf-8"))
        if prior["fit_manifest_sha256"] != sha256(fit_manifest) or prior["artifact_sha256"] != sha256(target):
            raise ValueError("ED8 MSE embedding changed")
        print(f"REUSE ED8 MSE embedding {mode}", flush=True)
        return
    if target.exists():
        raise FileExistsError("Partial ED8 MSE embedding exists")
    data = joblib.load(SOURCE)
    x = np.asarray(data["spikes"], dtype=np.float32)
    model = CEBRA.load(str(checkpoint))
    raw = np.asarray(model.transform(x), dtype=np.float32)
    if raw.shape != (len(x), 2) or not np.isfinite(raw).all():
        raise ValueError(f"Invalid MSE embedding {mode}: {raw.shape}")
    unit = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, embedding_raw=raw, embedding_unit=unit,
                        neural=x, position_xy=np.asarray(data["pos"], dtype=np.float32),
                        direction=np.repeat(np.asarray(data["movement_dir"], dtype=np.int8), 600),
                        trial_id=np.repeat(np.arange(193), 600),
                        time_id=np.tile(np.arange(600), 193),
                        valid_mask=np.ones(len(x), dtype=bool),
                        split=np.asarray("active_all_full_sample_descriptive"))
    manifest.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(),
                                    "fit_manifest_sha256": sha256(fit_manifest),
                                    "source_sha256": fit_record["source_sha256"],
                                    "script_sha256": sha256(Path(__file__)),
                                    "artifact_sha256": sha256(target),
                                    "embedding_shape": list(raw.shape),
                                    "trial_count": 193}, indent=2) + "\n", encoding="utf-8")
    print(f"EXPORTED ED8 MSE {mode}: {target}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("fit", "export"))
    parser.add_argument("mode", choices=MODES)
    args = parser.parse_args()
    (fit if args.phase == "fit" else export)(args.mode)
