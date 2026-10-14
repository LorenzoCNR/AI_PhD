"""Fit one official-code conv-piVAE and export its two Figure 3b emissions.

The authors' task.py exports output 0 (posterior with direction prior) and
output 6 (q(z|x), no inference-time label) from the SAME checkpoint. This
runner keeps that exact distinction; it does not pretend they are two fits.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from keras.callbacks import BackupAndRestore, CSVLogger, ModelCheckpoint
from keras.models import Model


PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data/monkey_reaching_preload_smth_40"
ROOT = PROJECT / "results/monkey_figure3_paper_v1"
FIT_DIR = ROOT / "stage01_fit_baselines/conv_pivae/seed_42"
EXPORT_DIR = ROOT / "stage02_embedding_baselines/conv_pivae/seed_42"
SEED = 42
EPOCHS = 600


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise FileExistsError(f"Existing immutable artifact differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load(split: str):
    source = DATA / f"active_{split}.jl"
    data = joblib.load(source)
    n = int(data["num_trials"])
    length = int(data["trial_len"])
    spikes = np.asarray(data["spikes"], dtype=np.float32)
    if length != 600 or spikes.shape != (n * length, 65) or not np.isfinite(spikes).all():
        raise ValueError(f"Invalid monkey source {source}")
    centers = np.clip(np.arange(length), 5, length - 5)
    local = centers[:, None] + np.arange(-5, 5)[None, :]
    rows = np.arange(n)[:, None, None] * length + local[None, :, :]
    x = spikes[rows].reshape(n * length, 10, 65)
    labels = np.repeat(np.asarray(data["movement_dir"], dtype=np.int32), length)[:, None]
    if labels.shape != (len(x), 1) or not np.isin(labels, np.arange(8)).all():
        raise ValueError("Invalid direction labels")
    return data, x, labels, sha256(source)


def batches(x, u):
    count = int(len(x) / 200)
    if count < 1:
        raise ValueError("Insufficient rows for official batch target 200")
    return np.array_split(x, count), np.array_split(u, count)


def loader(x_parts, u_parts):
    while True:
        for x, u in zip(x_parts, u_parts):
            yield ([x, u], None)


def model():
    source = PROJECT / "CEBRA/third_party/pivae"
    sys.path.insert(0, str(source))
    from pivae_code import conv_pi_vae
    # The vendored module defines this Lambda before clear_session; avoid a
    # generated-name collision without altering its tensor operation.
    conv_pi_vae.squeeze_func._name = "monkey_conv_pivae_shared_squeeze_42"
    return conv_pi_vae.conv_vae_mdl(dim_x=65, dim_z=4, dim_u=8,
                                     time_window=10, gen_nodes=60, n_blk=2,
                                     mdl="poisson", disc=True,
                                     learning_rate=0.0005)


def fit() -> None:
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = FIT_DIR / "FIT.json"
    checkpoint = FIT_DIR / "best_validation.weights.h5"
    history = FIT_DIR / "training_history.csv"
    source_hashes = {split: sha256(DATA / f"active_{split}.jl") for split in ("train", "valid")}
    config = {"source": "CEBRA/third_party/pivae/task.py MonkeyReachingActiveTask",
              "training_seed": SEED, "train": "active_train", "validation": "active_valid",
              "test_used_for_fit": False, "time_window": 10, "input_channels": 65,
              "latent_dimension": 4, "direction_categories": 8,
              "prior": "discrete direction", "gen_nodes": 60, "flow_blocks": 2,
              "observation": "poisson", "learning_rate": 0.0005,
              "batch_target": 200, "epochs": EPOCHS,
              "checkpoint": "best validation loss, no early stopping",
              "embedding_outputs": {"with_label": 0, "without_inference_label": 6}}
    if manifest.exists():
        prior = json.loads(manifest.read_text(encoding="utf-8"))
        if prior["config"] != config or prior["source_hashes"] != source_hashes:
            raise ValueError("Existing piVAE fit provenance changed")
        if sha256(checkpoint) != prior["checkpoint_sha256"] or sha256(history) != prior["history_sha256"]:
            raise ValueError("Existing piVAE fit artifacts changed")
        print("REUSE conv-piVAE fit: checkpoint/history verified", flush=True)
        return
    if checkpoint.exists() or history.exists():
        # An interrupted fit may be resumed by Keras BackupAndRestore only.
        if not (FIT_DIR / "training_backup").exists():
            raise FileExistsError("Partial piVAE output without resumable training backup")
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.backend.clear_session()
    _, x_train, u_train, _ = load("train")
    _, x_valid, u_valid, _ = load("valid")
    x_train_parts, u_train_parts = batches(x_train, u_train)
    x_valid_parts, u_valid_parts = batches(x_valid, u_valid)
    with (FIT_DIR / "model_summary.txt").open("w", encoding="utf-8") as stream:
        with contextlib.redirect_stdout(stream):
            vae = model()
    callbacks = [ModelCheckpoint(str(checkpoint), monitor="val_loss",
                                 save_best_only=True, save_weights_only=True, mode="min"),
                 CSVLogger(str(history), append=history.exists()),
                 BackupAndRestore(backup_dir=str(FIT_DIR / "training_backup"),
                                  save_freq="epoch", delete_checkpoint=False)]
    started = time.perf_counter()
    result = vae.fit_generator(loader(x_train_parts, u_train_parts),
                               steps_per_epoch=len(x_train_parts), epochs=EPOCHS,
                               verbose=2, validation_data=loader(x_valid_parts, u_valid_parts),
                               validation_steps=len(x_valid_parts), callbacks=callbacks,
                               workers=1, use_multiprocessing=False)
    if not checkpoint.is_file():
        raise RuntimeError("piVAE validation checkpoint missing")
    import csv
    with history.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EPOCHS:
        raise RuntimeError(f"Expected {EPOCHS} history epochs, found {len(rows)}")
    valid_losses = [float(row["val_loss"]) for row in rows]
    write_json(manifest, {"created_utc": datetime.now(timezone.utc).isoformat(),
                          "config": config, "source_hashes": source_hashes,
                          "script_sha256": sha256(Path(__file__)),
                          "vendored_model_sha256": sha256(PROJECT / "CEBRA/third_party/pivae/pivae_code/conv_pi_vae.py"),
                          "fit_seconds": time.perf_counter() - started,
                          "best_epoch": int(np.argmin(valid_losses)) + 1,
                          "best_validation_loss": min(valid_losses),
                          "checkpoint_sha256": sha256(checkpoint),
                          "history_sha256": sha256(history),
                          "tensorflow_version": tf.__version__,
                          "gpu": [gpu.name for gpu in tf.config.list_physical_devices("GPU")]})
    print(f"COMPLETE conv-piVAE best epoch {int(np.argmin(valid_losses)) + 1}", flush=True)


def export() -> None:
    manifest = FIT_DIR / "FIT.json"
    fit_record = json.loads(manifest.read_text(encoding="utf-8"))
    checkpoint = FIT_DIR / "best_validation.weights.h5"
    if sha256(checkpoint) != fit_record["checkpoint_sha256"]:
        raise ValueError("piVAE checkpoint hash changed")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = EXPORT_DIR / "all_trials.npz"
    provenance = EXPORT_DIR / "EMBEDDING.json"
    if provenance.exists():
        previous = json.loads(provenance.read_text(encoding="utf-8"))
        if previous["fit_manifest_sha256"] != sha256(manifest) or previous["artifact_sha256"] != sha256(output):
            raise ValueError("Existing piVAE embedding provenance changed")
        print("REUSE conv-piVAE embedding: hashes verified", flush=True)
        return
    if output.exists():
        raise FileExistsError("Partial piVAE export exists")
    data, x, u, source_hash = load("all")
    tf.keras.backend.clear_session()
    vae = model()
    vae.load_weights(str(checkpoint))
    with_label = Model(inputs=vae.inputs, outputs=vae.outputs[0])
    without_label = Model(inputs=vae.inputs, outputs=vae.outputs[6])
    with_parts = []
    without_parts = []
    for start in range(0, len(x), 256):
        stop = min(len(x), start + 256)
        with_parts.append(with_label.predict([x[start:stop], u[start:stop]], verbose=0))
        without_parts.append(without_label.predict([x[start:stop], np.zeros_like(u[start:stop])], verbose=0))
    with_raw = np.concatenate(with_parts).astype(np.float32)
    without_raw = np.concatenate(without_parts).astype(np.float32)
    if with_raw.shape != (len(x), 4) or without_raw.shape != (len(x), 4):
        raise ValueError("piVAE embedding shape mismatch")
    n = int(data["num_trials"])
    length = int(data["trial_len"])
    np.savez_compressed(output, with_label_raw=with_raw,
                        with_label_unit=with_raw / np.maximum(np.linalg.norm(with_raw, axis=1, keepdims=True), 1e-12),
                        without_label_raw=without_raw,
                        without_label_unit=without_raw / np.maximum(np.linalg.norm(without_raw, axis=1, keepdims=True), 1e-12),
                        direction=u[:, 0], position_xy=np.asarray(data["pos"], dtype=np.float32),
                        neural=np.asarray(data["spikes"], dtype=np.float32),
                        trial_id=np.repeat(np.arange(n), length),
                        time_id=np.tile(np.arange(length), n),
                        valid_mask=np.ones(len(x), dtype=bool),
                        split=np.asarray("full_sample_descriptive"))
    write_json(provenance, {"created_utc": datetime.now(timezone.utc).isoformat(),
                            "fit_manifest_sha256": sha256(manifest),
                            "active_all_source_sha256": source_hash,
                            "script_sha256": sha256(Path(__file__)),
                            "artifact_sha256": sha256(output),
                            "rows": len(x), "trial_count": n,
                            "with_label_means": "output 0: q(z|x,u) posterior mean",
                            "without_label_means": "output 6: q(z|x) encoder mean, same trained model",
                            "held_out_claim": False})
    print(f"EXPORTED conv-piVAE: {output}", flush=True)


def export_heldout() -> None:
    """Export q(z|x) from the frozen checkpoint for clean train/test decoding."""
    fit_manifest = FIT_DIR / "FIT.json"
    fit_record = json.loads(fit_manifest.read_text(encoding="utf-8"))
    checkpoint = FIT_DIR / "best_validation.weights.h5"
    if sha256(checkpoint) != fit_record["checkpoint_sha256"]:
        raise ValueError("piVAE checkpoint hash changed")
    split_manifest = PROJECT / "results/monkey_figure3_v1/stage00_data/DATA_AND_SPLIT_PROVENANCE.json"
    split_record = json.loads(split_manifest.read_text(encoding="utf-8"))
    tf.keras.backend.clear_session()
    vae = model()
    vae.load_weights(str(checkpoint))
    x_encoder = Model(inputs=vae.inputs, outputs=vae.outputs[6])
    for split in ("train", "valid", "test"):
        data, x, u, source_hash = load(split)
        expected_hash = split_record["source_file_hashes"][f"active_{split}.jl"]["sha256"]
        if source_hash != expected_hash:
            raise ValueError(f"Active {split} source hash mismatch")
        embedding_parts = []
        for start in range(0, len(x), 256):
            stop = min(len(x), start + 256)
            embedding_parts.append(x_encoder.predict([x[start:stop], np.zeros_like(u[start:stop])], verbose=0))
        raw = np.concatenate(embedding_parts).astype(np.float32)
        if raw.shape != (len(x), 4) or not np.isfinite(raw).all():
            raise ValueError(f"Invalid held-out piVAE embedding: {split} {raw.shape}")
        unit = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
        trial_ids = np.asarray(split_record["session_trial_indices_by_split"]["active"][split], dtype=np.int32)
        n, length = int(data["num_trials"]), int(data["trial_len"])
        if len(trial_ids) != n or length != 600:
            raise ValueError("piVAE split trial mapping mismatch")
        directory = PROJECT / f"results/monkey_extended8_paper_v1/stage02_embedding_heldout/conv_pivae/{split}/seed_{SEED}"
        target = directory / "embedding_and_trials.npz"
        provenance = directory / "EMBEDDING.json"
        if provenance.exists():
            old = json.loads(provenance.read_text(encoding="utf-8"))
            if old["artifact_sha256"] != sha256(target) or old["fit_manifest_sha256"] != sha256(fit_manifest):
                raise ValueError(f"Existing piVAE {split} embedding changed")
            print(f"REUSE conv-piVAE {split} embedding", flush=True)
            continue
        if target.exists():
            raise FileExistsError(f"Partial piVAE held-out export exists: {target}")
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, embedding_raw=raw, embedding_unit=unit,
                            neural=np.asarray(data["spikes"], dtype=np.float32),
                            position_xy=np.asarray(data["pos"], dtype=np.float32),
                            direction=np.repeat(np.asarray(data["movement_dir"], dtype=np.int8), length),
                            trial_id=np.repeat(trial_ids, length),
                            time_id=np.tile(np.arange(length, dtype=np.int16), n),
                            valid_mask=np.ones(len(raw), dtype=bool),
                            split=np.asarray(split))
        provenance.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(),
                                          "fit_manifest_sha256": sha256(fit_manifest),
                                          "checkpoint_sha256": fit_record["checkpoint_sha256"],
                                          "source_sha256": source_hash,
                                          "split_manifest_sha256": sha256(split_manifest),
                                          "script_sha256": sha256(Path(__file__)),
                                          "artifact_sha256": sha256(target),
                                          "embedding_shape": list(raw.shape),
                                          "embedding_definition": "q(z|x) encoder mean, no inference-time direction labels",
                                          "split": split}, indent=2) + "\n", encoding="utf-8")
        print(f"EXPORTED conv-piVAE {split}: {target}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("fit", "export", "export-heldout"))
    arguments = parser.parse_args()
    {"fit": fit, "export": export, "export-heldout": export_heldout}[arguments.phase]()
