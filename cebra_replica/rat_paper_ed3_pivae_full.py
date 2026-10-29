"""Full-recording rat conv-piVAE fits for ED3 / shared Figure 1d embeddings.

These are separate from the historical held-out fold fits. Training sees all
recording rows; q(z|x) is the label-free representation, while q(z|x,u) is
saved separately and explicitly flagged as behavior-informed inference.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RATS = ("achilles", "buddy", "cicero", "gatsby")
SEED = 42
DEFAULT_OUTPUT = ROOT / "results" / "rat_paper_replication_2023_v1" / "ED3" / "pivae_full_recording"
PROJECT_PIVAE = ROOT / "CEBRA" / "third_party" / "pivae"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def config_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def infinite_loader(x_batches, u_batches):
    while True:
        for xb, ub in zip(x_batches, u_batches):
            yield ([xb, ub], None)


def batches(x: np.ndarray, u: np.ndarray, requested_batch_size: int):
    count = int(len(x) / requested_batch_size)
    if count < 1:
        raise ValueError(f"Need >= {requested_batch_size} observations; got {len(x)}")
    return np.array_split(x, count), np.array_split(u, count)


def configure_seed(seed: int) -> None:
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_full_input(rat: str):
    import joblib
    import cebra
    from cebra.data.datatypes import Offset
    from cebra.datasets.hippocampus import SingleRatDataset

    source = ROOT / "data" / "rat_hippocampus" / f"{rat}.jl"
    raw = joblib.load(source)
    ds = SingleRatDataset(name=rat, root=ROOT / "data", download=False)
    if len(ds) != len(raw["spikes"]):
        raise ValueError(f"CEBRA loader/source row mismatch for {rat}: {len(ds)} vs {len(raw['spikes'])}")
    ds.offset = Offset(5, 5)
    import torch
    windows = ds[torch.as_tensor(np.arange(len(ds)), dtype=torch.long)].detach().cpu().numpy().transpose(0, 2, 1)
    behavior = ds.continuous_index.detach().cpu().numpy().astype(np.float32, copy=False)
    source_ids = np.arange(len(ds), dtype=np.int64)
    if windows.ndim != 3 or windows.shape[1] != 10 or behavior.shape != (len(ds), 3):
        raise ValueError(f"Unexpected full piVAE input for {rat}: x={windows.shape}, u={behavior.shape}")
    if not np.isfinite(windows).all() or not np.isfinite(behavior).all():
        raise FloatingPointError(f"Non-finite piVAE input for {rat}")
    return (windows.astype(np.float32, copy=False), behavior, source_ids,
            np.asarray(raw["position"], dtype=np.float32), source, cebra.__version__)


def prepare_input(output: Path, rat: str) -> dict:
    """Export exact full-recording CEBRA windows for use in a TF-only env."""
    x, u, source_ids, position, source_path, cebra_version = load_full_input(rat)
    input_dir = output / "input_cache" / rat
    input_path = input_dir / "full_recording.npz"
    manifest_path = input_dir / "manifest.json"
    expected = dict(rat=rat, rows=len(x), window_shape=list(x.shape[1:]),
                    behavior_shape=list(u.shape[1:]), source_data_sha256=sha256(source_path),
                    cebra_version=cebra_version, time_window_bins=10,
                    offset_bins=[-5, 4], bin_ms=25,
                    exporter_sha256=sha256(Path(__file__)))
    if manifest_path.exists():
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        if record.get("config") != expected or not input_path.is_file():
            raise RuntimeError(f"Full piVAE input cache provenance mismatch: {input_dir}")
        if sha256(input_path) != record.get("input_sha256"):
            raise RuntimeError(f"Full piVAE input cache hash mismatch: {input_path}")
        return record
    if input_dir.exists() and any(input_dir.iterdir()):
        raise RuntimeError(f"Partial piVAE input cache requires inspection: {input_dir}")
    input_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(input_path, x=x, u=u, source_id=source_ids, position=position)
    record = dict(config=expected, input_sha256=sha256(input_path),
                  created_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(manifest_path, record)
    print(f"PREPARED piVAE input {rat}: {len(x)} rows -> {input_path}", flush=True)
    return record


def load_cached_input(output: Path, rat: str):
    input_dir = output / "input_cache" / rat
    input_path = input_dir / "full_recording.npz"
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.is_file() or not input_path.is_file():
        raise RuntimeError(f"Missing piVAE full-recording input; prepare it with the CEBRA environment first: {input_dir}")
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256(input_path) != record.get("input_sha256"):
        raise RuntimeError(f"piVAE full-recording input hash mismatch: {input_path}")
    with np.load(input_path, allow_pickle=False) as stored:
        x, u = stored["x"].copy(), stored["u"].copy()
        source_ids, position = stored["source_id"].copy(), stored["position"].copy()
    config = record["config"]
    if len(x) != config["rows"] or x.shape[1:] != tuple(config["window_shape"]):
        raise RuntimeError(f"piVAE full-recording input shape mismatch: {input_path}")
    source_path = ROOT / "data" / "rat_hippocampus" / f"{rat}.jl"
    if sha256(source_path) != config["source_data_sha256"]:
        raise RuntimeError(f"Raw piVAE source changed since input export: {source_path}")
    return x, u, source_ids, position, source_path, config["cebra_version"], record


def make_valid_mask(position: np.ndarray, n_rows: int) -> np.ndarray:
    direction_change_idx = np.where(position[1:, 1] != position[:-1, 1])[0]
    bounds = np.append(np.insert(direction_change_idx[1::2], 0, 0), n_rows).astype(np.int64)
    trial_id = np.searchsorted(bounds[1:], np.arange(n_rows), side="right")
    valid = np.zeros(n_rows, dtype=bool)
    centers = np.arange(5, n_rows - 4, dtype=np.int64)
    support = centers[:, None] + np.arange(-5, 5, dtype=np.int64)[None, :]
    valid[centers] = (trial_id[support] == trial_id[centers, None]).all(axis=1)
    return valid


def model_source_hashes() -> dict:
    files = {
        "runner": Path(__file__),
        "pivae_conv": PROJECT_PIVAE / "pivae_code" / "conv_pi_vae.py",
        "pivae_core": PROJECT_PIVAE / "pivae_code" / "pi_vae.py",
        "pivae_task": PROJECT_PIVAE / "task.py",
        "hippocampus": ROOT / "CEBRA" / "cebra" / "datasets" / "hippocampus.py",
        "base_dataset": ROOT / "CEBRA" / "cebra" / "data" / "base.py",
    }
    return {key: {"path": str(path.resolve()), "sha256": sha256(path)} for key, path in files.items()}


def one_fit(rat: str, output: Path, epochs: int = 1000, dry_run: bool = False) -> dict | None:
    if dry_run:
        x, u, source_ids, position, source_path, cebra_version = load_full_input(rat)
        input_record = {"config": {"source_data_sha256": sha256(source_path),
                                   "cebra_version": cebra_version},
                        "input_sha256": None}
    else:
        x, u, source_ids, position, source_path, cebra_version, input_record = load_cached_input(output, rat)
    x_batches, u_batches = batches(x, u, 200)
    config = {
        "figure": "ED3 / shared Figure 1d full-recording baseline",
        "method": "conv-piVAE",
        "rat": rat,
        "seed": SEED,
        "input_scope": "all rows of one complete rat recording; no train/validation/test split",
        "source_rows": int(len(x)),
        "input_window_shape": [10, int(x.shape[2])],
        "time_window_bins": 10,
        "offset_left_bins": 5,
        "offset_right_bins": 5,
        "sampling_period_ms": 25,
        "latent_dim": 2,
        "behavior_dim": 3,
        "gen_nodes": 60,
        "flow_blocks": 2,
        "observation_model": "poisson",
        "continuous_behavior_prior": True,
        "learning_rate": 0.00025,
        "requested_batch_size": 200,
        "batch_count_per_epoch": len(x_batches),
        "epochs": int(epochs),
        "checkpoint_selection": "fixed terminal epoch; no validation/early stopping because all rows train",
        "embedding_types": ["z_mean_q(z|x)_no_behavior_at_inference",
                            "post_mean_q(z|x,u)_behavior_informed"],
        "window_edge_policy": "CEBRA SingleRatDataset Offset(5,5) expansion over the complete recording",
        "cebra_version": cebra_version,
        "source_data_sha256": input_record["config"]["source_data_sha256"],
        "full_input_manifest_sha256": (sha256(output / "input_cache" / rat / "manifest.json")
                                        if not dry_run else None),
        "full_input_sha256": input_record["input_sha256"],
        "code_hashes": model_source_hashes(),
    }
    dest = output / rat / f"seed_{SEED}"
    print(json.dumps({"config": config, "output": str(dest)}, indent=2), flush=True)
    if dry_run:
        return None
    signature = config_hash(config)
    manifest_path = dest / "manifest.json"
    if manifest_path.exists():
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        if record.get("signature") != signature:
            raise RuntimeError(f"Completed piVAE signature mismatch: {dest}")
        for name, digest in record.get("artifact_hashes", {}).items():
            if sha256(dest / name) != digest:
                raise RuntimeError(f"Completed piVAE artifact hash mismatch: {dest / name}")
        return record
    if dest.exists() and any(dest.iterdir()) and not (dest / "tf_ckpt" / "checkpoint").exists():
        raise RuntimeError(f"Partial piVAE output without a resumable TF checkpoint: {dest}")
    dest.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf
    from keras.callbacks import CSVLogger
    from keras.models import Model

    sys.path.insert(0, str(PROJECT_PIVAE))
    from pivae_code import conv_pi_vae

    configure_seed(SEED)
    tf.keras.backend.clear_session()
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    conv_pi_vae.squeeze_func._name = f"conv_pivae_full_squeeze_{rat}_{SEED}"
    summary_path = dest / "model_summary.txt"
    with summary_path.open("w", encoding="utf-8") as summary_stream:
        with contextlib.redirect_stdout(summary_stream):
            model = conv_pi_vae.conv_vae_mdl(
                dim_x=int(x.shape[2]), dim_z=2, dim_u=3, time_window=10,
                gen_nodes=60, n_blk=2, mdl="poisson", disc=False,
                learning_rate=0.00025,
            )

    checkpoint_dir = dest / "tf_ckpt"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
    manager = tf.train.CheckpointManager(checkpoint, str(checkpoint_dir), max_to_keep=1)
    state_path = dest / "training_state.json"
    initial_epoch = 0
    if manager.latest_checkpoint:
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        if state.get("signature") != signature:
            raise RuntimeError(f"piVAE resume signature mismatch: {dest}")
        initial_epoch = int(state["completed_epochs"])
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        if initial_epoch >= epochs:
            raise RuntimeError("piVAE terminal epoch recorded but final manifest is missing; inspect before resuming")

    class EpochCheckpoint(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            completed = int(epoch + 1)
            manager.save(checkpoint_number=completed)
            atomic_json(state_path, {"signature": signature, "completed_epochs": completed,
                                     "checkpoint": manager.latest_checkpoint,
                                     "last_loss": float((logs or {}).get("loss", float("nan"))),
                                     "updated_utc": datetime.now(timezone.utc).isoformat()})

    history_path = dest / "training_history.csv"
    callbacks = [CSVLogger(str(history_path), append=initial_epoch > 0), EpochCheckpoint()]
    print(f"START piVAE {rat}: rows={len(x)}, epochs={initial_epoch}->{epochs}, device=TensorFlow auto", flush=True)
    started = time.perf_counter()
    history = model.fit_generator(
        infinite_loader(x_batches, u_batches), steps_per_epoch=len(x_batches),
        epochs=epochs, initial_epoch=initial_epoch, verbose=0,
        callbacks=callbacks, workers=1, use_multiprocessing=False,
    )
    fit_seconds = time.perf_counter() - started
    terminal_weights = dest / "terminal.weights.h5"
    model.save_weights(str(terminal_weights))

    # q(z|x) is the primary label-free embedding; provide zeros in u because
    # this graph names u as an input although output 6 is mathematically x-only.
    z_encoder = Model(inputs=model.inputs, outputs=model.outputs[6])
    posterior_encoder = Model(inputs=model.inputs, outputs=model.outputs[0])
    dummy_u = np.zeros_like(u, dtype=np.float32)
    z_mean = z_encoder.predict([x, dummy_u], batch_size=256, verbose=0)
    probe_n = min(32, len(x))
    z_true_u = z_encoder.predict([x[:probe_n], u[:probe_n]], batch_size=256, verbose=0)
    z_zero_u = z_encoder.predict([x[:probe_n], dummy_u[:probe_n]], batch_size=256, verbose=0)
    if not np.allclose(z_true_u, z_zero_u, rtol=1e-6, atol=1e-7):
        raise AssertionError("q(z|x) unexpectedly depends on behavior during inference")
    post_mean = posterior_encoder.predict([x, u], batch_size=256, verbose=0)
    if z_mean.shape != (len(x), 2) or post_mean.shape != (len(x), 2):
        raise RuntimeError(f"Unexpected piVAE embedding shapes: {z_mean.shape}, {post_mean.shape}")
    if not np.isfinite(z_mean).all() or not np.isfinite(post_mean).all():
        raise FloatingPointError("Non-finite piVAE embedding")
    embedding_path = dest / "embedding_full.npz"
    valid_mask = make_valid_mask(position, len(x))
    np.savez_compressed(embedding_path,
        z_mean_raw=z_mean.astype(np.float32),
        post_mean_behavior_informed_raw=post_mean.astype(np.float32),
        behavior=u, source_id=source_ids, valid_mask=valid_mask,
        split="full_recording", rat=rat, seed=SEED,
        q_z_x_is_label_free=True, post_mean_uses_behavior=True)
    artifacts = {path.name: sha256(path) for path in
                 (terminal_weights, history_path, summary_path, embedding_path)}
    device_names = []
    for gpu in tf.config.list_physical_devices("GPU"):
        details = tf.config.experimental.get_device_details(gpu)
        device_names.append(details.get("device_name", gpu.name))
    record = dict(signature=signature, created_utc=datetime.now(timezone.utc).isoformat(),
        config=config, epochs_completed=int(initial_epoch + len(history.history["loss"])),
        fit_seconds=fit_seconds, trainable_parameters=int(model.count_params()),
        device=device_names or ["CPU"], tensorflow_version=tf.__version__,
        keras_version=tf.keras.__version__ if hasattr(tf.keras, "__version__") else "2.10.x",
        rows=len(x), embedding_shape=list(z_mean.shape), valid_window_rows=int(valid_mask.sum()),
        artifact_hashes=artifacts,
        q_z_x_semantics="z_mean=q(z|x), behavior-independent at inference",
        posterior_semantics="post_mean=q(z|x,u), behavior-informed; descriptive only, not label-free decoding evidence")
    atomic_json(manifest_path, record)
    print(f"COMPLETE piVAE {rat}: {len(x)} full rows, {fit_seconds:.1f}s", flush=True)
    return record


def process_lock(output: Path) -> Path:
    lock = output / "RUNNING.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text(encoding="ascii").strip())
            os.kill(pid, 0)
        except OSError:
            lock.unlink()
        except ValueError as exc:
            raise RuntimeError(f"Unverifiable piVAE run lock requires inspection: {lock}") from exc
        else:
            raise RuntimeError(f"piVAE runner may still be active (PID {pid}): {lock}")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode("ascii"))
    os.close(descriptor)
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Export complete CEBRA windows here; run this in the CEBRA/PyTorch environment")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--max-new-fits", type=int, default=4)
    args = parser.parse_args()
    if sum((args.execute, args.dry_run, args.prepare_only)) > 1:
        parser.error("choose only one of --execute, --dry-run, or --prepare-only")
    if not args.execute and not args.dry_run and not args.prepare_only:
        args.dry_run = True
    if args.epochs < 1 or not 1 <= args.max_new_fits <= len(RATS):
        parser.error("epochs must be positive and max-new-fits must be 1..4")
    output = args.output.resolve()
    results_root = (ROOT / "results").resolve()
    if results_root not in output.parents:
        parser.error("output must be inside this project's results directory")
    if args.prepare_only:
        output.mkdir(parents=True, exist_ok=True)
        for rat in RATS:
            prepare_input(output, rat)
        return
    if args.dry_run:
        for rat in RATS:
            one_fit(rat, output, args.epochs, dry_run=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    lock = process_lock(output)
    state_path = output / "CAMPAIGN_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
        "planned": len(RATS), "records": {}, "created_utc": datetime.now(timezone.utc).isoformat()}
    completed_new = 0
    try:
        for rat in RATS:
            try:
                record = one_fit(rat, output, args.epochs)
            except BaseException as exc:
                state.setdefault("records", {})[rat] = {"status": "failed_or_interrupted", "error": repr(exc)}
                state.update(status="interrupted", current=rat, updated_utc=datetime.now(timezone.utc).isoformat())
                atomic_json(state_path, state)
                raise
            state.setdefault("records", {})[rat] = {
                "status": "complete", "manifest": str(output / rat / f"seed_{SEED}" / "manifest.json"),
                "signature": record["signature"], "embedding_sha256": record["artifact_hashes"]["embedding_full.npz"]}
            state.update(status="running", current=None, completed=sum(
                value.get("status") == "complete" for value in state["records"].values()),
                updated_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(state_path, state)
            completed_new += 1
            if completed_new >= args.max_new_fits:
                break
        total = sum(value.get("status") == "complete" for value in state["records"].values())
        state.update(status="complete" if total == len(RATS) else "requested_limit",
                     current=None, completed=total, completed_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(state_path, state)
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
