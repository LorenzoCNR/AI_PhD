"""Reproducible, compartmentalized runner for the vendored rat conv-piVAE.

Training follows the local CEBRA conv-piVAE reference: 10-bin centered input,
2D latent, 3D continuous behavior covariate, Poisson likelihood, 60 generator
nodes, two flow blocks, Adam at 2.5e-4, batch-size target 200, and 600 epochs.
The best-validation checkpoint is used for frozen embeddings. Test rows are
not presented to fit/validation and are read only after fitting is complete.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import Model


RATS = ("achilles", "buddy", "cicero", "gatsby")
FOLDS = (0, 1, 2)
SEEDS = tuple(range(42, 47))
SPLITS = ("train", "valid", "test")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def batches(x: np.ndarray, u: np.ndarray, requested_batch_size: int):
    count = int(len(x) / requested_batch_size)
    if count < 1:
        raise ValueError(f"split has {len(x)} observations, fewer than batch target {requested_batch_size}")
    # Match the official notebook's np.array_split and infinite generator.
    return np.array_split(x, count), np.array_split(u, count)


def infinite_loader(x_batches, u_batches):
    while True:
        for xb, ub in zip(x_batches, u_batches):
            yield ([xb, ub], None)


def load_bundle(path: Path, expected_file_hash: str) -> dict[str, np.ndarray]:
    actual = sha256(path)
    if actual != expected_file_hash:
        raise ValueError(f"Input bundle SHA-256 mismatch for {path}: {actual} != {expected_file_hash}")
    with np.load(path, allow_pickle=False) as z:
        result = {key: z[key] for key in z.files}
    n = len(result["x"])
    for key in ("u", "source_id", "trial_id", "time_id", "window_source_id", "window_trial_id", "crosses_trial_boundary"):
        if len(result[key]) != n:
            raise ValueError(f"Input invariant failed: {key} has {len(result[key])} rows, x has {n}")
    if result["x"].ndim != 3 or result["x"].shape[1] != 10 or result["u"].shape != (n, 3):
        raise ValueError(f"Input shape invariant failed: x={result['x'].shape}, u={result['u'].shape}")
    return result


def configure_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def one_fit(project_root: Path, input_root: Path, output_root: Path,
            rat: str, fold: int, seed: int, epochs: int) -> dict:
    manifest_path = input_root / "INPUT_BUNDLE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rat_manifest = next(item for item in manifest["bundles"] if item["rat"] == rat)
    split_entries = {
        item["split"]: item for item in rat_manifest["splits"] if item["fold"] == fold
    }
    config = {
        "method": "conv-piVAE",
        "implementation": "vendored CEBRA third_party/pivae, local source hash in provenance",
        "rat": rat,
        "fold": fold,
        "seed": seed,
        "epochs": epochs,
        "time_window_bins": 10,
        "offset_left_bins": 5,
        "offset_right_bins": 5,
        "latent_dim": 2,
        "behavior_dim": 3,
        "gen_nodes": 60,
        "flow_blocks": 2,
        "observation_model": "poisson",
        "continuous_behavior_prior": True,
        "learning_rate": 0.00025,
        "requested_batch_size": 200,
        "checkpoint": "minimum validation loss; fixed epoch budget; no early stopping",
        "reproducibility": "Python/NumPy/TensorFlow RNGs seeded; GPU kernels are not guaranteed bitwise deterministic",
        "embedding_types": ["z_mean_q(z|x)_no_test_behavior", "post_mean_q(z|x,u)_behavior-informed"],
    }
    parents = {split: split_entries[split]["bundle_file_sha256"] for split in SPLITS}
    base_run_id = f"convpivae-{rat}-fold{fold}-seed{seed}-{json_hash({'config': config, 'parents': parents})[:12]}"
    run_parent = output_root / "runs" / rat / f"fold_{fold}" / f"seed_{seed}"
    attempt = 1
    while True:
        run_id = base_run_id if attempt == 1 else f"{base_run_id}-attempt{attempt:02d}"
        run_dir = run_parent / run_id
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        prior_manifest_path = run_dir / "run.json"
        prior = json.loads(prior_manifest_path.read_text(encoding="utf-8")) if prior_manifest_path.exists() else {}
        if prior.get("status") == "complete":
            artifacts = prior.get("artifact_hashes", {})
            expected = {
                "best_validation.weights.h5": artifacts.get("checkpoint_sha256"),
                "training_history.csv": artifacts.get("history_sha256"),
                "model_summary.txt": artifacts.get("model_summary_sha256"),
            }
            expected.update({f"embedding_{split}.npz": artifacts.get(f"embedding_{split}_sha256") for split in SPLITS})
            if all(value and (run_dir / name).is_file() and sha256(run_dir / name) == value for name, value in expected.items()):
                print(f"REUSE {run_id}: completed fit and artifacts hash-verified", flush=True)
                return {**prior, "reused": True}
        attempt += 1
    run_record = {
        "run_id": run_id,
        "status": "started",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "config_sha256": json_hash(config),
        "parent_input_hashes": parents,
        "input_manifest_sha256": sha256(manifest_path),
    }
    (run_dir / "run.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    code_path = Path(__file__).resolve()
    pivae_root = project_root / "CEBRA" / "third_party" / "pivae"
    sys.path.insert(0, str(pivae_root))
    try:
        from pivae_code import conv_pi_vae

        # Fit receives only train and validation bundles. Test is intentionally
        # loaded below, after the optimizer and validation checkpoint are done.
        train = load_bundle(input_root / rat / f"fold_{fold}" / "train" / "input.npz", parents["train"])
        valid = load_bundle(input_root / rat / f"fold_{fold}" / "valid" / "input.npz", parents["valid"])
        x_train, u_train = batches(train["x"], train["u"], 200)
        x_valid, u_valid = batches(valid["x"], valid["u"], 200)
        configure_seed(seed)
        tf.keras.backend.clear_session()
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                # The process may have already initialized the runtime.
                pass
        # The vendored source creates this shared Lambda at import time. A
        # clear_session resets Keras' generated-name counter, so namespace the
        # pre-existing layer before the source constructs its other Lambdas.
        # This changes no tensor operation or parameter; it only prevents a
        # duplicate-name graph-construction error under TensorFlow/Keras 2.10.
        conv_pi_vae.squeeze_func._name = f"conv_pivae_shared_squeeze_{seed}"
        model_summary_path = run_dir / "model_summary.txt"
        with model_summary_path.open("w", encoding="utf-8") as summary_stream:
            with contextlib.redirect_stdout(summary_stream):
                model = conv_pi_vae.conv_vae_mdl(
                    dim_x=int(train["x"].shape[2]), dim_z=2, dim_u=3,
                    time_window=10, gen_nodes=60, n_blk=2, mdl="poisson",
                    disc=False, learning_rate=0.00025,
                )
        best_weights = run_dir / "best_validation.weights.h5"
        callbacks = [
            ModelCheckpoint(str(best_weights), monitor="val_loss", save_best_only=True,
                            save_weights_only=True, mode="min", verbose=0),
            CSVLogger(str(run_dir / "training_history.csv"), append=False),
        ]
        print(f"START {run_id}: {rat} fold={fold} seed={seed}, epochs={epochs}, train_rows={len(train['x'])}, valid_rows={len(valid['x'])}", flush=True)
        started = time.perf_counter()
        history = model.fit_generator(
            infinite_loader(x_train, u_train),
            steps_per_epoch=len(x_train),
            epochs=epochs,
            verbose=0,
            validation_data=infinite_loader(x_valid, u_valid),
            validation_steps=len(x_valid),
            callbacks=callbacks,
            workers=1,
            use_multiprocessing=False,
        )
        fit_seconds = time.perf_counter() - started
        if not best_weights.is_file():
            raise RuntimeError("Validation checkpoint was not produced")
        best_epoch = int(np.argmin(history.history["val_loss"])) + 1
        best_valid_loss = float(np.min(history.history["val_loss"]))
        model.load_weights(str(best_weights))

        # Primary representation: q(z|x), not conditioned on held-out behavior.
        # The vendored Functional graph declares u as a required model input
        # even though output 6 (z_mean=q(z|x)) is mathematically x-only.
        # Keep both graph inputs, and feed a zero covariate for this output.
        z_encoder = Model(inputs=model.inputs, outputs=model.outputs[6])
        conditional_encoder = Model(inputs=model.inputs, outputs=model.outputs[0])
        inference_started = time.perf_counter()
        split_summaries = {}
        for split in SPLITS:
            bundle = train if split == "train" else valid if split == "valid" else load_bundle(
                input_root / rat / f"fold_{fold}" / "test" / "input.npz", parents["test"]
            )
            dummy_u = np.zeros_like(bundle["u"], dtype=np.float32)
            z_mean = z_encoder.predict([bundle["x"], dummy_u], batch_size=256, verbose=0)
            # Guard against accidental behavior dependence of the purported
            # no-label embedding. Compare the same x rows and batch shape so
            # this invariant is independent of batch-size numerical effects.
            n_probe = min(32, len(bundle["x"]))
            probe_x = bundle["x"][:n_probe]
            probe_zero_u = dummy_u[:n_probe]
            z_with_true_u = z_encoder.predict(
                [probe_x, bundle["u"][:n_probe]], batch_size=256, verbose=0
            )
            z_with_zero_u = z_encoder.predict([probe_x, probe_zero_u], batch_size=256, verbose=0)
            if not np.allclose(z_with_zero_u, z_with_true_u, rtol=1e-6, atol=1e-7):
                raise AssertionError("z_mean changed when only the behavior covariate changed")
            post_mean = conditional_encoder.predict([bundle["x"], bundle["u"]], batch_size=256, verbose=0)
            if z_mean.shape != (len(bundle["x"]), 2) or post_mean.shape != (len(bundle["x"]), 2):
                raise AssertionError(f"Embedding shape invariant failed for {split}: {z_mean.shape}, {post_mean.shape}")
            if not np.isfinite(z_mean).all() or not np.isfinite(post_mean).all():
                raise FloatingPointError(f"Non-finite embedding in {split}")
            np.savez_compressed(
                run_dir / f"embedding_{split}.npz",
                z_mean_raw=z_mean.astype(np.float32),
                post_mean_behavior_informed_raw=post_mean.astype(np.float32),
                **{key: bundle[key] for key in ("u", "source_id", "trial_id", "time_id", "crosses_trial_boundary")},
            )
            split_summaries[split] = {
                "rows": int(len(z_mean)),
                "boundary_crossing_rows": int(bundle["crosses_trial_boundary"].sum()),
                "embedding_shape": list(z_mean.shape),
                "embedding_sha256": sha256(run_dir / f"embedding_{split}.npz"),
            }
        inference_seconds = time.perf_counter() - inference_started
        device_names = []
        for device in tf.config.list_physical_devices("GPU"):
            details = tf.config.experimental.get_device_details(device)
            device_names.append(details.get("device_name", device.name))
        artifact_hashes = {
            "checkpoint_sha256": sha256(best_weights),
            "history_sha256": sha256(run_dir / "training_history.csv"),
            "model_summary_sha256": sha256(model_summary_path),
            **{f"embedding_{split}_sha256": split_summaries[split]["embedding_sha256"] for split in SPLITS},
        }
        run_record.update({
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "best_epoch": best_epoch,
            "best_validation_loss": best_valid_loss,
            "epochs_completed": int(len(history.history["loss"])),
            "fit_seconds": fit_seconds,
            "inference_seconds_all_splits": inference_seconds,
            "trainable_parameters": int(model.count_params()),
            "device": device_names,
            "tensorflow_version": tf.__version__,
            "keras_version": tf.keras.__version__ if hasattr(tf.keras, "__version__") else "2.10.x",
            "pivae_source_hashes": {
                "runner_sha256": sha256(code_path),
                "conv_pi_vae_py_sha256": sha256(pivae_root / "pivae_code" / "conv_pi_vae.py"),
                "pi_vae_py_sha256": sha256(pivae_root / "pivae_code" / "pi_vae.py"),
                "task_py_sha256": sha256(pivae_root / "task.py"),
            },
            "split_embeddings": split_summaries,
            "artifact_hashes": artifact_hashes,
            "scientific_note": "z_mean is the behavior-independent q(z|x) representation; post_mean uses u and must not be used as label-free decoding evidence.",
        })
        (run_dir / "run.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
        print(f"COMPLETE {run_id}: best epoch={best_epoch}/{epochs}, val={best_valid_loss:.6g}, fit={fit_seconds:.1f}s", flush=True)
        return run_record
    except Exception as exc:
        run_record.update({
            "status": "failed",
            "failed_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        (run_dir / "run.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
        print(f"FAILED {run_id}: {type(exc).__name__}: {exc}", flush=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rat", choices=RATS, default="achilles")
    parser.add_argument("--fold", type=int, choices=FOLDS, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--smoke", action="store_true", help="Run only Achilles/fold0/seed42 for 3 epochs in a separate qualification folder")
    parser.add_argument("--campaign", action="store_true", help="Run all 4 rats x 3 folds x seeds 42-46")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if args.smoke:
        output_root = output_root / "qualification_smoke"
        one_fit(project_root, input_root, output_root, "achilles", 0, 42, 3)
        return
    if args.campaign:
        grid = ((rat, fold, seed) for rat in RATS for fold in FOLDS for seed in SEEDS)
    else:
        grid = [(args.rat, args.fold, args.seed)]
    completed = 0
    failures = []
    for rat, fold, seed in grid:
        try:
            one_fit(project_root, input_root, output_root, rat, fold, seed, args.epochs)
            completed += 1
        except Exception as exc:
            failures.append({"rat": rat, "fold": fold, "seed": seed, "error": repr(exc)})
            # Failures are immutable run records; continue to independent fits.
            continue
    summary = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "completed_this_invocation": completed,
        "failures_this_invocation": failures,
        "campaign_size": len(RATS) * len(FOLDS) * len(SEEDS),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / f"INVOCATION_SUMMARY_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"INVOCATION FINISHED: completed={completed}, failed={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
