"""Recover frozen Conv-piVAE embeddings from already-trained checkpoints.

This utility never calls model.fit or an optimizer. It is restricted to
canonical runs that completed the configured epoch budget and failed only at
the post-fit q(z|x) behavior-invariance assertion. It writes a new immutable
recovery run directory; it does not change the source failed run.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.models import Model

import rat_conv_pivae_fit as fit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "results" / "rat_conv_pivae_v1" / "runs"
INPUT_ROOT = PROJECT_ROOT / "results" / "rat_conv_pivae_v1" / "input_cache"
SOURCE_ROOT = PROJECT_ROOT / "CEBRA" / "third_party" / "pivae"
EXPECTED_POSTFIT_ERROR = "z_mean changed when only the behavior covariate changed"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def list_candidates() -> list[Path]:
    candidates = []
    for manifest_path in RUN_ROOT.rglob("run.json"):
        manifest = load_json(manifest_path)
        if (manifest.get("status") == "failed"
                and manifest.get("error") == EXPECTED_POSTFIT_ERROR):
            candidates.append(manifest_path)
    return sorted(candidates)


def validate_parent(manifest_path: Path) -> tuple[dict, Path, dict]:
    manifest = load_json(manifest_path)
    if manifest.get("status") != "failed" or manifest.get("error") != EXPECTED_POSTFIT_ERROR:
        raise RuntimeError(f"Not an eligible post-fit recovery source: {manifest_path}")
    run_dir = manifest_path.parent
    checkpoint = run_dir / "best_validation.weights.h5"
    history = run_dir / "training_history.csv"
    summary = run_dir / "model_summary.txt"
    for artifact in (checkpoint, history, summary):
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise RuntimeError(f"Required trained artifact missing/empty: {artifact}")
    if manifest.get("config", {}).get("epochs") != 600:
        raise RuntimeError(f"Unexpected epoch budget in source manifest: {manifest_path}")
    history_rows = sum(1 for _ in history.open(encoding="utf-8")) - 1
    if history_rows != manifest["config"]["epochs"]:
        raise RuntimeError(f"History has {history_rows} epochs, expected {manifest['config']['epochs']}: {history}")
    input_manifest = INPUT_ROOT / "INPUT_BUNDLE_MANIFEST.json"
    if fit.sha256(input_manifest) != manifest["input_manifest_sha256"]:
        raise RuntimeError(f"Input-bundle manifest changed since training: {input_manifest}")
    bundle_manifest = load_json(input_manifest)
    rat = manifest["config"]["rat"]
    fold = int(manifest["config"]["fold"])
    rat_entry = next(item for item in bundle_manifest["bundles"] if item["rat"] == rat)
    split_entries = {item["split"]: item for item in rat_entry["splits"] if item["fold"] == fold}
    expected_inputs = {split: split_entries[split]["bundle_file_sha256"] for split in fit.SPLITS}
    if expected_inputs != manifest["parent_input_hashes"]:
        raise RuntimeError(f"Parent input hashes differ from the frozen bundle manifest: {manifest_path}")
    return manifest, run_dir, {"checkpoint": checkpoint, "history": history, "summary": summary,
                               "input_manifest": input_manifest, "split_entries": split_entries}


def allocate_recovery_dir(parent: Path, source_run_id: str) -> tuple[str, Path]:
    base = f"{source_run_id}-embedding-recovery-v1"
    attempt = 1
    while True:
        run_id = base if attempt == 1 else f"{base}-attempt{attempt:02d}"
        target = parent / run_id
        if not target.exists():
            return run_id, target
        existing = target / "run.json"
        if existing.is_file():
            record = load_json(existing)
            if record.get("status") == "complete" and record.get("recovery", {}).get("source_run_id") == source_run_id:
                expected = record.get("artifact_hashes", {})
                names = {"checkpoint_sha256": "best_validation.weights.h5",
                         "history_sha256": "training_history.csv",
                         "model_summary_sha256": "model_summary.txt"}
                names.update({f"embedding_{split}_sha256": f"embedding_{split}.npz" for split in fit.SPLITS})
                if all(digest and (target / names[key]).is_file()
                       and fit.sha256(target / names[key]) == digest for key, digest in expected.items() if key in names):
                    print(f"REUSE {run_id}: recovered embeddings hash-verified", flush=True)
                    return "", target
        attempt += 1


def recover_one(manifest_path: Path) -> dict:
    source, source_dir, files = validate_parent(manifest_path)
    c = source["config"]
    rat, fold, seed = c["rat"], int(c["fold"]), int(c["seed"])
    run_id, target = allocate_recovery_dir(source_dir.parent, source["run_id"])
    if not run_id:
        return load_json(target / "run.json")
    target.mkdir(parents=False, exist_ok=False)
    recovery_record = {
        "run_id": run_id,
        "status": "started",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": c,
        "config_sha256": source["config_sha256"],
        "parent_input_hashes": source["parent_input_hashes"],
        "input_manifest_sha256": source["input_manifest_sha256"],
        "recovery": {"source_run_id": source["run_id"],
                     "source_run_manifest": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
                     "source_run_manifest_sha256": fit.sha256(manifest_path),
                     "source_checkpoint_sha256": fit.sha256(files["checkpoint"]),
                     "source_history_sha256": fit.sha256(files["history"]),
                     "operation": "post-fit embedding extraction only; no optimizer/model.fit calls",
                     "invariance_check": "same x batch and shape evaluated with dummy-u versus true-u; q(z|x) must agree"},
    }
    atomic_json(target / "run.json", recovery_record)
    try:
        input_manifest = load_json(files["input_manifest"])
        rat_entry = next(item for item in input_manifest["bundles"] if item["rat"] == rat)
        split_entries = {item["split"]: item for item in rat_entry["splits"] if item["fold"] == fold}
        first = fit.load_bundle(INPUT_ROOT / rat / f"fold_{fold}" / "train" / "input.npz",
                                split_entries["train"]["bundle_file_sha256"])
        tf.keras.backend.clear_session()
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        sys.path.insert(0, str(SOURCE_ROOT))
        from pivae_code import conv_pi_vae
        conv_pi_vae.squeeze_func._name = f"conv_pivae_recovery_squeeze_{seed}"
        model = conv_pi_vae.conv_vae_mdl(
            dim_x=int(first["x"].shape[2]), dim_z=2, dim_u=3, time_window=10,
            gen_nodes=60, n_blk=2, mdl="poisson", disc=False, learning_rate=0.00025)
        model.load_weights(str(files["checkpoint"]))
        z_encoder = Model(inputs=model.inputs, outputs=model.outputs[6])
        conditional_encoder = Model(inputs=model.inputs, outputs=model.outputs[0])

        split_summaries = {}
        for split in fit.SPLITS:
            bundle_path = INPUT_ROOT / rat / f"fold_{fold}" / split / "input.npz"
            bundle = fit.load_bundle(bundle_path, split_entries[split]["bundle_file_sha256"])
            n_probe = min(32, len(bundle["x"]))
            probe_x = bundle["x"][:n_probe]
            probe_zero = np.zeros_like(bundle["u"][:n_probe], dtype=np.float32)
            probe_true = bundle["u"][:n_probe]
            # Both calls have exactly the same batch shape, avoiding the old
            # false mismatch between a full-split batch and a 32-row probe.
            z_zero = z_encoder.predict([probe_x, probe_zero], batch_size=256, verbose=0)
            z_true_u = z_encoder.predict([probe_x, probe_true], batch_size=256, verbose=0)
            max_abs_delta = float(np.max(np.abs(z_zero - z_true_u))) if n_probe else 0.0
            if not np.allclose(z_zero, z_true_u, rtol=1e-6, atol=1e-7):
                raise AssertionError(f"q(z|x) behavior-invariance failed for {split}; max_abs_delta={max_abs_delta}")

            dummy_u = np.zeros_like(bundle["u"], dtype=np.float32)
            z_mean = z_encoder.predict([bundle["x"], dummy_u], batch_size=256, verbose=0)
            post_mean = conditional_encoder.predict([bundle["x"], bundle["u"]], batch_size=256, verbose=0)
            if z_mean.shape != (len(bundle["x"]), 2) or post_mean.shape != (len(bundle["x"]), 2):
                raise RuntimeError(f"Embedding shape invariant failed for {split}: {z_mean.shape}, {post_mean.shape}")
            if not np.isfinite(z_mean).all() or not np.isfinite(post_mean).all():
                raise FloatingPointError(f"Non-finite embedding in {split}")
            output = target / f"embedding_{split}.npz"
            np.savez_compressed(output, z_mean_raw=z_mean.astype(np.float32),
                                post_mean_behavior_informed_raw=post_mean.astype(np.float32),
                                **{key: bundle[key] for key in ("u", "source_id", "trial_id", "time_id", "crosses_trial_boundary")})
            split_summaries[split] = {"rows": int(len(z_mean)), "embedding_shape": list(z_mean.shape),
                                      "boundary_crossing_rows": int(bundle["crosses_trial_boundary"].sum()),
                                      "embedding_sha256": fit.sha256(output),
                                      "q_z_x_behavior_invariance_probe_max_abs_delta": max_abs_delta,
                                      "bundle_sha256": split_entries[split]["bundle_file_sha256"]}

        for name in ("best_validation.weights.h5", "training_history.csv", "model_summary.txt"):
            shutil.copy2(files["checkpoint"] if name == "best_validation.weights.h5" else
                         files["history"] if name == "training_history.csv" else files["summary"], target / name)
        hashes = {"checkpoint_sha256": fit.sha256(target / "best_validation.weights.h5"),
                  "history_sha256": fit.sha256(target / "training_history.csv"),
                  "model_summary_sha256": fit.sha256(target / "model_summary.txt"),
                  **{f"embedding_{split}_sha256": split_summaries[split]["embedding_sha256"] for split in fit.SPLITS}}
        source_pivae = SOURCE_ROOT / "pivae_code"
        recovery_record.update({
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "best_epoch": int(np.argmin(np.genfromtxt(target / "training_history.csv", delimiter=",", names=True)["val_loss"])) + 1,
            "epochs_completed": int(sum(1 for _ in (target / "training_history.csv").open(encoding="utf-8")) - 1),
            "trainable_parameters": int(model.count_params()),
            "device": [tf.config.experimental.get_device_details(gpu).get("device_name", gpu.name)
                       for gpu in tf.config.list_physical_devices("GPU")],
            "tensorflow_version": tf.__version__,
            "pivae_source_hashes": {"runner_sha256": fit.sha256(Path(fit.__file__)),
                                    "recovery_runner_sha256": fit.sha256(Path(__file__)),
                                    "conv_pi_vae_py_sha256": fit.sha256(source_pivae / "conv_pi_vae.py"),
                                    "pi_vae_py_sha256": fit.sha256(source_pivae / "pi_vae.py"),
                                    "task_py_sha256": fit.sha256(SOURCE_ROOT / "task.py")},
            "split_embeddings": split_summaries,
            "artifact_hashes": hashes,
            "scientific_note": "Training/checkpoint/history are copied byte-for-byte from the failed post-fit run. No training was performed. z_mean is q(z|x); post_mean uses u and is not label-free evidence.",
            "recovery": {**recovery_record["recovery"], "status": "complete", "optimizer_updates": 0},
        })
        atomic_json(target / "run.json", recovery_record)
        print(f"RECOVERED {run_id}: embeddings extracted from existing checkpoint; no refit", flush=True)
        return recovery_record
    except Exception as exc:
        recovery_record.update({"status": "failed", "failed_utc": datetime.now(timezone.utc).isoformat(),
                                "error_type": type(exc).__name__, "error": str(exc)})
        atomic_json(target / "run.json", recovery_record)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="List eligible sources and verify checkpoint/history/input hashes only")
    parser.add_argument("--run-id", action="append", help="Source failed run ID; may be repeated. Default: all eligible post-fit failures")
    args = parser.parse_args()
    candidates = list_candidates()
    if args.run_id:
        candidates = [path for path in candidates if load_json(path)["run_id"] in args.run_id]
        missing = sorted(set(args.run_id) - {load_json(path)["run_id"] for path in candidates})
        if missing:
            raise SystemExit(f"No eligible post-fit failed source found for run IDs: {missing}")
    for path in candidates:
        record, run_dir, files = validate_parent(path)
        print(f"ELIGIBLE {record['run_id']} checkpoint={fit.sha256(files['checkpoint'])} history={fit.sha256(files['history'])}", flush=True)
    if args.dry_run:
        print(f"DRY RUN: {len(candidates)} sources; no embeddings or model fits executed", flush=True)
        return
    for path in candidates:
        recover_one(path)


if __name__ == "__main__":
    main()
