"""Resumable 6-seed conv-piVAE fit stage for the isolated rat paper campaign.

This wrapper reuses the existing, hash-verified 10-bin input cache and the
single-fit implementation in ``rat_conv_pivae_fit.py``. It writes to a new
campaign-specific output directory and never touches the older 600-epoch runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rat_conv_pivae_fit as pivae


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "results" / "rat_conv_pivae_v1" / "input_cache"
OUTPUT_ROOT = ROOT / "results" / "rat_paper_fig1e_ed4_6seed_v1" / "pivae"
SEEDS = tuple(range(42, 48))
EPOCHS = 1000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def input_inventory() -> dict:
    manifest_path = INPUT_ROOT / "INPUT_BUNDLE_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing frozen piVAE input manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {}
    source_indices = {}
    trial_indices = {}
    window_source_indices = {}
    for rat in pivae.RATS:
        rat_entry = next((item for item in manifest["bundles"] if item["rat"] == rat), None)
        if rat_entry is None:
            raise RuntimeError(f"Missing piVAE input bundle for {rat}")
        for fold in pivae.FOLDS:
            splits = {item["split"]: item for item in rat_entry["splits"] if item["fold"] == fold}
            if set(splits) != {"train", "valid", "test"}:
                raise RuntimeError(f"Incomplete piVAE split bundle: {rat}/fold_{fold}")
            for split, item in splits.items():
                path = INPUT_ROOT / rat / f"fold_{fold}" / split / "input.npz"
                if not path.is_file() or sha256(path) != item["bundle_file_sha256"]:
                    raise RuntimeError(f"PiVAE input hash mismatch: {path}")
                with np.load(path, allow_pickle=False) as bundle:
                    source_id = np.asarray(bundle["source_id"], dtype=np.int64)
                    trial_id = np.asarray(bundle["trial_id"], dtype=np.int64)
                    window_source_id = np.asarray(bundle["window_source_id"], dtype=np.int64)
                    if len(source_id) != len(bundle["x"]) or len(trial_id) != len(source_id):
                        raise RuntimeError(f"PiVAE source-ID cardinality mismatch: {path}")
                    if len(np.unique(source_id)) != len(source_id):
                        raise RuntimeError(f"Repeated source IDs in PiVAE split: {path}")
                key = f"{rat}/fold_{fold}/{split}"
                source_indices[key] = source_id
                trial_indices[key] = trial_id
                window_source_indices[key] = window_source_id
                item["_source_id_sha256"] = hashlib.sha256(source_id.tobytes()).hexdigest()
                item["_trial_id_sha256"] = hashlib.sha256(trial_id.tobytes()).hexdigest()
                item["_window_source_id_sha256"] = hashlib.sha256(window_source_id.tobytes()).hexdigest()
            entries[f"{rat}/fold_{fold}"] = {
                split: {
                    "bundle_sha256": item["bundle_file_sha256"],
                    "source_id_sha256": item["_source_id_sha256"],
                    "trial_id_sha256": item["_trial_id_sha256"],
                    "window_source_id_sha256": item["_window_source_id_sha256"],
                    "rows": item.get("rows"),
                    "trials": int(len(np.unique(trial_indices[f"{rat}/fold_{fold}/{split}"]))),
                    "crosses_trial_boundary_rows": int(np.load(
                        INPUT_ROOT / rat / f"fold_{fold}" / split / "input.npz",
                        allow_pickle=False,
                    )["crosses_trial_boundary"].sum()),
                }
                for split, item in splits.items()
            }
            split_names = [f"{rat}/fold_{fold}/{split}" for split in ("train", "valid", "test")]
            for i, left in enumerate(split_names):
                for right in split_names[i + 1:]:
                    if np.intersect1d(source_indices[left], source_indices[right]).size:
                        raise RuntimeError(f"Source-row leakage across PiVAE splits: {left}, {right}")
                    if np.intersect1d(trial_indices[left], trial_indices[right]).size:
                        raise RuntimeError(f"Trial leakage across PiVAE splits: {left}, {right}")
                    if np.intersect1d(window_source_indices[left].ravel(), window_source_indices[right].ravel()).size:
                        raise RuntimeError(f"Window-row leakage across PiVAE splits: {left}, {right}")
    return {
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(manifest_path),
        "folds": entries,
        "_source_indices": source_indices,
        "_trial_indices": trial_indices,
    }


def completed_run(rat: str, fold: int, seed: int) -> dict | None:
    parent = OUTPUT_ROOT / "runs" / rat / f"fold_{fold}" / f"seed_{seed}"
    if not parent.is_dir():
        return None
    for run_path in parent.glob("*/run.json"):
        record = json.loads(run_path.read_text(encoding="utf-8"))
        config = record.get("config", {})
        if (record.get("status") != "complete" or config.get("epochs") != EPOCHS
                or config.get("latent_dim") != 2 or config.get("time_window_bins") != 10):
            continue
        artifacts = record.get("artifact_hashes", {})
        expected = {
            "best_validation.weights.h5": artifacts.get("checkpoint_sha256"),
            "training_history.csv": artifacts.get("history_sha256"),
            "model_summary.txt": artifacts.get("model_summary_sha256"),
        }
        for split in pivae.SPLITS:
            expected[f"embedding_{split}.npz"] = artifacts.get(f"embedding_{split}_sha256")
        if all(digest and (run_path.parent / name).is_file()
               and sha256(run_path.parent / name) == digest for name, digest in expected.items()):
            return record
    return None


def main() -> None:
    global INPUT_ROOT, OUTPUT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-new-fits", type=int, default=None)
    args = parser.parse_args()

    INPUT_ROOT = args.input_root.resolve()
    OUTPUT_ROOT = args.output_root.resolve()
    inventory = input_inventory()
    index_arrays = {}
    for index_type in ("source", "trial"):
        records = inventory.pop(f"_{index_type}_indices")
        for key, values in records.items():
            array_key = f"{key.replace('/', '__')}__{index_type}_id"
            index_arrays[array_key] = values
    grid = [(rat, fold, seed) for seed in SEEDS for fold in pivae.FOLDS for rat in pivae.RATS]
    existing = {(rat, fold, seed): completed_run(rat, fold, seed) for rat, fold, seed in grid}
    todo = [item for item in grid if existing[item] is None]
    print(json.dumps({
        "campaign": "rat_paper_fig1e_ed4_6seed_v1",
        "method": "conv-piVAE",
        "status": "dry_run" if args.dry_run else "ready_to_execute",
        "planned_fits": len(grid),
        "verified_reusable_fits": len(grid) - len(todo),
        "remaining_fits": len(todo),
        "rats": list(pivae.RATS),
        "folds": list(pivae.FOLDS),
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "latent_dim": 2,
        "batch_size": 200,
        "learning_rate": 0.00025,
        "input_inventory": inventory,
        "output_root": str(OUTPUT_ROOT),
    }, indent=2), flush=True)
    if args.dry_run:
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    split_index_path = OUTPUT_ROOT / "SPLIT_INDICES.npz"
    if split_index_path.exists():
        if sha256(split_index_path) != json.loads(
                (OUTPUT_ROOT / "SPLIT_INDEX_MANIFEST.json").read_text(encoding="utf-8")
        )["artifact_sha256"]:
            raise RuntimeError(f"Existing PiVAE split-index hash mismatch: {split_index_path}")
    else:
        np.savez_compressed(split_index_path, **index_arrays)
        atomic_json(OUTPUT_ROOT / "SPLIT_INDEX_MANIFEST.json", {
            "campaign": "rat_paper_fig1e_ed4_6seed_v1",
            "method": "conv-piVAE",
            "source_manifest_sha256": inventory["manifest_sha256"],
            "artifact": split_index_path.name,
            "artifact_sha256": sha256(split_index_path),
            "indices": inventory["folds"],
            "note": "Source-row and trial-ID lists for train/valid/test input bundles; pairwise checks found no row, trial, or 10-bin window-source overlap across splits.",
        })
    lock = OUTPUT_ROOT / "RUNNING.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()}))
    except FileExistsError as exc:
        raise RuntimeError(f"piVAE campaign lock already exists: {lock}") from exc

    state_path = OUTPUT_ROOT / "CAMPAIGN_STATE.json"
    state = {
        "campaign": "rat_paper_fig1e_ed4_6seed_v1",
        "method": "conv-piVAE",
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "planned_fits": len(grid),
        "completed_fits": len(grid) - len(todo),
        "failed_fits": 0,
        "epochs": EPOCHS,
        "input_inventory": inventory,
        "records": {},
    }
    for key, record in existing.items():
        if record is not None:
            state["records"][f"{key[0]}/fold_{key[1]}/seed_{key[2]}"] = {
                "status": "reused_verified",
                "run_id": record["run_id"],
                "config_sha256": record.get("config_sha256"),
            }
    atomic_json(state_path, state)
    started = 0
    try:
        for rat, fold, seed in todo:
            if args.max_new_fits is not None and started >= args.max_new_fits:
                state["status"] = "paused_at_max_new_fits"
                break
            key = f"{rat}/fold_{fold}/seed_{seed}"
            try:
                result = pivae.one_fit(ROOT, INPUT_ROOT, OUTPUT_ROOT, rat, fold, seed, EPOCHS)
                state["records"][key] = {
                    "status": "complete",
                    "run_id": result["run_id"],
                    "config_sha256": result.get("config_sha256"),
                }
                state["completed_fits"] += 1
                print(f"COMPLETE {key} {result['run_id']}", flush=True)
            except Exception as exc:
                state["records"][key] = {"status": "failed", "error": repr(exc)}
                state["failed_fits"] += 1
                print(f"FAILED {key}: {type(exc).__name__}: {exc}", flush=True)
            started += 1
            state["updated_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(state_path, state)
        else:
            state["status"] = "complete" if state["completed_fits"] == len(grid) else "incomplete"
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(state_path, state)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
