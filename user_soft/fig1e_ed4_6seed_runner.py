"""Resumable Soft Figure 1e / Extended Data Figure 4 six-seed runner.

Dry-run is read-only. Training is possible only with the explicit --execute flag.
Each fit calls the existing rat_runner.fit_cell API and keeps a condition- and
architecture-specific manifest, data lineage, state record, and lock.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from . import behavior_only_runner as behavior_runner
from . import rat_runner as rat_runner
from . import time_only_runner as time_runner
from . import transformer_runner as transformer_runner
from .data import config_hash, load_rows, sha256


ROOT = rat_runner.ROOT
OUTPUT_ROOT = ROOT / "results" / "rat_paper_fig1e_ed4_6seed_v1" / "soft"
CAMPAIGN_ID = "rat_paper_fig1e_ed4_soft_6seed_v1"
CONDITIONS = ("time_direction_position", "time_only", "direction_position")
ARCHITECTURES = ("cnn", "transformer")
VERSIONS = ("soft_v1_stable", "soft_v1_frozen")
SEEDS = tuple(range(42, 48))
FOLDS = (0, 1, 2)
OUTPUT_DIMENSION = 3
BATCH_SIZE = 512
LEARNING_RATE = 3e-4
MAX_UPDATES = 10000
EMBEDDING_TEMPERATURE = 1.0
METADATA_TEMPERATURE = 0.5
PLANNED_FITS = 864
PLAN_PATH = OUTPUT_ROOT / "CAMPAIGN_PLAN.json"
STATE_PATH = OUTPUT_ROOT / "CAMPAIGN_STATE.json"
CAMPAIGN_LOCK = OUTPUT_ROOT / "RUNNING.lock"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    """Publish JSON atomically, using a unique temporary file per writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def iter_configs():
    """Stable, explicit cell grid; no generic/joint condition label."""
    for rat, fold, condition, architecture, version, seed in itertools.product(
        rat_runner.RATS, FOLDS, CONDITIONS, ARCHITECTURES, VERSIONS, SEEDS
    ):
        yield config_for(condition, architecture, rat, fold, seed, version)


def config_for(condition: str, architecture: str, rat: str, fold: int,
               seed: int, version: str) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown explicit condition: {condition}")
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}")
    if version not in VERSIONS:
        raise ValueError(f"Unknown Soft version: {version}")

    factory = rat_runner.frozen_config if architecture == "cnn" else transformer_runner.frozen_config
    config = factory(rat, fold, seed, version, updates=MAX_UPDATES)
    if condition == "time_direction_position":
        config.update(
            condition=condition,
            metadata_specs=[dict(spec) for spec in rat_runner.SPECS],
            metadata_supervision="time_and_direction_position",
        )
    elif condition == "time_only":
        config.update(
            condition=condition,
            metadata_specs=[dict(spec) for spec in time_runner.SPECS],
            temporal_coordinate="official trial-relative 25 ms bin",
            cross_trial_candidates=True,
            metadata_supervision="local_time_only_no_behavior",
        )
    else:
        behavior_config = behavior_runner.frozen_config(
            architecture, rat, fold, seed, updates=MAX_UPDATES
        )
        config.update(behavior_config)
        config.update(
            condition=condition,
            metadata_specs=[dict(spec) for spec in behavior_runner.SPECS],
            metadata_supervision="raw_direction_and_position_no_time",
        )

    config.update(
        rat=rat,
        fold=fold,
        seed=seed,
        version=version,
        output_dimension=OUTPUT_DIMENSION,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_updates=MAX_UPDATES,
        embedding_temperature=EMBEDDING_TEMPERATURE,
        metadata_temperature=METADATA_TEMPERATURE,
        campaign_id=CAMPAIGN_ID,
        campaign_architecture_key=architecture,
    )
    return config


def cell_id(config: dict) -> str:
    return (
        f"{config['condition']}/{config['campaign_architecture_key']}/"
        f"{config['rat']}/fold_{config['fold']}/{config['version']}/seed_{config['seed']}"
    )


def cell_folder(config: dict) -> Path:
    return OUTPUT_ROOT / "stage01_fit" / cell_id(config)


def validate_grid(configs: list[dict]) -> dict:
    expected_ids = {
        f"{condition}/{architecture}/{rat}/fold_{fold}/{version}/seed_{seed}"
        for rat, fold, condition, architecture, version, seed in itertools.product(
            rat_runner.RATS, FOLDS, CONDITIONS, ARCHITECTURES, VERSIONS, SEEDS
        )
    }
    observed_ids = [cell_id(config) for config in configs]
    if len(observed_ids) != PLANNED_FITS or set(observed_ids) != expected_ids:
        raise RuntimeError("Cell identifiers do not match the full 864-cell protocol")
    expected_specs = {
        "time_direction_position": rat_runner.SPECS,
        "time_only": time_runner.SPECS,
        "direction_position": behavior_runner.SPECS,
    }
    for config in configs:
        expected_arch = "offset10-model" if config["campaign_architecture_key"] == "cnn" else "TemporalTransformerEncoder"
        if (config["condition"] not in CONDITIONS
                or config["version"] not in VERSIONS
                or config["output_dimension"] != OUTPUT_DIMENSION
                or config["batch_size"] != BATCH_SIZE
                or config["learning_rate"] != LEARNING_RATE
                or config["max_updates"] != MAX_UPDATES
                or config["embedding_temperature"] != EMBEDDING_TEMPERATURE
                or config["metadata_temperature"] != METADATA_TEMPERATURE
                or config["architecture"] != expected_arch
                or config["metadata_specs"] != expected_specs[config["condition"]]):
            raise RuntimeError(f"Cell config violates approved protocol: {cell_id(config)}")
    return {
        "per_condition": {condition: sum(c["condition"] == condition for c in configs) for condition in CONDITIONS},
        "per_architecture": {architecture: sum(c["campaign_architecture_key"] == architecture for c in configs)
                             for architecture in ARCHITECTURES},
        "per_loss_version": {version: sum(c["version"] == version for c in configs) for version in VERSIONS},
        "per_condition_architecture_version": {
            f"{condition}/{architecture}/{version}": sum(
                c["condition"] == condition and c["campaign_architecture_key"] == architecture
                and c["version"] == version for c in configs
            )
            for condition, architecture, version in itertools.product(CONDITIONS, ARCHITECTURES, VERSIONS)
        },
    }


def collect_source_record() -> dict:
    """Hash the runner, model/loss/data APIs, Transformer, and relevant versions."""
    file_hashes = {}
    for label, source_map in (
        ("cnn_fit_api", rat_runner.sources()),
        ("transformer_fit_api", transformer_runner.sources()),
        ("time_condition", time_runner.sources()),
        ("behavior_condition", behavior_runner.sources()),
    ):
        for source, digest in source_map.items():
            file_hashes[f"{label}::{source}"] = digest
    file_hashes["campaign_runner"] = sha256(Path(__file__).resolve())
    versions = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}
    for package in ("cebra",):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {"files_and_functions": file_hashes, "dependencies": versions}


def validate_plan() -> dict:
    if not PLAN_PATH.is_file():
        raise RuntimeError(f"Campaign plan is missing: {PLAN_PATH}")
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    protocol = plan.get("user_protocol", {})
    checks = {
        "campaign_id": plan.get("campaign_id") == CAMPAIGN_ID,
        "output_root": plan.get("output_root") == str(OUTPUT_ROOT.relative_to(ROOT)).replace("\\", "/"),
        "rats": tuple(protocol.get("rats", ())) == tuple(rat_runner.RATS),
        "folds": tuple(protocol.get("folds", ())) == FOLDS,
        "seeds": tuple(protocol.get("seeds", ())) == SEEDS,
        "conditions": tuple(protocol.get("conditions", ())) == CONDITIONS,
        "architectures": protocol.get("architectures") == {
            "cnn": "offset10-model", "transformer": "TemporalTransformerEncoder"
        },
        "versions": tuple(protocol.get("loss_versions", ())) == VERSIONS,
        "dimension": protocol.get("output_dimension") == OUTPUT_DIMENSION,
        "batch_size": protocol.get("optimization", {}).get("batch_size") == BATCH_SIZE,
        "learning_rate": protocol.get("optimization", {}).get("learning_rate") == LEARNING_RATE,
        "max_updates": protocol.get("optimization", {}).get("max_updates") == MAX_UPDATES,
        "embedding_temperature": protocol.get("optimization", {}).get("embedding_temperature") == EMBEDDING_TEMPERATURE,
        "metadata_temperature": protocol.get("optimization", {}).get("metadata_temperature") == METADATA_TEMPERATURE,
        "planned_fit_count": plan.get("planned_fit_counts", {}).get("total") == PLANNED_FITS,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Campaign plan disagrees with runner protocol: {failed}")
    return plan


def condition_specs(condition: str):
    if condition == "time_direction_position":
        return [dict(spec) for spec in rat_runner.SPECS], None
    if condition == "time_only":
        return [dict(spec) for spec in time_runner.SPECS], time_runner.project_scales
    if condition == "direction_position":
        return [dict(spec) for spec in behavior_runner.SPECS], behavior_runner.project_scales
    raise ValueError(condition)


def build_condition_manifest(condition: str, rat: str, fold: int,
                             parent_folder: Path, parent: dict) -> dict:
    specs, scale_projector = condition_specs(condition)
    scales = scale_projector(parent["train_scales"]) if scale_projector else parent["train_scales"]
    parent_manifest = parent_folder / "manifest.json"
    record = {
        "schema": "soft_fig1e_ed4_condition_data_v1",
        "campaign_id": CAMPAIGN_ID,
        "condition": condition,
        "rat": rat,
        "fold": fold,
        "parent_cache_manifest": str(parent_manifest.resolve()),
        "parent_cache_sha256": sha256(parent_manifest),
        "parent_cache_signature": parent["signature"],
        "split_artifacts": parent["splits"],
        "metadata_specs": specs,
        "train_scales": scales,
        "scale_refit": False,
        "test_rows_used_for_fit": False,
    }
    record["signature"] = config_hash(record)
    return record


def prepare_all() -> dict:
    """Create/reuse hashed split caches and condition manifests; no optimizer calls."""
    prepared = {}
    for rat, fold in itertools.product(rat_runner.RATS, FOLDS):
        parent_folder, parent = rat_runner.prepare(OUTPUT_ROOT, rat, fold)
        for condition in CONDITIONS:
            record = build_condition_manifest(condition, rat, fold, parent_folder, parent)
            manifest_path = OUTPUT_ROOT / "stage00_data" / condition / rat / f"fold_{fold}" / "manifest.json"
            if manifest_path.exists():
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                if existing != record:
                    raise RuntimeError(f"Condition data manifest mismatch; refusing overwrite: {manifest_path}")
            else:
                atomic_json(manifest_path, record)
            prepared[(condition, rat, fold)] = (parent_folder, record, manifest_path)
    return prepared


def bind_condition(condition: str, architecture: str):
    stack = ExitStack()
    if condition == "time_only":
        stack.enter_context(time_runner.context_bindings())
        if architecture == "transformer":
            stack.enter_context(transformer_runner.architecture_bindings())
    elif condition == "direction_position":
        stack.enter_context(behavior_runner.context_bindings(architecture))
    elif architecture == "transformer":
        stack.enter_context(transformer_runner.architecture_bindings())
    return stack


def expected_fit_signature(config: dict, condition_manifest_path: Path,
                           source_record: dict) -> str:
    return config_hash(dict(config=config, parent_id=sha256(condition_manifest_path), sources=source_record))


def verify_cell(folder: Path, config: dict, condition_manifest_path: Path,
                source_record: dict, require_sidecar: bool = True) -> dict:
    fit_path = folder / "manifest.json"
    checkpoint = folder / "checkpoint.pt"
    sidecar_path = folder / "CELL_MANIFEST.json"
    if not fit_path.is_file() or not checkpoint.is_file():
        raise RuntimeError(f"Incomplete completed-cell artifacts: {cell_id(config)}")
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    signature = expected_fit_signature(config, condition_manifest_path, source_record)
    if fit.get("signature") != signature:
        raise RuntimeError(f"Fit source/config/data signature mismatch: {cell_id(config)}")
    checkpoint_hash = sha256(checkpoint)
    if fit.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError(f"Checkpoint hash mismatch: {cell_id(config)}")
    if require_sidecar and not sidecar_path.is_file():
        raise RuntimeError(f"Fit manifest exists but cell sidecar is missing: {cell_id(config)}")
    if sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        expected = {
            "cell_id": cell_id(config),
            "config_signature": config_hash(config),
            "fit_signature": signature,
            "fit_manifest_sha256": sha256(fit_path),
            "checkpoint_sha256": checkpoint_hash,
            "condition_manifest_sha256": sha256(condition_manifest_path),
            "source_signature": config_hash(source_record),
        }
        for key, value in expected.items():
            if sidecar.get(key) != value:
                raise RuntimeError(f"Cell sidecar {key} mismatch: {cell_id(config)}")
    return fit


def sidecar_record(config: dict, fit: dict, folder: Path,
                   condition_manifest: dict, condition_manifest_path: Path,
                   source_record: dict) -> dict:
    fit_path = folder / "manifest.json"
    checkpoint = folder / "checkpoint.pt"
    return {
        "schema": "soft_fig1e_ed4_fit_cell_v1",
        "cell_id": cell_id(config),
        "condition": config["condition"],
        "architecture": config["campaign_architecture_key"],
        "loss_version": config["version"],
        "rat": config["rat"],
        "fold": config["fold"],
        "seed": config["seed"],
        "config": config,
        "config_signature": config_hash(config),
        "fit_signature": fit["signature"],
        "fit_manifest_sha256": sha256(fit_path),
        "checkpoint_sha256": sha256(checkpoint),
        "condition_manifest_path": str(condition_manifest_path.resolve()),
        "condition_manifest_sha256": sha256(condition_manifest_path),
        "parent_cache_manifest_sha256": condition_manifest["parent_cache_sha256"],
        "split_source_identity_hashes": {
            split: {
                "source_id_hash": details["source_id_hash"],
                "trial_ids": details["trials"],
                "split_npz_sha256": details["sha256"],
            }
            for split, details in condition_manifest["split_artifacts"].items()
        },
        "source_hashes": source_record,
        "source_signature": config_hash(source_record),
        "actual_updates": fit["actual_updates"],
        "recorded_utc": now(),
    }


def check_partial_folder(folder: Path, config: dict) -> None:
    if not folder.exists():
        return
    allowed = {"checkpoint.pt", "manifest.json", "attempts"}
    unexpected = {entry.name for entry in folder.iterdir()} - allowed
    if unexpected:
        raise RuntimeError(f"Unrecognized artifacts in cell folder; inspect before resume: {cell_id(config)}: {sorted(unexpected)}")
    if (folder / "attempts").exists() and not (folder / "checkpoint.pt").is_file():
        raise RuntimeError(f"Attempt records exist without a resumable checkpoint: {cell_id(config)}")


def dry_run(configs: list[dict], source_record: dict, plan: dict,
            max_new_fits: int | None) -> dict:
    grid_counts = validate_grid(configs)
    completed = 0
    recoverable_complete = 0
    partial = 0
    pending = 0
    partial_cells = []
    for config in configs:
        folder = cell_folder(config)
        fit_path = folder / "manifest.json"
        sidecar = folder / "CELL_MANIFEST.json"
        checkpoint = folder / "checkpoint.pt"
        if fit_path.exists():
            if checkpoint.exists() and sidecar.exists():
                # Parent manifest is already sealed into the sidecar.
                sidecar_record = json.loads(sidecar.read_text(encoding="utf-8"))
                fit_record = json.loads(fit_path.read_text(encoding="utf-8"))
                if (sidecar_record.get("cell_id") != cell_id(config)
                        or sidecar_record.get("config_signature") != config_hash(config)
                        or sidecar_record.get("source_signature") != config_hash(source_record)
                        or fit_record.get("config") != config
                        or fit_record.get("checkpoint_sha256") != sha256(checkpoint)
                        or fit_record.get("signature") != sidecar_record.get("fit_signature")
                        or sha256(fit_path) != sidecar_record.get("fit_manifest_sha256")):
                    raise RuntimeError(f"Existing cell artifacts fail dry-run integrity check: {cell_id(config)}")
                completed += 1
            elif checkpoint.exists():
                recoverable_complete += 1
                partial_cells.append(cell_id(config))
            else:
                raise RuntimeError(f"Manifest exists without checkpoint: {cell_id(config)}")
        elif folder.exists() and any(folder.iterdir()):
            check_partial_folder(folder, config)
            if checkpoint.exists():
                partial += 1
                partial_cells.append(cell_id(config))
            else:
                raise RuntimeError(f"Nonempty cell folder has no checkpoint: {cell_id(config)}")
        else:
            pending += 1

    raw_data_hashes = {}
    missing_data = []
    for rat in rat_runner.RATS:
        path = ROOT / "data" / "rat_hippocampus" / f"{rat}.jl"
        if path.is_file():
            raw_data_hashes[rat] = sha256(path)
        else:
            missing_data.append(str(path))
    if missing_data:
        raise RuntimeError(f"Dry-run preflight cannot hash input recordings: {missing_data}")

    signature_payload = {
        "campaign_id": CAMPAIGN_ID,
        "cell_ids": [cell_id(config) for config in configs],
        "configs": configs,
        "sources": source_record,
        "raw_data_hashes": raw_data_hashes,
    }
    state_signature = config_hash(signature_payload)
    if STATE_PATH.is_file():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if state.get("campaign_signature") != state_signature:
            raise RuntimeError("Existing campaign state signature differs; refusing to mix cohorts")
    if CAMPAIGN_LOCK.exists():
        raise RuntimeError(f"Campaign lock exists; inspect its PID before proceeding: {CAMPAIGN_LOCK}")

    result = {
        "mode": "dry_run_read_only",
        "campaign_id": CAMPAIGN_ID,
        "output_root": str(OUTPUT_ROOT),
        "planned_fits": len(configs),
        "grid_validation": "PASS",
        "grid_counts": grid_counts,
        "completed_validated": completed,
        "recoverable_complete_without_sidecar": recoverable_complete,
        "partial_checkpoint_cells": partial,
        "pending_fits": pending,
        "max_new_fits_if_executed": max_new_fits,
        "conditions": list(CONDITIONS),
        "architectures": list(ARCHITECTURES),
        "loss_versions": list(VERSIONS),
        "seeds": list(SEEDS),
        "output_dimension": OUTPUT_DIMENSION,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_updates": MAX_UPDATES,
        "embedding_temperature": EMBEDDING_TEMPERATURE,
        "metadata_temperature": METADATA_TEMPERATURE,
        "optimizer_updates": 0,
        "cell_id_grid_sha256": config_hash([cell_id(config) for config in configs]),
        "campaign_signature": state_signature,
        "raw_recording_sha256": raw_data_hashes,
        "source_signature": config_hash(source_record),
        "source_file_count": len(source_record["files_and_functions"]),
        "partial_cell_ids": partial_cells,
        "plan_status": plan.get("status"),
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def condition_manifest_path(config: dict) -> Path:
    return (OUTPUT_ROOT / "stage00_data" / config["condition"] / config["rat"]
            / f"fold_{config['fold']}" / "manifest.json")


def acquire_lock(path: Path, description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"pid": os.getpid(), "created_utc": now(), "description": description})
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"{description} lock exists; inspect before resuming: {path}") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def execute(configs: list[dict], source_record: dict, max_new_fits: int | None) -> None:
    acquire_lock(CAMPAIGN_LOCK, "Campaign")
    state = None
    try:
        raw_data_hashes = {
            rat: sha256(ROOT / "data" / "rat_hippocampus" / f"{rat}.jl")
            for rat in rat_runner.RATS
        }
        signature_payload = {
            "campaign_id": CAMPAIGN_ID,
            "cell_ids": [cell_id(config) for config in configs],
            "configs": configs,
            "sources": source_record,
            "raw_data_hashes": raw_data_hashes,
        }
        campaign_signature = config_hash(signature_payload)
        if STATE_PATH.exists():
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if state.get("campaign_signature") != campaign_signature:
                raise RuntimeError("Existing campaign state signature differs; refusing to mix cohorts")
            if state.get("planned_fits") != PLANNED_FITS or not isinstance(state.get("records"), dict):
                raise RuntimeError("Existing campaign state schema/count mismatch")
        prepared = prepare_all()
        if state is None:
            state = {
                "schema": "soft_fig1e_ed4_campaign_state_v1",
                "campaign_id": CAMPAIGN_ID,
                "campaign_signature": campaign_signature,
                "planned_fits": PLANNED_FITS,
                "records": {},
                "status": "prepared",
                "created_utc": now(),
                "raw_recording_sha256": raw_data_hashes,
                "source_signature": config_hash(source_record),
            }
            atomic_json(STATE_PATH, state)

        new_fits = 0
        completed = 0
        current_fold = None
        rows_cache = None
        for config in configs:
            key = cell_id(config)
            folder = cell_folder(config)
            data_folder, data_manifest, data_manifest_file = prepared[
                (config["condition"], config["rat"], config["fold"])
            ]
            if (config["rat"], config["fold"]) != current_fold:
                current_fold = (config["rat"], config["fold"])
                rows_cache = {
                    split: load_rows(data_folder / f"{split}.npz")
                    for split in ("train", "valid")
                }

            if (folder / "manifest.json").is_file() and (folder / "CELL_MANIFEST.json").is_file():
                fit = verify_cell(folder, config, data_manifest_file, source_record)
                cell_record = {
                    "status": "complete",
                    "cell_id": key,
                    "fit_manifest_sha256": sha256(folder / "manifest.json"),
                    "checkpoint_sha256": fit["checkpoint_sha256"],
                }
                state["records"][key] = cell_record
                completed += 1
                atomic_json(STATE_PATH, state)
                continue

            check_partial_folder(folder, config)
            cell_lock = folder.parent / f".{folder.name}.RUNNING.lock"
            acquire_lock(cell_lock, f"Cell {key}")
            try:
                state.update(status="running", current_cell=key, updated_utc=now())
                state["records"][key] = {"status": "running", "cell_id": key, "started_utc": now()}
                atomic_json(STATE_PATH, state)
                parent_id = sha256(data_manifest_file)
                with bind_condition(config["condition"], config["campaign_architecture_key"]):
                    fit, created = rat_runner.fit_cell(
                        folder,
                        rows_cache["train"],
                        rows_cache["valid"],
                        config,
                        data_manifest["train_scales"],
                        parent_id,
                        source_record,
                        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    )
                sidecar = sidecar_record(
                    config, fit, folder, data_manifest, data_manifest_file, source_record
                )
                atomic_json(folder / "CELL_MANIFEST.json", sidecar)
                verify_cell(folder, config, data_manifest_file, source_record)
                new_fits += int(created)
                completed += 1
                state["records"][key] = {
                    "status": "complete",
                    "cell_id": key,
                    "fit_manifest_sha256": sha256(folder / "manifest.json"),
                    "checkpoint_sha256": fit["checkpoint_sha256"],
                    "actual_updates": fit["actual_updates"],
                    "newly_completed_this_invocation": bool(created),
                    "completed_utc": now(),
                }
                state.update(completed=completed, updated_utc=now())
                atomic_json(STATE_PATH, state)
                print(json.dumps({
                    "completed": completed,
                    "planned": PLANNED_FITS,
                    "new_fits_this_invocation": new_fits,
                    "last_cell": key,
                    "fit_seconds": fit.get("fit_seconds"),
                }), flush=True)
            finally:
                cell_lock.unlink(missing_ok=True)

            if max_new_fits is not None and new_fits >= max_new_fits:
                state.update(status="requested_limit", updated_utc=now(),
                             new_fits_this_invocation=new_fits, completed=completed)
                atomic_json(STATE_PATH, state)
                break
        else:
            state.update(status="complete", completed=completed, updated_utc=now(),
                         new_fits_this_invocation=new_fits)
            atomic_json(STATE_PATH, state)
        print(json.dumps({
            "status": state.get("status"),
            "completed": state.get("completed", completed),
            "planned": PLANNED_FITS,
            "new_fits_this_invocation": new_fits,
            "output_root": str(OUTPUT_ROOT),
        }, indent=2), flush=True)
    except BaseException as exc:
        if state is not None:
            state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                         last_error=repr(exc), updated_utc=now())
            atomic_json(STATE_PATH, state)
        raise
    finally:
        CAMPAIGN_LOCK.unlink(missing_ok=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Read-only plan and artifact audit; never fits")
    mode.add_argument("--execute", action="store_true", help="Run/resume fits; explicit opt-in required")
    parser.add_argument("--max-new-fits", type=int, default=None,
                        help="Stop after at most this many newly completed fit cells")
    args = parser.parse_args(argv)
    if args.max_new_fits is not None and not 1 <= args.max_new_fits <= PLANNED_FITS:
        parser.error(f"--max-new-fits must be in 1..{PLANNED_FITS}")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    plan = validate_plan()
    configs = list(iter_configs())
    validate_grid(configs)
    source_record = collect_source_record()
    if args.dry_run:
        dry_run(configs, source_record, plan, args.max_new_fits)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing to start the requested fit campaign")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    execute(configs, source_record, args.max_new_fits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
