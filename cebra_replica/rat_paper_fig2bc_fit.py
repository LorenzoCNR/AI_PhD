"""Isolated, resumable CEBRA fits for the rat Figure 2b/2c demo conditions.

The default invocation is a read-only plan. Training requires --execute.
Each rat/condition/seed (42-47) has its own directory below the versioned output root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RATS = ("achilles", "buddy", "cicero", "gatsby")
SEEDS = tuple(range(42, 48))
BEHAVIOR_CONDITIONS = (
    "position",
    "direction",
    "position_direction",
    "position_shuffled",
    "direction_shuffled",
    "position_direction_shuffled",
)
CONDITIONS = BEHAVIOR_CONDITIONS + ("time",)
OUTPUT_DIMENSION = 32
MAX_ITERATIONS = 5000
OUT = ROOT / "results" / "rat_paper_fig2bc_32d_v1"
SOURCE_ROOT = ROOT / "data" / "rat_hippocampus"
HYBRID_FIT_ROOT = ROOT / "results" / "rat_figure_gap_closure_v1" / "figure2b_fits"
HYBRID_CONDITION = "hybrid_position_time_5d"
SCRIPT_PATH = Path(__file__).resolve()
# This saved revision differs only in JSON tuple/list comparison for manifests;
# its fit code and parameters are unchanged, so verified cells remain reusable.
COMPATIBLE_HISTORICAL_CODE_SHA256 = frozenset({
    "f6c5c34bd832d43dc78dd04ffda6a940f05d07f59b19dad6e8b116b4f6da833e",
})

ARTIFACT_NAMES = (
    "fit_targets.npz",
    "checkpoint.pt",
    "loss_history.csv",
    "train_embedding.npz",
    "test_embedding.npz",
    "all_embedding.npz",
)
EMBEDDING_ROWS = {
    "train_embedding.npz": "train",
    "test_embedding.npz": "test",
    "all_embedding.npz": "all",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_loss_csv(path: Path, loss: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("update", "loss"))
            writer.writerows((i + 1, float(value)) for i, value in enumerate(loss))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for package in ("cebra", "numpy", "joblib", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def model_parameters(condition: str) -> dict[str, Any]:
    is_time = condition == "time"
    return {
        "model_architecture": "offset10-model",
        "device": "cuda_if_available",
        "criterion": "infonce",
        "distance": "cosine",
        "conditional": "time" if is_time else "time_delta",
        "temperature": 1.12 if is_time else 1.0,
        "temperature_mode": "constant",
        "min_temperature": 0.1,
        "time_offsets": 10,
        "delta": None,
        "max_iterations": MAX_ITERATIONS,
        "max_adapt_iterations": 500,
        "batch_size": 512,
        "learning_rate": 3e-4,
        "optimizer": "adam",
        "output_dimension": OUTPUT_DIMENSION,
        "verbose": False,
        "num_hidden_units": 32,
        "pad_before_transform": True,
        "hybrid": False,
        "optimizer_kwargs": (
            ("betas", (0.9, 0.999)),
            ("eps", 1e-8),
            ("weight_decay", 0),
            ("amsgrad", False),
        ),
        "masking_kwargs": None,
    }


def load_recording(rat: str) -> tuple[Path, np.ndarray, np.ndarray, dict[str, str]]:
    source = SOURCE_ROOT / f"{rat}.jl"
    if not source.is_file():
        raise FileNotFoundError(f"Missing rat recording: {source}")
    raw = joblib.load(source)
    if not isinstance(raw, dict) or "spikes" not in raw or "position" not in raw:
        raise ValueError(f"Expected spikes and position arrays in {source}")
    neural = np.asarray(raw["spikes"], dtype=np.float32)
    behavior = np.asarray(raw["position"], dtype=np.float32)
    if neural.ndim != 2 or behavior.ndim != 2 or behavior.shape[1] < 2:
        raise ValueError(f"Unexpected data shapes for {rat}: neural={neural.shape}, behavior={behavior.shape}")
    if neural.shape[0] != behavior.shape[0] or neural.shape[0] <= 0:
        raise ValueError(f"Neural/behavior row mismatch for {rat}: {neural.shape} vs {behavior.shape}")
    split_at = int(0.8 * len(neural))
    if split_at < 512 or split_at >= len(neural):
        raise ValueError(f"Insufficient observations for 80/20 split of {rat}: n={len(neural)}")
    source_hashes = {
        "input_file_sha256": sha256_file(source),
        "neural_array_sha256": sha256_array(neural),
        "behavior_array_sha256": sha256_array(behavior),
    }
    return source, neural, behavior, source_hashes


def make_identity(
    rat: str,
    condition: str,
    seed: int,
    source: Path,
    neural: np.ndarray,
    behavior: np.ndarray,
    source_hashes: dict[str, str],
    code_sha256: str,
    versions: dict[str, str],
) -> dict[str, Any]:
    split_at = int(0.8 * len(neural))
    target_description = {
        "position": "position[:, 0]",
        "direction": "position[:, 1]",
        "position_direction": "all columns of position",
        "position_shuffled": f"position[:, 0], training rows permuted with NumPy RandomState({seed})",
        "direction_shuffled": f"position[:, 1], training rows permuted with NumPy RandomState({seed})",
        "position_direction_shuffled": f"all columns of position, training rows permuted together with NumPy RandomState({seed})",
        "time": None,
    }[condition]
    config = {
        "rat": rat,
        "condition": condition,
        "seed": seed,
        "parameters": model_parameters(condition),
        "fit_target": target_description,
        "shuffle_scope": "training labels only; neural row order unchanged" if "shuffled" in condition else None,
        "shuffle_rng": f"numpy.random.RandomState({seed}).permutation" if "shuffled" in condition else None,
        "split": {
            "kind": "contiguous_temporal_80_20",
            "train_source_id_range_half_open": [0, split_at],
            "test_source_id_range_half_open": [split_at, len(neural)],
            "source_id_kind": "zero-based row index in original recording",
        },
        "versions": versions,
    }
    return {
        "rat": rat,
        "condition": condition,
        "seed": seed,
        "source_path": str(source.resolve()),
        "source_hashes": source_hashes,
        "script_path": str(SCRIPT_PATH),
        "code_sha256": code_sha256,
        "config": config,
        "config_sha256": sha256_json(config),
        "neural_shape": list(neural.shape),
        "behavior_shape": list(behavior.shape),
        "n_observations": int(len(neural)),
        "split_at": split_at,
    }


def cell_path(identity: dict[str, Any]) -> Path:
    return OUT / identity["rat"] / identity["condition"] / f"seed_{identity['seed']}"


def validate_embedding(path: Path, expected_split: str, identity: dict[str, Any]) -> None:
    data = np.load(path, allow_pickle=False)
    embedding = data["embedding"]
    ids = data["source_id"]
    n = identity["n_observations"]
    split_at = identity["split_at"]
    expected_ids = {
        "train": np.arange(split_at, dtype=np.int64),
        "test": np.arange(split_at, n, dtype=np.int64),
        "all": np.arange(n, dtype=np.int64),
    }[expected_split]
    if embedding.shape != (len(expected_ids), OUTPUT_DIMENSION) or not np.isfinite(embedding).all():
        raise RuntimeError(f"Invalid embedding shape/content: {path} ({embedding.shape})")
    if not np.array_equal(ids, expected_ids):
        raise RuntimeError(f"Source-ID mismatch in {path}")
    if str(data["rat"].item()) != identity["rat"] or str(data["condition"].item()) != identity["condition"]:
        raise RuntimeError(f"Cell metadata mismatch in {path}")
    if int(data["seed"].item()) != identity["seed"] or str(data["split"].item()) != expected_split:
        raise RuntimeError(f"Seed/split metadata mismatch in {path}")


def verify_artifacts(folder: Path, artifact_hashes: dict[str, str], identity: dict[str, Any]) -> None:
    for name, digest in artifact_hashes.items():
        path = folder / name
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"Artifact missing or hash mismatch: {path}")
    for name, split in EMBEDDING_ROWS.items():
        if name in artifact_hashes:
            validate_embedding(folder / name, split, identity)
    if "loss_history.csv" in artifact_hashes:
        with (folder / "loss_history.csv").open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != MAX_ITERATIONS:
            raise RuntimeError(f"Expected {MAX_ITERATIONS} loss updates in {folder / 'loss_history.csv'}")
        losses = np.asarray([float(row["loss"]) for row in rows], dtype=np.float64)
        if not np.isfinite(losses).all():
            raise RuntimeError(f"Non-finite training loss in {folder / 'loss_history.csv'}")


def read_cell_state(identity: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    folder = cell_path(identity)
    manifest_path = folder / "manifest.json"
    status_path = folder / "RUNNING.json"
    if not folder.exists():
        return "pending_fit", None
    if not folder.is_dir():
        raise RuntimeError(f"Cell output path is not a directory: {folder}")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("rat", "condition", "seed", "source_path", "source_hashes", "script_path", "code_sha256", "config", "config_sha256", "neural_shape", "behavior_shape", "n_observations", "split_at"):
            # JSON round-trips tuples as lists (e.g. optimizer betas); compare
            # config in its canonical JSON form so equivalent saved fits remain reusable.
            if key == "config":
                mismatch = sha256_json(manifest.get(key)) != identity.get("config_sha256")
            elif key == "code_sha256":
                mismatch = manifest.get(key) != identity.get(key) and manifest.get(key) not in COMPATIBLE_HISTORICAL_CODE_SHA256
            else:
                mismatch = manifest.get(key) != identity.get(key)
            if mismatch:
                raise RuntimeError(f"Completed cell provenance mismatch at {manifest_path} ({key})")
        artifact_hashes = manifest.get("artifact_sha256")
        if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != set(ARTIFACT_NAMES):
            raise RuntimeError(f"Manifest artifact list is incomplete or unexpected: {manifest_path}")
        verify_artifacts(folder, artifact_hashes, identity)
        return "complete", manifest
    if not any(folder.iterdir()):
        return "pending_fit", None
    if not status_path.is_file():
        raise RuntimeError(f"Untracked partial cell; preserving it for inspection: {folder}")
    state = json.loads(status_path.read_text(encoding="utf-8"))
    for key in ("rat", "condition", "seed", "config_sha256", "source_hashes", "code_sha256"):
        expected = identity.get(key) if key != "config_sha256" else identity["config_sha256"]
        if key == "code_sha256":
            mismatch = state.get(key) != expected and state.get(key) not in COMPATIBLE_HISTORICAL_CODE_SHA256
        else:
            mismatch = state.get(key) != expected
        if mismatch:
            raise RuntimeError(f"Partial cell provenance mismatch at {status_path} ({key})")
    artifact_hashes = state.get("artifact_sha256", {})
    if not isinstance(artifact_hashes, dict):
        raise RuntimeError(f"Invalid partial artifact index: {status_path}")
    allowed = set(ARTIFACT_NAMES) | {"RUNNING.json"}
    extras = {p.name for p in folder.iterdir()} - allowed
    if extras:
        raise RuntimeError(f"Unrecognized partial files; preserving them: {sorted(extras)} in {folder}")
    verify_artifacts(folder, artifact_hashes, identity)
    checkpoint_exists = (folder / "checkpoint.pt").is_file()
    checkpoint_indexed = "checkpoint.pt" in artifact_hashes
    if checkpoint_exists != checkpoint_indexed:
        raise RuntimeError(f"Checkpoint/state mismatch; preserving partial cell: {folder}")
    if not checkpoint_exists and set(artifact_hashes) - {"fit_targets.npz"}:
        raise RuntimeError(f"Post-fit artifacts exist without a checkpoint: {folder}")
    return ("resume_from_checkpoint" if checkpoint_exists else "resume_fit"), state


def build_plans(code_sha256: str, versions: dict[str, str]) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    for rat in RATS:
        source, neural, behavior, source_hashes = load_recording(rat)
        for seed in SEEDS:
            for condition in CONDITIONS:
                identity = make_identity(rat, condition, seed, source, neural, behavior, source_hashes, code_sha256, versions)
                state, previous = read_cell_state(identity)
                plans.append({"identity": identity, "state": state, "previous": previous})
        del neural, behavior
    return plans


def audit_existing_hybrid() -> tuple[list[dict[str, Any]], dict[int, list[str]]]:
    """Read-only integrity check for the legacy 5D Hybrid fits; never writes them."""
    summaries: list[dict[str, Any]] = []
    historical_script = ROOT / "cebra_replica" / "rat_figure_gap_closure.py"
    current_historical_script_sha = sha256_file(historical_script) if historical_script.is_file() else None
    expected_artifacts = {
        "checkpoint.pt",
        "train_embedding.npz",
        "test_embedding.npz",
        "all_embedding.npz",
        "training_history.csv",
    }
    for rat in RATS:
        folder = HYBRID_FIT_ROOT / rat / HYBRID_CONDITION / "seed_42"
        manifest_path = folder / "result.json"
        if not manifest_path.is_file():
            summaries.append({"rat": rat, "status": "missing", "path": str(folder)})
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_path = SOURCE_ROOT / f"{rat}.jl"
        source_matches = source_path.is_file() and manifest.get("source_sha256") == sha256_file(source_path)
        params = manifest.get("parameters", {})
        config_matches = (
            manifest.get("rat") == rat
            and manifest.get("condition") == HYBRID_CONDITION
            and manifest.get("seed") == 42
            and manifest.get("max_iterations") == MAX_ITERATIONS
            and manifest.get("output_dimension") == 5
            and params.get("output_dimension") == 5
            and params.get("max_iterations") == MAX_ITERATIONS
            and params.get("hybrid") is True
        )
        artifacts = manifest.get("artifacts", {})
        hashes_match = set(artifacts) == expected_artifacts and all(
            (folder / name).is_file() and sha256_file(folder / name) == digest
            for name, digest in artifacts.items()
        )
        split = manifest.get("split", {})
        train_end = split.get("train", [None, None])[1]
        n_rows = split.get("test", [None, None])[1]
        embeddings_match = train_end is not None and n_rows is not None
        if embeddings_match:
            for split_name, start, end in (
                ("train", 0, train_end),
                ("test", train_end, n_rows),
                ("all", 0, n_rows),
            ):
                with np.load(folder / f"{split_name}_embedding.npz", allow_pickle=False) as archive:
                    embedding = archive["embedding"]
                    ids = archive["source_id"]
                    expected_ids = np.arange(start, end, dtype=np.int64)
                    embeddings_match &= (
                        embedding.shape == (len(expected_ids), 5)
                        and np.isfinite(embedding).all()
                        and np.array_equal(ids, expected_ids)
                        and int(archive["seed"].item()) == 42
                        and str(archive["rat"].item()) == rat
                        and str(archive["condition"].item()) == HYBRID_CONDITION
                    )
        history_rows = 0
        history_path = folder / "training_history.csv"
        if history_path.is_file():
            with history_path.open("r", encoding="utf-8", newline="") as stream:
                history_rows = sum(1 for _ in csv.DictReader(stream))
        history_matches = history_rows == manifest.get("training_history_updates") == MAX_ITERATIONS
        code_hash_matches = manifest.get("script_sha256") == current_historical_script_sha
        integrity = source_matches and config_matches and hashes_match and embeddings_match and history_matches
        status = "verified_reusable" if integrity and code_hash_matches else (
            "artifact_integrity_ok_historical_code_hash_differs" if integrity else "needs_manual_review"
        )
        summaries.append({
            "rat": rat,
            "status": status,
            "path": str(folder),
            "source_matches": source_matches,
            "config_matches": config_matches,
            "artifact_hashes_match": hashes_match,
            "embedding_support_matches": bool(embeddings_match),
            "loss_updates": history_rows,
            "recorded_script_sha256": manifest.get("script_sha256"),
            "current_script_sha256": current_historical_script_sha,
        })
    other_seed_dirs = {
        seed: [
            rat for rat in RATS
            if (HYBRID_FIT_ROOT / rat / HYBRID_CONDITION / f"seed_{seed}").exists()
        ]
        for seed in SEEDS[1:]
    }
    return summaries, other_seed_dirs


def make_fit_target(condition: str, behavior: np.ndarray, split_at: int, seed: int) -> tuple[np.ndarray, np.ndarray, bool]:
    source_ids = np.arange(split_at, dtype=np.int64)
    if condition == "time":
        return np.empty((split_at, 0), dtype=np.float32), np.empty(0, dtype=np.int64), False
    if condition.startswith("position_direction"):
        labels = behavior[:split_at].copy()
    elif condition.startswith("position"):
        labels = behavior[:split_at, 0].copy()
    else:
        labels = behavior[:split_at, 1].copy()
    target_source_ids = source_ids.copy()
    if "shuffled" in condition:
        permutation = np.random.RandomState(seed).permutation(split_at)
        labels = labels[permutation]
        target_source_ids = source_ids[permutation]
    return np.asarray(labels), target_source_ids, True


def seed_runtime(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def write_status(folder: Path, identity: dict[str, Any], stage: str, artifact_hashes: dict[str, str]) -> dict[str, Any]:
    status = {
        "status": "running",
        "stage": stage,
        "rat": identity["rat"],
        "condition": identity["condition"],
        "seed": identity["seed"],
        "config_sha256": identity["config_sha256"],
        "source_hashes": identity["source_hashes"],
        "code_sha256": identity["code_sha256"],
        "artifact_sha256": dict(artifact_hashes),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(folder / "RUNNING.json", status)
    return status


def record_artifact(folder: Path, name: str, artifact_hashes: dict[str, str], identity: dict[str, Any], stage: str) -> None:
    artifact_hashes[name] = sha256_file(folder / name)
    write_status(folder, identity, stage, artifact_hashes)


def save_targets(
    folder: Path,
    condition: str,
    behavior: np.ndarray,
    split_at: int,
    seed: int,
    identity: dict[str, Any],
    artifact_hashes: dict[str, str],
) -> None:
    path = folder / "fit_targets.npz"
    if path.exists():
        if "fit_targets.npz" not in artifact_hashes or sha256_file(path) != artifact_hashes["fit_targets.npz"]:
            raise RuntimeError(f"Unindexed fit-target file; preserving it: {path}")
        return
    target, target_source_ids, has_target = make_fit_target(condition, behavior, split_at, seed)
    atomic_npz(
        path,
        fit_source_id=np.arange(split_at, dtype=np.int64),
        target_source_id=target_source_ids,
        target=target,
        has_behavior_target=np.asarray(has_target),
        rat=np.asarray(identity["rat"]),
        condition=np.asarray(condition),
        seed=np.asarray(seed, dtype=np.int64),
    )
    record_artifact(folder, path.name, artifact_hashes, identity, "targets_saved")


def save_checkpoint(model: Any, path: Path) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp.pt", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        temp_path.unlink()
        model.save(str(temp_path))
        if not temp_path.is_file():
            raise RuntimeError(f"CEBRA did not create checkpoint at {temp_path}")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def extract_loss(model: Any) -> np.ndarray:
    state = getattr(model, "state_dict_", None)
    if not isinstance(state, dict) or "loss" not in state:
        raise RuntimeError("Fitted CEBRA model has no state_dict_['loss']; refusing to finalize without loss history")
    loss = np.asarray(state["loss"], dtype=np.float64).reshape(-1)
    if len(loss) != MAX_ITERATIONS or not np.isfinite(loss).all():
        raise RuntimeError(f"Expected {MAX_ITERATIONS} finite loss updates, got {len(loss)}")
    return loss


def save_embedding(
    model: Any,
    folder: Path,
    split_name: str,
    neural: np.ndarray,
    behavior: np.ndarray,
    ids: np.ndarray,
    identity: dict[str, Any],
    artifact_hashes: dict[str, str],
) -> None:
    name = f"{split_name}_embedding.npz"
    path = folder / name
    if path.exists():
        if name not in artifact_hashes or sha256_file(path) != artifact_hashes[name]:
            raise RuntimeError(f"Unindexed embedding; preserving it: {path}")
        validate_embedding(path, split_name, identity)
        return
    embedding = np.asarray(model.transform(neural[ids]), dtype=np.float32)
    if embedding.shape != (len(ids), OUTPUT_DIMENSION) or not np.isfinite(embedding).all():
        raise RuntimeError(f"Invalid {split_name} embedding: {embedding.shape}")
    atomic_npz(
        path,
        embedding=embedding,
        source_id=ids,
        behavior=behavior[ids],
        split=np.asarray(split_name),
        source_id_kind=np.asarray("zero_based_row_index_in_original_recording"),
        rat=np.asarray(identity["rat"]),
        condition=np.asarray(identity["condition"]),
        seed=np.asarray(identity["seed"], dtype=np.int64),
        output_dimension=np.asarray(OUTPUT_DIMENSION, dtype=np.int64),
    )
    validate_embedding(path, split_name, identity)
    record_artifact(folder, name, artifact_hashes, identity, f"{split_name}_embedding_saved")


def finish_cell(folder: Path, identity: dict[str, Any], artifact_hashes: dict[str, str], fit_seconds: float | None) -> None:
    if set(artifact_hashes) != set(ARTIFACT_NAMES):
        raise RuntimeError(f"Cannot finalize incomplete artifact set in {folder}: {sorted(artifact_hashes)}")
    verify_artifacts(folder, artifact_hashes, identity)
    manifest = {
        **identity,
        "status": "complete",
        "artifact_sha256": dict(sorted(artifact_hashes.items())),
        "fit_seconds_this_invocation": fit_seconds,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Fit uses only contiguous first 80 percent; final 20 percent is held out. Train/test/all embeddings are transforms of the indicated source rows.",
        "source_qualification": {
            "figure2c_repository": "official figure repository supplies plotting code for precomputed arrays; hippocampus demo supplies the hypothesis-testing fit recipe",
            "direction_demo_caveat": "Official demo's direction constructor omits explicit distance, conditional, and time_offsets arguments; this script sets cosine, time_delta, and 10 explicitly to match the requested shared configuration.",
            "hybrid": {
                "seed42_reuse_root": str(HYBRID_FIT_ROOT),
                "paper_figure_reference": "https://www.nature.com/articles/s41586-023-06031-6#Fig2",
                "demo_reference": "https://cebra.ai/docs/demo_notebooks/Demo_hippocampus.html",
                "dimension_note": "Nature Fig. 2b caption specifies a 5D hybrid space and plots its first 3 coordinates; the current hippocampus demo notebook's hybrid example is output_dimension=3, so it is not the same exact dimensional recipe.",
                "other_seeds": "not included in this 32D runner; the official Fig. 2b panel is one qualitative 5D hybrid embedding, so seeds 43-47 are optional multi-seed robustness fits, not required to reproduce that panel",
                "implementation": "the existing fit_figure2b_hybrid function accepts a seed and writes seed-specific folders; a 6-seed hybrid extension would be 20 new fits (5 seeds x 4 rats) and is deliberately not launched here",
            },
        },
    }
    atomic_json(folder / "manifest.json", manifest)
    write_status(folder, identity, "complete", artifact_hashes)


def execute_cell(
    plan: dict[str, Any],
    neural: np.ndarray,
    behavior: np.ndarray,
    allow_new_fit: bool,
) -> tuple[bool, str]:
    identity = plan["identity"]
    seed = identity["seed"]
    folder = cell_path(identity)
    state_kind, previous = read_cell_state(identity)
    if state_kind == "complete":
        return False, "complete"
    checkpoint_path = folder / "checkpoint.pt"
    has_checkpoint = state_kind == "resume_from_checkpoint"
    if not has_checkpoint and not allow_new_fit:
        return False, "deferred_by_max_new_fits"
    folder.mkdir(parents=True, exist_ok=True)
    artifact_hashes = dict((previous or {}).get("artifact_sha256", {}))
    write_status(folder, identity, "preparing", artifact_hashes)
    split_at = identity["split_at"]
    save_targets(folder, identity["condition"], behavior, split_at, seed, identity, artifact_hashes)

    import cebra
    import torch

    fit_seconds: float | None = None
    if has_checkpoint:
        model = cebra.CEBRA.load(str(checkpoint_path))
        if "loss_history.csv" not in artifact_hashes:
            atomic_loss_csv(folder / "loss_history.csv", extract_loss(model))
            record_artifact(folder, "loss_history.csv", artifact_hashes, identity, "loss_history_saved")
    else:
        seed_runtime(seed, torch)
        parameters = model_parameters(identity["condition"])
        model = cebra.CEBRA(**parameters)
        labels, _, has_behavior_target = make_fit_target(identity["condition"], behavior, split_at, seed)
        start = time.perf_counter()
        if has_behavior_target:
            model.fit(neural[:split_at], labels)
        else:
            model.fit(neural[:split_at])
        fit_seconds = time.perf_counter() - start
        save_checkpoint(model, checkpoint_path)
        record_artifact(folder, "checkpoint.pt", artifact_hashes, identity, "checkpoint_saved")
        atomic_loss_csv(folder / "loss_history.csv", extract_loss(model))
        record_artifact(folder, "loss_history.csv", artifact_hashes, identity, "loss_history_saved")

    ids = {
        "train": np.arange(split_at, dtype=np.int64),
        "test": np.arange(split_at, len(neural), dtype=np.int64),
        "all": np.arange(len(neural), dtype=np.int64),
    }
    for split_name in ("train", "test", "all"):
        save_embedding(model, folder, split_name, neural, behavior, ids[split_name], identity, artifact_hashes)
    finish_cell(folder, identity, artifact_hashes, fit_seconds)
    return not has_checkpoint, "complete"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="explicitly request the read-only plan (the default)")
    parser.add_argument("--execute", action="store_true", help="perform missing fits and resume valid partial cells")
    parser.add_argument("--max-new-fits", type=int, default=None, help="cap new model.fit calls; checkpoint-based resumes do not count")
    args = parser.parse_args()
    if args.execute and args.dry_run:
        parser.error("--execute and --dry-run cannot be combined")
    if args.max_new_fits is not None and args.max_new_fits < 0:
        parser.error("--max-new-fits must be non-negative")

    code_sha256 = sha256_file(SCRIPT_PATH)
    versions = dependency_versions()
    plans = build_plans(code_sha256, versions)
    hybrid_summaries, hybrid_other_seeds = audit_existing_hybrid()
    counts: dict[str, int] = {}
    for plan in plans:
        kind = plan["state"]
        counts[kind] = counts.get(kind, 0) + 1
        identity = plan["identity"]
        print(f"PLAN {identity['rat']}/{identity['condition']}/seed_{identity['seed']}: {kind}; config={identity['config_sha256'][:12]}")
    print(f"MATRIX cells={len(plans)} rats={len(RATS)} conditions={len(CONDITIONS)} seeds={SEEDS}; states={counts}")
    print(f"OUTPUT_ROOT {OUT}")
    for item in hybrid_summaries:
        print(f"HYBRID_SEED42 {item['rat']}: {item['status']}; {item['path']}")
    print(f"HYBRID_SEED42_AUDIT verified_artifact_cells={sum(item['status'] in ('verified_reusable', 'artifact_integrity_ok_historical_code_hash_differs') for item in hybrid_summaries)}/{len(RATS)}")
    print(f"HYBRID_SEEDS43_47_EXISTING {hybrid_other_seeds}")
    print("HYBRID_POLICY paper Fig2b specifies one qualitative 5D hybrid panel, not a six-seed statistic; seed42 is retained for reuse. Seeds43-47 are optional if extending seed-wise robustness to Hybrid too (20 fits across four rats), technically supported by the legacy seed-parameterized fitter, but excluded from this 32D runner and not launched.")
    if not args.execute:
        cap_text = "unbounded" if args.max_new_fits is None else str(args.max_new_fits)
        print(f"DRY_RUN no files written; --max-new-fits cap={cap_text}; pass --execute to fit")
        return

    available_new_fits = args.max_new_fits
    completed_or_resumed = 0
    new_fit_calls = 0
    deferred = 0
    for rat in RATS:
        rat_plans = [plan for plan in plans if plan["identity"]["rat"] == rat]
        if not rat_plans:
            continue
        source, neural, behavior, source_hashes = load_recording(rat)
        for plan in rat_plans:
            identity = plan["identity"]
            if source_hashes != identity["source_hashes"]:
                raise RuntimeError(f"Source changed after planning: {source}")
            current_state, _ = read_cell_state(identity)
            needs_new_fit = current_state in ("pending_fit", "resume_fit")
            if needs_new_fit and available_new_fits == 0:
                deferred += 1
                print(f"DEFER {rat}/{identity['condition']}: max-new-fits reached")
                continue
            did_new_fit, result = execute_cell(plan, neural, behavior, allow_new_fit=not needs_new_fit or available_new_fits is None or available_new_fits > 0)
            if did_new_fit:
                new_fit_calls += 1
                if available_new_fits is not None:
                    available_new_fits -= 1
            completed_or_resumed += 1
            print(f"{result.upper()} {rat}/{identity['condition']}/seed_{identity['seed']}")
        del neural, behavior
    print(f"SUMMARY processed={completed_or_resumed} new_fit_calls={new_fit_calls} deferred={deferred} output={OUT}")


if __name__ == "__main__":
    main()
