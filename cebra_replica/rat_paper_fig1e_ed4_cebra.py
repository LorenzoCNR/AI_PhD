"""Resumable trial-wise CEBRA fits for rat Figure 1e / Extended Data 4.

The default invocation is read-only. Training requires the explicit --execute
flag. Each rat/fold/condition/seed has an isolated, immutable artifact folder.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import time
import uuid

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "results" / "rat_paper_fig1e_ed4_6seed_v1" / "cebra"
RATS = ("achilles", "buddy", "cicero", "gatsby")
FOLDS = (0, 1, 2)
SEEDS = tuple(range(42, 48))
CONDITIONS = ("behavior", "time")
SPLITS = ("train", "valid", "test")
OFFSET_LEFT = 5
OFFSET_RIGHT = 5
OUTPUT_DIMENSION = 3
MAX_ITERATIONS = 1000

CONDITION_PARAMS = {
    "behavior": {
        "display_name": "CEBRA-Behavior",
        "conditional": "time_delta",
        "temperature": 1.265961498923956,
        "learning_rate": 0.0020507814090484,
    },
    "time": {
        "display_name": "CEBRA-Time",
        "conditional": "time",
        "temperature": 3.212991682008965,
        "learning_rate": 0.0028521675677231,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_log(path: Path, event: str, **details) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"utc": utc_now(), "event": event, **details}
    encoded = (json.dumps(record, sort_keys=True, ensure_ascii=False,
                          default=str) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        if getattr(error, "winerror", None) == 87:
            return False
        return True
    return True


@contextmanager
def cell_lock(path: Path, run_id: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"pid": os.getpid(), "host": socket.gethostname(),
                          "run_id": run_id, "created_utc": utc_now()}).encode("utf-8")
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        try:
            prior = json.loads(path.read_text(encoding="utf-8"))
            stale = (prior.get("host") == socket.gethostname()
                     and not _pid_alive(int(prior.get("pid", -1))))
        except Exception:
            stale = False
        if not stale:
            raise RuntimeError(f"Cell is locked or lock needs inspection: {path}")
        archived = path.with_name(f"{path.name}.stale.{uuid.uuid4().hex}")
        os.replace(path, archived)
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
            if current.get("run_id") == run_id and int(current.get("pid", -1)) == os.getpid():
                path.unlink()
        except FileNotFoundError:
            pass


def fit_parameters(condition: str) -> dict:
    condition_params = CONDITION_PARAMS[condition]
    return {
        "model_architecture": "offset10-model",
        "num_hidden_units": 64,
        "batch_size": 20000,
        "learning_rate": condition_params["learning_rate"],
        "temperature": condition_params["temperature"],
        "output_dimension": OUTPUT_DIMENSION,
        "max_iterations": MAX_ITERATIONS,
        "distance": "cosine",
        "conditional": condition_params["conditional"],
        "time_offsets": 10,
        "device": "cuda_if_available",
        "verbose": False,
    }


def cell_config(condition: str) -> dict:
    return {
        "campaign_id": "rat_paper_fig1e_ed4_6seed_v1",
        "figure_scope": ["Figure 1e", "Extended Data Figure 4"],
        "condition": condition,
        "display_name": CONDITION_PARAMS[condition]["display_name"],
        "fit_parameters": fit_parameters(condition),
        "dataset_offset": {"left": OFFSET_LEFT, "right": OFFSET_RIGHT,
                           "context_indices": list(range(-OFFSET_LEFT, OFFSET_RIGHT))},
        "fit_rows_policy": "all rows from the official training trial split; context-valid mask is stored, not used to drop fit rows",
    }


def source_paths() -> dict[str, Path]:
    return {
        "hippocampus_splitter": ROOT / "CEBRA" / "cebra" / "datasets" / "hippocampus.py",
        "dataset_base": ROOT / "CEBRA" / "cebra" / "data" / "base.py",
        "split_audit_helper": ROOT / "cebra_replica" / "rat_manifold_benchmark.py",
    }


def load_split_data(rat: str, fold: int) -> dict[str, dict]:
    """Load the official CEBRA trial split and independently verify row/trial IDs."""
    import cebra
    import joblib
    from cebra.datasets.hippocampus import SingleRatTrialSplitDataset
    from rat_manifold_benchmark import row_ids, trial_bounds, trial_split

    source = ROOT / "data" / "rat_hippocampus" / f"{rat}.jl"
    raw = joblib.load(source)
    position = np.asarray(raw["position"], dtype=np.float32)
    bounds = trial_bounds(position)
    trial_sets = trial_split(bounds, fold)
    all_trial = np.searchsorted(bounds[1:], np.arange(len(position)), side="right").astype(np.int64)
    source_hash = sha256_file(source)
    result = {}
    for split in SPLITS:
        dataset = SingleRatTrialSplitDataset(name=rat, split_no=fold, split=split,
                                             root=ROOT / "data")
        dataset.offset = cebra.data.datatypes.Offset(OFFSET_LEFT, OFFSET_RIGHT)
        source_id = np.concatenate([
            np.arange(section.start, section.stop, dtype=np.int64)
            for section in dataset.selected_indices
        ])
        expected_source_id = row_ids(bounds, trial_sets[split])
        if not np.array_equal(source_id, expected_source_id):
            raise AssertionError(f"Loader/support row mismatch: {rat}/fold_{fold}/{split}")
        trial_id = all_trial[source_id]
        if not np.array_equal(np.unique(trial_id), trial_sets[split]):
            raise AssertionError(f"Loader/support trial mismatch: {rat}/fold_{fold}/{split}")
        neural = dataset.neural.detach().cpu().numpy().astype(np.float32, copy=False)
        behavior = dataset.continuous_index.detach().cpu().numpy().astype(np.float32, copy=False)
        if neural.ndim != 2 or len(neural) != len(source_id) or behavior.shape != (len(source_id), 3):
            raise AssertionError(f"Unexpected split shapes: {rat}/fold_{fold}/{split}")
        local_time = source_id - bounds[trial_id]
        trial_length = bounds[trial_id + 1] - bounds[trial_id]
        # For the requested [-5, ..., +4] context, the final valid row is
        # trial_length - 5 (exclusive upper bound trial_length - 4).
        context_valid = (local_time >= OFFSET_LEFT) & (
            local_time < trial_length - (OFFSET_RIGHT - 1)
        )
        if np.any(context_valid):
            offsets = np.arange(-OFFSET_LEFT, OFFSET_RIGHT, dtype=np.int64)
            window_trial = all_trial[source_id[context_valid, None] + offsets[None, :]]
            if not np.all(window_trial == trial_id[context_valid, None]):
                raise AssertionError(f"A valid 5/5 context crosses a trial: {rat}/fold_{fold}/{split}")
        result[split] = {
            "neural": neural,
            "behavior": behavior,
            "source_id": source_id,
            "trial_id": trial_id,
            "context_valid": context_valid.astype(np.bool_),
            "trial_ids": trial_sets[split],
            "source_sha256": source_hash,
        }
    if any(np.intersect1d(result[a]["source_id"], result[b]["source_id"]).size
           for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise AssertionError(f"Official row splits overlap: {rat}/fold_{fold}")
    return result


def support_record(data: dict) -> dict:
    source_id = data["source_id"]
    trial_id = data["trial_id"]
    context_valid = data["context_valid"]
    trial_ids = np.asarray(data["trial_ids"], dtype=np.int64)
    return {
        "rows": int(len(source_id)),
        "trial_ids": trial_ids.tolist(),
        "source_id_sha256": sha256_array(source_id),
        "trial_id_sha256": sha256_array(trial_id),
        "context_valid_sha256": sha256_array(context_valid),
        "support_sha256": sha256_json({
            "source_id": sha256_array(source_id),
            "trial_id": sha256_array(trial_id),
            "context_valid": sha256_array(context_valid),
        }),
    }


def split_support(data_by_split: dict[str, dict]) -> dict:
    return {split: support_record(data_by_split[split]) for split in SPLITS}


def load_validated_checkpoint(path: Path, cebra, condition: str):
    """Load a saved model and validate its exact fit configuration/history."""
    model = cebra.CEBRA.load(str(path), backend="sklearn", weights_only=False)
    parameters = model.get_params(deep=False)
    mismatches = {
        name: (parameters.get(name), expected)
        for name, expected in fit_parameters(condition).items()
        if parameters.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"Checkpoint parameter mismatch at {path}: {mismatches}")
    losses = np.asarray(model.state_dict_["loss"], dtype=np.float32)
    if len(losses) != MAX_ITERATIONS or not np.isfinite(losses).all():
        raise RuntimeError(f"Checkpoint has invalid loss history at {path}: {len(losses)} rows")
    return model, losses


def cell_dir(rat: str, fold: int, condition: str, seed: int) -> Path:
    return OUTPUT_ROOT / rat / f"fold_{fold}" / condition / f"seed_{seed}"


def verify_checkpoint(folder: Path, config: dict, config_sha: str,
                      support: dict, source_hash: str, cebra_version: str,
                      torch_version: str) -> tuple[dict, bool]:
    checkpoint = folder / "checkpoint.pt"
    record_path = folder / "checkpoint_manifest.json"
    if not checkpoint.exists():
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("status") != "fit_in_progress":
                raise RuntimeError(f"Checkpoint missing for recorded fit: {record_path}")
            expected = {
                "config_sha256": config_sha,
                "config": config,
                "split_support": support,
                "source_data_sha256": source_hash,
                "cebra_version": cebra_version,
                "torch_version": torch_version,
            }
            if any(record.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f"In-progress checkpoint provenance mismatch: {record_path}")
            return record, False
        return {}, False
    if not record_path.is_file():
        raise RuntimeError(f"Unmanifested checkpoint; refusing to overwrite: {checkpoint}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    expected = {
        "config_sha256": config_sha,
        "config": config,
        "split_support": support,
        "source_data_sha256": source_hash,
        "cebra_version": cebra_version,
        "torch_version": torch_version,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Checkpoint provenance mismatch: {record_path}")
    if record.get("status") not in ("fit_in_progress", "checkpoint_ready"):
        raise RuntimeError(f"Unexpected checkpoint stage: {record_path}")
    digest = sha256_file(checkpoint)
    if record.get("status") == "checkpoint_ready" and digest != record.get("checkpoint_sha256"):
        raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
    if record.get("status") == "checkpoint_ready":
        loss_path = folder / "training_loss.npy"
        if not loss_path.is_file() or sha256_file(loss_path) != record.get("training_loss_sha256"):
            raise RuntimeError(f"Training loss missing/changed: {loss_path}")
    return record, True


def verify_embedding(path: Path, data: dict, condition: str, rat: str,
                     fold: int, seed: int, config_sha: str,
                     checkpoint_sha: str, expected_sha: str | None = None) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if expected_sha and digest != expected_sha:
        raise RuntimeError(f"Embedding hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as saved:
        required = ("embedding", "behavior", "source_id", "trial_id", "context_valid",
                    "rat", "fold", "seed", "condition", "config_sha256", "checkpoint_sha256")
        if any(key not in saved.files for key in required):
            raise RuntimeError(f"Incomplete embedding artifact: {path}")
        if not np.array_equal(saved["source_id"], data["source_id"]):
            raise RuntimeError(f"source_id mismatch: {path}")
        if not np.array_equal(saved["trial_id"], data["trial_id"]):
            raise RuntimeError(f"trial_id mismatch: {path}")
        if not np.array_equal(saved["context_valid"], data["context_valid"]):
            raise RuntimeError(f"context support mismatch: {path}")
        if not np.array_equal(saved["behavior"], data["behavior"]):
            raise RuntimeError(f"behavior labels mismatch: {path}")
        if (saved["rat"].item() != rat or int(saved["fold"].item()) != fold
                or int(saved["seed"].item()) != seed or saved["condition"].item() != condition
                or saved["config_sha256"].item() != config_sha
                or saved["checkpoint_sha256"].item() != checkpoint_sha):
            raise RuntimeError(f"Embedding identity/provenance mismatch: {path}")
        embedding = saved["embedding"]
        if embedding.shape != (len(data["source_id"]), OUTPUT_DIMENSION):
            raise RuntimeError(f"Embedding shape mismatch: {path}: {embedding.shape}")
        if not np.isfinite(embedding).all():
            raise RuntimeError(f"Non-finite embedding values: {path}")
    return {"path": path.name, "sha256": digest,
            "shape": [len(data["source_id"]), OUTPUT_DIMENSION],
            **support_record(data)}


def verify_complete(folder: Path, identity: dict, config: dict, config_sha: str,
                    data_by_split: dict, support: dict, source_hash: str,
                    cebra_version: str, torch_version: str) -> bool:
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, value in {**identity, "config": config, "config_sha256": config_sha,
                       "split_support": support, "source_data_sha256": source_hash,
                       "cebra_version": cebra_version, "torch_version": torch_version,
                       "status": "complete"}.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Completed manifest provenance mismatch: {manifest_path} ({key})")
    checkpoint_record, has_checkpoint = verify_checkpoint(
        folder, config, config_sha, support, source_hash, cebra_version, torch_version)
    if not has_checkpoint or manifest.get("checkpoint_sha256") != checkpoint_record.get("checkpoint_sha256"):
        raise RuntimeError(f"Completed manifest/checkpoint mismatch: {manifest_path}")
    loss_path = folder / "training_loss.npy"
    if sha256_file(loss_path) != manifest.get("training_loss_sha256"):
        raise RuntimeError(f"Training-loss hash mismatch: {loss_path}")
    for split in SPLITS:
        entry = manifest.get("embeddings", {}).get(split, {})
        verify_embedding(folder / "embeddings" / f"{split}.npz", data_by_split[split],
                         identity["condition"], identity["rat"], identity["fold"],
                         identity["seed"], config_sha, checkpoint_record["checkpoint_sha256"],
                         entry.get("sha256"))
    return True


def inspect_cell(folder: Path, identity: dict, config: dict, config_sha: str,
                 data_by_split: dict, support: dict, source_hash: str,
                 cebra_version: str, torch_version: str, cebra) -> str:
    if verify_complete(folder, identity, config, config_sha, data_by_split, support,
                       source_hash, cebra_version, torch_version):
        return "complete_verified"
    record, has_checkpoint = verify_checkpoint(
        folder, config, config_sha, support, source_hash, cebra_version, torch_version)
    if has_checkpoint:
        return "resume_embeddings"
    if record.get("status") == "fit_in_progress":
        candidates = sorted(folder.glob(".checkpoint.*.tmp"))
        if candidates:
            if len(candidates) != 1:
                raise RuntimeError(f"Ambiguous uncommitted checkpoints; inspect manually: {candidates}")
            load_validated_checkpoint(candidates[0], cebra, identity["condition"])
            return "resume_checkpoint_commit"
    if (folder / "manifest.json").exists():
        raise RuntimeError(f"Invalid final manifest: {folder / 'manifest.json'}")
    return "new_fit"


def write_checkpoint(model, folder: Path, config: dict, config_sha: str,
                     support: dict, source_hash: str, cebra_version: str,
                     torch_version: str, run_id: str) -> dict:
    checkpoint = folder / "checkpoint.pt"
    temporary = folder / f".checkpoint.{uuid.uuid4().hex}.tmp"
    model.save(str(temporary))
    with temporary.open("r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, checkpoint)
    losses = np.asarray(model.state_dict_["loss"], dtype=np.float32)
    if len(losses) != MAX_ITERATIONS or not np.isfinite(losses).all():
        raise RuntimeError(f"Unexpected CEBRA loss history length/content: {len(losses)}")
    loss_path = folder / "training_loss.npy"
    atomic_npy(loss_path, losses)
    record = {
        "status": "checkpoint_ready", "run_id": run_id,
        "config": config, "config_sha256": config_sha,
        "split_support": support, "source_data_sha256": source_hash,
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_loss_sha256": sha256_file(loss_path),
        "training_loss_rows": int(len(losses)), "completed_utc": utc_now(),
        "script_sha256": sha256_file(Path(__file__)),
        "cebra_version": cebra_version, "torch_version": torch_version,
    }
    atomic_json(folder / "checkpoint_manifest.json", record)
    return record


def load_or_fit_model(folder: Path, identity: dict, config: dict, config_sha: str,
                      data_by_split: dict, support: dict, source_hash: str,
                      cebra, torch, cebra_version: str, torch_version: str,
                      run_id: str, log_path: Path) -> tuple[object, dict, bool]:
    record, has_checkpoint = verify_checkpoint(
        folder, config, config_sha, support, source_hash, cebra_version, torch_version)
    checkpoint = folder / "checkpoint.pt"
    if has_checkpoint:
        model, losses = load_validated_checkpoint(checkpoint, cebra, identity["condition"])
        if record.get("status") == "fit_in_progress":
            loss_path = folder / "training_loss.npy"
            if not loss_path.exists():
                atomic_npy(loss_path, losses)
            recovered = {
                **record, "status": "checkpoint_ready",
                "checkpoint_sha256": sha256_file(checkpoint),
                "training_loss_sha256": sha256_file(loss_path),
                "training_loss_rows": int(len(losses)), "recovered_utc": utc_now(),
            }
            atomic_json(folder / "checkpoint_manifest.json", recovered)
            record = recovered
        append_log(log_path, "checkpoint_reused", **identity,
                   checkpoint_sha256=record["checkpoint_sha256"])
        return model, record, False

    checkpoint_manifest = folder / "checkpoint_manifest.json"
    if checkpoint_manifest.exists():
        prior = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
        if prior.get("status") != "fit_in_progress":
            raise RuntimeError(f"Refusing to retrain over a recorded checkpoint: {checkpoint_manifest}")
        candidates = sorted(folder.glob(".checkpoint.*.tmp"))
        if candidates:
            if len(candidates) != 1:
                raise RuntimeError(f"Ambiguous uncommitted checkpoints; inspect manually: {candidates}")
            temporary = candidates[0]
            model, losses = load_validated_checkpoint(temporary, cebra, identity["condition"])
            with temporary.open("r+b") as stream:
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, checkpoint)
            loss_path = folder / "training_loss.npy"
            atomic_npy(loss_path, losses)
            record = {
                **prior, "status": "checkpoint_ready",
                "checkpoint_sha256": sha256_file(checkpoint),
                "training_loss_sha256": sha256_file(loss_path),
                "training_loss_rows": int(len(losses)),
                "recovered_from_temporary_checkpoint": temporary.name,
                "recovered_utc": utc_now(),
                "script_sha256": sha256_file(Path(__file__)),
            }
            atomic_json(checkpoint_manifest, record)
            append_log(log_path, "orphan_checkpoint_recovered", **identity,
                       checkpoint_sha256=record["checkpoint_sha256"],
                       recovered_from=temporary.name)
            return model, record, False
    else:
        atomic_json(checkpoint_manifest, {
            "status": "fit_in_progress", "run_id": run_id,
            "config": config, "config_sha256": config_sha,
            "split_support": support, "source_data_sha256": source_hash,
            "script_sha256": sha256_file(Path(__file__)),
            "cebra_version": cebra_version, "torch_version": torch_version,
            "started_utc": utc_now(),
        })
    atomic_json(folder / "run_state.json", {"status": "fitting", "run_id": run_id,
                                              "identity": identity, "updated_utc": utc_now()})
    append_log(log_path, "fit_started", **identity, config_sha256=config_sha)
    random.seed(identity["seed"])
    np.random.seed(identity["seed"])
    torch.manual_seed(identity["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(identity["seed"])
    parameters = fit_parameters(identity["condition"])
    model = cebra.CEBRA(**parameters)
    train = data_by_split["train"]
    label = train["behavior"] if identity["condition"] == "behavior" else None
    started = time.perf_counter()
    model.fit(train["neural"], label) if label is not None else model.fit(train["neural"])
    elapsed = time.perf_counter() - started
    record = write_checkpoint(model, folder, config, config_sha, support, source_hash,
                              cebra_version, torch_version, run_id)
    record["fit_seconds"] = elapsed
    atomic_json(folder / "checkpoint_manifest.json", record)
    append_log(log_path, "checkpoint_ready", **identity,
               checkpoint_sha256=record["checkpoint_sha256"], fit_seconds=elapsed)
    return model, record, True


def export_cell(folder: Path, identity: dict, config: dict, config_sha: str,
                data_by_split: dict, support: dict, source_hash: str,
                model, checkpoint_record: dict, cebra_version: str,
                torch_version: str, run_id: str, log_path: Path) -> dict:
    checkpoint_sha = checkpoint_record["checkpoint_sha256"]
    embedding_dir = folder / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    output_records = {}
    for split in SPLITS:
        data = data_by_split[split]
        path = embedding_dir / f"{split}.npz"
        if path.exists():
            output_records[split] = verify_embedding(
                path, data, identity["condition"], identity["rat"], identity["fold"],
                identity["seed"], config_sha, checkpoint_sha)
            append_log(log_path, "embedding_reused", **identity, split=split,
                       embedding_sha256=output_records[split]["sha256"])
            continue
        embedding = np.asarray(model.transform(data["neural"]), dtype=np.float32)
        if embedding.shape != (len(data["source_id"]), OUTPUT_DIMENSION) or not np.isfinite(embedding).all():
            raise RuntimeError(f"Invalid embedding for {identity}/{split}: {embedding.shape}")
        arrays = {
            "embedding": embedding,
            "behavior": data["behavior"],
            "source_id": data["source_id"],
            "trial_id": data["trial_id"],
            "context_valid": data["context_valid"],
            "rat": np.asarray(identity["rat"]), "fold": np.asarray(identity["fold"], dtype=np.int64),
            "seed": np.asarray(identity["seed"], dtype=np.int64),
            "condition": np.asarray(identity["condition"]),
            "config_sha256": np.asarray(config_sha),
            "checkpoint_sha256": np.asarray(checkpoint_sha),
        }
        atomic_npz(path, arrays)
        output_records[split] = verify_embedding(
            path, data, identity["condition"], identity["rat"], identity["fold"],
            identity["seed"], config_sha, checkpoint_sha)
        atomic_json(folder / "run_state.json", {
            "status": f"embedding_{split}_ready", "run_id": run_id,
            "identity": identity, "embeddings": output_records,
            "updated_utc": utc_now(),
        })
        append_log(log_path, "embedding_ready", **identity, split=split,
                   embedding_sha256=output_records[split]["sha256"])

    manifest = {
        "schema_version": 1, "status": "complete", "identity": identity,
        **identity, "config": config, "config_sha256": config_sha,
        "source_data_path": f"data/rat_hippocampus/{identity['rat']}.jl",
        "source_data_sha256": source_hash,
        "source_code_sha256": {name: sha256_file(path) for name, path in source_paths().items()},
        "split_support": support,
        "checkpoint_path": "checkpoint.pt", "checkpoint_sha256": checkpoint_sha,
        "checkpoint_manifest_sha256": sha256_file(folder / "checkpoint_manifest.json"),
        "training_loss_path": "training_loss.npy",
        "training_loss_sha256": checkpoint_record["training_loss_sha256"],
        "embeddings": output_records,
        "fit_parameters": fit_parameters(identity["condition"]),
        "cebra_version": cebra_version, "torch_version": torch_version,
        "runner_sha256": sha256_file(Path(__file__)), "completed_utc": utc_now(),
        "fit_seconds": checkpoint_record.get("fit_seconds"),
        "held_out_policy": "fit on train rows only; transform train/valid/test; no test-based selection",
    }
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"Final manifest appeared during run; refusing overwrite: {manifest_path}")
    atomic_json(manifest_path, manifest)
    atomic_json(folder / "run_state.json", {"status": "complete", "run_id": run_id,
                                             "identity": identity,
                                             "manifest_sha256": sha256_file(manifest_path),
                                             "updated_utc": utc_now()})
    append_log(log_path, "cell_complete", **identity,
               manifest_sha256=sha256_file(manifest_path))
    return manifest


def cells(args):
    for rat in RATS:
        if args.rat and args.rat != rat:
            continue
        for fold in FOLDS:
            if args.fold is not None and args.fold != fold:
                continue
            for seed in SEEDS:
                if args.seed is not None and args.seed != seed:
                    continue
                for condition in CONDITIONS:
                    if args.condition and args.condition != condition:
                        continue
                    yield {"rat": rat, "fold": fold, "seed": seed, "condition": condition}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Allow new fits; omitted means dry-run")
    parser.add_argument("--dry-run", action="store_true", help="Explicitly force read-only planning/verification")
    parser.add_argument("--max-new-fits", type=int, default=None,
                        help="Upper bound on new model.fit calls; checkpoint resumes do not count")
    parser.add_argument("--rat", choices=RATS)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--condition", choices=CONDITIONS)
    args = parser.parse_args()
    if args.execute and args.dry_run:
        parser.error("--execute and --dry-run are mutually exclusive")
    if args.max_new_fits is not None and args.max_new_fits < 0:
        parser.error("--max-new-fits must be >= 0")

    import cebra
    import torch

    version_cebra = getattr(cebra, "__version__", "unknown")
    version_torch = str(torch.__version__)
    selected = list(cells(args))
    cache = {}
    inspected = []
    for identity in selected:
        key = (identity["rat"], identity["fold"])
        if key not in cache:
            cache[key] = load_split_data(*key)
        data_by_split = cache[key]
        config = cell_config(identity["condition"])
        config_sha = sha256_json(config)
        source_hashes = {data["source_sha256"] for data in data_by_split.values()}
        if len(source_hashes) != 1:
            raise AssertionError(f"Source hash differs between splits: {key}")
        support = split_support(data_by_split)
        folder = cell_dir(identity["rat"], identity["fold"], identity["condition"], identity["seed"])
        state = inspect_cell(folder, identity, config, config_sha, data_by_split, support,
                             next(iter(source_hashes)), version_cebra, version_torch, cebra)
        inspected.append((identity, state, config_sha, data_by_split, support,
                          next(iter(source_hashes)), config, folder))

    counts = {name: sum(state == name for _, state, *_ in inspected)
              for name in ("complete_verified", "resume_embeddings",
                           "resume_checkpoint_commit", "new_fit")}
    limit = args.max_new_fits
    permitted_new = counts["new_fit"] if limit is None else min(counts["new_fit"], limit)
    if not args.execute:
        print(json.dumps({
            "mode": "dry_run", "output_root": str(OUTPUT_ROOT),
            "selected_cells": len(inspected), "counts": counts,
            "new_fits_permitted_by_limit": permitted_new,
            "new_fits_deferred_by_limit": counts["new_fit"] - permitted_new,
            "seed_range": [min(SEEDS), max(SEEDS)], "conditions": list(CONDITIONS),
            "fit_parameters": {condition: fit_parameters(condition) for condition in CONDITIONS},
            "verified_split_support_cells": len(cache) * len(SPLITS),
            "writes_performed": False,
        }, indent=2), flush=True)
        return

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + f"_pid{os.getpid()}"
    run_dir = OUTPUT_ROOT / "runs" / run_id
    log_path = run_dir / "campaign.log"
    campaign_state_path = run_dir / "campaign_state.json"
    run_dir.mkdir(parents=True, exist_ok=False)
    campaign_state = {"campaign_id": "rat_paper_fig1e_ed4_6seed_v1",
                      "run_id": run_id, "runner_sha256": sha256_file(Path(__file__)),
                      "target_cells": len(inspected), "cells": {},
                      "created_utc": utc_now(), "status": "running"}
    atomic_json(campaign_state_path, campaign_state)
    new_fits = 0
    try:
        for identity, state, config_sha, data_by_split, support, source_hash, config, folder in inspected:
            key = "/".join((identity["rat"], f"fold_{identity['fold']}",
                            identity["condition"], f"seed_{identity['seed']}"))
            if state == "complete_verified":
                campaign_state["cells"][key] = "complete_reused"
                append_log(log_path, "complete_reused", **identity)
                atomic_json(campaign_state_path, campaign_state)
                continue
            if state == "new_fit" and limit is not None and new_fits >= limit:
                campaign_state["cells"][key] = "deferred_max_new_fits"
                atomic_json(campaign_state_path, campaign_state)
                continue
            folder.mkdir(parents=True, exist_ok=True)
            lock = folder / ".cell.lock"
            with cell_lock(lock, run_id):
                # Recheck after taking the per-cell lock in case another process finished it.
                latest = inspect_cell(folder, identity, config, config_sha, data_by_split,
                                      support, source_hash, version_cebra, version_torch, cebra)
                if latest == "complete_verified":
                    campaign_state["cells"][key] = "complete_reused"
                else:
                    model, checkpoint_record, did_fit = load_or_fit_model(
                        folder, identity, config, config_sha, data_by_split, support,
                        source_hash, cebra, torch, version_cebra, version_torch,
                        run_id, log_path)
                    new_fits += int(did_fit)
                    export_cell(folder, identity, config, config_sha, data_by_split,
                                support, source_hash, model, checkpoint_record,
                                version_cebra, version_torch, run_id, log_path)
                    campaign_state["cells"][key] = "complete_new_fit" if did_fit else "complete_resumed"
                campaign_state["new_fits_started"] = new_fits
                atomic_json(campaign_state_path, campaign_state)
        campaign_state["new_fits_started"] = new_fits
        campaign_state["status"] = ("stopped_at_max_new_fits"
                                     if "deferred_max_new_fits" in campaign_state["cells"] else "complete")
        campaign_state["completed_utc"] = utc_now()
        atomic_json(campaign_state_path, campaign_state)
        append_log(log_path, "campaign_finished", status=campaign_state["status"],
                   new_fits_started=new_fits, cells=len(campaign_state["cells"]))
        print(json.dumps({"run_id": run_id, "status": campaign_state["status"],
                          "new_fits_started": new_fits, "run_dir": str(run_dir)}, indent=2))
    except Exception as error:
        campaign_state["status"] = "failed"
        campaign_state["error"] = repr(error)
        campaign_state["new_fits_started"] = new_fits
        campaign_state["updated_utc"] = utc_now()
        atomic_json(campaign_state_path, campaign_state)
        append_log(log_path, "campaign_failed", error=repr(error), new_fits_started=new_fits)
        raise


if __name__ == "__main__":
    main()
