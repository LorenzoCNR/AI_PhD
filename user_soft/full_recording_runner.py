"""Full-recording, native-3D Stable Soft fits for rat figure embeddings.

This is an isolated descriptive-fit study. It does not overwrite the existing
split-based 32D campaigns, and its full-recording metadata scales must not be
interpreted as held-out statistics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import itertools
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from . import rat_runner as rr, transformer_runner as tr, time_only_runner as tm
from . import behavior_only_runner as bo
from .data import config_hash, eligible_centers, load_official_splits, sha256
from .loss import evaluate_soft, fit_distance_scales


ROOT = rr.ROOT
DEFAULT_OUTPUT = ROOT / "results" / "user_soft_rat_fullrecording_stable3d_v1"
RATS = rr.RATS
ARCHITECTURES = ("cnn", "transformer")
CONDITIONS = ("joint", "time_only", "behavior_only", "position_direction_shuffled")
FIGURE1D_CONDITIONS = ("time_direction_position", "time_only", "direction_position")
SEED = 42
LEFT_CONTEXT = 5
RIGHT_CONTEXT = 5


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_full_recording(rat: str):
    """Join the exact official fold-0 trial splits back by source row ID."""
    splits = load_official_splits(ROOT, rat, 0)
    fields = ("neural", "behavior", "source_id", "trial_id", "time_id", "session_id")
    merged = {key: np.concatenate([getattr(rows, key) for rows in splits.values()], axis=0)
              for key in fields}
    order = np.argsort(merged["source_id"], kind="stable")
    rows = SimpleNamespace(**{key: np.asarray(value[order]) for key, value in merged.items()})
    expected = np.arange(len(rows.source_id), dtype=np.int64)
    if not np.array_equal(rows.source_id, expected):
        raise ValueError(f"Official splits do not reconstruct the complete {rat} source in order")
    source_path = ROOT / "data" / "rat_hippocampus" / f"{rat}.jl"
    if len(rows.neural) != joblib_length(source_path):
        raise ValueError(f"Full-recording row count differs from source archive for {rat}")
    if rows.neural.ndim != 2 or rows.behavior.shape != (len(rows.neural), 3):
        raise ValueError(f"Unexpected full-recording shapes for {rat}: {rows.neural.shape}, {rows.behavior.shape}")
    if len(rows.neural) <= LEFT_CONTEXT + RIGHT_CONTEXT:
        raise ValueError(f"Not enough observations for temporal context: {rat}")
    return rows, source_path


def joblib_length(source_path: Path) -> int:
    import joblib
    data = joblib.load(source_path)
    return len(data["spikes"])


def condition_specs(condition: str):
    if condition in ("joint", "time_direction_position", "position_direction_shuffled"):
        return [dict(spec) for spec in rr.SPECS]
    if condition == "time_only":
        return [dict(spec) for spec in tm.SPECS]
    if condition in ("behavior_only", "direction_position"):
        return [dict(spec) for spec in bo.SPECS]
    raise ValueError(f"Unknown Stable Soft condition: {condition}")


def condition_config(architecture: str, rat: str, condition: str,
                     embedding_temperature: float = 0.1) -> dict:
    factory = rr.frozen_config if architecture == "cnn" else tr.frozen_config
    config = factory(rat, 0, SEED, "soft_v1_stable", updates=10000)
    config.update(
        condition=condition,
        fit_scope="full_recording",
        output_dimension=3,
        metadata_specs=condition_specs(condition),
        metadata_temperature=0.5,
        embedding_temperature=embedding_temperature,
        batch_size=512,
        learning_rate=3e-4,
        max_updates=10000,
        early_stopping=False,
        checkpoint_interval=500,
        diagnostic_interval=500,
        full_recording_scale_policy="ranges_or_feature_bounding_boxes_from_all_rows; no held-out split",
        input_policy="legacy_concat",
        offset_left=LEFT_CONTEXT,
        offset_right=RIGHT_CONTEXT,
        time_bin_ms=25,
        shuffled_behavior_scope=("all_rows_within_one_rat_session" if condition == "position_direction_shuffled" else None),
        shuffle_seed=(SEED + 10000 if condition == "position_direction_shuffled" else None),
    )
    config["condition_interpretation"] = {
        "joint": "local trial time and raw (position,right,left), weights 0.5/0.5",
        "time_direction_position": "local trial time and raw (position,right,left), weights 0.5/0.5",
        "time_only": "local trial time only, weight 1",
        "behavior_only": "raw (position,right,left) Euclidean distance only, weight 1",
        "direction_position": "raw (position,right,left) Euclidean distance only, weight 1",
        "position_direction_shuffled": "joint target; complete behavior rows permuted, local time unchanged",
    }[condition]
    return config


def source_record(architecture: str) -> dict:
    paths = [Path(__file__), Path(rr.__file__), Path(tr.__file__), Path(tm.__file__),
             Path(bo.__file__), Path(rr.__file__).with_name("shuffle_runner.py"),
             Path(rr.__file__).with_name("loss.py"), Path(rr.__file__).with_name("data.py"),
             Path(rr.__file__).with_name("legacy_infonce.py"),
             Path(rr.__file__).with_name("legacy_batch_similarity.py")]
    record = {str(path.resolve()): sha256(path) for path in paths}
    import cebra
    import cebra.models
    model = cebra.models.init("offset10-model", num_neurons=1, num_units=32, num_output=3)
    record["cebra_version"] = cebra.__version__
    record["cebra_model_source"] = str(Path(inspect.getfile(model.__class__)).resolve())
    record["cebra_model_source_sha256"] = sha256(Path(record["cebra_model_source"]))
    record["architecture_source"] = "cebra.models.init(offset10-model)" if architecture == "cnn" else str(Path(tr.__file__).with_name("recovered_encoders.py").resolve())
    if architecture == "transformer":
        record[record["architecture_source"]] = sha256(Path(record["architecture_source"]))
    return record


def rows_with_behavior(rows, behavior: np.ndarray):
    return SimpleNamespace(neural=rows.neural, behavior=behavior,
                           source_id=rows.source_id, trial_id=rows.trial_id,
                           time_id=rows.time_id, session_id=rows.session_id)


def prepare_condition(output: Path, rows, rat: str, condition: str):
    specs = condition_specs(condition)
    permutation = None
    training_rows = rows
    if condition == "position_direction_shuffled":
        permutation = np.random.default_rng(SEED + 10000).permutation(len(rows.behavior))
        training_rows = rows_with_behavior(rows, rows.behavior[permutation].copy())
        perm_dir = output / "shuffle_provenance" / rat / "seed_42"
        perm_path = perm_dir / "behavior_permutation.npz"
        expected_hash = config_hash(permutation.tolist())
        if perm_path.exists():
            with np.load(perm_path, allow_pickle=False) as stored:
                if not np.array_equal(stored["permutation"], permutation):
                    raise RuntimeError(f"Existing full-recording shuffle differs: {perm_path}")
        else:
            if perm_dir.exists() and any(perm_dir.iterdir()):
                raise RuntimeError(f"Partial full-recording shuffle provenance: {perm_dir}")
            perm_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(perm_path, permutation=permutation,
                                recipient_source_id=rows.source_id,
                                behavior_donor_source_id=rows.source_id[permutation],
                                seed=SEED, shuffle_seed=SEED + 10000,
                                scope="all_rows_within_one_rat_session")
            atomic_json(perm_dir / "manifest.json", {
                "rat": rat, "seed": SEED, "shuffle_seed": SEED + 10000,
                "scope": "all rows within this rat/session; time and neural rows fixed",
                "permutation_sha256": sha256(perm_path),
                "permutation_config_hash": expected_hash,
                "rows": len(rows.behavior),
            })
    cpu_values = rr.tensors(training_rows, torch.device("cpu"))
    scales = fit_distance_scales(
        {key: cpu_values[key] for key in ("behavior", "time_id")}, specs,
        split="full_recording", source_ids=training_rows.source_id,
    )
    return training_rows, specs, scales, permutation


def make_encoder(architecture: str, n_features: int, config: dict, device):
    if architecture == "cnn":
        return rr.make_encoder(n_features, config, device)
    if architecture == "transformer":
        return tr.make_encoder(n_features, config, device)
    raise ValueError(f"Unknown architecture: {architecture}")


def batch_objective(model, values, indices, config, specs, scales):
    z, batch = rr.encode_batch(model, values, indices)
    return evaluate_soft(z, batch, specs, version="soft_v1_stable", scales=scales,
                         temperature=config["embedding_temperature"],
                         metadata_temperature=config["metadata_temperature"],
                         allow_full_recording_scales=True)


def save_checkpoint(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def write_history(path: Path, history: list[dict]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("update", "train_loss"))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def fit_cell(output: Path, rows, training_rows, specs, scales, rat: str,
             condition: str, architecture: str, config: dict, sources: dict, device):
    folder = output / rat / condition / architecture / "seed_42"
    checkpoint = folder / "checkpoint.pt"
    manifest = folder / "manifest.json"
    embedding_path = folder / "embedding_full.npz"
    signature = config_hash(dict(config=config, sources=sources,
                                 source_data_sha256=sha256(ROOT / "data" / "rat_hippocampus" / f"{rat}.jl"),
                                 source_ids_sha256=config_hash(rows.source_id.tolist()),
                                 scale_record=scales))
    if manifest.exists():
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if record.get("signature") != signature:
            raise RuntimeError(f"Completed fit signature mismatch: {folder}")
        if not checkpoint.is_file() or sha256(checkpoint) != record.get("checkpoint_sha256"):
            raise RuntimeError(f"Completed checkpoint hash mismatch: {checkpoint}")
        if not embedding_path.is_file() or sha256(embedding_path) != record.get("embedding_sha256"):
            raise RuntimeError(f"Completed embedding hash mismatch: {embedding_path}")
        return record, False
    if folder.exists() and not checkpoint.exists() and any(folder.iterdir()):
        raise RuntimeError(f"Partial output without resumable checkpoint requires inspection: {folder}")
    folder.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats(device)
    model = make_encoder(architecture, rows.neural.shape[1], config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 betas=tuple(config["betas"]), eps=config["eps"],
                                 weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(SEED)
    values = rr.tensors(training_rows, device)
    step, history, diagnostics, elapsed = 0, [], [], 0.0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved.get("signature") != signature:
            raise RuntimeError(f"Partial checkpoint signature mismatch; refusing overwrite: {checkpoint}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator"].cpu())
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda" and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
        step, history = saved["step"], saved["history"]
        diagnostics, elapsed = saved["diagnostics"], saved["seconds"]
    evaluation_generator = torch.Generator().manual_seed(SEED + 20000)
    evaluation_indices = torch.randint(len(training_rows.neural), (config["batch_size"],),
                                       generator=evaluation_generator)
    started = time.perf_counter()

    def commit():
        nonlocal elapsed, started
        if device.type == "cuda":
            torch.cuda.synchronize()
        total = elapsed + time.perf_counter() - started
        save_checkpoint(checkpoint, dict(signature=signature, model=model.state_dict(),
            optimizer=optimizer.state_dict(), generator=generator.get_state(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            step=step, history=history, diagnostics=diagnostics, seconds=total,
            config=config, scales=scales, sources=sources))
        write_history(folder / "training_history.csv", history)
        elapsed = total
        started = time.perf_counter()

    model.train()
    if step == 0:
        commit()
    while step < config["max_updates"]:
        indices = torch.randint(len(training_rows.neural), (config["batch_size"],), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        result = batch_objective(model, values, indices, config, specs, scales)
        if not torch.isfinite(result.loss):
            raise FloatingPointError(f"Non-finite full-recording loss: {rat}/{condition}/{architecture}")
        result.loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in model.parameters()):
            raise FloatingPointError(f"Non-finite gradient: {rat}/{condition}/{architecture}")
        optimizer.step()
        step += 1
        history.append({"update": step, "train_loss": float(result.loss.detach())})
        if step % config["diagnostic_interval"] == 0 or step == config["max_updates"]:
            model.eval()
            with torch.no_grad():
                diagnostic = batch_objective(model, values, evaluation_indices, config, specs, scales)
            diagnostics.append({"update": step, "full_recording_batch_loss": float(diagnostic.loss),
                                **diagnostic.diagnostics()})
            model.train()
        if step % config["checkpoint_interval"] == 0 or step == config["max_updates"]:
            commit()
            print(f"UPDATE {rat}/{condition}/{architecture}: {step}/{config['max_updates']}", flush=True)

    model.eval()
    display_values = rr.tensors(rows, device)
    parts = []
    with torch.no_grad():
        for start in range(0, len(rows.neural), config["batch_size"]):
            indices = torch.arange(start, min(start + config["batch_size"], len(rows.neural)))
            z, _ = rr.encode_batch(model, display_values, indices)
            parts.append(z.cpu().numpy())
    raw = np.concatenate(parts).astype(np.float32, copy=False)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    if raw.shape != (len(rows.neural), config['output_dimension']) or not np.isfinite(raw).all() or np.any(norms <= 0):
        raise RuntimeError(f"Invalid native embedding for {rat}/{condition}/{architecture}: {raw.shape}")
    unit = raw / norms
    valid_mask = np.zeros(len(rows.neural), dtype=bool)
    valid_mask[eligible_centers(rows)] = True
    np.savez_compressed(embedding_path, embedding=raw, embedding_raw=raw,
                        embedding_unit=unit, behavior=rows.behavior,
                        source_id=rows.source_id, trial_id=rows.trial_id,
                        time_id=rows.time_id, session_id=rows.session_id,
                        valid_mask=valid_mask, rat=rat, condition=condition,
                        architecture=architecture, seed=SEED,
                        fit_scope="full_recording", output_dimension=config['output_dimension'])
    record = dict(signature=signature, created_utc=now(), config=config, sources=sources,
                  scales=scales, fit_scope="full_recording_all_rows_no_heldout_split",
                  rows=len(rows.neural), valid_window_rows=int(valid_mask.sum()),
                  condition=condition, architecture=architecture, rat=rat, seed=SEED,
                  fit_seconds=elapsed, actual_updates=step,
                  parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                  device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                  cuda_available=torch.cuda.is_available(), torch_version=torch.__version__,
                  peak_gpu_allocated=(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None),
                  peak_gpu_reserved=(torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None),
                  checkpoint_sha256=sha256(checkpoint), embedding_sha256=sha256(embedding_path),
                  objective="soft_v1_stable", checkpoint_selection="fixed terminal checkpoint; no validation or early stopping",
                  scale_scope="all rows of this rat recording, including rows used to fit the embedding",
                  plotted_coordinates="first three native coordinates; no dimensionality reduction")
    atomic_json(manifest, record)
    print(f"COMPLETE {rat}/{condition}/{architecture}: {config['output_dimension']}D, {len(rows.neural)} full rows, {elapsed:.1f}s", flush=True)
    return record, True


def process_lock(output: Path):
    lock = output / "RUNNING.lock"
    if lock.exists():
        try:
            previous_pid = int(lock.read_text(encoding="ascii").strip())
        except (ValueError, OSError):
            raise RuntimeError(f"Unverifiable run lock requires inspection: {lock}")
        try:
            os.kill(previous_pid, 0)
        except OSError:
            lock.unlink()
        else:
            raise RuntimeError(f"A runner may still be active (PID {previous_pid}): {lock}")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode("ascii"))
    os.close(descriptor)
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run/resume all 32 full-recording fits")
    parser.add_argument("--dry-run", action="store_true", help="Print shapes/configs without training or writes")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--conditions", nargs="+",
                        choices=tuple(dict.fromkeys(CONDITIONS + FIGURE1D_CONDITIONS)),
                        default=list(CONDITIONS))
    parser.add_argument("--embedding-temperature", type=float, default=0.1)
    parser.add_argument("--max-new-fits", type=int, default=32,
                        help="Stop after this many fits newly completed in this invocation")
    args = parser.parse_args()
    if args.execute and args.dry_run:
        parser.error("choose either --execute or --dry-run")
    if not args.execute and not args.dry_run:
        args.dry_run = True
    planned_count = len(RATS) * len(args.conditions) * len(ARCHITECTURES)
    if not 1 <= args.max_new_fits <= planned_count:
        parser.error(f"--max-new-fits must be between 1 and {planned_count}")
    output = args.output.resolve()
    results_root = (ROOT / "results").resolve()
    if results_root not in output.parents:
        parser.error("output must be a new study root inside this project's results directory")
    if args.device == "cuda" and not torch.cuda.is_available() and args.execute:
        parser.error("CUDA was requested but is unavailable; choose --device cpu explicitly")
    sources = {architecture: source_record(architecture) for architecture in ARCHITECTURES}
    configs = [(rat, condition, architecture,
                condition_config(architecture, rat, condition, args.embedding_temperature))
               for rat, condition, architecture in itertools.product(RATS, args.conditions, ARCHITECTURES)]
    plan = dict(created_utc=now(), planned=len(configs), rats=list(RATS), conditions=list(args.conditions),
                architectures=list(ARCHITECTURES), seed=SEED,
                sources=sources, configs=[dict(rat=rat, condition=condition,
                                               architecture=architecture, config=config)
                                          for rat, condition, architecture, config in configs],
                training_protocol="all recording rows; 10-bin offsets [-5,+4]; 10,000 updates; batch 512; Adam 3e-4; stable scale fitted from same full recording; no validation/test split",
                output_dimension=3, fits_are_descriptive_not_heldout=True)
    print(json.dumps({"planned": len(configs), "output": str(output),
                      "fits": [dict(rat=r, condition=c, architecture=a,
                                   output_dimension=cfg["output_dimension"], rows="all rows")
                               for r, c, a, cfg in configs]}, indent=2), flush=True)
    if args.dry_run:
        for rat in RATS:
            rows, source_path = load_full_recording(rat)
            print(json.dumps({"rat": rat, "rows": len(rows.neural),
                              "features": rows.neural.shape[1],
                              "source": str(source_path),
                              "source_sha256": sha256(source_path)}, indent=2), flush=True)
        return

    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "CAMPAIGN_PLAN.json"
    if plan_path.exists():
        prior_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if prior_plan.get("configs") != plan["configs"] or prior_plan.get("sources") != sources:
            raise RuntimeError("Existing full-recording plan differs; refusing to overwrite or resume")
    else:
        atomic_json(plan_path, plan)
    lock = process_lock(output)
    state_path = output / "CAMPAIGN_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
        "planned": len(configs), "records": {}, "created_utc": now()}
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    completed_new = 0
    try:
        rows_by_rat = {rat: load_full_recording(rat)[0] for rat in RATS}
        for rat, condition, architecture, config in configs:
            key = f"{rat}/{condition}/{architecture}/seed_42"
            rows = rows_by_rat[rat]
            training_rows, specs, scales, _ = prepare_condition(output, rows, rat, condition)
            state.update(status="running", current=key, updated_utc=now())
            atomic_json(state_path, state)
            try:
                record, created = fit_cell(output, rows, training_rows, specs, scales,
                                           rat, condition, architecture, config,
                                           sources[architecture], device)
            except BaseException as exc:
                state.setdefault("records", {})[key] = {"status": "failed_or_interrupted",
                                                         "error": repr(exc), "updated_utc": now()}
                state.update(status="interrupted", updated_utc=now())
                atomic_json(state_path, state)
                raise
            state.setdefault("records", {})[key] = {"status": "complete",
                "manifest": str(output / rat / condition / architecture / "seed_42" / "manifest.json"),
                "signature": record["signature"], "checkpoint_sha256": record["checkpoint_sha256"],
                "embedding_sha256": record["embedding_sha256"]}
            if created:
                completed_new += 1
            state.update(current=None, completed=sum(v.get("status") == "complete" for v in state["records"].values()),
                         updated_utc=now())
            atomic_json(state_path, state)
            if completed_new >= args.max_new_fits:
                break
        total = sum(v.get("status") == "complete" for v in state.get("records", {}).values())
        state.update(status="complete" if total == len(configs) else "requested_limit",
                     current=None, completed=total, completed_utc=now())
        atomic_json(state_path, state)
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
