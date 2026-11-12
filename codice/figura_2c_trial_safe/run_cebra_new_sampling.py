"""Run the isolated CEBRA-only Figure 2c sampling correction campaign."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import cebra
from cebra.integrations.sklearn.dataset import SklearnDataset
from cebra.integrations.sklearn.cebra import _init_loader

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))

from prepare_data import DEFAULT_OUTPUT, RATS, sha256, _check_prepared, _read_permutation
from cebra_trial_safe import trial_bounds
from cebra_trial_safe_sampling import TrialSafeDeltaCEBRA

SOURCE_ROOT = DEFAULT_OUTPUT
PREVIOUS_SAFE_ROOT = SOURCE_ROOT / "2c_new_sampling scheme"
DEFAULT_ROOT = SOURCE_ROOT / "2c_cebra_continuous_trial_safe"
ROOT = DEFAULT_ROOT
SCOPES = ("split", "full")
CONDITIONS = ("position", "direction", "position_direction", "position_shuffled",
              "direction_shuffled", "position_direction_shuffled")
RUN_CONDITIONS = tuple(condition for condition in CONDITIONS if condition != "position")
SPLITS = {"split": ("train", "valid", "test"), "full": ("full",)}
SHUFFLED = frozenset(condition for condition in CONDITIONS if condition.endswith("_shuffled"))
NAMES = {"position": "Position", "direction": "Direction",
         "position_direction": "Position+Direction"}
UPDATES = 10_000
BATCH_SIZE = 512
TIME_OFFSET = 10


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def copy_frozen_inputs():
    """Copy the immutable prepared inputs and verify every copied hash."""
    source_manifest_path = SOURCE_ROOT / "data" / "PREPARED.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    _check_prepared(SOURCE_ROOT, source_manifest)
    ROOT.mkdir(parents=True, exist_ok=True)
    for relative in ["data/PREPARED.json", *source_manifest["artifacts"].keys()]:
        source = SOURCE_ROOT / relative
        target = ROOT / relative
        if target.exists():
            if sha256(target) != sha256(source):
                raise RuntimeError(f"new campaign input differs from frozen source: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if sha256(target) != sha256(source):
                raise RuntimeError(f"input copy hash mismatch: {target}")
    _check_prepared(ROOT, source_manifest)
    write_json(ROOT / "INPUT_PROVENANCE.json", {
        "source_root": str(SOURCE_ROOT),
        "prepared_manifest_sha256": sha256(source_manifest_path),
        "source_prepared_manifest": str(source_manifest_path.relative_to(PROJECT)),
        "inputs_verified": True,
        "source_fit_artifacts_reused": False,
    })
    readme = ROOT / "LEGGIMI.md"
    if not readme.exists():
        readme.write_text(
            "# Figura 2c — CEBRA con nuovo campionamento\n\n"
            "Campagna CEBRA isolata. I fit precedenti e i fit Soft non vengono modificati o rilanciati.\n\n"
            "Per Position e Position + Direction, time_delta=10 estrae variazioni da coppie "
            "donatrici (i-10, i) interne allo stesso trial. Il prior di riferimenti e negativi, "
            "la ricerca del positivo, l'encoder, InfoNCE, gli aggiornamenti e la finestra neurale "
            "restano quelli stabiliti nella campagna precedente.\n\n"
            "La modalità Time di CEBRA (offset fisso +10, senza etichette comportamentali) "
            "resta disponibile per altri esperimenti, ma non fa parte di questa campagna 2c.\n\n"
            "Gli output sono in fits/ e lo stato in CAMPAIGN_STATE.json. data/ e permutations/ "
            "sono copie verificate degli input congelati della campagna precedente; le "
            "permutazioni non entrano nei fit senza shuffle.\n",
            encoding="utf-8",
        )
    return source_manifest


def read_rows(root: Path, scope: str, rat: str, split: str) -> dict:
    path = root / "data" / scope / rat / f"{split}.npz"
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def inspect_donor_pools(root: Path) -> dict:
    """Prove every eligible displacement is ten bins inside one trial."""
    report = {}
    for scope in SCOPES:
        for rat in RATS:
            train = read_rows(root, scope, rat, "train" if scope == "split" else "full")
            trial = train["trial_id"]
            time = train["time_id"]
            source = train["source_id"]
            n = len(trial)
            endpoints = np.arange(TIME_OFFSET, n, dtype=np.int64)
            same_trial = trial[endpoints] == trial[endpoints - TIME_OFFSET]
            valid = endpoints[same_trial]
            if not len(valid):
                raise RuntimeError(f"empty valid donor pool for {scope}/{rat}")
            if not np.all(time[valid] - time[valid - TIME_OFFSET] == TIME_OFFSET):
                raise RuntimeError(f"donor time gap is not {TIME_OFFSET} bins for {scope}/{rat}")
            if not np.all(source[valid] - source[valid - TIME_OFFSET] == TIME_OFFSET):
                raise RuntimeError(f"donor source gap is not {TIME_OFFSET} bins for {scope}/{rat}")
            report[f"{scope}/{rat}"] = {
                "training_rows": int(n),
                "trials": int(len(np.unique(trial))),
                "candidate_nonzero_donor_pairs": int(n - TIME_OFFSET),
                "excluded_cross_trial_pairs": int((~same_trial).sum()),
                "eligible_within_trial_pairs": int(len(valid)),
                "within_trial_time_gaps_are_exactly_10_bins": True,
                "within_trial_source_gaps_are_exactly_10_bins": True,
            }
    write_json(root / "SAMPLER_PREFLIGHT.json", {
        "status": "pass",
        "sampler": "uniform over nonzero time_delta donors whose endpoints share a trial",
        "time_delta_bins": TIME_OFFSET,
        "first_10_zero_placeholders_in_stock_pool": "excluded; they are not donor pairs",
        "reference_and_final_positive_trial_constraint": "none; stock CEBRA behavior retained",
        "data": report,
    })
    return report


def reuse_position_fits() -> dict:
    """Copy the eight already-correct Position fits after verifying their provenance."""
    preflight_path = PREVIOUS_SAFE_ROOT / "SAMPLER_PREFLIGHT.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "pass":
        raise RuntimeError("the previous Position donor-safe preflight did not pass")
    reused = {}
    for scope in SCOPES:
        for rat in RATS:
            source = PREVIOUS_SAFE_ROOT / "fits" / scope / rat / "cebra_cnn" / "position"
            target = ROOT / "fits" / scope / rat / "cebra_cnn" / "position"
            manifest_path = source / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"missing reusable Position fit: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            settings = manifest.get("fit_settings", {})
            if (manifest.get("condition") != "position" or settings.get("conditional") != "time_delta"
                    or "within one trial" not in settings.get("donor_sampling", "")
                    or settings.get("labels_passed_to_fit") != ["position"]):
                raise RuntimeError(f"Position fit is not the requested continuous donor-safe fit: {source}")
            for name, digest in manifest.get("artifacts", {}).items():
                file = source / name
                if not file.is_file() or sha256(file) != digest:
                    raise RuntimeError(f"Position fit artifact failed hash verification: {file}")
            if target.exists():
                target_manifest = target / "manifest.json"
                if not target_manifest.is_file() or sha256(target_manifest) != sha256(manifest_path):
                    raise RuntimeError(f"conflicting Position fit already exists: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, target)
            reused[f"{scope}/{rat}/position"] = {
                "source": str(source),
                "manifest_sha256": sha256(manifest_path),
                "artifacts": manifest["artifacts"],
            }
    write_json(ROOT / "REUSED_POSITION_FITS.json", {
        "count": len(reused), "reason": "Position already has the requested continuous donor-safe fits",
        "fits": reused,
    })
    return reused


def inspect_effective_loaders(root: Path) -> dict:
    """Verify dtype-based CEBRA dispatch for every condition without fitting."""
    records = {}
    for scope in SCOPES:
        for rat in RATS:
            split = "train" if scope == "split" else "full"
            data = read_rows(root, scope, rat, split)
            for condition in CONDITIONS:
                permutation = np.arange(len(data["source_id"]), dtype=np.int64)
                if condition in SHUFFLED:
                    permutation = _read_permutation(
                        root / "permutations" / scope / rat / f"{condition}.csv",
                        data["source_id"],
                    )
                behavior = data["behavior"][permutation]
                y = labels(behavior, condition)
                dataset = SklearnDataset(data["neural"], y, device="cpu")
                loader, _ = _init_loader(
                    is_cont=dataset.continuous_index is not None,
                    is_disc=dataset.discrete_index is not None,
                    is_full=False, is_multi=False, is_hybrid=False,
                    shared_kwargs=dict(dataset=dataset, batch_size=BATCH_SIZE, num_steps=1),
                    extra_kwargs=dict(time_offsets=TIME_OFFSET, conditional="time_delta", delta=None),
                )
                key = f"{scope}/{rat}/{condition}"
                if (type(loader).__name__ != "ContinuousDataLoader" or dataset.discrete_index is not None
                        or dataset.continuous_index is None or not hasattr(loader.distribution, "time_difference")):
                    raise RuntimeError(f"unexpected CEBRA loader for {key}: {type(loader).__name__}")
                records[key] = {
                    "label_arrays": [{"dtype": str(a.dtype), "shape": list(a.shape)} for a in y],
                    "continuous_index_dtype": str(dataset.continuous_index.dtype),
                    "continuous_index_shape": list(dataset.continuous_index.shape),
                    "discrete_index": None,
                    "dataloader": type(loader).__name__,
                    "distribution": type(loader.distribution).__name__,
                    "conditional": "time_delta",
                    "time_offset_bins": TIME_OFFSET,
                }
    result = {"status": "pass", "all_fit_labels_continuous": True, "records": records}
    write_json(ROOT / "DATA_LOADER_PREFLIGHT.json", result)
    return result


def labels(behavior, condition):
    base = condition.removesuffix("_shuffled")
    position = behavior[:, 0:1].astype(np.float32)
    direction = behavior[:, 1:2].astype(np.float32)
    if base == "position":
        return (position,)
    if base == "direction":
        return (direction,)
    if base == "position_direction":
        return position, direction
    raise ValueError(f"unknown CEBRA condition: {condition}")


def export_embedding(path: Path, model, data: dict, condition: str,
                     training_behavior: np.ndarray, target_source_id: np.ndarray):
    embedding = model.transform(data["neural"], trial_ids=data["trial_id"], batch_size=BATCH_SIZE)
    embedding = np.asarray(embedding, dtype=np.float32)
    if embedding.shape != (len(data["source_id"]), 3) or not np.isfinite(embedding).all():
        raise ValueError(f"invalid embedding {path}: {embedding.shape}")
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError(f"zero embedding vector in {path}")
    start, end = trial_bounds(data["trial_id"], len(embedding))
    center = np.maximum(np.minimum(np.arange(len(embedding)), end - 4), start + 5)
    indices = center[:, None] + np.arange(-5, 5)
    if not np.all(data["trial_id"][indices] == data["trial_id"][:, None]):
        raise ValueError("exported neural windows cross a trial boundary")
    target = np.concatenate(labels(training_behavior, condition), axis=1).astype(np.float32)
    temporary = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(
        temporary,
        embedding=embedding,
        embedding_unit=embedding / norms,
        true_position=data["behavior"][:, 0],
        true_direction=data["behavior"][:, 1].astype(np.int64),
        training_target=target,
        source_id=data["source_id"],
        trial_id=data["trial_id"],
        time_id=data["time_id"],
        target_source_id=target_source_id,
        window_source_ids=data["source_id"][indices],
    )
    temporary.replace(path)


def fit_one(scope: str, rat: str, condition: str, signature: str):
    output = ROOT / "fits" / scope / rat / "cebra_cnn" / condition
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") != signature:
            raise RuntimeError(f"completed fit signature mismatch: {output}")
        for name, digest in manifest["artifacts"].items():
            file = output / name
            if not file.is_file() or sha256(file) != digest:
                raise RuntimeError(f"completed fit artifact changed: {file}")
        return "reused"

    train_split = "train" if scope == "split" else "full"
    datasets = {split: read_rows(ROOT, scope, rat, split) for split in SPLITS[scope]}
    train = datasets[train_split]
    permutation = np.arange(len(train["source_id"]), dtype=np.int64)
    if condition in SHUFFLED:
        permutation = _read_permutation(
            ROOT / "permutations" / scope / rat / f"{condition}.csv",
            train["source_id"],
        )
    target_behavior = train["behavior"][permutation]
    checkpoint = output / "checkpoint.pt"
    history_path = output / "training_history.csv"
    intent_path = output / "fit_intent.json"
    if intent_path.exists():
        if json.loads(intent_path.read_text(encoding="utf-8"))["signature"] != signature:
            raise RuntimeError(f"incomplete fit signature mismatch: {output}")
    elif checkpoint.exists() or history_path.exists():
        raise RuntimeError(f"incomplete fit has no intent record: {output}")
    else:
        write_json(intent_path, {
            "signature": signature, "scope": scope, "rat": rat,
            "condition": condition, "started": now(),
        })

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if checkpoint.exists() and history_path.exists():
        with history_path.open(newline="", encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
        if len(history) != UPDATES or [int(row["update"]) for row in history] != list(range(1, UPDATES + 1)):
            raise RuntimeError(f"incomplete history: {history_path}")
        model = TrialSafeDeltaCEBRA.load(str(checkpoint), weights_only=False)
    elif checkpoint.exists() or history_path.exists():
        raise RuntimeError(f"checkpoint/history pair is incomplete: {output}")
    else:
        model = TrialSafeDeltaCEBRA(
            model_architecture="offset10-model",
            num_hidden_units=32,
            output_dimension=3,
            batch_size=BATCH_SIZE,
            learning_rate=3e-4,
            temperature=1.0,
            temperature_mode="constant",
            optimizer="adam",
            criterion="infonce",
            distance="cosine",
            conditional="time_delta",
            time_offsets=TIME_OFFSET,
            max_iterations=UPDATES,
            device="cuda_if_available",
            verbose=False,
        )
        model.fit(train["neural"], *labels(target_behavior, condition), trial_ids=train["trial_id"])
        losses = model.state_dict_["loss"]
        if len(losses) != UPDATES or not np.isfinite(np.asarray(losses, dtype=np.float64)).all():
            raise ValueError(f"CEBRA did not produce {UPDATES} finite losses: {output}")
        partial_checkpoint = output / "checkpoint.partial.pt"
        model.save(str(partial_checkpoint))
        partial_checkpoint.replace(checkpoint)
        with history_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("update", "train_loss"))
            writer.writerows((i + 1, float(loss)) for i, loss in enumerate(losses))

    for split, data in datasets.items():
        if split == train_split:
            fit_behavior = target_behavior
            target_ids = train["source_id"][permutation]
        else:
            fit_behavior = data["behavior"]
            target_ids = data["source_id"]
        export_embedding(output / f"{split}_embedding.npz", model, data, condition,
                         fit_behavior, target_ids)

    fit_settings = {
        "condition": condition,
        "labels_passed_to_fit": ["position"] if condition.removesuffix("_shuffled") == "position"
                                else (["direction"] if condition.removesuffix("_shuffled") == "direction"
                                      else ["position", "direction"]),
        "all_labels_dtype": "float32",
        "all_labels_shape": ["N x 1"] if condition.removesuffix("_shuffled") != "position_direction"
                            else ["N x 1", "N x 1"],
        "effective_dataloader": "ContinuousDataLoader",
        "continuous_index_shape": "N x 1" if condition.removesuffix("_shuffled") != "position_direction" else "N x 2",
        "conditional": "time_delta",
        "time_offset_bins": TIME_OFFSET,
        "donor_sampling": "uniform donors i-10 to i whose endpoints share a trial; first 10 placeholders excluded",
        "shuffle": "fit labels permuted by shared saved mapping" if condition in SHUFFLED else "none",
        "training_target_export": "fit labels for train; true labels for valid/test, used only for alignment",
    }
    artifacts = {file.name: sha256(file) for file in output.iterdir()
                 if file.is_file() and file.name not in ("manifest.json",)}
    write_json(manifest_path, {
        "signature": signature,
        "scope": scope,
        "rat": rat,
        "model": "cebra_cnn",
        "condition": condition,
        "seed": 42,
        "embedding_dimension": 3,
        "updates": UPDATES,
        "batch_size": BATCH_SIZE,
        "learning_rate": 0.0003,
        "temperature": 1.0,
        "architecture": "offset10-model",
        "criterion": "infonce",
        "distance": "cosine",
        "optimizer": "adam",
        "window_offsets": list(range(-5, 5)),
        "window_rule": "shift entire 10-bin window within trial; label/bin index remains original",
        "cebra_version": cebra.__version__,
        "torch_version": torch.__version__,
        "device": str(model.device_),
        "fit_settings": fit_settings,
        "artifacts": artifacts,
    })
    return "completed"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=SCOPES, action="append")
    parser.add_argument("--rat", choices=RATS, action="append")
    parser.add_argument("--condition", choices=CONDITIONS, action="append")
    parser.add_argument("--output-root", type=Path,
                        help="separata destinazione per esperimenti futuri; richiesta per --condition time")
    parser.add_argument("--preflight-only", action="store_true",
                        help="copy/hash-check frozen inputs and verify donor pools, without fitting")
    args = parser.parse_args(argv)

    global ROOT
    selected_conditions = args.condition or list(RUN_CONDITIONS)
    if "position" in selected_conditions:
        parser.error("Position ha già 8 fit donor-safe validi e non va rifatta")
    if "time" in selected_conditions and args.output_root is None:
        parser.error("la modalità CEBRA time non appartiene alla campagna 2c; specificare --output-root separata")
    if args.output_root is not None:
        requested_root = args.output_root
        ROOT = (requested_root if requested_root.is_absolute() else PROJECT / requested_root).resolve()
        if "time" in selected_conditions and ROOT == DEFAULT_ROOT.resolve():
            parser.error("la modalità CEBRA time deve scrivere fuori dalla cartella 2c_new_sampling scheme")

    prepared = copy_frozen_inputs()
    sampler_preflight = inspect_donor_pools(ROOT)
    reused_position = reuse_position_fits()
    loader_preflight = inspect_effective_loaders(ROOT)
    if args.preflight_only:
        print(json.dumps({"status": "preflight_pass", "root": str(ROOT),
                          "source_prepared_manifest_sha256": sha256(SOURCE_ROOT / "data" / "PREPARED.json"),
                          "sampler": sampler_preflight,
                          "reused_position_fit_count": len(reused_position),
                          "loader_preflight": loader_preflight}, indent=2))
        return
    inputs = [Path(__file__), HERE / "cebra_trial_safe.py",
              HERE / "cebra_trial_safe_sampling.py", SOURCE_ROOT / "data" / "PREPARED.json"]
    config = json.dumps({
        "scopes": list(SCOPES),
        "rats": list(RATS),
        "conditions": list(RUN_CONDITIONS),
        "updates": UPDATES, "batch_size": BATCH_SIZE,
        "time_offset": TIME_OFFSET, "seed": 42,
        "sampler_preflight": sampler_preflight,
        "loader_preflight": loader_preflight,
        "reused_position_fits": reused_position,
    }, sort_keys=True).encode("utf-8")
    campaign_signature = hashlib.sha256(config + "".join(sha256(path) for path in inputs).encode("ascii")).hexdigest()
    combinations = [(scope, rat, condition)
                    for scope in (args.scope or SCOPES)
                    for rat in (args.rat or RATS)
                    for condition in selected_conditions]
    state_path = ROOT / "CAMPAIGN_STATE.json"
    state = {
        "status": "running", "signature": campaign_signature, "total": len(combinations),
        "done": 0, "current": None, "started": now(),
        "source_campaign_root": str(SOURCE_ROOT),
        "frozen_prepared_manifest_sha256": sha256(SOURCE_ROOT / "data" / "PREPARED.json"),
        "models": ["cebra_cnn"], "conditions": list(selected_conditions),
        "reused_position_fit_count": len(reused_position),
        "sampler": "donor-safe only",
        "label_representation": "all passed labels float32 continuous; Direction N x 1",
    }
    write_json(state_path, state)
    try:
        for scope, rat, condition in combinations:
            state["current"] = {"scope": scope, "rat": rat, "condition": condition}
            write_json(state_path, state)
            print(f"START {scope} {rat} {condition}", flush=True)
            fit_signature = hashlib.sha256(
                f"{campaign_signature}|{scope}|{rat}|{condition}".encode("utf-8")
            ).hexdigest()
            result = fit_one(scope, rat, condition, fit_signature)
            state["done"] += 1
            print(f"{result.upper()} {scope} {rat} {condition}", flush=True)
            write_json(state_path, state)
        state.update(status="complete", current=None, finished=now())
        write_json(state_path, state)
    except Exception:
        state.update(status="error", finished=now(), error=traceback.format_exc())
        write_json(state_path, state)
        raise
    print(json.dumps({"status": state["status"], "root": str(ROOT),
                      "total": state["total"], "done": state["done"]}, indent=2))


if __name__ == "__main__":
    main()
