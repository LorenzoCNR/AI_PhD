"""Export trial-resolved frozen embeddings for completed Figure 3 paper fits."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from cebra import CEBRA

from fit_figure3_paper import CONFIGS, DATA, PROJECT, ROOT as FIT_ROOT, SEED, SPLITS


ROOT = PROJECT / "results" / "monkey_figure3_paper_v1" / "stage02_embedding"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_one(item: tuple[str, str, str, int]) -> str:
    name, session, _, dimension = item
    fit_dir = FIT_ROOT / name / f"seed_{SEED}"
    manifest = fit_dir / "FIT_PROVENANCE.json"
    if not manifest.exists():
        return f"pending {name}"
    record = json.loads(manifest.read_text(encoding="utf-8"))
    checkpoint = fit_dir / "model.pt"
    if sha256(checkpoint) != record["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {name}")
    source = DATA / f"{session}_all.jl"
    if sha256(source) != record["source_sha256"]:
        raise ValueError(f"Source checksum mismatch: {name}")
    mapping = json.loads(SPLITS.read_text(encoding="utf-8"))
    if session == "all":
        ids = np.arange(int(mapping["trial_counts"]["all_all.jl"]), dtype=np.int32)
    else:
        ids = np.asarray(mapping["session_trial_indices_by_split"][session]["all"], dtype=np.int32)
    dest = ROOT / name / f"seed_{SEED}"
    target = dest / "all_trials.npz"
    provenance = dest / "EMBEDDING_PROVENANCE.json"
    if provenance.exists():
        old = json.loads(provenance.read_text(encoding="utf-8"))
        if old["checkpoint_sha256"] != record["checkpoint_sha256"] or old["embedding_sha256"] != sha256(target):
            raise ValueError(f"Existing embedding mismatch: {name}")
        return f"reused {name}"
    if target.exists():
        raise FileExistsError(f"Embedding without provenance: {target}")
    data = joblib.load(source)
    trial_len = int(data["trial_len"])
    n_trials = int(data["num_trials"])
    neural = np.asarray(data["spikes"], dtype=np.float32)
    if trial_len != 600 or len(ids) != n_trials or len(neural) != n_trials * trial_len:
        raise ValueError(f"Input metadata mismatch: {name}")
    model = CEBRA.load(str(checkpoint))
    embedding = np.asarray(model.transform(neural), dtype=np.float32)
    if embedding.shape != (len(neural), dimension) or not np.isfinite(embedding).all():
        raise ValueError(f"Embedding shape/values mismatch: {name}: {embedding.shape}")
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    unit = embedding / np.maximum(norms, np.finfo(np.float32).eps)
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        embedding_raw=embedding,
        embedding_unit=unit,
        neural=neural,
        trial_id=np.repeat(ids, trial_len),
        time_id=np.tile(np.arange(trial_len, dtype=np.int16), n_trials),
        valid_mask=np.ones(len(neural), dtype=bool),
        position_xy=np.asarray(data["pos"], dtype=np.float32),
        velocity_xy=np.asarray(data["vel"], dtype=np.float32),
        direction=np.repeat(np.asarray(data["movement_dir"], dtype=np.int8), trial_len),
        direction_active_passive=np.repeat(np.asarray(data["movement_dir_actpas"], dtype=np.int8), trial_len),
        passive=np.repeat(np.asarray(data["passive"], dtype=bool), trial_len),
        split=np.asarray("all_descriptive"),
        checkpoint_sha256=np.asarray(record["checkpoint_sha256"]),
    )
    with np.load(target) as saved:
        n = len(saved["embedding_raw"])
        for key in ("embedding_unit", "neural", "trial_id", "time_id", "valid_mask", "position_xy", "direction", "direction_active_passive", "passive"):
            if len(saved[key]) != n:
                raise ValueError(f"Cardinality mismatch: {name}/{key}")
    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fit_manifest_sha256": sha256(manifest),
        "checkpoint_sha256": record["checkpoint_sha256"],
        "source_sha256": record["source_sha256"],
        "stage00_sha256": sha256(SPLITS),
        "script_sha256": sha256(Path(__file__)),
        "embedding_sha256": sha256(target),
        "embedding_shape": list(embedding.shape),
        "trial_count": n_trials,
        "training_repeated": False,
        "split": "all_descriptive",
    }
    provenance.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return f"complete {name} shape={embedding.shape}"


def main() -> None:
    for item in CONFIGS:
        print(export_one(item), flush=True)


if __name__ == "__main__":
    main()
