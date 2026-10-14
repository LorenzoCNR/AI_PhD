"""Export frozen Figure 3c embeddings and plot with the official visual grammar."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import numpy as np
from cebra import CEBRA

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data" / "monkey_reaching_preload_smth_40"
ROOT = PROJECT / "results" / "monkey_figure3_v1"
FIT = ROOT / "stage01_fit" / "figure3c_position_3d_demo" / "seed_42"
OUTPUT = ROOT / "stage02_embedding" / "figure3c_position_3d_demo" / "seed_42"
FIGURES = ROOT / "stage03_figures" / "figure3c_position_3d_demo" / "seed_42"
SPLITS = ROOT / "stage00_data" / "DATA_AND_SPLIT_PROVENANCE.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_embeddings() -> dict:
    fit = json.loads((FIT / "FIT_PROVENANCE.json").read_text(encoding="utf-8"))
    if sha256(FIT / "model.pt") != fit["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    mapping = json.loads(SPLITS.read_text(encoding="utf-8"))
    model = CEBRA.load(str(FIT / "model.pt"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    exports = {}
    for split in ("all", "train", "valid", "test"):
        source = DATA / f"active_{split}.jl"
        if sha256(source) != mapping["source_file_hashes"][source.name]["sha256"]:
            raise ValueError(f"Source hash mismatch: {source.name}")
        target = OUTPUT / f"active_{split}.npz"
        if not target.exists():
            data = joblib.load(source)
            neural = np.asarray(data["spikes"], dtype=np.float32)
            raw = np.asarray(model.transform(neural), dtype=np.float32)
            if raw.shape != (len(neural), 3) or not np.isfinite(raw).all():
                raise ValueError(f"Invalid embedding shape/values: {split}: {raw.shape}")
            norm = np.linalg.norm(raw, axis=1, keepdims=True)
            unit = raw / np.maximum(norm, np.finfo(np.float32).eps)
            n_trials, trial_len = int(data["num_trials"]), int(data["trial_len"])
            ids = np.asarray(mapping["session_trial_indices_by_split"]["active"][split], dtype=np.int32)
            if len(ids) != n_trials or len(raw) != n_trials * trial_len:
                raise ValueError(f"Trial mapping/embedding cardinality mismatch: {split}")
            np.savez_compressed(
                target,
                embedding_raw=raw,
                embedding_unit=unit,
                trial_id=np.repeat(ids, trial_len),
                time_id=np.tile(np.arange(trial_len, dtype=np.int16), n_trials),
                valid_mask=np.ones(len(raw), dtype=bool),
                position_xy=np.asarray(data["pos"], dtype=np.float32),
                velocity_xy=np.asarray(data["vel"], dtype=np.float32),
                direction=np.repeat(np.asarray(data["movement_dir"], dtype=np.int8), trial_len),
                passive=np.repeat(np.asarray(data["passive"], dtype=bool), trial_len),
                split=np.asarray(split),
                checkpoint_sha256=np.asarray(fit["checkpoint_sha256"]),
                source_sha256=np.asarray(sha256(source)),
            )
        with np.load(target) as saved:
            n = len(saved["embedding_raw"])
            for key in ("embedding_unit", "trial_id", "time_id", "valid_mask", "position_xy", "direction", "passive"):
                if len(saved[key]) != n:
                    raise ValueError(f"Saved {split} {key} cardinality mismatch")
            if saved["embedding_raw"].shape[1] != 3:
                raise ValueError(f"Saved {split} embedding dimension mismatch")
            if saved["checkpoint_sha256"].item() != fit["checkpoint_sha256"]:
                raise ValueError(f"Saved {split} parent checkpoint mismatch")
        exports[split] = {"path": str(target), "sha256": sha256(target), "rows": n}
    manifest = OUTPUT / "EMBEDDING_PROVENANCE.json"
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)),
        "checkpoint_sha256": fit["checkpoint_sha256"],
        "split_manifest_sha256": sha256(SPLITS),
        "embeddings": exports,
        "training_repeated": False,
    }
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old["checkpoint_sha256"] != record["checkpoint_sha256"] or old["embeddings"] != exports:
            raise ValueError("Existing embedding provenance conflict")
    else:
        manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def plot(split: str) -> None:
    source = OUTPUT / f"active_{split}.npz"
    with np.load(source) as data:
        embedding = data["embedding_raw"]
        labels = data["position_xy"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"projection": "3d"})
        fig.suptitle("CEBRA-Behavior, labels - continuous position", fontsize=20)
        for axis, column, title in zip(axes, (0, 1), ("x pos", "y pos")):
            axis.set_title(title)
            points = axis.scatter(
                embedding[:, 0], embedding[:, 1], embedding[:, 2],
                c=labels[:, column], cmap="seismic", s=0.05, vmin=-15, vmax=15,
            )
            axis.xaxis.set_pane_color((1, 1, 1, 0))
            axis.yaxis.set_pane_color((1, 1, 1, 0))
            axis.zaxis.set_pane_color((1, 1, 1, 0))
            axis.xaxis.set_ticks([])
            axis.yaxis.set_ticks([])
            axis.zaxis.set_ticks([])
            axis.grid(False)
        colorbar = fig.colorbar(points, ax=axes[1], fraction=0.03, pad=0.05, ticks=np.linspace(-15, 15, 7))
        colorbar.ax.tick_params(labelsize=15)
        colorbar.ax.set_title("(cm)", fontsize=10)
        FIGURES.mkdir(parents=True, exist_ok=True)
        path = FIGURES / f"FIGURE3C_{split.upper()}.png"
        if not path.exists():
            fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    result = export_embeddings()
    for split in ("all", "test"):
        plot(split)
    print(json.dumps({"status": "complete", "embeddings": result["embeddings"], "figures": str(FIGURES)}))


if __name__ == "__main__":
    main()
