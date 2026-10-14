"""Render the three MSE-loss embedding panels at the top of Extended Data 8."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "results/monkey_extended8_paper_v1"
EMBED = ROOT / "stage02_embedding_mse"
OUTPUT = ROOT / "stage03_figures/mse_panels"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(mode: str) -> tuple[dict[str, np.ndarray], str]:
    directory = EMBED / mode / "seed_42"
    path = directory / "all_trials.npz"
    manifest_path = directory / "EMBEDDING.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256(path) != manifest["artifact_sha256"]:
        raise ValueError(f"Embedding hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    n = 193 * 600
    for key in ("embedding_raw", "embedding_unit", "neural", "position_xy", "direction", "trial_id", "time_id", "valid_mask"):
        if len(data[key]) != n:
            raise ValueError(f"Cardinality mismatch: {mode}/{key}")
    if data["embedding_raw"].shape != (n, 2) or not np.all(data["valid_mask"]):
        raise ValueError(f"Invalid MSE embedding {mode}")
    return data, sha256(path)


def save(fig, stem: str) -> dict[str, str]:
    png, pdf = OUTPUT / f"{stem}.png", OUTPUT / f"{stem}.pdf"
    if png.exists() or pdf.exists():
        raise FileExistsError(f"Refusing to overwrite {stem}")
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {path.name: sha256(path) for path in (png, pdf)}


def render() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    loaded = {name: load(name) for name in ("target", "position", "time")}
    hashes = {name: value[1] for name, value in loaded.items()}
    if (OUTPUT / "PROVENANCE.json").exists():
        old = json.loads((OUTPUT / "PROVENANCE.json").read_text(encoding="utf-8"))
        if old["parent_embedding_sha256"] != hashes:
            raise ValueError("Existing ED8 figure parents differ")
        for name, expected in old["figure_sha256"].items():
            if sha256(OUTPUT / name) != expected:
                raise ValueError("Existing ED8 figure hash mismatch")
        print("REUSE ED8 MSE figures: hashes verified")
        return

    target = loaded["target"][0]
    target_data = target["embedding_raw"]
    target_label = target["direction"]
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle("CEBRA-behavior trained with target label using MSE loss", fontsize=20)
    ax = fig.add_subplot(121)
    ax.set_title("All trials embedding", fontsize=20, y=-0.1)
    ax.scatter(target_data[:, 0], target_data[:, 1], c=target_label, cmap=plt.cm.hsv, s=0.05)
    ax.axis("off")
    ax = fig.add_subplot(122)
    ax.set_title("Post-averaged embs by direction", fontsize=20, y=-0.1)
    trials = target_data.reshape(193, 600, 2)
    trial_labels = target_label.reshape(193, 600)[:, 1]
    for direction in range(8):
        mean = trials[trial_labels == direction].mean(axis=0)
        ax.scatter(mean[:, 0], mean[:, 1], color=plt.cm.hsv(direction / 8), s=3)
    ax.axis("off")
    output_hashes = save(fig, "EXTENDED8_MSE_TARGET")

    for mode in ("position", "time"):
        data = loaded[mode][0]
        embedding, pos = data["embedding_raw"], data["position_xy"]
        fig = plt.figure(figsize=(12, 5))
        title = "CEBRA-behavior trained with position label using MSE loss" if mode == "position" else "CEBRA-time trained using MSE loss"
        fig.suptitle(title, fontsize=20)
        for slot, column, label in ((121, 0, "x"), (122, 1, "y")):
            ax = fig.add_subplot(slot)
            ax.set_title(label, fontsize=20, y=0)
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=pos[:, column],
                                cmap="seismic", s=0.05, vmin=-15, vmax=15)
            ax.axis("off")
            if column == 1:
                bar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.05,
                                   ticks=np.linspace(-15, 15, 7))
                bar.ax.tick_params(labelsize=15)
                bar.ax.set_title("(cm)", fontsize=10)
        output_hashes.update(save(fig, f"EXTENDED8_MSE_{mode.upper()}"))

    record = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "script_sha256": sha256(Path(__file__)),
              "parent_embedding_sha256": hashes,
              "figure_sha256": output_hashes,
              "reference_plot_source": "CEBRA cebra-figures/src/ExtendedDataFigure8.py",
              "reference_hdf5_used_as_local_data": False,
              "split": "active_all_full_sample_descriptive"}
    (OUTPUT / "PROVENANCE.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"COMPLETE ED8 MSE panels: {OUTPUT}")


if __name__ == "__main__":
    with plt.rc_context({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"}):
        render()
