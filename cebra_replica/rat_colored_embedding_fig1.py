"""Figure-1d-style colored CEBRA panels from frozen rat embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FITS = ROOT / "results" / "rat_trial_benchmark_v1"
OUT = ROOT / "results" / "rat_colored_fig1d_32d_first3_v2"
PANELS = (("position_direction", "CEBRA-Behavior"), ("time", "CEBRA-Time"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rat", choices=("achilles", "buddy", "cicero", "gatsby"), default="achilles")
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    parents = {}
    fig = plt.figure(figsize=(10, 5.2))
    color_norm = None
    for panel, (condition, title) in enumerate(PANELS, start=1):
        folder = FITS / args.rat / f"fold_{args.fold}" / condition / f"seed_{args.seed}"
        embeddings, labels, ids = [], [], []
        for split in ("train", "valid", "test"):
            path = folder / f"{split}_embedding.npz"
            with np.load(path, allow_pickle=False) as data:
                embeddings.append(data["embedding"])
                labels.append(data["behavior"])
                ids.append(data["source_id"])
            parents[str(path)] = sha256(path)
        order = np.argsort(np.concatenate(ids))
        embedding = np.concatenate(embeddings)[order]
        label = np.concatenate(labels)[order]
        if embedding.shape[1] != 32 or label.shape != (len(embedding), 3):
            raise AssertionError(f"Unexpected shape: {folder}")
        r_ind = label[:, 1] == 1
        l_ind = label[:, 2] == 1
        if not np.all(r_ind ^ l_ind):
            raise AssertionError(f"Invalid direction one-hot: {folder}")
        this_norm = Normalize(vmin=float(label[:, 0].min()), vmax=float(label[:, 0].max()))
        if color_norm is None:
            color_norm = this_norm
        elif (not np.isclose(color_norm.vmin, this_norm.vmin) or
              not np.isclose(color_norm.vmax, this_norm.vmax)):
            raise AssertionError("Inconsistent position scale across panels")
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        ax.scatter(*embedding[l_ind, :3].T, c=label[l_ind, 0], cmap="cool",
                   norm=color_norm, s=0.5)
        ax.scatter(*embedding[r_ind, :3].T, c=label[r_ind, 0], cmap="viridis",
                   norm=color_norm, s=0.5)
        ax.set_xlim((-0.7, 0.7))
        ax.set_ylim((-0.7, 0.7))
        ax.set_zlim((-0.7, 0.7))
        ax.axis("off")
        ax.set_title(title)
    fig.suptitle(f"{args.rat.title()} — first 3 coordinates of 32D embeddings "
                 f"(fold {args.fold}, seed {args.seed})", fontsize=12)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.23)
    for xpos, cmap, direction in ((0.16, "cool", "Leftward trials"),
                                  (0.56, "viridis", "Rightward trials")):
        bar_axis = fig.add_axes((xpos, 0.12, 0.28, 0.025))
        bar = fig.colorbar(ScalarMappable(norm=color_norm, cmap=cmap),
                           cax=bar_axis, orientation="horizontal")
        bar.set_label(f"{direction}: position (cm)", fontsize=9)
        ticks_m = np.linspace(color_norm.vmin, color_norm.vmax, 3)
        bar.set_ticks(ticks_m)
        bar.set_ticklabels([f"{value * 100:.0f}" for value in ticks_m])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"fig1d_cebra_panels_with_key_{args.rat}_fold{args.fold}_seed{args.seed}.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    (OUT / f"PROVENANCE_{args.rat}_fold{args.fold}_seed{args.seed}.json").write_text(
        json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(),
                    "script_sha256": sha256(Path(__file__)), "parents": parents,
                    "figure_source": "https://cebra.ai/docs/cebra-figures/figures/Figure1.html",
                    "scope": "Only CEBRA Behavior/Time columns; 32D first-three-coordinate "
                             "diagnostic from trial-split fits, not the paper's final 3D cohort."},
                   indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
