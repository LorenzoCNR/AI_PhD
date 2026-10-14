"""Figure 2c layout from the first three coordinates of frozen 32D pilots.

This is an explicitly labelled diagnostic, not a claim that the paper used
the same 32D models or split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "rat_figure2c_32d_first3_diagnostic_v1"
CONDITIONS = ("position", "direction", "position_direction",
              "position_shuffled", "direction_shuffled", "position_direction_shuffled")
TITLES = ("position_only", "direction_only", "position+direction",
          "position, shuffled", "direction, shuffled", "p+d, shuffled")


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
    source = ROOT / "results" / "rat_trial_benchmark_v1" / args.rat / f"fold_{args.fold}"
    parents = {}
    fig = plt.figure(figsize=(15, 10))
    for i, (condition, title) in enumerate(zip(CONDITIONS, TITLES)):
        folder = source / condition / f"seed_{args.seed}"
        arrays = []
        source_ids = []
        for split in ("train", "valid", "test"):
            path = folder / f"{split}_embedding.npz"
            with np.load(path, allow_pickle=False) as data:
                arrays.append(data["embedding"])
                source_ids.append(data["source_id"])
            parents[str(path)] = sha256(path)
        embedding = np.concatenate(arrays)
        indices = np.concatenate(source_ids)
        if embedding.shape[1] != 32 or len(np.unique(indices)) != len(indices):
            raise AssertionError(f"Invalid frozen embedding: {condition}")
        embedding = embedding[np.argsort(indices)]
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c="gray", s=0.1)
        ax.axis("off")
        ax.set_title(title, fontsize=20)
    fig.suptitle(f"{args.rat.title()} — first 3 coordinates of 32D frozen models "
                 f"(fold {args.fold}, seed {args.seed}; diagnostic only)", fontsize=15)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"figure2c_layout_32d_first3_{args.rat}_fold{args.fold}_seed{args.seed}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    (OUT / f"PROVENANCE_{args.rat}_fold{args.fold}_seed{args.seed}.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)), "parents": parents,
        "meaning": "Official Figure 2c six-condition 2x3 plotting layout applied to "
                   "first three coordinates of independently trained 32D trial-split models; "
                   "not an exact replication of paper model dimensionality or split."
    }, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
