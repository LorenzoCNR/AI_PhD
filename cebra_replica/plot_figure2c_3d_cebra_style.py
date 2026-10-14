"""Render Figure 2c-style panels from frozen 3D CEBRA rat embeddings.

The panel layout and gray 3D scatter follow the official cebra-figures
Figure 2 notebook. This script only plots existing embeddings; it never fits.
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


ROOT = Path(__file__).resolve().parents[1]
FITS = ROOT / "results" / "rat_figure2c_3d_v1"
OUT = ROOT / "results" / "rat_figure2c_3d_cebra_style_v1"
RATS = ("achilles", "buddy", "cicero", "gatsby")
CONDITIONS = (
    "position",
    "direction",
    "position_direction",
    "position_shuffled",
    "direction_shuffled",
    "position_direction_shuffled",
)
TITLES = (
    "position_only",
    "direction_only",
    "position+direction",
    "position, shuffled",
    "direction, shuffled",
    "p+d, shuffled",
)
SOURCE_URL = "https://cebra.ai/docs/cebra-figures/figures/Figure2.html"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_figure(rat: str, seed: int) -> Path:
    parent_hashes: dict[str, str] = {}
    panels: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    expected_ids: np.ndarray | None = None
    expected_behavior: np.ndarray | None = None
    expected_trial_id: np.ndarray | None = None
    expected_time_id: np.ndarray | None = None
    for condition in CONDITIONS:
        run = FITS / rat / condition / f"seed_{seed}"
        manifest_path = run / "result.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing completed fit manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("rat") != rat or manifest.get("condition") != condition
                or manifest.get("seed") != seed or manifest.get("output_dimension") != 3):
            raise RuntimeError(f"Unexpected fit metadata: {manifest_path}")
        parent_hashes[str(manifest_path)] = sha256(manifest_path)
        embedding_path = run / "all_embedding.npz"
        expected_hash = manifest.get("artifacts", {}).get("all_embedding.npz")
        if not expected_hash or sha256(embedding_path) != expected_hash:
            raise RuntimeError(f"Embedding hash mismatch: {embedding_path}")
        parent_hashes[str(embedding_path)] = expected_hash
        with np.load(embedding_path, allow_pickle=False) as data:
            embedding = data["embedding"]
            source_id = data["source_id"]
            behavior = data["behavior"]
            trial_id = data["trial_id"]
            time_id = data["time_id"]
            valid_mask = data["valid_mask"]
        n = len(embedding)
        if (embedding.shape != (n, 3) or not np.isfinite(embedding).all()
                or len(source_id) != n or len(behavior) != n
                or len(trial_id) != n or len(time_id) != n
                or len(valid_mask) != n or not np.all(valid_mask)):
            raise RuntimeError(f"Invalid embedding metadata/cardinality: {embedding_path}")
        if len(np.unique(source_id)) != n:
            raise RuntimeError(f"Duplicate source rows: {embedding_path}")
        order = np.argsort(source_id)
        sorted_ids = source_id[order]
        sorted_behavior = behavior[order]
        sorted_trial_id = trial_id[order]
        sorted_time_id = time_id[order]
        if expected_ids is None:
            expected_ids, expected_behavior = sorted_ids, sorted_behavior
            expected_trial_id, expected_time_id = sorted_trial_id, sorted_time_id
        elif (not np.array_equal(expected_ids, sorted_ids)
              or not np.array_equal(expected_behavior, sorted_behavior)
              or not np.array_equal(expected_trial_id, sorted_trial_id)
              or not np.array_equal(expected_time_id, sorted_time_id)):
            raise RuntimeError(f"Conditions do not share aligned samples/labels for {rat}")
        panels.append((embedding, source_id, trial_id, time_id))

    destination = OUT / f"figure2c_3d_{rat}_seed{seed}"
    png_path, svg_path = destination.with_suffix(".png"), destination.with_suffix(".svg")
    provenance_path = destination.with_name(destination.name + "_PROVENANCE.json")
    if png_path.exists() or svg_path.exists() or provenance_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing figure output: {destination}")

    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(15, 10), dpi=300)
    for index, (embedding, _ids, _trials, _times) in enumerate(panels):
        ax = fig.add_subplot(2, 3, index + 1, projection="3d")
        # Mirrors the official Figure 2c scatter: first three latents, gray dots.
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c="gray", s=0.5)
        ax.axis("off")
        ax.set_title(TITLES[index], fontsize=20)
    fig.savefig(png_path, dpi=300, transparent=False, facecolor="white", bbox_inches="tight")
    fig.savefig(svg_path, transparent=False, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)),
        "source_figure_code": SOURCE_URL,
        "rat": rat,
        "seed": seed,
        "conditions": list(CONDITIONS),
        "layout": "official Figure 2c 2x3; gray 3D scatter; dimensions 0,1,2",
        "scope": "descriptive visualization from all transformed rows; models fit on contiguous first 80%; test rows were not used for fitting or selection",
        "parents": parent_hashes,
        "outputs": {png_path.name: sha256(png_path), svg_path.name: sha256(svg_path)},
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return png_path


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--rat", choices=RATS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="New destination directory; existing figure outputs are never overwritten")
    args = parser.parse_args()
    if args.out_dir is not None:
        OUT = args.out_dir
    print(make_figure(args.rat, args.seed), flush=True)


if __name__ == "__main__":
    main()
