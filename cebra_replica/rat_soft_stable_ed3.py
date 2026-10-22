"""Extended Data Figure 3 analogue with frozen rat Soft CNN/Transformer fits.

The layout, two direction-dependent colormaps and first-two-coordinate view
follow the official CEBRA ED3 notebook. All points are held-out test rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "cebra_replica"))
from rat_figure_gap_closure_analysis import index_manifests, load_fit, RATS  # noqa: E402
from rat_soft_stable_tables import locate, load_valid, verify_pair  # noqa: E402


OUT = PROJECT / "results/RAT_FIGURES/Extended_Data/ED3"
OFFICIAL_URL = "https://cebra.ai/docs/cebra-figures/figures/ExtendedDataFigure3.html"
SEED = 42
FOLD = 0

# Keep the order from the official ED3 notebook; autoLFADS is unavailable in
# the current frozen rat replication. New rows are appended, with red labels.
METHODS = (
    ("CEBRA-Behavior", "cebra", "CEBRA-Behavior-32D", "raw", "42"),
    ("conv-piVAE w/labels", "pivae", "Conv-piVAE-z_mean", "posterior", "42"),
    ("CEBRA-Time", "cebra", "CEBRA-Time-32D", "raw", "42"),
    ("conv-piVAE", "pivae", "Conv-piVAE-z_mean", "raw", "42"),
    ("tSNE", "baseline", "opentsne", "raw", "42"),
    ("UMAP", "baseline", "umap", "raw", "42"),
    ("Soft CNN · joint", "soft", "CNN-joint", "raw", "42"),
    ("Soft CNN · time", "soft", "CNN-time", "raw", "42"),
    ("Soft Transformer · joint", "soft", "Transformer-joint", "raw", "42"),
    ("Soft Transformer · time", "soft", "Transformer-time", "raw", "42"),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(PROJECT).as_posix()


def load_panel(index: dict, row: tuple, rat: str, representation: str) -> dict:
    display, kind, method, variant, seed = row
    if kind != "soft":
        data = load_fit(index, kind, method, rat, FOLD, seed, "test", variant)
        mask = data["valid_mask"]
        return {"embedding": data["embedding"][mask], "position": data["behavior"][mask, 0],
                "direction": data["behavior"][mask, 1], "source_id": data["source_id"][mask],
                "parent_path": relative(data["embedding_path"]),
                "parent_sha256": data["embedding_sha256"], "fitted_dim": data["embedding"].shape[1],
                "label_informed_inference": kind == "pivae" and variant == "posterior"}
    found = locate(method, rat, FOLD, SEED)
    if found is None:
        raise FileNotFoundError(f"Missing stable frozen fit: {method}/{rat}/fold{FOLD}/seed{SEED}")
    emb_path, metrics_path = found
    verify_pair(emb_path, metrics_path)
    data = load_valid(emb_path, representation)
    with np.load(emb_path, allow_pickle=False) as z:
        direction = z["behavior"][z["valid_mask"].astype(bool), 1]
    return {"embedding": data["embedding"], "position": data["position"],
            "direction": direction, "source_id": data["source_id"],
            "parent_path": relative(emb_path), "parent_sha256": sha256(emb_path),
            "fitted_dim": data["embedding"].shape[1], "label_informed_inference": False}


def scatter(ax, data: dict) -> None:
    xy = data["embedding"][:, :2]
    if xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("ED3 requires two finite displayed coordinates")
    direction = data["direction"] > 0
    # Exact ED3 choices: viridis for direction > 0, cool otherwise, s=.5,
    # alpha=.7. Rasterization keeps the vector PDF useful at this panel count.
    ax.scatter(*xy[direction].T, c=data["position"][direction], s=0.5,
               cmap="viridis", alpha=0.7, linewidths=0, rasterized=True)
    ax.scatter(*xy[~direction].T, c=data["position"][~direction], s=0.5,
               cmap="cool", alpha=0.7, linewidths=0, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def render(representation: str, index: dict, out: Path) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "text.color": "black",
                         "axes.labelcolor": "black"})
    fig, axes = plt.subplots(len(METHODS), len(RATS), figsize=(12, 30), dpi=200)
    rows = []
    for ri, spec in enumerate(METHODS):
        for ci, rat in enumerate(RATS):
            data = load_panel(index, spec, rat, representation)
            scatter(axes[ri, ci], data)
            if ri == 0:
                axes[ri, ci].set_title(f"Rat {rat.capitalize()}", fontsize=16)
            if ci == 0:
                axes[ri, ci].set_ylabel(spec[0], fontsize=11,
                                        color="red" if spec[1] == "soft" else "black")
            rows.append({"method": spec[0], "rat": rat, "seed": spec[4], "fold": FOLD,
                         "split": "test", "representation": representation if spec[1] == "soft" else "native",
                         "n_rows": len(data["embedding"]), "fit_dimension": data["fitted_dim"],
                         "plotted_dimension": 2, "parent_embedding": data["parent_path"],
                         "parent_sha256": data["parent_sha256"],
                         "label_informed_inference": data["label_informed_inference"]})
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    stem = f"ED3_RAT_EMBEDDINGS_{'RAW' if representation == 'embedding_raw' else 'UNIT'}"
    fig.savefig(out / f"{stem}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Immutable ED3 output already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    index = index_manifests()
    rows = []
    for representation in ("embedding_raw", "embedding_unit"):
        rows.extend(render(representation, index, out))
    with (out / "ED3_PANEL_INDEX.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "script_sha256": sha256(Path(__file__)), "official_reference": OFFICIAL_URL,
                  "official_layout": "methods as rows, four rats as columns; first two coordinates; direction dependent viridis/cool, s=0.5, alpha=0.7",
                  "differences_from_official": ["autoLFADS unavailable and omitted", "four stable Soft rows appended",
                                                "held-out fold-0 test rows", "32D Soft fits display first two coordinates",
                                                "white opaque export to avoid dark viewer backgrounds"],
                  "soft_method_provenance": "Internal soft_v1_stable loss; presentation labels Soft CNN and Soft Transformer",
                  "raw_unit_policy": "Separate exports. Official reference rows have native representations in both exports; only Soft rows vary by representation field.",
                  "posterior_warning": "conv-piVAE with-labels row uses test behavior at inference; descriptive display only, excluded from decoding comparisons.",
                  "panel_index_sha256": sha256(out / "ED3_PANEL_INDEX.csv"),
                  "outputs": {p.name: sha256(p) for p in out.iterdir() if p.suffix in (".png", ".pdf")},
                  "rat_count": len(RATS), "methods": [row[0] for row in METHODS],
                  "panels_per_representation": len(RATS) * len(METHODS)}
    (out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": relative(out), "panels_per_representation": provenance["panels_per_representation"]}))


if __name__ == "__main__":
    main()
