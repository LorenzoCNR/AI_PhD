"""Render Figure 1E consistency heatmaps from audited saved metric tables."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "rat_rebuild_20261027" / "Figure_1E_ED4"
OUTPUT_PAPER = ROOT / "results" / "figura_1_2026-10-27" / "e" / "Paper"
OUTPUT_SOFT = ROOT / "results" / "figura_1_2026-10-27" / "e" / "Soft"
RATS = ("achilles", "buddy", "cicero", "gatsby")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def display_name(name: str) -> str:
    replacements = (
        ("CEBRA behavior (position + direction)", "CEBRA — Position + Direction"),
        ("CEBRA time", "CEBRA — Time"),
        ("conv-piVAE q(z|x,u)", "conv-piVAE — Position + Direction at inference"),
        ("conv-piVAE q(z|x)", "conv-piVAE — no labels at inference"),
        ("Soft cnn ", "Soft CNN — "),
        ("Soft transformer ", "Soft Transformer — "),
        ("time_direction_position", "Time + Direction + Position"),
        ("direction_position", "Position + Direction"),
        ("time_only", "Time"),
    )
    for before, after in replacements:
        name = name.replace(before, after)
    return name


def annotation_color(image, value: float) -> str:
    """Choose whichever of black or white has greater WCAG contrast."""
    red, green, blue, *_ = image.cmap(image.norm(value))

    def linearize(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    luminance = (0.2126 * linearize(red) + 0.7152 * linearize(green)
                 + 0.0722 * linearize(blue))
    black_contrast = (luminance + 0.05) / 0.05
    white_contrast = 1.05 / (luminance + 0.05)
    return "black" if black_contrast >= white_contrast else "white"


def render(rows, methods, output_dir: Path, filename: str, heading: str,
           source_hashes: dict[str, str]):
    target = output_dir / filename
    outputs = [target.with_suffix(".png"), target.with_suffix(".pdf")]
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    image = None
    matrix_records = []
    for ax, method in zip(axes.flat, methods):
        matrix = np.full((len(RATS), len(RATS)), np.nan)
        for i, rat in enumerate(RATS):
            for j, peer in enumerate(RATS):
                values = [float(row["train_cross_subject_r2"]) for row in rows
                          if row["method"] == method and row["rat"] == rat and row["peer_rat"] == peer]
                if values:
                    matrix[i, j] = float(np.mean(values) * 100)
        image = ax.imshow(matrix, vmin=0, vmax=100, cmap="viridis")
        for i in range(len(RATS)):
            for j in range(len(RATS)):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                            fontsize=7, color=annotation_color(image, matrix[i, j]))
        ax.set_xticks(range(len(RATS)), [rat.capitalize() for rat in RATS], rotation=40,
                      ha="right", fontsize=7)
        ax.set_yticks(range(len(RATS)), [rat.capitalize() for rat in RATS], fontsize=7)
        ax.set_title(display_name(method), fontsize=8)
        matrix_records.append({"method": method, "display_name": display_name(method),
                               "matrix_percent": [[None if not np.isfinite(v) else float(v)
                                                   for v in row] for row in matrix]})
    for ax in axes.flat[len(methods):]:
        ax.set_visible(False)
    fig.subplots_adjust(left=.06, right=.89, bottom=.08, top=.91, wspace=.43, hspace=.60)
    if image is not None:
        color_ax = fig.add_axes([.925, .31, .018, .38])
        fig.colorbar(image, cax=color_ax, label="Train cross-subject consistency R² (%)")
    fig.suptitle(heading, fontsize=13)
    for path in outputs:
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "renderer": Path(__file__).relative_to(ROOT).as_posix(),
        "renderer_sha256": sha256(Path(__file__).resolve()),
        "training_performed": False,
        "metric": "train_cross_subject_r2",
        "aggregation": "mean over available fold-seed records per directed rat pair",
        "seed_count": 6,
        "fold_count": 3,
        "source_hashes": source_hashes,
        "methods": matrix_records,
        "outputs": {path.relative_to(ROOT).as_posix(): sha256(path) for path in outputs},
    }
    (output_dir / "PROVENANCE.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main():
    input_path = SOURCE / "train_cross_subject_consistency.csv"
    audit_path = SOURCE / "support_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise RuntimeError("Support audit is not passing; refusing to plot")
    rows = read_csv(input_path)
    methods = list(dict.fromkeys(row["method"] for row in rows))
    paper_methods = [method for method in methods if not method.startswith("Soft ")]
    soft_methods = [method for method in methods if method.startswith("Soft ")]
    source_hashes = {input_path.relative_to(ROOT).as_posix(): sha256(input_path),
                     audit_path.relative_to(ROOT).as_posix(): sha256(audit_path)}
    render(rows, paper_methods, OUTPUT_PAPER, "FIGURE_1E_PAPER_METHODS",
           "Figure 1E — saved methods; train cross-subject consistency (6 seeds, 3 folds)", source_hashes)
    render(rows, soft_methods, OUTPUT_SOFT, "FIGURE_1E_SOFT_STABLE",
           "Figure 1E — Soft Stable; train cross-subject consistency (6 seeds, 3 folds)", source_hashes)
    print(json.dumps({"paper_method_count": len(paper_methods), "soft_method_count": len(soft_methods),
                      "metric_rows": len(rows), "support_audit": audit["status"],
                      "training_performed": False}, indent=2))


if __name__ == "__main__":
    main()
