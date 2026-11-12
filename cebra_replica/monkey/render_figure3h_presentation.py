"""Cosmetic refresh of the saved Figure 3H plots; no fits or scores are rerun."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "results" / "figura_3_2026-10-29" / "plot" / "h"
SCORES = PROJECT / "results" / "figura_3_2026-10-29" / "processo" / "metriche" / "h" / "Figure_3h_seed_scores.csv"
SUMMARY = PROJECT / "results" / "figura_3_2026-10-29" / "processo" / "metriche" / "h" / "Figure_3h_median_iqr.csv"
MAIN_PROVENANCE = PROJECT / "results" / "figura_3_2026-10-29" / "processo" / "provenienza" / "h" / "PROVENANCE.json"
DOTS_PROVENANCE = PROJECT / "results" / "figura_3_2026-10-29" / "processo" / "provenienza" / "h" / "Figure_3h_model_dots_PROVENANCE.json"

MODELS = ("CEBRA offset10-model", "Soft Stable CNN", "Soft Stable Transformer")
DISPLAY_MODELS = ("CEBRA offset10-model", "Soft CNN", "Soft Transformer")
TASKS = ("position", "direction", "ap")
DIMS = (3, 4, 8, 16, 32)
DIM_COLORS = {
    3: "#4C78A8",
    4: "#F58518",
    8: "#54A24B",
    16: "#E45756",
    32: "#72B7B2",
}
SOFT_RED = "#C62828"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def color_soft_ticklabels(axis) -> None:
    for label in axis.get_yticklabels():
        if label.get_text().startswith("Soft "):
            label.set_color(SOFT_RED)


def render_matrix(summary: list[dict[str, str]]) -> None:
    panel_titles = {
        "position": "Position R² (X,Y)",
        "direction": "Direction accuracy",
        "ap": "Active / passive accuracy",
    }
    values = {
        (row["model"], row["task"], int(row["dimension"])): float(row["median_test_score_pct"])
        for row in summary
    }
    plt.rcParams.update({"font.size": 12, "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(18, 8.2), sharey=True, constrained_layout=False)
    image = None
    for axis, task in zip(axes, TASKS):
        matrix = np.asarray([[values[(model, task, dimension)] for dimension in DIMS] for model in MODELS])
        image = axis.imshow(matrix, cmap="viridis", vmin=50, vmax=90, aspect="auto")
        axis.set_title(panel_titles[task], fontsize=17, pad=14)
        axis.set_xticks(range(len(DIMS)), [str(dimension) for dimension in DIMS])
        axis.set_xlabel("Embedding dimension", fontsize=14)
        axis.set_yticks(range(len(MODELS)))
        if axis is axes[0]:
            axis.set_yticklabels(list(DISPLAY_MODELS))
        else:
            axis.tick_params(axis="y", labelleft=False)
        axis.tick_params(axis="both", labelsize=12)
        for row_index in range(len(MODELS)):
            for column_index in range(len(DIMS)):
                value = matrix[row_index, column_index]
                color = "white" if value < 67 else "black"
                axis.text(column_index, row_index, f"{value:.1f}", ha="center", va="center",
                          fontsize=12, fontweight="bold", color=color)
        axis.set_xticks(np.arange(-0.5, len(DIMS), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(MODELS), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.5)
        axis.tick_params(which="minor", bottom=False, left=False)
    color_soft_ticklabels(axes[0])

    fig.suptitle(
        "Held-out decoding by model and embedding dimension\n"
        "Cell: median across five seeds (42–46); seed-level scores and IQR in the CSV",
        fontsize=17,
        y=0.98,
    )
    colorbar_axis = fig.add_axes([0.94, 0.2, 0.018, 0.62])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Median held-out test score (%)", fontsize=12)
    fig.text(
        0.5,
        0.015,
        "Active/Passive is decoded from Position X,Y-trained embeddings. Position/Direction test rows differ slightly: "
        "CEBRA=15,600, Soft=15,366 (window-boundary mask). Seed-level scores and IQR are in the CSV.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.23, right=0.91, top=0.85, bottom=0.12, wspace=0.22)
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"Figure_3h_models_by_dimension.{extension}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_dots(scores: list[dict[str, str]]) -> None:
    titles = {
        "position": "Position R² (X,Y) — active trials",
        "direction": "Direction accuracy — active trials",
        "ap": "Active/passive accuracy — mixed trials",
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.3), sharey=True)
    model_y = {model: len(MODELS) - 1 - index for index, model in enumerate(MODELS)}
    dimension_offsets = np.linspace(0.24, -0.24, len(DIMS))
    seed_jitter = np.linspace(-0.026, 0.026, 5)

    for axis, task in zip(axes, TASKS):
        task_rows = [row for row in scores if row["task"] == task]
        for model in MODELS:
            for dimension_index, dimension in enumerate(DIMS):
                cell = [row for row in task_rows
                        if row["model"] == model and int(row["dimension"]) == dimension]
                cell.sort(key=lambda row: int(row["seed"]))
                if len(cell) != 5:
                    raise ValueError(f"Expected five seed rows for {model}/{task}/{dimension}; got {len(cell)}")
                values = np.asarray([float(row["score_pct"]) for row in cell], dtype=float)
                y = model_y[model] + dimension_offsets[dimension_index]
                q1, median, q3 = np.percentile(values, [25, 50, 75])
                axis.hlines(y, q1, q3, color=DIM_COLORS[dimension], linewidth=2.2, zorder=2)
                axis.scatter(values, y + seed_jitter, s=32, color=DIM_COLORS[dimension],
                             edgecolors="none", zorder=3)
                axis.scatter([median], [y], marker="D", s=75, color=DIM_COLORS[dimension],
                             edgecolors="#333333", linewidths=0.8, zorder=4)

        axis.set_title(titles[task], fontsize=13, pad=12)
        axis.set_xlim(50, 90)
        axis.set_ylim(-0.48, len(MODELS) - 0.52)
        axis.set_yticks([model_y[model] for model in MODELS], list(DISPLAY_MODELS))
        axis.set_xlabel("Held-out test score (%)", fontsize=12)
        axis.grid(axis="x", alpha=0.22)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].set_ylabel("Model", fontsize=12)
    color_soft_ticklabels(axes[0])
    for axis in axes[1:]:
        axis.tick_params(axis="y", labelleft=False)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=DIM_COLORS[dimension],
               label=f"{dimension}D", markersize=8)
        for dimension in DIMS
    ]
    legend_handles.append(Line2D([0], [0], marker="D", linestyle="", markerfacecolor="white",
                                 markeredgecolor="#333333", label="Median", markersize=8))
    fig.suptitle("Held-out decoding by model", fontsize=17, y=0.97)
    fig.legend(handles=legend_handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.045), title="Embedding dimension")
    fig.text(
        0.5,
        0.005,
        "Dots: individual seeds 42–46; bars: interquartile range; diamonds: median. "
        "Active/passive is decoded from Position X,Y-trained embeddings.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.18, right=0.99, top=0.88, bottom=0.23, wspace=0.15)
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"Figure_3h_model_dots.{extension}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def update_provenance(scores: list[dict[str, str]]) -> None:
    main = json.loads(MAIN_PROVENANCE.read_text(encoding="utf-8"))
    main["presentation_update_2026_11_05"] = {
        "scope": "Figure 3H plot presentation only",
        "title_prefix_removed": "Figure 3H",
        "soft_method_labels": "displayed as Soft CNN and Soft Transformer; stored model IDs and data unchanged",
        "training_or_decoder_refit_performed": False,
        "source_data_unchanged": True,
        "renderer": str(Path(__file__).resolve().relative_to(PROJECT)).replace("\\", "/"),
        "renderer_sha256": sha256(Path(__file__).resolve()),
    }
    main["plot_output_sha256"] = {
        name: sha256(OUT / name)
        for name in (
            "Figure_3h_models_by_dimension.png",
            "Figure_3h_models_by_dimension.pdf",
        )
    }
    MAIN_PROVENANCE.write_text(json.dumps(main, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    dots = json.loads(DOTS_PROVENANCE.read_text(encoding="utf-8"))
    dots["presentation_update_2026_11_05"] = main["presentation_update_2026_11_05"]
    for output in dots["outputs"]:
        path = PROJECT / output["path"]
        output["sha256"] = sha256(path)
    DOTS_PROVENANCE.write_text(json.dumps(dots, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    for path in (SCORES, SUMMARY, MAIN_PROVENANCE, DOTS_PROVENANCE):
        if not path.is_file():
            raise FileNotFoundError(path)
    scores = load_csv(SCORES)
    summary = load_csv(SUMMARY)
    if len(scores) != 225 or len(summary) != 45:
        raise ValueError(f"Unexpected Figure 3H inputs: {len(scores)} scores, {len(summary)} summary cells")
    render_matrix(summary)
    render_dots(scores)
    update_provenance(scores)
    print("Updated Figure 3H PNG/PDF presentation only; saved score CSVs and fit artifacts unchanged.")


if __name__ == "__main__":
    main()
