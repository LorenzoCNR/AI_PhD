"""Rerender Figure 2c graphics from saved plot CSVs; never reads fits or embeddings.

This renderer copies an existing completed plot tree, redraws the Soft panels and
model-comparison charts from the saved point/loss/metric CSVs, and updates plot
hashes/provenance. It does not run encoders or recalculate diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

import diagnostics_plots as diagnostics
import summary_comparison as comparison


MODELS = ("cebra_cnn", "soft_stable_cnn", "soft_stable_transformer")
SOFT_MODELS = ("soft_stable_cnn", "soft_stable_transformer")
SCOPES = ("split", "full")
RATS = ("achilles", "gatsby")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(fig, directory: Path, stem: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{stem}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def point_table(directory: Path, support: str, condition: str) -> pd.DataFrame:
    path = directory / f"points_{support}_{condition}.csv"
    frame = pd.read_csv(path)
    required = {"source_id", "true_position", "true_direction", "embedding_1", "embedding_2", "embedding_3"}
    if not required <= set(frame.columns) or frame.source_id.duplicated().any():
        raise ValueError(f"Invalid saved point table: {path}")
    return frame


def render_embeddings(directory: Path, scope: str, rat: str, model: str) -> None:
    supports = ("train", "test") if scope == "split" else ("full",)
    for support in supports:
        fig = plt.figure(figsize=(15, 9))
        norm = Normalize(0, 1.6)
        for i, condition in enumerate(diagnostics.CONDITIONS):
            frame = point_table(directory, support, condition)
            z = frame[["embedding_1", "embedding_2", "embedding_3"]].to_numpy()
            ax = fig.add_subplot(2, 3, i + 1, projection="3d")
            for label, cmap in (("0", "cool"), ("1", "viridis")):
                mask = frame.true_direction.astype(str).to_numpy() == label
                ax.scatter(*z[mask].T, c=frame.true_position.to_numpy()[mask],
                           cmap=cmap, norm=norm, s=2, rasterized=True)
            ax.set(title=diagnostics.CONDITION_LABELS[condition], xlabel="embedding 1",
                   ylabel="embedding 2", zlabel="embedding 3")
        fig.subplots_adjust(right=.85, hspace=.3)
        for j, (name, cmap) in enumerate((("Left (0)", "cool"), ("Right (1)", "viridis"))):
            bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                               cax=fig.add_axes([.88 + j * .055, .2, .018, .6]),
                               ticks=[0, .5, 1, 1.5])
            bar.set_label(f"{name}: position (m)")
        title = "test held-out" if support == "test" else support
        fig.suptitle(f"{rat.capitalize()} / {diagnostics.MODEL_LABELS[model]} — {title}")
        save(fig, directory, f"embedding_{support}")


def render_loss(directory: Path, rat: str, model: str) -> None:
    data = pd.read_csv(directory / "loss_data.csv")
    required = {"condition", "update", "train_loss"}
    if not required <= set(data.columns):
        raise ValueError(f"Invalid saved loss table: {directory}")
    fig, ax = plt.subplots(figsize=(9, 5))
    for condition in diagnostics.CONDITIONS:
        rows = data[data.condition == condition]
        if not rows.empty:
            ax.plot(rows["update"], rows["train_loss"], label=diagnostics.CONDITION_LABELS[condition])
    ax.set(xlabel="update", ylabel="train loss",
           title=f"{rat.capitalize()} / {diagnostics.MODEL_LABELS[model]}: training loss")
    ax.legend()
    save(fig, directory, "loss")


def aligned_direction_points(directory: Path, support: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    partitions = pd.read_csv(directory / "partitions.csv")
    required = {"source_id", "pseudo_direction", "kmeans_direction", "kmeans_direction_shuffled"}
    if not required <= set(partitions.columns) or partitions.source_id.duplicated().any():
        raise ValueError(f"Invalid saved partition table: {directory}")
    extra = partitions[["source_id", "pseudo_direction", "kmeans_direction", "kmeans_direction_shuffled"]]
    original = point_table(directory, support, "direction").merge(extra, on="source_id", validate="one_to_one")
    shuffled = point_table(directory, support, "direction_shuffled").merge(extra, on="source_id", validate="one_to_one")
    if len(original) != len(partitions) or len(shuffled) != len(partitions):
        raise ValueError(f"Saved partition and point supports differ: {directory}")
    return original, shuffled


def render_shuffle_diagnostics(directory: Path, scope: str, rat: str, model: str) -> None:
    support = "train" if scope == "split" else "full"
    original, shuffled = aligned_direction_points(directory, support)
    fig = plt.figure(figsize=(13, 8))
    specifications = (
        (original, "true_direction", "Real direction"),
        (original, "pseudo_direction", "Shuffled training labels"),
        (original, "kmeans_direction", "K-means groups"),
        (shuffled, "true_direction", "Real direction"),
        (shuffled, "pseudo_direction", "Shuffled training labels"),
        (shuffled, "kmeans_direction_shuffled", "K-means groups"),
    )
    for i, (frame, label_column, title) in enumerate(specifications):
        z = frame[["embedding_1", "embedding_2", "embedding_3"]].to_numpy()
        _, codes = np.unique(frame[label_column].to_numpy(), return_inverse=True)
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        ax.scatter(*z.T, c=codes, cmap="coolwarm", s=2, rasterized=True)
        ax.set(title=title, xlabel="embedding 1", ylabel="embedding 2", zlabel="embedding 3")
    fig.text(.012, .70, "Original Direction fit", rotation=90, ha="left", va="center", fontsize=11)
    fig.text(.012, .28, "Shuffled Direction fit", rotation=90, ha="left", va="center", fontsize=11)
    fig.suptitle(f"{rat.capitalize()} / {diagnostics.MODEL_LABELS[model]} — {support}; same source bins\n"
                 "Columns: real direction | shuffled labels | K-means groups", y=.99)
    save(fig, directory, "true_pseudo_kmeans")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, frame, condition, fit_label in (
        (axes[0], original, "direction", "original labels"),
        (axes[1], shuffled, "direction_shuffled", "shuffled labels"),
    ):
        z = frame[["embedding_1", "embedding_2"]].to_numpy()
        ax.scatter(z[:, 0], z[:, 1], c=frame.pseudo_direction, cmap="coolwarm", s=2, rasterized=True)
        ax.set(title=f"Direction fit ({fit_label})", xlabel="embedding 1", ylabel="embedding 2")
    fig.suptitle(f"{rat.capitalize()} / {diagnostics.MODEL_LABELS[model]} — color shows shuffled labels")
    save(fig, directory, "comparative_scatter")


def render_comparisons(directory: Path, scope: str, rat: str) -> None:
    models = list(MODELS)
    note = "source/trial supports match across models"
    prefix = f"{rat.capitalize()} | {scope} | {note}"
    support = "train" if scope == "split" else "full"
    specs = (
        ("ari", "comparison", "ari", f"{prefix}\nARI partitions on {support}", "Adjusted Rand index"),
        ("silhouette", "comparison", "silhouette", f"{prefix}\nSilhouette on {support}: fit | partition", "Silhouette score"),
        ("cka", "comparison", "linear_cka", f"{prefix}\nLinear CKA between conditions within each model ({support})", "Linear CKA"),
    )
    for filename, category, value, title, xlabel in specs:
        table = pd.read_csv(directory / f"{filename}.csv")
        comparison._plot(table, category, value, title, xlabel, directory, filename, models)
    if scope == "split":
        decoding = pd.read_csv(directory / "decoding.csv")
        for metric in ("test_accuracy", "test_balanced_accuracy"):
            label = metric.replace("_", " ")
            comparison._plot(decoding, "condition", metric, f"{prefix}\nkNN {label} | test held-out",
                             label, directory, metric, models)


def refresh_comparison_provenance(directory: Path, canonical_plot: Path, scope: str, rat: str) -> None:
    path = directory / "comparison_provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    for item in provenance["diagnostics"]:
        diag_dir = canonical_plot / scope / rat / item["model"]
        item["path"] = str(diag_dir.resolve())
        item["run_sha256"] = sha256(diag_dir / "diagnostics_run.json")
        item["complete_sha256"] = sha256(diag_dir / "diagnostics_complete.json")
    path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def refresh_diagnostic_manifest(directory: Path) -> None:
    complete_path = directory / "diagnostics_complete.json"
    artifacts = json.loads(complete_path.read_text(encoding="utf-8"))
    for name in artifacts:
        artifact = directory / name
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        artifacts[name] = sha256(artifact)
    complete_path.write_text(json.dumps(artifacts, indent=2) + "\n", encoding="utf-8")


def render(source_plot: Path, staged_plot: Path, canonical_plot: Path) -> None:
    if staged_plot.exists():
        raise FileExistsError(staged_plot)
    shutil.copytree(source_plot, staged_plot)
    rendered = []
    for scope in SCOPES:
        for rat in RATS:
            for model in SOFT_MODELS:
                print(f"Rerendering saved plots: {scope} / {rat} / {diagnostics.MODEL_LABELS[model]}", flush=True)
                directory = staged_plot / scope / rat / model
                render_embeddings(directory, scope, rat, model)
                render_loss(directory, rat, model)
                render_shuffle_diagnostics(directory, scope, rat, model)
                refresh_diagnostic_manifest(directory)
                for path in directory.iterdir():
                    if path.suffix.lower() in (".png", ".pdf"):
                        rendered.append(f"{scope}/{rat}/{diagnostics.MODEL_LABELS[model]}/{path.name}")
            comp = staged_plot / scope / rat / "confronto_modelli"
            print(f"Rerendering saved comparisons: {scope} / {rat}", flush=True)
            render_comparisons(comp, scope, rat)
            refresh_comparison_provenance(comp, canonical_plot, scope, rat)
            for path in comp.iterdir():
                if path.suffix.lower() in (".png", ".pdf"):
                    rendered.append(f"{scope}/{rat}/Model comparison/{path.name}")
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "plot-only rerender from saved point, loss, and metric CSVs",
        "model_fitting_performed": False,
        "diagnostic_metrics_recomputed": False,
        "display_labels": ["Soft CNN", "Soft Transformer"],
        "renderer": str(Path(__file__).name),
        "renderer_sha256": sha256(Path(__file__)),
        "rendered_files": rendered,
    }
    (staged_plot / "plot_label_render_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plot", type=Path, required=True)
    parser.add_argument("--staged-plot", type=Path, required=True)
    parser.add_argument("--canonical-plot", type=Path, required=True)
    args = parser.parse_args()
    render(args.source_plot.resolve(), args.staged_plot.resolve(), args.canonical_plot.resolve())


if __name__ == "__main__":
    main()
