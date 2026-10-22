"""Render the rat paper panels from frozen stable-soft artifacts.

This is a separate visual delivery. The original paper-replica figures, metrics,
fits, and embeddings are never written by this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "RAT_FIGURES"
FIG_OUT = OUT
PAPER = ROOT / "results" / "RAT_FIGURES" / "Reference_components"
RATS = ("achilles", "buddy", "cicero", "gatsby")
SOURCES = {
    ("CNN", "joint"): ROOT / "results" / "user_soft_rat_v1",
    ("CNN", "time_only"): ROOT / "results" / "user_soft_rat_time_only_v1",
    ("Transformer", "joint"): ROOT / "results" / "user_soft_rat_transformer_5seed_v1" / "joint",
    ("Transformer", "time_only"): ROOT / "results" / "user_soft_rat_transformer_5seed_v1" / "time_only",
}
DISPLAY = {
    ("CNN", "joint"): "Soft CNN\nTime + Behavior",
    ("CNN", "time_only"): "Soft CNN\nTime",
    ("Transformer", "joint"): "Soft Transformer\nTime + Behavior",
    ("Transformer", "time_only"): "Soft Transformer\nTime",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def embedding(architecture: str, condition: str, rat: str, seed: int = 42):
    base = SOURCES[(architecture, condition)] / "stage02_embedding" / rat / "fold_0" / "soft_v1_stable" / f"seed_{seed}"
    if not (base / "test_embedding.npz").is_file() and architecture == "Transformer" and condition == "joint":
        base = ROOT / "results" / "user_soft_rat_transformer_v1" / "stage02_embedding" / rat / "fold_0" / "soft_v1_stable" / f"seed_{seed}"
    manifest_path = base / "manifest.json"
    file = base / "test_embedding.npz"
    if not (manifest_path.is_file() and file.is_file()):
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["embeddings"]["test"]["embedding_sha256"]
    observed = sha(file)
    if observed != expected:
        raise ValueError(f"Embedding hash mismatch: {file}")
    with np.load(file) as data:
        arr = {key: data[key].copy() for key in ("embedding_raw", "embedding_unit", "behavior", "source_id", "valid_mask")}
    n = len(arr["source_id"])
    if any(len(arr[key]) != n for key in ("embedding_raw", "embedding_unit", "behavior", "valid_mask")):
        raise ValueError(f"Embedding/metadata cardinality mismatch: {file}")
    return arr, {"path": file.relative_to(ROOT).as_posix(), "sha256": observed,
                 "manifest": manifest_path.relative_to(ROOT).as_posix(), "manifest_sha256": sha(manifest_path)}


def save(fig, base: Path, parents: list[dict], panel: str, metadata: dict | None = None):
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        target = base.with_suffix("." + ext)
        if target.exists():
            raise FileExistsError(f"Immutable visual output exists: {target}")
        fig.savefig(target, dpi=300 if ext == "png" else None, bbox_inches="tight",
                    transparent=False, facecolor="white")
    record = {"created_utc": datetime.now(timezone.utc).isoformat(), "panel": panel,
              "script_sha256": sha(Path(__file__)), "parents": parents,
              "files": {ext: sha(base.with_suffix("." + ext)) for ext in ("png", "pdf")},
              "fit_version": "soft_v1_stable", "seed": 42, "fold": 0,
              "split": "held-out test", "representation": "embedding_raw; native encoder output is already L2 normalized",
              "display_highlight": "red method title only; data colors match existing paper-style gallery"}
    record.update(metadata or {})
    base.with_name(base.name + "_PROVENANCE.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def composite(rat: str, panel: str, added: Path, original: Path, destination: Path):
    """Keep every original comparison panel visibly intact above the addition."""
    from PIL import Image, JpegImagePlugin  # PDF writer requires JPEG registration.

    if not original.is_file():
        raise FileNotFoundError(original)
    if destination.with_name(destination.name + "_PROVENANCE.json").exists():
        raise FileExistsError(destination.with_suffix(".png"))
    with Image.open(original) as first, Image.open(added) as second:
        first = first.convert("RGB")
        second = second.convert("RGB")
        width = max(first.width, second.width)
        gap = 35
        canvas = Image.new("RGB", (width, first.height + second.height + gap), "white")
        canvas.paste(first, ((width - first.width) // 2, 0))
        canvas.paste(second, ((width - second.width) // 2, first.height + gap))
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination.with_suffix(".png"), dpi=(300, 300))
        canvas.save(destination.with_suffix(".pdf"), resolution=300)
    destination.with_name(destination.name + "_PROVENANCE.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(), "panel": panel, "rat": rat,
        "script_sha256": sha(Path(__file__)),
        "layout": "original comparison above; added stable-soft methods below; original pixels preserved",
        "parents": [{"path": x.relative_to(ROOT).as_posix(), "sha256": sha(x)} for x in (original, added)],
        "files": {ext: sha(destination.with_suffix("." + ext)) for ext in ("png", "pdf")},
    }, indent=2), encoding="utf-8")


def copy_unchanged_reference():
    source = PAPER / "Figure_1" / "Figure_1C"
    target = FIG_OUT / "Figure_1" / "Figure_1C"
    target.mkdir(parents=True, exist_ok=True)
    parents = []
    for name in ("FIGURE1C_RAT_DATA_OVERVIEW.png", "FIGURE1C_RAT_DATA_OVERVIEW.pdf", "FIGURE1C_DATA_SUMMARY.csv"):
        src = source / name
        dst = target / name
        if not src.is_file():
            raise FileNotFoundError(src)
        if dst.exists():
            raise FileExistsError(dst)
        shutil.copy2(src, dst)
        if sha(src) != sha(dst):
            raise ValueError(f"Figure 1C reference copy mismatch: {name}")
        parents.append({"path": src.relative_to(ROOT).as_posix(), "sha256": sha(src)})
    (target / "PROVENANCE.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(), "panel": "1C",
        "operation": "byte-identical inclusion of existing rat data overview; soft objective has no role",
        "parents": parents, "script_sha256": sha(Path(__file__)),
    }, indent=2), encoding="utf-8")


def embedding_panels():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for rat in RATS:
        available = []
        for key in SOURCES:
            item = embedding(*key, rat)
            if item is not None:
                available.append((key, *item))
        if not available:
            continue
        fig = plt.figure(figsize=(5 * len(available), 5), dpi=300)
        parents = []
        for i, (key, data, parent) in enumerate(available):
            mask = data["valid_mask"]
            emb, beh = data["embedding_raw"][mask, :3], data["behavior"][mask]
            if emb.shape[1] != 3:
                raise ValueError(f"Expected at least three embedding coordinates: {rat}/{key}")
            ax = fig.add_subplot(1, len(available), i + 1, projection="3d")
            left, right = beh[:, 2] == 1, beh[:, 1] == 1
            ax.scatter(*emb[left].T, c=beh[left, 0], cmap="cool", s=5, linewidths=0, rasterized=True)
            ax.scatter(*emb[right].T, c=beh[right, 0], cmap="viridis", s=5, linewidths=0, rasterized=True)
            ax.axis("off")
            ax.set_title(DISPLAY[key] + "\n3 shown / 32D fit", fontsize=12, color="red")
            parents.append(parent)
        save(fig, FIG_OUT / "Figure_1" / "Figure_1D" / f"FIGURE1D_{rat.upper()}_SOFT_STABLE",
             parents, "1D", {"rat": rat, "point_size": 5,
                             "label_color_maps": {"left": "cool", "right": "viridis"}})
        plt.close(fig)
        composite(rat, "1D", FIG_OUT / "Figure_1" / "Figure_1D" / f"FIGURE1D_{rat.upper()}_SOFT_STABLE.png",
                  PAPER / "Figure_1" / "Figure_1D" / f"FIGURE1D_{rat.upper()}_METHOD_EMBEDDINGS.png",
                  FIG_OUT / "Figure_1" / "Figure_1D" / f"FIGURE1D_{rat.upper()}_ALL_METHODS")
        # Figure 2B uses the same official 3D scatter style with the hypothesis
        # labels visible. All panels use identical held-out row support.
        fig = plt.figure(figsize=(5 * len(available), 5.8), dpi=240)
        ids_ref = None
        for i, (key, data, _parent) in enumerate(available):
            mask = data["valid_mask"]
            ids = data["source_id"][mask]
            if ids_ref is None:
                ids_ref = ids
            elif not np.array_equal(ids_ref, ids):
                raise ValueError(f"Different held-out support across methods: {rat}/{key}")
            emb, beh = data["embedding_raw"][mask, :3], data["behavior"][mask]
            ax = fig.add_subplot(1, len(available), i + 1, projection="3d")
            left, right = beh[:, 2] == 1, beh[:, 1] == 1
            ax.scatter(*emb[left].T, c=beh[left, 0], cmap="cool", s=0.5, rasterized=True)
            ax.scatter(*emb[right].T, c=beh[right, 0], cmap="viridis", s=0.5, rasterized=True)
            ax.axis("off")
            ax.set_title(DISPLAY[key], fontsize=13, color="red")
            ax.text2D(0.02, 0.02, "32D fit; first 3 coordinates shown", transform=ax.transAxes,
                      fontsize=7, alpha=0.8)
        fig.tight_layout(rect=(0, 0.16, 1, 0.94))
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        beh = available[0][1]["behavior"][available[0][1]["valid_mask"]]
        for bounds, cmap, label, selection in (
            ([0.25, 0.07, 0.20, 0.022], "viridis", "Label 1 (right): position [cm]", beh[:, 1] == 1),
            ([0.55, 0.07, 0.20, 0.022], "cool", "Label 2 (left): position [cm]", beh[:, 2] == 1),
        ):
            values = beh[selection, 0] * 100
            if len(values):
                bar = fig.colorbar(ScalarMappable(norm=Normalize(vmin=float(values.min()),
                                    vmax=float(values.max())), cmap=cmap),
                                   cax=fig.add_axes(bounds), orientation="horizontal")
                bar.set_label(label, fontsize=9)
        save(fig, FIG_OUT / "Figure_2" / "Figure_2B" / f"FIGURE2B_{rat.upper()}_SOFT_STABLE",
             parents, "2B", {"rat": rat, "scope": "held-out stable soft methods; not the original four-hypothesis ablation"})
        plt.close(fig)
        composite(rat, "2B", FIG_OUT / "Figure_2" / "Figure_2B" / f"FIGURE2B_{rat.upper()}_SOFT_STABLE.png",
                  PAPER / "Figure_2" / "Figure_2B" / f"FIGURE2B_{rat.upper()}_seed42.png",
                  FIG_OUT / "Figure_2" / "Figure_2B" / f"FIGURE2B_{rat.upper()}_ALL_METHODS")


def loss_panels():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for rat in RATS:
        fig, ax = plt.subplots(figsize=(7, 3.5), dpi=300)
        parents = []
        for key, source in SOURCES.items():
            file = source / "stage01_fit" / rat / "fold_0" / "soft_v1_stable" / "seed_42" / "manifest.json"
            if not file.is_file() and key == ("Transformer", "joint"):
                file = ROOT / "results" / "user_soft_rat_transformer_v1" / "stage01_fit" / rat / "fold_0" / "soft_v1_stable" / "seed_42" / "manifest.json"
            if not file.is_file():
                continue
            doc = json.loads(file.read_text(encoding="utf-8"))
            history = doc.get("history", [])
            if not history:
                continue
            ax.plot([int(p["update"]) for p in history], [float(p["train_loss"]) for p in history],
                    label=DISPLAY[key].replace("\n", " "), linewidth=1.4)
            parents.append({"path": file.relative_to(ROOT).as_posix(), "sha256": sha(file)})
        if not parents:
            plt.close(fig)
            continue
        ax.set_xlabel("Optimizer updates")
        ax.set_ylabel("Stable soft objective (training)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.patch.set_facecolor("white")
        save(fig, FIG_OUT / "Figure_2" / "Figure_2C" / f"FIGURE2C_{rat.upper()}_SOFT_STABLE_LOSS",
             parents, "2C", {"rat": rat, "limitation": "Soft loss scale differs from native CEBRA InfoNCE; curves cannot be overlaid numerically as one objective."})
        plt.close(fig)
        composite(rat, "2C", FIG_OUT / "Figure_2" / "Figure_2C" / f"FIGURE2C_{rat.upper()}_SOFT_STABLE_LOSS.png",
                  PAPER / "Figure_2" / "Figure_2C" / f"FIGURE2C_LOSS_{rat.upper()}.png",
                  FIG_OUT / "Figure_2" / "Figure_2C" / f"FIGURE2C_{rat.upper()}_ALL_METHODS")


def consistency_panels():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    new_path = OUT / "Tables" / "SOFT_CROSS_RAT_CONSISTENCY_BY_SEED.csv"
    new_rows = csv_rows(new_path)
    old_path = PAPER / "Figure_1" / "Figure_1E" / "CROSS_RAT_CONSISTENCY_BY_SEED.csv"
    old_rows = csv_rows(old_path)
    if not new_rows:
        return
    new_rows = [r for r in new_rows if r.get("representation", "embedding_raw") == "embedding_raw"]
    methods = [(architecture, condition) for architecture in ("CNN1D", "Transformer")
               for condition in ("joint", "time_only")
               if any(r.get("architecture") == architecture and r.get("condition") == condition for r in new_rows)]
    old_methods = tuple(x for x in ("CEBRA-Behavior", "CEBRA-Time",
                        "Conv-piVAE with test labels", "Conv-piVAE no test labels",
                        "t-SNE", "UMAP", "PCA") if any(r["method"] == x for r in old_rows))
    panels = [(x, None) for x in old_methods] + methods
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(1.7 * (len(panels) + 1), 1.4),
                             gridspec_kw={"width_ratios": [1] * len(panels) + [0.08]}, dpi=600)
    for ix, (method, condition) in enumerate(panels):
        rows = old_rows if condition is None else new_rows
        matrix = np.full((len(RATS), len(RATS)), np.nan)
        for i, rat_a in enumerate(RATS):
            for j, rat_b in enumerate(RATS):
                values = [float(r["mean_across_folds"]) for r in rows
                          if (r["method"] == method if condition is None else
                              r.get("architecture") == method and r.get("condition") == condition)
                          and r["source_rat"] == rat_a and r["target_rat"] == rat_b]
                if values:
                    matrix[i, j] = np.mean(values)
        sns.heatmap(np.minimum(matrix * 100, 99), vmin=20, vmax=100,
                    xticklabels=[], yticklabels=[], cmap=sns.color_palette("gray_r", as_cmap=True),
                    annot=True, annot_kws={"fontsize": 12}, fmt=".0f", ax=axes[ix],
                    cbar=ix == len(panels) - 1, cbar_ax=axes[-1] if ix == len(panels) - 1 else None)
        axes[ix].axis("off")
        baseline_titles = {"Conv-piVAE with test labels": "conv-piVAE\nw/labels",
                           "Conv-piVAE no test labels": "conv-piVAE",
                           "t-SNE": "tSNE"}
        title = baseline_titles.get(method, method) if condition is None else DISPLAY[("CNN" if method == "CNN1D" else "Transformer", condition)]
        axes[ix].set_title(f"{title}\n({100*np.nanmean(matrix):.1f})", fontsize=8,
                           color="black" if condition is None else "red")
    fig.subplots_adjust(wspace=0.5)
    save(fig, FIG_OUT / "Figure_1" / "Figure_1E" / "FIGURE1E_STABLE_SOFT_CONSISTENCY_FINAL",
         [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in (new_path, old_path)],
         "1E", {"metric": "CEBRA consistency_score between datasets, position-aligned directed R2, held-out"})
    plt.close(fig)


def performance_panels():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    new_path = OUT / "Tables" / "SOFT_DECODING_BY_SEED.csv"
    base_path = PAPER / "Figure_2" / "Figure_2D" / "FIGURE2D_METHOD_METRICS_SEED_LEVEL.csv"
    new_rows = [r for r in csv_rows(new_path) if r.get("representation") == "embedding_raw"
                and r.get("split") == "test" and r.get("decoder") == "knn"]
    base_rows = [r for r in csv_rows(base_path) if r.get("split") == "test"]
    if not new_rows:
        return
    baseline_methods = tuple(x for x in ("CEBRA-Behavior-32D", "Conv-piVAE-z_mean",
                             "Conv-piVAE-test-labels", "CEBRA-Time-32D", "PCA", "UMAP", "OPENTSNE")
                             if any(r["method"] == x for r in base_rows))
    panels = [(x, None) for x in baseline_methods] + [
              ("CNN1D", "joint"), ("CNN1D", "time_only"),
              ("Transformer", "joint"), ("Transformer", "time_only")]
    labels = {"position_median_absolute_error_cm": "Median absolute position error (cm)",
              "position_r2": "Position $R^2$", "direction_balanced_accuracy": "Direction balanced accuracy"}
    for name, metrics in (("Figure_2/ Figure_2D", ("position_median_absolute_error_cm",)),
                          ("Extended_Data/ED4", tuple(labels))):
        parent_dir = FIG_OUT / name.replace("/ ", "/")
        fig, axes = plt.subplots(len(RATS), len(metrics), figsize=(4.1 * len(metrics),
                                  2.7 * len(RATS)), squeeze=False, dpi=220)
        for ri, rat in enumerate(RATS):
            for ci, metric in enumerate(metrics):
                ax = axes[ri, ci]
                for yi, (method, condition) in enumerate(panels):
                    rows = base_rows if condition is None else new_rows
                    vals = [float(r["mean_across_folds"]) for r in rows
                            if r.get("rat") == rat and r.get("metric") == metric
                            and (r.get("method") == method if condition is None else
                                 r.get("architecture") == method and r.get("condition") == condition)]
                    if not vals:
                        continue
                    ax.scatter(vals, np.full(len(vals), yi), s=13, color="black", alpha=0.74,
                               edgecolors="white", linewidths=0.25, zorder=3)
                    med = float(np.median(vals))
                    ax.plot([med, med], [yi-0.22, yi+0.22], color="#e88d25", linewidth=2.5, zorder=4)
                ax.set_yticks(range(len(panels)))
                if ci == 0:
                    tick_labels = [method if condition is None else
                                   DISPLAY[("CNN" if method == "CNN1D" else "Transformer", condition)].replace("\n", " ")
                                   for method, condition in panels]
                    ax.set_yticklabels(tick_labels, fontsize=8)
                    for tick, (_method, condition) in zip(ax.get_yticklabels(), panels):
                        if condition is not None:
                            tick.set_color("red")
                else:
                    ax.set_yticklabels([])
                ax.invert_yaxis()
                ax.grid(axis="x", alpha=0.2)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                if ri == 0:
                    ax.set_title(labels[metric])
                if ci == 0:
                    ax.set_ylabel(rat.capitalize())
                if ri == len(RATS)-1:
                    ax.set_xlabel(labels[metric])
        fig.tight_layout()
        save(fig, parent_dir / ("FIGURE2D_SOFT_STABLE_KNN_DECODING" if "Figure_2" in name else
                                "ED4_SOFT_STABLE_KNN_ADDITIONAL_METRICS"),
             [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in (new_path, base_path)],
             "2D" if "Figure_2" in name else "ED4",
             {"style": "black per-seed dots and orange median mark; red labels identify added methods",
              "replication_note": "actual seed counts, not 10 runs; rows with no score remain empty"})
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", choices=("embeddings", "loss", "consistency", "performance", "all"), default="all")
    args = parser.parse_args()
    if args.panel in ("embeddings", "all"):
        copy_unchanged_reference()
        embedding_panels()
    if args.panel in ("loss", "all"):
        loss_panels()
    if args.panel in ("consistency", "all"):
        consistency_panels()
    if args.panel in ("performance", "all"):
        performance_panels()


if __name__ == "__main__":
    main()
