"""Assemble monkey Figure 3 and Extended Data Figure 8 from frozen artifacts.

No encoder training is performed. Paper-reference loss traces are read from
the checked-in-location HDF5 reference container; user-Soft histories and
held-out outputs are read from their immutable result manifests.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
import shutil

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DELIVERY = RESULTS / "PAPER_FIGURE_DELIVERY"
FIG3 = DELIVERY / "Figure_3"
ED8 = DELIVERY / "Extended_Data" / "ED8"
SOFT = RESULTS / "user_soft_monkey_v1"
SEEDS = (42, 43, 44, 45, 46)
DIMS = (3, 4, 8, 16, 32)
CONDITIONS = ("time_only", "behavior_only", "joint")
ARCHS = ("cnn", "transformer")
PROVENANCE_PATH = FIG3 / "MONKEY_FIGURE_DELIVERY_PROVENANCE_2026-10-22.json"
PREVIOUS_OUTPUT_HASHES: dict[str, str] = {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def previous_output_is_ours(path: Path) -> bool:
    try:
        rel = str(path.relative_to(ROOT))
    except ValueError:
        return False
    expected = PREVIOUS_OUTPUT_HASHES.get(rel)
    return bool(expected and path.is_file() and sha256(path) == expected)


def save_new(fig, stem: Path, *, dpi: int = 300) -> list[Path]:
    made = []
    for suffix in (".png", ".pdf"):
        dest = stem.with_suffix(suffix)
        if dest.exists() and not previous_output_is_ours(dest):
            raise FileExistsError(f"Refusing to overwrite existing figure: {dest}")
        temp = dest.with_name(f"{dest.stem}.rendering{suffix}")
        fig.savefig(temp, dpi=dpi, bbox_inches="tight", facecolor="white")
        os.replace(temp, dest)
        made.append(dest)
    plt.close(fig)
    return made


def copy_existing(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return
    if target.exists():
        if sha256(source) == sha256(target):
            return
        raise FileExistsError(f"Refusing to overwrite existing figure: {target}")
    shutil.copy2(source, target)


def copy_panel(panel: str, source_dir: Path, source_stem: str) -> list[Path]:
    made = []
    for suffix in (".png", ".pdf"):
        src = source_dir / f"{source_stem}{suffix}"
        dst = FIG3 / f"FIGURE3{panel}{suffix}"
        copy_existing(src, dst)
        made.append(dst)
    return made


def figure3_paper_panels() -> list[Path]:
    base = RESULTS / "monkey_figure3_paper_v1" / "stage03_figures"
    mapping = {
        "A": (RESULTS / "monkey_figure3_v1" / "stage00_data", "FIGURE3A_BEHAVIOR_FROM_OFFICIAL_RAW"),
        "B": (base / "figure3b_comparison", "FIGURE3B_LOCAL_NO_AUTOLFADS"),
        "C": (base, "FIGURE3C_PAPER_SETTINGS"),
        "D": (base, "FIGURE3D_PAPER_SETTINGS"),
        "E": (base, "FIGURE3E_PAPER_SETTINGS"),
        "F": (base, "FIGURE3F_PAPER_SETTINGS"),
        "G": (base, "FIGURE3G_PAPER_SETTINGS"),
        "H": (base / "figure3hi", "FIGURE3H_LOCAL_REPLICATION"),
    }
    return [out for panel, (directory, stem) in mapping.items()
            for out in copy_panel(panel, directory, stem)]


def read_loss_manifest(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    hist = data.get("history")
    if not hist:
        raise ValueError(f"No stored training history in {path}")
    losses = np.asarray([row["train_loss"] for row in hist], dtype=float)
    if len(losses) != 10_000 or not np.isfinite(losses).all():
        raise ValueError(f"Unexpected/non-finite loss history in {path}")
    return losses


def soft_loss_curves() -> tuple[list[Path], list[dict]]:
    output_files: list[Path] = []
    summary: list[dict] = []
    cache: dict[tuple[str, str, int, str], list[np.ndarray]] = {}
    parents: dict[str, str] = {}
    for arch in ARCHS:
        for condition in CONDITIONS:
            for dim in DIMS:
                for branch in ("observed", "behavior_shuffle"):
                    # Time-only was not assigned a redundant behavior shuffle.
                    if branch == "behavior_shuffle" and condition == "time_only":
                        continue
                    curves = []
                    for seed in SEEDS:
                        manifest = (SOFT / "stage01_fit" / branch / arch / condition /
                                    f"dim_{dim}" / f"seed_{seed}" / "manifest.json")
                        if not manifest.is_file():
                            if branch == "behavior_shuffle" and condition == "time_only":
                                continue
                            raise FileNotFoundError(manifest)
                        curves.append(read_loss_manifest(manifest))
                        parents[str(manifest.relative_to(ROOT))] = sha256(manifest)
                    if curves:
                        if len(curves) != len(SEEDS):
                            raise ValueError(f"Expected five seed curves: {arch}/{condition}/{dim}/{branch}")
                        cache[(arch, condition, dim, branch)] = curves

    sample_idx = np.arange(0, 10_000, 25)
    for (arch, condition, dim, branch), curves in sorted(cache.items()):
        arr = np.stack(curves)
        mean = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=1)
        for i in sample_idx:
            summary.append({
                "architecture": arch,
                "condition": condition,
                "dimension": dim,
                "control": branch,
                "update": int(i + 1),
                "mean_train_loss": float(mean[i]),
                "sd_train_loss": float(sd[i]),
                "n_seeds": int(arr.shape[0]),
            })

    csv_path = ED8 / "SOFT_TRAINING_LOSS_SEED_SUMMARY.csv"
    if csv_path.exists() and not previous_output_is_ours(csv_path):
        raise FileExistsError(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temp_csv = csv_path.with_name(csv_path.name + ".rendering")
    with temp_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    os.replace(temp_csv, csv_path)
    output_files.append(csv_path)

    colors = {"observed": "#1769aa", "behavior_shuffle": "#c44e52"}
    for dim in DIMS:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
        fig.suptitle(f"Monkey Soft training objective — output dimension {dim}", fontsize=17)
        for row, arch in enumerate(ARCHS):
            for col, condition in enumerate(CONDITIONS):
                ax = axes[row, col]
                for branch in ("observed", "behavior_shuffle"):
                    curves = cache.get((arch, condition, dim, branch))
                    if not curves:
                        continue
                    arr = np.stack(curves)
                    mean = arr.mean(axis=0)
                    sd = arr.std(axis=0, ddof=1)
                    x = np.arange(1, len(mean) + 1)
                    style = "-" if branch == "observed" else "--"
                    label = "observed labels" if branch == "observed" else "behavior-shuffled control"
                    ax.plot(x, mean, linestyle=style, color=colors[branch], lw=1.5, label=label)
                    ax.fill_between(x, mean - sd, mean + sd, color=colors[branch], alpha=0.16, linewidth=0)
                ax.set_title(f"{arch.upper()} — {condition.replace('_', ' ')}")
                ax.set_xlabel("Optimizer updates")
                ax.set_ylabel("Training loss")
                ax.grid(True, alpha=0.18)
                if ax.lines:
                    ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        stem = ED8 / f"EXTENDED8_SOFT_TRAINING_LOSS_DIM_{dim}"
        output_files.extend(save_new(fig, stem))
    return output_files, [{"path": key, "sha256": value}
                          for key, value in parents.items()]


def soft_embeddings() -> tuple[list[Path], dict[str, str]]:
    made: list[Path] = []
    parents: dict[str, str] = {}
    for seed in SEEDS:
        fig = plt.figure(figsize=(16, 9), facecolor="white")
        grid = fig.add_gridspec(2, 3, left=0.015, right=0.90, bottom=0.02,
                                top=0.88, wspace=0.01, hspace=0.025)
        axes = [fig.add_subplot(grid[i // 3, i % 3], projection="3d") for i in range(6)]
        sm = None
        for ax, (arch, condition) in zip(axes, [(a, c) for a in ARCHS for c in CONDITIONS]):
            npz = (SOFT / "stage02_embedding" / "observed" / arch / condition /
                   "dim_3" / f"seed_{seed}" / "test_embedding.npz")
            if not npz.is_file():
                raise FileNotFoundError(npz)
            parents[str(npz.relative_to(ROOT))] = sha256(npz)
            data = np.load(npz, allow_pickle=False)
            mask = data["valid_mask"].astype(bool)
            z = data["embedding_raw"][mask]
            posx = data["position_xy"][mask, 0]
            ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=posx, cmap="seismic",
                       vmin=-15, vmax=15, s=0.45, alpha=0.75, rasterized=True)
            ax.set_title(f"{arch.upper()} — {condition.replace('_', ' ')}", fontsize=11)
            ax.set_axis_off()
            ax.view_init(elev=24, azim=-58)
            sm = plt.cm.ScalarMappable(cmap="seismic", norm=plt.Normalize(-15, 15))
        fig.suptitle(f"Monkey Soft embeddings — active held-out test, seed {seed}, dim 3\n"
                     "Colour: hand position x (cm); axes hidden as in the paper panels", fontsize=15)
        cax = fig.add_axes([0.925, 0.18, 0.018, 0.62])
        fig.colorbar(sm, cax=cax, label="Position x (cm)")
        stem = FIG3 / f"FIGURE3_SOFT_HELDOUT_EMBEDDINGS_SEED_{seed}"
        made.extend(save_new(fig, stem))
    return made, parents


def soft_decoding() -> tuple[list[Path], list[dict], dict[str, str]]:
    rows: list[dict] = []
    parents: dict[str, str] = {}
    for arch in ARCHS:
        for condition in CONDITIONS:
            for dim in DIMS:
                for seed in SEEDS:
                    path = (SOFT / "stage03_metrics" / "observed" / arch / condition /
                            f"dim_{dim}" / f"seed_{seed}" / "RESULTS.json")
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    parents[str(path.relative_to(ROOT))] = sha256(path)
                    data = json.loads(path.read_text(encoding="utf-8"))
                    rec = next((r for r in data["records"]
                                if r["split"] == "test" and r["representation"] == "embedding_raw"
                                and r["decoder"] == "knn"), None)
                    if rec is None:
                        raise ValueError(f"Missing raw-test-kNN result in {path}")
                    rows.append({
                        "architecture": arch,
                        "condition": condition,
                        "dimension": dim,
                        "seed": seed,
                        "split": "test",
                        "representation": "embedding_raw (native model output)",
                        "decoder": "knn36 cosine",
                        "n_trials": rec["n_trials"],
                        "n_samples": rec["n_samples"],
                        "position_r2": rec["position_r2"],
                        "position_mae": rec["position_mae"],
                        "direction_accuracy": rec["direction_accuracy"],
                        "direction_balanced_accuracy": rec["direction_balanced_accuracy"],
                        "support_sha256": rec["support_sha256"],
                    })
    csv_path = FIG3 / "FIGURE3H_SOFT_HELDOUT_SEED_VALUES.csv"
    if csv_path.exists() and not previous_output_is_ours(csv_path):
        raise FileExistsError(csv_path)
    temp_csv = csv_path.with_name(csv_path.name + ".rendering")
    pd.DataFrame(rows).to_csv(temp_csv, index=False)
    os.replace(temp_csv, csv_path)

    colors = {"cnn": "#222222", "transformer": "#2166ac"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for col, condition in enumerate(CONDITIONS):
        for row_idx, metric in enumerate(("position_r2", "direction_accuracy")):
            ax = axes[row_idx, col]
            for arch in ARCHS:
                for j, dim in enumerate(DIMS):
                    vals = [r[metric] * 100 for r in rows if r["condition"] == condition
                            and r["architecture"] == arch and r["dimension"] == dim]
                    offset = -0.08 if arch == "cnn" else 0.08
                    jitter = np.linspace(-0.025, 0.025, len(vals))
                    ax.scatter(j + offset + jitter, vals, s=24, facecolors="none",
                               edgecolors=colors[arch], linewidths=1.0,
                               label=arch.upper() if (j == 0 and row_idx == 0) else None)
                    ax.plot(j + offset, np.median(vals), marker="_", markersize=18,
                            markeredgewidth=2.5, color=colors[arch])
            ax.set_title(f"{condition.replace('_', ' ')} — " +
                         ("position R²" if row_idx == 0 else "direction accuracy"))
            ax.set_xticks(range(len(DIMS)), DIMS)
            ax.set_xlabel("Embedding dimension")
            ax.set_ylabel("Score (%)")
            ax.grid(True, axis="y", alpha=0.18)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Soft held-out decoding — active monkey, five seeds (42–46)\n"
                 "Test results shown descriptively; no model selection", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    files = [csv_path]
    files.extend(save_new(fig, FIG3 / "FIGURE3H_SOFT_HELDOUT_DECODING_5SEED"))
    return files, rows, parents


def official_ed8_loss_reference() -> tuple[list[Path], str]:
    path = (ROOT / "cebra_replica" / "monkey" / "official_reference" / "EDFigure8.h5")
    with h5py.File(path, "r") as h5:
        values = pickle.loads(bytes(h5["data/block0_values"][0]))
        index = [x.decode("utf-8") for x in h5["data/axis1"][:]]
        columns = [x.decode("utf-8") for x in h5["data/axis0"][:]]
    data = pd.DataFrame(values, index=index, columns=columns)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7.5),
                             gridspec_kw={"width_ratios": [1.35, 1, 1]})
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.30, top=0.86, wspace=0.24)
    all_labels = ["target", "position", "a/p", "position+target", "a/p+position",
                  "a/p+target", "a/p+target+position"]
    all_keys = ["target", "pos", "ap", "pos_target", "ap_pos", "ap_target", "ap_pos_target"]
    all_colors = [plt.cm.tab10(i) for i in range(7)]
    ax = axes[0]
    for key, label, color in zip(all_keys, all_labels, all_colors):
        y = np.asarray(data.loc[key, "all_trial_loss"], dtype=float)
        ax.plot(savgol_filter(y, 9, 7), label=label, color=color, lw=1.3)
    for key, label, color in zip(("target_shuffle", "pos_shuffle", "ap_shuffle"),
                                 ("target shuffled", "position shuffled", "a/p shuffled"),
                                 all_colors[:3]):
        y = np.asarray(data.loc[key, "all_trial_loss"], dtype=float)
        ax.plot(savgol_filter(y, 9, 7), label=label, color=color, alpha=0.48, lw=1.1)
    ax.set_title("InfoNCE — all trials")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Training steps")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))

    def active_panel(ax, group: str, mse: bool) -> None:
        labels = (("target", "position", "position+target") if not mse else
                  ("target", "position", "position+target"))
        keys = ("target", "pos", "pos_target")
        colors_here = (plt.cm.tab10(0), plt.cm.tab10(1), plt.cm.tab10(3))
        for key, label, color in zip(keys, labels, colors_here):
            y = np.asarray(data.loc[key, group], dtype=float)
            ax.plot(savgol_filter(y, 9, 7), label=label, color=color, lw=1.3)
        shuffle_suffix = "shuffled" if mse else "shuffle"
        for key, label, color in ((f"target_{shuffle_suffix}", "target shuffled", colors_here[0]),
                                  (f"pos_{shuffle_suffix}", "position shuffled", colors_here[1])):
            y = np.asarray(data.loc[key, group], dtype=float)
            ax.plot(savgol_filter(y, 9, 7), label=label, color=color, alpha=0.48, lw=1.1)
        ax.set_title("InfoMSE — active trials" if mse else "InfoNCE — active trials")
        ax.set_xlabel("Training steps")
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16))

    active_panel(axes[1], "active_trial_loss", False)
    active_panel(axes[2], "active_mse_loss", True)
    for ax in axes:
        ax.set_xlim(0, 10_000)
        ax.grid(True, alpha=0.16)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Extended Data Figure 8 — published CEBRA loss traces (reference data)", fontsize=15)
    fig.subplots_adjust(top=0.86, bottom=0.30)
    return save_new(fig, ED8 / "EXTENDED8_CEBRA_REFERENCE_LOSS_CURVES"), sha256(path)


def main() -> None:
    global PREVIOUS_OUTPUT_HASHES
    previous_provenance = {}
    if PROVENANCE_PATH.is_file():
        previous_provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
        PREVIOUS_OUTPUT_HASHES = {item["path"]: item["sha256"]
                                  for item in previous_provenance.get("outputs", [])}
    FIG3.mkdir(parents=True, exist_ok=True)
    ED8.mkdir(parents=True, exist_ok=True)
    existing_panels = figure3_paper_panels()
    soft_loss_files, soft_loss_parents = soft_loss_curves()
    soft_embedding_files, embedding_parents = soft_embeddings()
    soft_decode_files, decode_rows, decode_parents = soft_decoding()
    ref_loss_files, hdf_hash = official_ed8_loss_reference()

    all_outputs = existing_panels + soft_loss_files + soft_embedding_files + soft_decode_files + ref_loss_files
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)),
        "project": str(ROOT),
        "training_performed": False,
        "training_seed_set": list(SEEDS),
        "embedding_dimensions": list(DIMS),
        "soft_conditions": list(CONDITIONS),
        "soft_campaign_state": str((SOFT / "CAMPAIGN_STATE.json").relative_to(ROOT)),
        "soft_fit_parent_manifests": soft_loss_parents,
        "soft_embedding_parent_sha256": embedding_parents,
        "soft_metric_parent_sha256": decode_parents,
        "official_ed8_hdf5": str((ROOT / "cebra_replica/monkey/official_reference/EDFigure8.h5").relative_to(ROOT)),
        "official_ed8_hdf5_sha256": hdf_hash,
        "outputs": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in all_outputs],
        "interpretation": [
            "Figure 3 A-H paper-style panels are copied from their verified canonical local artifacts.",
            "The five Soft embedding sheets use active held-out test trials, dim=3, one sheet per existing seed.",
            "Soft decoding plot summarizes five held-out seeds per dimension; test results are descriptive only.",
            "Soft loss curves show training loss mean and SD across the five existing seeds; shuffled controls are separate.",
            "The CEBRA ED8 loss plot reuses published reference curves from the official local HDF5; it is not a local fit.",
            "Soft fits cover active joint XY only; they do not supply passive/all-trial label conditions for Figure 3e-g."
        ],
    }
    provenance["supersedes_render"] = {
        "created_utc": previous_provenance.get("created_utc"),
        "script_sha256": previous_provenance.get("script_sha256"),
        "prior_output_hashes": previous_provenance.get("outputs", []),
    } if previous_provenance else None
    prov_path = PROVENANCE_PATH
    prov_temp = prov_path.with_name(prov_path.name + ".rendering")
    prov_temp.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    os.replace(prov_temp, prov_path)

    index = """# Monkey Figure 3 and Extended Data Figure 8 — 22 October 2026

All plots use white backgrounds. Existing scientific artifacts are reused;
no encoder was trained or refit by this delivery step.

## Figure 3 (paper panels)

`FIGURE3A.png` through `FIGURE3H.png` are the canonical paper-style local
replications. The panel-source map and caveats are in
`results/PAPER_FIGURE_DELIVERY/Figure_3/MONKEY_CURRENT_STATE_2026-10-19.md`.

## Added Soft held-out outputs

- `FIGURE3_SOFT_HELDOUT_EMBEDDINGS_SEED_42` through `_46`: six panels per
  seed (CNN/Transformer × time-only/behavior-only/joint), active test trials,
  3D output, colored by x-position. One image is provided per seed to keep the
  trajectories readable instead of superimposing seed trajectories.
- `FIGURE3H_SOFT_HELDOUT_DECODING_5SEED`: position R² and direction accuracy,
  all dimensions and five individual seeds. The matching CSV preserves each
  seed value. These test scores are descriptive, not selection evidence.

Soft was fit on active trials with joint XY behavior. It cannot be inserted as
an equivalent method in panels requiring passive or all-trial AP labels (3e-g).
Those panels therefore remain the existing paper-protocol outputs; this is a
data-scope limitation, not a missing fit within the completed Soft campaign.

## Extended Data Figure 8

The existing MSE embedding, held-out AP decoding and direction-timecourse
figures remain in this folder. `EXTENDED8_CEBRA_REFERENCE_LOSS_CURVES` is a
replot of the published reference loss arrays in the official HDF5. The five
`EXTENDED8_SOFT_TRAINING_LOSS_DIM_*` figures summarize the locally completed
Soft training histories, with mean ± SD over seeds 42–46 and separate
shuffled-label controls. Different objectives use different loss scales and
are not compared numerically across panels.

Full lineage and output hashes: `Figure_3/MONKEY_FIGURE_DELIVERY_PROVENANCE_2026-10-22.json`.
"""
    index_path = DELIVERY / "MONKEY_FIGURE_DELIVERY_INDEX_2026-10-22.md"
    if index_path.exists() and index_path.read_text(encoding="utf-8") != index:
        raise FileExistsError(f"Refusing to overwrite modified delivery index: {index_path}")
    index_temp = index_path.with_name(index_path.name + ".rendering")
    index_temp.write_text(index, encoding="utf-8")
    os.replace(index_temp, index_path)
    print(f"completed panels copied: {len(existing_panels)} files")
    print(f"new PNG/PDF/CSV outputs: {len(soft_loss_files) + len(soft_embedding_files) + len(soft_decode_files) + len(ref_loss_files)}")
    print(f"soft decoding seed rows: {len(decode_rows)}")
    print(f"provenance: {prov_path}")
    print(f"index: {index_path}")


if __name__ == "__main__":
    main()
