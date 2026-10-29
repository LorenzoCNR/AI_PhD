"""Create held-out Soft analogues of monkey Figure 3b/d/e/g/i.

This is a downstream renderer only: it reads frozen test embeddings and saved
decoder predictions. It never fits or selects an encoder/decoder.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from figure3_style import DIRECTION_COLORS, POSITION_CMAP, POSITION_LIMITS, SOFT_OBJECTIVE_LABELS


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results"
SOFT_ROOT = RESULTS / "user_soft_monkey_v1"
OUT = RESULTS / "PAPER_FIGURE_DELIVERY" / "Figure_3"
SEEDS = (42, 43, 44, 45, 46)
ARCHITECTURES = ("cnn", "transformer")
CONDITIONS = ("time_only", "behavior_only", "joint")
PANEL_CONDITION = {"D": "time_only"}
TRAJECTORY_CONDITION = {"E": "behavior_only", "F": "joint", "G": "behavior_only"}
COLORS = DIRECTION_COLORS


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _embedding_path(arch: str, condition: str, seed: int) -> Path:
    return (SOFT_ROOT / "stage02_embedding" / "observed" / arch / condition /
            "dim_3" / f"seed_{seed}" / "test_embedding.npz")


def _prediction_path(arch: str, condition: str, seed: int) -> Path:
    return (SOFT_ROOT / "stage03_metrics" / "observed" / arch / condition /
            "dim_3" / f"seed_{seed}" / "predictions.npz")


def load_embedding(arch: str, condition: str, seed: int) -> dict:
    path = _embedding_path(arch, condition, seed)
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in (
            "embedding_raw", "position_xy", "direction", "trial_id", "time_id",
            "source_id", "valid_mask", "split", "seed", "model", "objective",
        )}
    n = len(data["embedding_raw"])
    if data["embedding_raw"].shape != (n, 3):
        raise ValueError(f"Expected N×3 Soft test embedding: {path}")
    for key in ("position_xy", "direction", "trial_id", "time_id", "source_id", "valid_mask"):
        if len(data[key]) != n:
            raise ValueError(f"Cardinality mismatch for {key}: {path}")
    if not np.isfinite(data["embedding_raw"]).all():
        raise ValueError(f"Non-finite embedding: {path}")
    mask = data["valid_mask"].astype(bool)
    data = {key: value[mask] if value.ndim and len(value) == n else value
            for key, value in data.items()}
    if not np.isin(data["direction"], np.arange(8)).all():
        raise ValueError(f"Invalid held-out direction labels: {path}")
    if not np.isfinite(data["position_xy"]).all():
        raise ValueError(f"Non-finite held-out position: {path}")
    data["_path"] = path
    data["_support"] = np.column_stack((data["source_id"], data["trial_id"], data["time_id"]))
    return data


def _ordered_trial_curves(data: dict) -> dict[int, np.ndarray]:
    z = data["embedding_raw"]
    trial, time, direction = data["trial_id"], data["time_id"], data["direction"]
    curves: dict[int, np.ndarray] = {}
    times_reference = None
    for trial_id in np.unique(trial):
        rows = np.flatnonzero(trial == trial_id)
        order = np.argsort(time[rows], kind="stable")
        rows = rows[order]
        times = time[rows]
        if len(np.unique(times)) != len(times):
            raise ValueError(f"Repeated time IDs in test trial {trial_id}")
        if times_reference is None:
            times_reference = times
        elif not np.array_equal(times, times_reference):
            raise ValueError("Held-out trial temporal supports are not equal")
        labels = np.unique(direction[rows])
        if len(labels) != 1:
            raise ValueError(f"Direction changes within held-out trial {trial_id}")
        curves[int(labels[0])] = np.vstack((curves.get(int(labels[0]), np.empty((0, len(times), z.shape[1]))), z[rows][None, ...]))
    return {label: values.mean(axis=0) for label, values in curves.items()}


def save_pair(fig, stem: Path, output_hashes: dict[str, str]) -> None:
    png, pdf = stem.with_suffix(".png"), stem.with_suffix(".pdf")
    if png.exists() or pdf.exists():
        raise FileExistsError(f"Refusing to overwrite existing figure: {stem}")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    output_hashes[png.name] = sha256(png)
    output_hashes[pdf.name] = sha256(pdf)


def plot_3b(seed: int, cache: dict, output_hashes: dict[str, str]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="white")
    for row, arch in enumerate(ARCHITECTURES):
        for col, condition in enumerate(CONDITIONS):
            ax = axes[row, col]
            data = cache[(arch, condition, seed)]
            curves = _ordered_trial_curves(data)
            if set(curves) != set(range(8)):
                raise ValueError(f"Missing direction trajectory: {arch}/{condition}/seed_{seed}")
            for label in range(8):
                trajectory = curves[label]
                # The paper panel uses 2D latent trajectories. Soft has a
                # 3D output; its first two frozen coordinates are shown here.
                ax.plot(trajectory[:, 0], trajectory[:, 1], color=COLORS[label], lw=2)
            ax.set_title(f"{arch.upper()} — {SOFT_OBJECTIVE_LABELS[condition]}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    handles = [Line2D([0], [0], color=COLORS[i], lw=2, label=f"Direction {i + 1}") for i in range(8)]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(f"Figure 3B Soft — output 3D, plotted z1–z2; held-out active test; seed {seed}", fontsize=17)
    fig.text(0.5, 0.055,
             "Direction colors/grouping are post hoc, not Soft loss targets. "
             "Behavior targets: none / position X,Y / time + position X,Y, as titled.",
             ha="center", fontsize=9)
    fig.subplots_adjust(top=0.91, bottom=0.11, wspace=0.12, hspace=0.2)
    save_pair(fig, OUT / f"FIGURE3B_SOFT_HELDOUT_SEED_{seed}", output_hashes)


def plot_embedding_panel(panel: str, seed: int, cache: dict, output_hashes: dict[str, str]) -> None:
    condition = PANEL_CONDITION[panel]
    fig = plt.figure(figsize=(13, 10), facecolor="white")
    axes = [[fig.add_subplot(2, 2, row * 2 + col + 1, projection="3d") for col in range(2)] for row in range(2)]
    scatters = [None, None]
    targets = [None, None]
    for row, arch in enumerate(ARCHITECTURES):
        data = cache[(arch, condition, seed)]
        z = data["embedding_raw"]
        for col, label_idx in enumerate((0, 1)):
            ax = axes[row][col]
            target = data["position_xy"][:, label_idx]
            scatters[col] = ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=target,
                                       cmap=POSITION_CMAP, s=0.8, alpha=0.55,
                                       vmin=POSITION_LIMITS[0], vmax=POSITION_LIMITS[1],
                                       rasterized=True)
            targets[col] = target
            ax.set_title(f"{arch.upper()} — {'x position' if col == 0 else 'y position'}", pad=4)
            ax.set_xlabel("z1", labelpad=-8)
            ax.set_ylabel("z2", labelpad=-8)
            ax.set_zlabel("z3", labelpad=-8)
            ax.tick_params(axis="both", which="major", labelsize=7, pad=-2)
    fig.suptitle(f"Figure 3{panel} Soft — {SOFT_OBJECTIVE_LABELS[condition]}; "
                 f"output 3D; position colors; held-out active TEST; seed {seed}", fontsize=15)
    for col, label in enumerate(("x position", "y position")):
        fig.colorbar(scatters[col], ax=[axes[0][col], axes[1][col]], fraction=0.035, pad=0.04,
                     label=f"{label} (native units)")
    fig.text(0.5, 0.015,
             "Soft-specific held-out panel; active joint-XY data only. Not an exact replication of the paper's passive/AP panel.",
             ha="center", fontsize=9)
    fig.subplots_adjust(top=0.9, bottom=0.07, wspace=0.08, hspace=0.17)
    save_pair(fig, OUT / f"FIGURE3{panel}_SOFT_HELDOUT_SEED_{seed}", output_hashes)


def set_clean_3d_axes(ax) -> None:
    for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        dim.set_pane_color((1.0, 1.0, 1.0, 0.0))
        dim.set_ticks([])
        dim._axinfo["grid"]["color"] = (1, 1, 1, 0)


def plot_trajectory_panel(panel: str, seed: int, cache: dict,
                          output_hashes: dict[str, str]) -> None:
    condition = TRAJECTORY_CONDITION[panel]
    fig = plt.figure(figsize=(14, 7), facecolor="white")
    for col, arch in enumerate(ARCHITECTURES, start=1):
        ax = fig.add_subplot(1, 2, col, projection="3d")
        data = cache[(arch, condition, seed)]
        curves = _ordered_trial_curves(data)
        if set(curves) != set(range(8)):
            raise ValueError(f"Missing direction trajectory: {arch}/{condition}/seed_{seed}")
        for direction in range(8):
            trajectory = curves[direction]
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                    color=COLORS[direction], lw=1.8)
        set_clean_3d_axes(ax)
        ax.set_title(arch.upper())
    handles = [Line2D([0], [0], color=COLORS[i], lw=2, label=f"Direction {i + 1}")
               for i in range(8)]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False,
               bbox_to_anchor=(0.5, 0.035))
    condition_label = SOFT_OBJECTIVE_LABELS[condition]
    fig.suptitle(f"Figure 3{panel} Soft — {condition_label}; output 3D; "
                 f"held-out active TEST; seed {seed}", fontsize=15)
    fig.text(0.5, 0.012,
             "Direction colors/grouping are post hoc. Soft was not trained on direction or active/passive labels; "
             "this is an active-only analogue, not an exact CEBRA 3E/3F/3G fit.",
             ha="center", fontsize=8.5)
    fig.subplots_adjust(top=0.87, bottom=0.14, wspace=0.04)
    save_pair(fig, OUT / f"FIGURE3{panel}_SOFT_HELDOUT_SEED_{seed}", output_hashes)


def plot_3i(seed: int, cache: dict, output_hashes: dict[str, str], prediction_hashes: dict[str, str]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor="white")
    for row, arch in enumerate(ARCHITECTURES):
        for col, condition in enumerate(CONDITIONS):
            data = cache[(arch, condition, seed)]
            path = _prediction_path(arch, condition, seed)
            prediction_hashes[f"{arch}/{SOFT_OBJECTIVE_LABELS[condition]}/seed_{seed}"] = sha256(path)
            with np.load(path, allow_pickle=False) as predictions:
                prefix = "embedding_raw_test_"
                true_xy = predictions[prefix + "position_true"].copy()
                pred_xy = predictions[prefix + "knn_position"].copy()
                source = predictions[prefix + "source_id"].copy()
                trial = predictions[prefix + "trial_id"].copy()
                time = predictions[prefix + "time_id"].copy()
            if true_xy.shape != pred_xy.shape or true_xy.ndim != 2 or true_xy.shape[1] != 2:
                raise ValueError(f"Bad saved held-out predictions: {path}")
            if not np.array_equal(np.column_stack((source, trial, time)), data["_support"]):
                raise ValueError(f"Decoder/embedding held-out support mismatch: {path}")
            ax = axes[row, col]
            ax.scatter(true_xy[:, 0], true_xy[:, 1], s=1.0, color="0.65", alpha=0.6,
                       label="Observed position X,Y")
            ax.scatter(pred_xy[:, 0], pred_xy[:, 1], s=1.0, color="red", alpha=0.55,
                       label="Decoded position X,Y (kNN36)")
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(f"{arch.upper()} — {SOFT_OBJECTIVE_LABELS[condition]}", fontsize=11)
            ax.set_xlabel("Position X (cm)")
            ax.set_ylabel("Position Y (cm)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(f"Figure 3I Soft — held-out position X,Y decoding; seed {seed}", fontsize=16)
    fig.text(0.5, 0.025, "Raw 3D embeddings; kNN36/cosine decoder fitted on TRAIN only; saved TEST predictions reused, not refitted.",
             ha="center", fontsize=9)
    fig.subplots_adjust(top=0.91, bottom=0.1, wspace=0.23, hspace=0.25)
    save_pair(fig, OUT / f"FIGURE3I_SOFT_HELDOUT_DECODED_TRAJECTORY_SEED_{seed}", output_hashes)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = {}
    support_reference = None
    parent_hashes = {}
    prediction_hashes = {}
    for seed in SEEDS:
        for arch in ARCHITECTURES:
            for condition in CONDITIONS:
                data = load_embedding(arch, condition, seed)
                support = data["_support"]
                if support_reference is None:
                    support_reference = support
                elif not np.array_equal(support, support_reference):
                    raise ValueError("Soft conditions/architectures/seeds do not share held-out trial/time support")
                cache[(arch, condition, seed)] = data
                parent_hashes[str(data["_path"].relative_to(PROJECT))] = sha256(data["_path"])
    output_hashes = {}
    with plt.rc_context({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.size": 10, "axes.titlepad": 8,
    }):
        for seed in SEEDS:
            plot_3b(seed, cache, output_hashes)
            for panel in ("D",):
                plot_embedding_panel(panel, seed, cache, output_hashes)
            for panel in ("E", "F", "G"):
                plot_trajectory_panel(panel, seed, cache, output_hashes)
            plot_3i(seed, cache, output_hashes, prediction_hashes)
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "renderer": str(Path(__file__).relative_to(PROJECT)),
        "renderer_sha256": sha256(Path(__file__)),
        "parent_embedding_sha256": parent_hashes,
        "parent_prediction_sha256": prediction_hashes,
        "figure_sha256": output_hashes,
        "seeds": list(SEEDS), "architectures": list(ARCHITECTURES),
        "conditions": list(CONDITIONS), "embedding_dimension": 3,
        "split": "held-out active test only",
        "input_representation": "raw frozen embedding",
        "encoder_training_or_decoder_refit": False,
        "panel_mapping": {"3b": "mean 2D latent trajectory by movement direction; first two frozen latent coordinates",
                          "3d": "time-only objective; held-out active XY used for color only, fixed seismic scale -15..15 cm",
                          "3e": "position XY objective; direction-colored mean 3D trajectories; direction is post hoc",
                          "3f": "time+position XY objective; direction-colored mean 3D trajectories; no direction/AP target",
                          "3g": "position XY objective; direction-colored mean 3D trajectories; direction is post hoc",
                          "3i": "saved train-fitted kNN36/cosine held-out position predictions"},
        "comparability_caveat": "Soft uses active trials only. It has no direction target or active/passive target; 3e/3f/3g are descriptive active-only analogues, not exact fits to the corresponding CEBRA objectives.",
        "direction_colors": "shared with CEBRA Figure 3: matplotlib hsv(index/8), direction array values 0..7",
        "position_colors": "shared with CEBRA Figure 3: seismic, fixed limits -15..15 cm",
        "trajectory_projection_caveat": "Figure 3b Soft uses frozen coordinates 1-2 for the paper-style 2D trajectory display; no post-hoc projection fit.",
    }
    path = OUT / "SOFT_FIGURE3_BDEGI_PROVENANCE.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite provenance: {path}")
    path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Created held-out Soft Figure 3 panels (PNG+PDF; including active-only 3E/3F/3G analogues) under {OUT}")
    print(f"Support rows: {len(support_reference)}; test trials: {len(np.unique(support_reference[:, 1]))}")


if __name__ == "__main__":
    main()
