"""Render Figure 3B-G-style panels for frozen Soft monkey fits.

The renderer joins already-saved train/validation/test embeddings per fit,
checks common trial/time support, and writes descriptive all-active-trial
figures. It does not load checkpoints, fit models, or fit decoders.
"""

from __future__ import annotations

import csv
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
SOFT_ROOT = PROJECT / "results" / "user_soft_monkey_v1"
OUT = PROJECT / "results" / "PAPER_FIGURE_DELIVERY" / "Figure_3"
SEEDS = (42, 43, 44, 45, 46)
ARCHITECTURES = ("cnn", "transformer")
OBJECTIVES = ("time_only", "behavior_only", "joint")
SPLITS = ("train", "valid", "test")
COLORS = DIRECTION_COLORS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_full_support(architecture: str, objective: str, seed: int, dimension: int):
    """Join cached split exports in metadata order; no model inference needed."""
    base = (SOFT_ROOT / "stage02_embedding" / "observed" / architecture /
            objective / f"dim_{dimension}" / f"seed_{seed}")
    arrays = []
    parents = []
    for split in SPLITS:
        path = base / f"{split}_embedding.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        parents.append(path)
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in (
                "embedding_raw", "position_xy", "direction", "trial_id",
                "time_id", "source_id", "session_id", "valid_mask", "split", "seed",
                "model", "objective",
            )}
        n = len(data["embedding_raw"])
        if data["embedding_raw"].shape != (n, dimension):
            raise ValueError(f"Expected N x {dimension} embedding: {path}")
        for key in ("position_xy", "direction", "trial_id", "time_id",
                    "source_id", "session_id", "valid_mask"):
            if len(data[key]) != n:
                raise ValueError(f"Cardinality mismatch for {key}: {path}")
        if data["split"].item() != split or int(data["seed"].item()) != seed:
            raise ValueError(f"Split/seed metadata mismatch: {path}")
        if not np.isfinite(data["embedding_raw"]).all():
            raise ValueError(f"Non-finite embedding: {path}")
        if not np.isfinite(data["position_xy"]).all():
            raise ValueError(f"Non-finite position: {path}")
        keep = data["valid_mask"].astype(bool)
        arrays.append({key: value[keep] if value.ndim and len(value) == n else value
                       for key, value in data.items()})

    merged = {}
    row_keys = ("embedding_raw", "position_xy", "direction", "trial_id",
                "time_id", "source_id", "session_id")
    for key in row_keys:
        merged[key] = np.concatenate([part[key] for part in arrays], axis=0)
    order = np.lexsort((merged["time_id"], merged["trial_id"], merged["session_id"]))
    merged = {key: value[order] for key, value in merged.items()}
    support = np.column_stack((merged["source_id"], merged["trial_id"], merged["time_id"]))
    if len(np.unique(support, axis=0)) != len(support):
        raise ValueError(f"Duplicate source/trial/time rows: {architecture}/{objective}/seed_{seed}")
    if not np.isin(merged["direction"], np.arange(8)).all():
        raise ValueError(f"Direction labels outside 0..7: {architecture}/{objective}/seed_{seed}")
    merged["_support"] = support
    merged["_parent_paths"] = parents
    return merged


def assert_common_support(cache):
    reference = None
    for key, data in cache.items():
        support = data["_support"]
        if reference is None:
            reference = support
        elif not np.array_equal(reference, support):
            raise ValueError(f"Full active trial/time support differs at {key}")
    return reference


def mean_direction_trajectories(data):
    """Average complete, time-aligned trials within each of eight directions."""
    embedding = data["embedding_raw"]
    session, trial, time = data["session_id"], data["trial_id"], data["time_id"]
    direction = data["direction"]
    trial_curves = {label: [] for label in range(8)}
    trial_counts = {label: 0 for label in range(8)}
    # source_id is a per-sample source row index, not a trial/session ID.
    # Use explicit session_id + trial_id to reconstruct complete trials.
    groups = np.unique(np.column_stack((session, trial)), axis=0)
    reference_time = None
    for session_id, trial_id in groups:
        rows = np.flatnonzero((session == session_id) & (trial == trial_id))
        rows = rows[np.argsort(time[rows], kind="stable")]
        times = time[rows]
        if reference_time is None:
            reference_time = times
        elif not np.array_equal(times, reference_time):
            raise ValueError("Trial time support is not identical; refusing unsafe averaging")
        labels = np.unique(direction[rows])
        if len(labels) != 1:
            raise ValueError(f"Direction changes within trial {trial_id}")
        label = int(labels[0])
        trial_curves[label].append(embedding[rows])
        trial_counts[label] += 1
    if any(trial_counts[label] == 0 for label in range(8)):
        raise ValueError(f"A direction has no active trials: {trial_counts}")
    means = {label: np.mean(np.stack(trial_curves[label]), axis=0)
             for label in range(8)}
    return means, trial_counts


def save_pair(fig, stem: Path, output_hashes: dict[str, str], prior_hashes: dict[str, str]):
    for suffix in (".png", ".pdf"):
        target = stem.with_suffix(suffix)
        expected = prior_hashes.get(target.name)
        if expected is not None and (not target.is_file() or sha256(target) != expected):
            raise RuntimeError(f"Previously generated figure changed; refusing replacement: {target}")
        if target.exists() and expected is None:
            raise FileExistsError(f"Refusing to overwrite unrelated output: {target}")
    if all(stem.with_suffix(suffix).is_file()
           and prior_hashes.get(stem.with_suffix(suffix).name) == sha256(stem.with_suffix(suffix))
           for suffix in (".png", ".pdf")):
        for suffix in (".png", ".pdf"):
            target = stem.with_suffix(suffix)
            output_hashes[target.name] = prior_hashes[target.name]
        plt.close(fig)
        return
    for suffix, fmt, options in (
        (".png", "png", {"dpi": 300}),
        (".pdf", "pdf", {}),
    ):
        target = stem.with_suffix(suffix)
        temporary = stem.with_name(stem.name + ".tmp" + suffix)
        fig.savefig(temporary, format=fmt, bbox_inches="tight",
                    facecolor="white", **options)
        temporary.replace(target)
        output_hashes[target.name] = sha256(target)
    plt.close(fig)


def plot_3b(seed: int, cache, output_hashes, prior_hashes):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor="white")
    trial_counts_by_config = {}
    for row, architecture in enumerate(ARCHITECTURES):
        for col, objective in enumerate(OBJECTIVES):
            data = cache[(architecture, objective, seed)]
            curves, counts = mean_direction_trajectories(data)
            trial_counts_by_config[f"{architecture}/{SOFT_OBJECTIVE_LABELS[objective]}"] = counts
            ax = axes[row, col]
            for label in range(8):
                # The paper displays 2D mean trajectories from a 4D embedding.
                # Use the same fixed first-two-coordinate view for every Soft fit.
                curve = curves[label]
                ax.plot(curve[:, 0], curve[:, 1], color=COLORS[label], linewidth=2)
            ax.set_title(f"{architecture.upper()} — {SOFT_OBJECTIVE_LABELS[objective]}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_aspect("equal", adjustable="datalim")
    handles = [Line2D([0], [0], color=COLORS[i], lw=2, label=f"{i + 1}")
               for i in range(8)]
    fig.legend(handles=handles, title="Direction", loc="lower center", ncol=8,
               frameon=False, bbox_to_anchor=(0.5, 0.012))
    fig.suptitle(f"Figure 3B Soft — output 4D, plotted z1–z2; all active rows; seed {seed}",
                 fontsize=16)
    fig.text(0.5, 0.075,
             "Line color: direction 1–8 (post-hoc grouping, not a Soft loss target). Train/validation/test rows shown; descriptive only.",
             ha="center", fontsize=9)
    fig.subplots_adjust(top=0.91, bottom=0.16, wspace=0.12, hspace=0.2)
    stem = OUT / f"FIGURE3B_SOFT_FULL_SAMPLE_SEED_{seed}"
    save_pair(fig, stem, output_hashes, prior_hashes)
    return trial_counts_by_config


def set_clean_3d_axes(ax):
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def plot_3c_or_d(panel: str, objective: str, seed: int, cache, output_hashes, prior_hashes):
    fig = plt.figure(figsize=(14, 11), facecolor="white")
    scatter_by_position = [None, None]
    for row, architecture in enumerate(ARCHITECTURES):
        data = cache[(architecture, objective, seed)]
        z = data["embedding_raw"]
        for col, axis_label in enumerate(("x", "y")):
            ax = fig.add_subplot(2, 2, row * 2 + col + 1, projection="3d")
            target = data["position_xy"][:, col]
            sc = ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=target,
                            cmap=POSITION_CMAP, s=0.05, alpha=0.85,
                            vmin=POSITION_LIMITS[0], vmax=POSITION_LIMITS[1],
                            rasterized=True, linewidths=0)
            scatter_by_position[col] = sc
            ax.set_title(f"{architecture.upper()} — {axis_label} position", fontsize=11, pad=1)
            set_clean_3d_axes(ax)
    for col, axis_label in enumerate(("x", "y")):
        fig.colorbar(scatter_by_position[col], ax=[fig.axes[0 + col], fig.axes[2 + col]],
                     fraction=0.035, pad=0.04, label=f"{axis_label} position (cm)")
    model_label = ("Position X,Y" if panel == "C" else "Time")
    fig.suptitle(f"Figure 3{panel} Soft — {model_label}; output 3D; all active rows; seed {seed}",
                 fontsize=15)
    fig.text(0.5, 0.015,
             "Color: position x/y (visualization only for time-only). Shared seismic scale: −15 to +15 cm. "
             "Train/validation/test rows shown; fit used saved train split only.",
             ha="center", fontsize=9)
    fig.subplots_adjust(top=0.89, bottom=0.07, wspace=0.15, hspace=0.42)
    stem = OUT / f"FIGURE3{panel}_SOFT_FULL_SAMPLE_SEED_{seed}"
    save_pair(fig, stem, output_hashes, prior_hashes)


def plot_3efg(panel: str, objective: str, seed: int, cache, output_hashes, prior_hashes):
    fig = plt.figure(figsize=(14, 7), facecolor="white")
    for slot, architecture in zip((121, 122), ARCHITECTURES):
        data = cache[(architecture, objective, seed)]
        curves, _ = mean_direction_trajectories(data)
        ax = fig.add_subplot(slot, projection="3d")
        for direction in range(8):
            curve = curves[direction]
            color = COLORS[direction]
            ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=color, linewidth=1.5)
            ax.scatter(*curve[0], color=color, marker="o", s=20)
            ax.scatter(*curve[-1], color=color, marker="<", s=20)
        ax.set_title(architecture.upper(), fontsize=12)
        set_clean_3d_axes(ax)
        ax.view_init(elev=24, azim=-58)
    handles = [Line2D([0], [0], color=COLORS[i], lw=2, label=f"Direction {i + 1}")
               for i in range(8)]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False,
               bbox_to_anchor=(0.5, 0.04))
    objective_label = SOFT_OBJECTIVE_LABELS[objective]
    fig.suptitle(
        f"Figure 3{panel} Soft — {objective_label}; output 3D; all active rows; seed {seed}",
        fontsize=15,
    )
    fig.text(0.5, 0.01,
             "Train/validation/test rows shown; fit used saved train split. "
             "Direction 1–8 is post-hoc grouping/color, not a Soft loss target. "
             "Active-only descriptive analogue; not held-out.",
             ha="center", fontsize=8)
    stem = OUT / f"FIGURE3{panel}_SOFT_FULL_SAMPLE_SEED_{seed}"
    save_pair(fig, stem, output_hashes, prior_hashes)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    output_hashes = {}
    parent_hashes = {}
    support_counts = {}
    support_reference = None
    manifest_path = OUT / "FIGURE3_SOFT_FULL_SAMPLE_PROVENANCE.json"
    prior_hashes = {}
    prior_manifest_sha256 = None
    if manifest_path.is_file():
        prior_manifest_sha256 = sha256(manifest_path)
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        prior_hashes = prior.get("figure_sha256", {})
        if len(prior_hashes) not in (30, 60):
            raise RuntimeError("Existing figure manifest is incomplete; refusing replacement")
        for name, expected in prior_hashes.items():
            path = OUT / name
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"Existing generated artifact does not match its manifest: {path}")
        output_hashes.update(prior_hashes)
    for seed in SEEDS:
        cache = {}
        for architecture in ARCHITECTURES:
            for objective in OBJECTIVES:
                data = load_full_support(architecture, objective, seed, dimension=4)
                for path in data["_parent_paths"]:
                    key = f"{architecture}/{SOFT_OBJECTIVE_LABELS[objective]}/dim_4/seed_{seed}/{path.name}"
                    parent_hashes[key] = sha256(path)
                support = data["_support"]
                if support_reference is None:
                    support_reference = support.copy()
                elif not np.array_equal(support_reference, support):
                    raise ValueError(
                        f"Architectures/objectives/seeds do not share full support: "
                        f"{architecture}/{objective}/seed_{seed}"
                    )
                cache[(architecture, objective, seed)] = data
        cache_3d = {}
        for architecture in ARCHITECTURES:
            for objective in OBJECTIVES:
                data_3d = load_full_support(architecture, objective, seed, dimension=3)
                for path in data_3d["_parent_paths"]:
                    key = f"{architecture}/{SOFT_OBJECTIVE_LABELS[objective]}/dim_3/seed_{seed}/{path.name}"
                    parent_hashes[key] = sha256(path)
                if not np.array_equal(support_reference, data_3d["_support"]):
                    raise ValueError(
                        f"3D panel support differs from 4D reference: "
                        f"{architecture}/{objective}/seed_{seed}"
                    )
                cache_3d[(architecture, objective, seed)] = data_3d
        support_counts[f"seed_{seed}"] = {
            "rows": int(len(support_reference)),
            "trials": int(len(np.unique(np.column_stack((
                cache[("cnn", "behavior_only", seed)]["session_id"],
                cache[("cnn", "behavior_only", seed)]["trial_id"],
            )), axis=0))),
        }
        direction_counts = plot_3b(seed, cache, output_hashes, prior_hashes)
        plot_3c_or_d("C", "behavior_only", seed, cache_3d, output_hashes, prior_hashes)
        plot_3c_or_d("D", "time_only", seed, cache_3d, output_hashes, prior_hashes)
        for panel, objective in (("E", "behavior_only"), ("F", "joint"), ("G", "behavior_only")):
            plot_3efg(panel, objective, seed, cache_3d, output_hashes, prior_hashes)
        support_counts[f"seed_{seed}"]["trials_per_direction"] = direction_counts

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "renderer": str(Path(__file__).relative_to(PROJECT)),
        "renderer_sha256": sha256(Path(__file__)),
        "parent_embedding_sha256": parent_hashes,
        "figure_sha256": output_hashes,
        "seeds": list(SEEDS),
        "architectures": list(ARCHITECTURES),
        "objectives": [SOFT_OBJECTIVE_LABELS[name] for name in OBJECTIVES],
        "embedding_dimensions_by_panel": {"3B": 4, "3C": 3, "3D": 3, "3E": 3, "3F": 3, "3G": 3},
        "displayed_coordinates": "raw exported embedding; first two for 3B, all three for 3C/3D/3E/3F/3G",
        "split_support": "cached train + validation + test embeddings concatenated; no rows refit or resampled",
        "support": support_counts,
        "positions_for_color": "original position_xy labels; fixed limits [-15, 15] cm as in official Figure 3 source",
        "encoder_or_decoder_training": False,
        "supersedes_previous_provenance_sha256": prior_manifest_sha256,
        "superseded_figure_sha256": prior_hashes if prior_manifest_sha256 else {},
        "panels": {
            "3B": "direction-averaged active-trial trajectories for CNN/Transformer x time only/position X,Y only/time + position X,Y",
            "3C": "Soft position X,Y-only embeddings, x/y position colors, CNN and Transformer",
            "3D": "Soft time-only embeddings, x/y position colors, CNN and Transformer",
            "3E": "active-only position X,Y fit; direction grouping is post hoc; full active trial support",
            "3F": "active-only time + position X,Y fit; direction grouping is post hoc; full active trial support",
            "3G": "active-only position X,Y fit; direction grouping is post hoc; full active trial support",
        },
        "comparability_note": "The official Figure 3b-d visualizations use cached embeddings without an explicit held-out filter. These Soft plots likewise show all available active rows, but Soft checkpoints were trained under the saved train split; the all-row display is descriptive and is not a held-out generalization claim or a full-sample refit.",
    }
    if prior_manifest_sha256 is not None and sha256(manifest_path) != prior_manifest_sha256:
        raise RuntimeError("Provenance changed during rendering; refusing to replace it")
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary_manifest.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    print(f"Created 30 Soft full-support figures (PNG+PDF): {OUT}")
    print(f"Common support: {len(support_reference)} rows, {support_counts['seed_42']['trials']} active trials")


if __name__ == "__main__":
    main()
