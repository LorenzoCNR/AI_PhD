"""Build a clean, English Figure 3 delivery from frozen local artifacts only.

No encoder training, embedding transform, dimensionality reduction, or decoder
fit is performed. Panels 3b and 3h/3i are plotted from saved arrays/results;
the other CEBRA panels are copied from the preserved 2026-10-28 delivery.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "results" / "figura_3_2026-10-29"
OLD = PROJECT / "results" / "figura_3_2026-10-28"
PAPER = PROJECT / "results" / "monkey_figure3_paper_v1"
H_SOFT = PROJECT / "results" / "monkey_fig3h_soft_matched_decoding_v1"
SOFT_FIT = PROJECT / "results" / "monkey_fig3_soft_descriptive_matched_v1"
CEBRA_H_SCORE = PAPER / "stage03_decoding_heldout"
CEBRA_H_EMBED = PAPER / "stage02_embedding_heldout"

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(PROJECT / "cebra_replica" / "monkey") not in sys.path:
    sys.path.insert(0, str(PROJECT / "cebra_replica" / "monkey"))

import plot_figure3b_comparison as comparison  # noqa: E402
from figure3_style import DIRECTION_COLORS  # noqa: E402
from user_soft import monkey_fig3_descriptive_plots as soft_plots  # noqa: E402


ARCH_COLORS = {"cnn": "#222222", "transformer": "#2166ac"}
SOFT_TASKS = {
    "position": "monkey_fig3h_soft_position_v1",
    "direction": "monkey_fig3h_soft_direction_v1",
    "ap": "monkey_fig3h_soft_ap_v1",
}
DIMS = (3, 4, 8, 16, 32)
SEEDS = (42, 43, 44, 45, 46)
ARCHS = ("cnn", "transformer")
TASKS = ("position", "direction", "ap")
TASK_TITLES = {
    "position": "Position decoding (X,Y)",
    "direction": "Direction decoding",
    "ap": "Active / Passive decoding",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT).as_posix()


def source(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": rel(path), "sha256": sha256(path)}


def source_many(paths) -> list[dict]:
    unique = {str(Path(p).resolve()): Path(p) for p in paths}
    return [source(unique[k]) for k in sorted(unique)]


def copy_pair(source_stem: Path, target_stem: Path) -> list[Path]:
    outputs = []
    for ext in (".png", ".pdf"):
        src, dst = source_stem.with_suffix(ext), target_stem.with_suffix(ext)
        if not src.is_file():
            raise FileNotFoundError(src)
        if dst.exists():
            if sha256(src) != sha256(dst):
                raise FileExistsError(f"Existing output differs from its source: {dst}")
        else:
            shutil.copy2(src, dst)
        outputs.append(dst)
    return outputs


def write_panel_provenance(panel: str, description: str, sources: list[Path],
                           outputs: list[Path], renderers: list[Path], **details) -> None:
    folder = OUT / panel
    presentation = {
        "description": description,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": source_many(sources),
        "renderers": source_many(renderers),
        "outputs": source_many(outputs),
        **details,
    }
    path = folder / "PROVENANCE.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("figure") != "3" + panel:
            raise ValueError(f"Unexpected provenance already exists: {path}")
        matches = [i for i, old in enumerate(record["presentations"])
                   if old.get("description") == description]
        if matches:
            first = matches[0]
            record["presentations"][first] = presentation
            record["presentations"] = [
                old for i, old in enumerate(record["presentations"])
                if old.get("description") != description or i == first
            ]
        else:
            record["presentations"].append(presentation)
    else:
        record = {
            "figure": "3" + panel,
            "training_or_refitting_performed": False,
            "presentations": [presentation],
        }
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def cebra_fit_sources(names: list[str]) -> list[Path]:
    files = []
    for name in names:
        folder = PAPER / "stage02_embedding" / name / "seed_42"
        files.extend([folder / "all_trials.npz", folder / "EMBEDDING_PROVENANCE.json"])
    return files


def render_cebra_b() -> tuple[list[Path], list[Path]]:
    base = PAPER / "stage02_embedding"
    baselines = PAPER / "stage02_embedding_baselines"
    specs = [
        ("CEBRA\nDirection", base / "figure3b_e_active_target_4d/seed_42/all_trials.npz",
         "embedding_raw", base / "figure3b_e_active_target_4d/seed_42/EMBEDDING_PROVENANCE.json", (2, 0), False),
        ("conv-piVAE\nq(z|x, direction)", baselines / "conv_pivae/seed_42/all_trials.npz",
         "with_label_raw", baselines / "conv_pivae/seed_42/EMBEDDING.json", (2, 3), True),
        ("CEBRA\nTime", base / "figure3b_active_time_4d/seed_42/all_trials.npz",
         "embedding_raw", base / "figure3b_active_time_4d/seed_42/EMBEDDING_PROVENANCE.json", (0, 1), False),
        ("conv-piVAE\nq(z|x)", baselines / "conv_pivae/seed_42/all_trials.npz",
         "without_label_raw", baselines / "conv_pivae/seed_42/EMBEDDING.json", (2, 3), True),
        ("t-SNE", baselines / "opentsne/seed_42/all_trials.npz",
         "embedding_raw", baselines / "opentsne/seed_42/EMBEDDING.json", (0, 1), True),
        ("UMAP", baselines / "umap/seed_42/all_trials.npz",
         "embedding_raw", baselines / "umap/seed_42/EMBEDDING.json", (0, 1), True),
    ]
    provenance = json.loads(comparison.TRIAL_PROVENANCE.read_text(encoding="utf-8"))
    active_trial_ids = np.asarray(
        provenance["session_trial_indices_by_split"]["active"]["all"], dtype=np.int64)
    length = int(provenance["trial_length_bins"])
    expected = np.sort(np.repeat(active_trial_ids, length) * length
                       + np.tile(np.arange(length, dtype=np.int64), len(active_trial_ids)))
    loaded = [comparison.read(path, key, manifest, ordinal, active_trial_ids, length, expected)
              for _, path, key, manifest, _, ordinal in specs]
    if not all(np.array_equal(item[2], loaded[0][2]) for item in loaded):
        raise ValueError("Figure 3b methods do not share aligned trial/time rows")
    if not all(np.array_equal(item[1], loaded[0][1]) for item in loaded):
        raise ValueError("Figure 3b methods do not share direction labels")

    target = OUT / "b" / "Figure_3b_CEBRA_and_baselines"
    outputs = [target.with_suffix(".png"), target.with_suffix(".pdf")]
    sources = [comparison.TRIAL_PROVENANCE]
    for _, data_path, _, manifest, _, _ in specs:
        sources.extend([data_path, manifest])
    existing = [p.exists() for p in outputs]
    if all(existing):
        return sources, outputs
    if any(existing):
        raise FileExistsError(f"Partial CEBRA 3b output: {outputs}")

    fig, axes = plt.subplots(1, 6, figsize=(24, 4.8))
    for ax, (title, _, _, _, coords, _), (embedding, direction, _, _) in zip(axes, specs, loaded):
        trials = embedding.reshape(-1, length, embedding.shape[1])
        labels = direction.reshape(-1, length)[:, 0]
        for direction_id in range(8):
            chosen = trials[labels == direction_id]
            if not len(chosen):
                raise ValueError(f"Missing direction {direction_id}: {title}")
            mean = chosen.mean(axis=0)
            ax.plot(mean[:, coords[0]], mean[:, coords[1]], color=DIRECTION_COLORS[direction_id], lw=2.2)
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
    fig.text(.5, .02, "Direction-averaged trajectories; full active-trial set",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.01, right=.99, top=.83, bottom=.1, wspace=.08)
    for p in outputs:
        ext = p.suffix
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return sources, outputs


def soft_panel_sources(panel: str, found: dict) -> list[Path]:
    keys = {
        "b": ("active_direction", "b_time_active_4d"),
        "c": ("c_position_active_3d",),
        "d": ("d_time_active_3d",),
        "e": ("active_direction", "e_direction_passive_4d"),
        "f": ("f_direction_ap_all_4d",),
        "g": ("g_position_all_4d",),
    }[panel]
    paths = []
    for key in keys:
        for arch in ARCHS:
            item = found[(key, arch)]
            folder = item["folder"]
            paths.extend([item["manifest"], folder / "embedding.npz",
                          folder / "checkpoint.pt", folder / "training_history.csv"])
    return paths


def render_cebra_h() -> tuple[list[Path], list[Path]]:
    rows = []
    sources = [PROJECT / "results/PAPER_FIGURE_DELIVERY/Figure_3/FIGURE3H_PROVENANCE.json"]
    for task in TASKS:
        for dim in DIMS:
            for seed in SEEDS:
                path = CEBRA_H_SCORE / task / f"dim_{dim}" / f"seed_{seed}" / "DECODING_RESULT.json"
                record = json.loads(path.read_text(encoding="utf-8"))
                if (record["task"], record["dimension"], record["seed"]) != (task, dim, seed):
                    raise ValueError(f"Unexpected CEBRA 3h score record: {path}")
                rows.append(record)
                sources.append(path)
                embed_folder = CEBRA_H_EMBED / task / f"dim_{dim}" / f"seed_{seed}"
                sources.extend(embed_folder / split / "embedding_and_trials.npz" for split in ("train", "test"))
    if len(rows) != len(TASKS) * len(DIMS) * len(SEEDS):
        raise ValueError(f"Incomplete CEBRA 3h score grid: {len(rows)}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)
    ylabels = {
        "position": "R² Position × 100",
        "direction": "Accuracy Direction (%)",
        "ap": "Accuracy Active / Passive (%)",
    }
    for ax, task in zip(axes, TASKS):
        for ix, dim in enumerate(DIMS):
            vals = [float(row["score"]) for row in rows
                    if row["task"] == task and row["dimension"] == dim]
            if len(vals) != len(SEEDS):
                raise ValueError(f"Incomplete CEBRA seed set for {task}/{dim}")
            ax.scatter(np.full(len(vals), ix), vals, s=38, facecolors="none", edgecolors="black")
            ax.hlines(np.median(vals), ix - .23, ix + .23, color="#e07b00", lw=2.4)
        ax.set_title(TASK_TITLES[task], fontsize=12, pad=10)
        ax.set_xticks(range(len(DIMS)), [str(dim) for dim in DIMS])
        ax.set_xlabel("Embedding dimension")
        ax.set_ylabel(ylabels[task])
        ax.spines[["top", "right"]].set_visible(False)
    outputs = []
    stem = OUT / "h" / "Figure_3h_CEBRA_Held_out_Decoding"
    for ext in ("png", "pdf"):
        path = stem.with_suffix("." + ext)
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    plt.close(fig)
    return sources, outputs


def render_soft_h() -> tuple[list[Path], list[Path]]:
    summary_path = H_SOFT / "SUMMARY.json"
    records = json.loads(summary_path.read_text(encoding="utf-8"))
    if len(records) != 150 or {(r["task"], r["architecture"], r["dimension"], r["seed"])
                               for r in records} != {
        (task, arch, dim, seed) for task in TASKS for arch in ARCHS for dim in DIMS for seed in SEEDS
    }:
        raise ValueError("Soft 3h grid is incomplete or has duplicate/missing cells")
    for row in records:
        p = H_SOFT / row["task"] / row["architecture"] / f"dim_{row['dimension']}" / f"seed_{row['seed']}" / "RESULT.json"
        result = json.loads(p.read_text(encoding="utf-8"))
        if result != row:
            raise ValueError(f"Summary and individual 3h result disagree: {p}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    for ax, task in zip(axes, TASKS):
        for arch in ARCHS:
            color = ARCH_COLORS[arch]
            shift = -.11 if arch == "cnn" else .11
            for ix, dim in enumerate(DIMS):
                vals = [100 * r["score"] for r in records if r["task"] == task
                        and r["architecture"] == arch and r["dimension"] == dim]
                if len(vals) != len(SEEDS):
                    raise ValueError(f"Incomplete seed set for {task}/{arch}/{dim}")
                jitter = np.linspace(-.035, .035, len(vals))
                ax.scatter(ix + shift + jitter, vals, s=31, marker="o" if arch == "cnn" else "s",
                           facecolors="none", edgecolors=color, linewidths=1.1, zorder=3)
                ax.hlines(np.median(vals), ix + shift - .075, ix + shift + .075,
                          color="#f28e2b", lw=2.5, zorder=4)
        ax.set_title(TASK_TITLES[task])
        ax.set_xticks(range(len(DIMS)), [str(d) for d in DIMS])
        ax.set_xlabel("Embedding dimension")
        ax.set_ylabel("R² Position (%)" if task == "position" else "Accuracy (%)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
    handles = [Line2D([], [], marker="o", linestyle="", color=ARCH_COLORS["cnn"], label="CNN"),
               Line2D([], [], marker="s", linestyle="", color=ARCH_COLORS["transformer"], label="Transformer"),
               Line2D([], [], color="#f28e2b", lw=2.5, label="Median")]
    axes[0].legend(handles=handles, frameon=False, loc="best")
    fig.suptitle("Figure 3H — Soft held-out decoding across five seeds (42–46)")
    outputs = []
    stem = OUT / "h" / "Figure_3h_Soft_held_out_decoding"
    for ext in ("png", "pdf"):
        p = stem.with_suffix("." + ext)
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        outputs.append(p)
    plt.close(fig)
    sources = [summary_path]
    for row in records:
        folder = H_SOFT / row["task"] / row["architecture"] / f"dim_{row['dimension']}" / f"seed_{row['seed']}"
        sources.extend([folder / "RESULT.json"])
        fit_folder = PROJECT / "results" / SOFT_TASKS[row["task"]] / "stage02_embedding" / row["architecture"] / f"dim_{row['dimension']}" / f"seed_{row['seed']}"
        for split in ("train", "test"):
            path = fit_folder / f"{split}_embedding.npz"
            if sha256(path) != row["source_sha256"][split]:
                raise ValueError(f"Soft 3h source embedding hash mismatch: {path}")
            sources.append(path)
    return sources, outputs


def rebuild_panel_h() -> None:
    """Regenerate only Figure 3h from saved scores; never fit or rewrite other panels."""
    (OUT / "h").mkdir(parents=True, exist_ok=True)
    cebra_sources, cebra_outputs = render_cebra_h()
    write_panel_provenance(
        "h", "CEBRA held-out decoding: Position X,Y, Direction, and Active/Passive",
        cebra_sources, cebra_outputs, [Path(__file__)],
        score_support="Saved TRAIN-fitted decoder scores on held-out TEST; five seeds per embedding dimension",
    )
    soft_sources, soft_outputs = render_soft_h()
    write_panel_provenance(
        "h", "Soft held-out decoding: Position X,Y, Direction, and Active/Passive",
        soft_sources, soft_outputs, [Path(__file__)],
        score_support="150 saved results: 3 tasks × 2 architectures × 5 dimensions × 5 seeds",
    )
    print(f"Regenerated only Figure 3h outputs and provenance in {OUT / 'h'}; no fit or score recomputation.")


def render_soft_i() -> tuple[list[Path], list[Path], dict]:
    loaded = {}
    for arch in ARCHS:
        folder = H_SOFT / "position" / arch / "dim_16" / "seed_42"
        result_path, pred_path = folder / "RESULT.json", folder / "test_predictions.npz"
        record = json.loads(result_path.read_text(encoding="utf-8"))
        if record["task"] != "position" or record["dimension"] != 16 or record["seed"] != 42:
            raise ValueError(f"Unexpected 3i result: {result_path}")
        if sha256(pred_path) != record["predictions_sha256"]:
            raise ValueError(f"Saved 3i prediction hash mismatch: {pred_path}")
        fit_folder = PROJECT / "results" / SOFT_TASKS["position"] / "stage02_embedding" / arch / "dim_16" / "seed_42"
        for split in ("train", "test"):
            if sha256(fit_folder / f"{split}_embedding.npz") != record["source_sha256"][split]:
                raise ValueError(f"3i source embedding hash mismatch: {fit_folder}/{split}")
        with np.load(pred_path, allow_pickle=False) as z:
            loaded[arch] = {k: z[k].copy() for k in ("true", "predicted", "source_id", "trial_id")}
        loaded[arch]["record"] = record
        loaded[arch]["sources"] = [result_path, pred_path,
                                   fit_folder / "train_embedding.npz", fit_folder / "test_embedding.npz"]

    reference = loaded["cnn"]
    for key in ("true", "source_id", "trial_id"):
        if not np.array_equal(reference[key], loaded["transformer"][key]):
            raise ValueError(f"CNN and Transformer 3i test support differs for {key}")
    selected_trial = int(np.min(reference["trial_id"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)
    for ax, arch in zip(axes, ARCHS):
        data = loaded[arch]
        ids = np.flatnonzero(data["trial_id"] == selected_trial)
        ids = ids[np.argsort(data["source_id"][ids], kind="stable")]
        if len(ids) < 2:
            raise ValueError("Selected held-out test trial has too few samples")
        actual, predicted = data["true"][ids], data["predicted"][ids]
        ax.plot(actual[:, 0], actual[:, 1], color="#777777", lw=1.7, label="Observed Position X,Y")
        ax.plot(predicted[:, 0], predicted[:, 1], color=ARCH_COLORS[arch], lw=1.5,
                label="Decoded Position X,Y")
        ax.set_title(arch.upper())
        ax.set_xlabel("Position X (cm)"); ax.set_ylabel("Position Y (cm)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.16); ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f"Figure 3I — Soft Position X,Y decoding, held-out test trial {selected_trial}\n16D; seed 42")
    outputs = []
    stem = OUT / "i" / "Figure_3i_Soft_Position_XY_16D_seed42"
    for ext in ("png", "pdf"):
        p = stem.with_suffix("." + ext)
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        outputs.append(p)
    plt.close(fig)
    sources = [p for arch in ARCHS for p in loaded[arch]["sources"]]
    details = {
        "dimension": 16,
        "seed": 42,
        "decoder": "saved uniform kNN predictions, cosine distance, k=36; no decoder fit or prediction recomputation",
        "decoder_training_support": "TRAIN split only",
        "evaluation_support": "held-out TEST split; 26 trials in the full score; plotted trial selected as minimum test trial_id, independent of scores",
        "plotted_trial_id": selected_trial,
        "test_r2_position_xy": {arch: loaded[arch]["record"]["score"] for arch in ARCHS},
        "color_mapping": {"observed": "#777777", "CNN decoded": ARCH_COLORS["cnn"],
                          "Transformer decoded": ARCH_COLORS["transformer"]},
    }
    return sources, outputs, details


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    unexpected = [p.name for p in OUT.iterdir() if p.is_file() or p.name not in set("abcdefghi")]
    if unexpected:
        raise FileExistsError(f"Unexpected files in canonical destination: {unexpected}")
    for panel in "abcdefghi":
        (OUT / panel).mkdir(parents=True, exist_ok=True)

    # All current Soft descriptive artifacts are validated against their manifests.
    found, missing = soft_plots.cells()
    if missing or len(found) != 14:
        raise RuntimeError(f"Soft panels 3b–3g are not complete: found={len(found)}, missing={missing}")
    soft_plots.FIGURES = OUT
    existing_soft = {
        panel: [p for p in (OUT / panel).iterdir() if p.name.startswith(f"Figure_3{panel}_Soft_")
                and p.suffix in (".png", ".pdf")]
        for panel in "bcdefg"
    }
    if not all(len(files) == 4 for files in existing_soft.values()):
        if any(existing_soft.values()):
            raise FileExistsError(f"Partial Soft descriptive outputs: {existing_soft}")
        soft_plots.render(found)

    # 3a: saved movement trajectories (no model fit, no text embedded in source image).
    a_stem = OLD / "a" / "Figura_3a_movimenti_attivi_passivi"
    a_outputs = copy_pair(a_stem, OUT / "a" / "Figure_3a_Active_Passive_Movement")
    write_panel_provenance("a", "Active and passive movement trajectories",
                           [a_stem.with_suffix(".png"), a_stem.with_suffix(".pdf")], a_outputs, [],
                           plot_text="No embedded labels; source has no non-English plot text.")

    # 3b CEBRA/baselines is rendered anew to replace old Italian labels.
    b_sources, b_outputs = render_cebra_b()
    write_panel_provenance("b", "Direction-averaged embeddings for CEBRA, conv-piVAE, t-SNE, and UMAP",
                           b_sources, b_outputs, [Path(comparison.__file__)],
                           direction_palette="Shared matplotlib HSV mapping for Direction 1–8",
                           note="conv-piVAE panels show q(z|x, direction) and q(z|x), both from the same direction-prior fit.")

    # 3b–3g Soft panels come from the completed 14 fit artifacts checked above.
    for panel, description in {
        "b": "Direction and Time", "c": "Position X,Y", "d": "Time",
        "e": "Direction", "f": "Direction and Active/Passive", "g": "Position X,Y",
    }.items():
        output_files = [p for p in (OUT / panel).iterdir() if p.is_file()
                        and p.name.startswith(f"Figure_3{panel}_Soft_") and p.suffix in (".png", ".pdf")]
        if len(output_files) != 4:
            raise RuntimeError(f"Expected CNN and Transformer PNG/PDF for Soft panel {panel}: {output_files}")
        renderer_sources = [Path(soft_plots.__file__), Path(soft_plots.__file__).parent / "monkey_fig3_descriptive_matched.py"]
        write_panel_provenance(panel, description,
            soft_panel_sources(panel, found), output_files, renderer_sources,
            architecture="Soft CNN and Soft Transformer", seed=42,
            direction_palette="Same direction-to-HSV mapping used by the CEBRA panels",
            plotted_support="Full-recording descriptive embeddings, as recorded in the fit manifests")

    # CEBRA panels c–g: preserve old dated files; copy English plot text to new canonical paths.
    cebra_panel_stems = {
        "c": (OLD / "c" / "Figura_3c_CEBRA", "CEBRA Position X,Y"),
        "d": (OLD / "d" / "Figura_3d_CEBRA", "CEBRA Time"),
        "e": (OLD / "e" / "Figura_3e_CEBRA", "CEBRA Direction"),
        "f": (OLD / "f" / "Figura_3f_CEBRA", "CEBRA Direction and Active/Passive"),
        "g": (OLD / "g" / "Figura_3g_CEBRA_Position_XY", "CEBRA Position X,Y"),
    }
    fits = {
        "c": ["figure3c_active_position_3d"],
        "d": ["figure3d_active_time_3d"],
        "e": ["figure3b_e_active_target_4d", "figure3e_passive_target_4d"],
        "f": ["figure3f_all_target_ap_4d"],
    }
    g_folder = PAPER / "figure3g_position_only_correction_v1" / "stage02_embedding" / "figure3g_all_position_only_4d_corrected" / "seed_42"
    for panel, (src, description) in cebra_panel_stems.items():
        safe_description = description.replace("/", "_").replace(",", "").replace(" ", "_")
        outputs = copy_pair(src, OUT / panel / f"Figure_3{panel}_CEBRA_{safe_description}")
        if panel == "g":
            inputs = [g_folder / "all_trials.npz", g_folder / "EMBEDDING_PROVENANCE.json"]
        else:
            inputs = cebra_fit_sources(fits[panel])
        write_panel_provenance(panel, description, inputs + [src.with_suffix(".png"), src.with_suffix(".pdf")],
                               outputs, [PROJECT / "cebra_replica/monkey/plot_figure3_paper.py"],
                               note="CEBRA local fit; source plot was copied without modifying its English plot text.")

    # 3h CEBRA and Soft: redraw only from saved scores/results, without fitting.
    cebra_h_sources, h_cebra_outputs = render_cebra_h()
    write_panel_provenance("h", "CEBRA held-out decoding: Position X,Y, Direction, and Active/Passive",
                           cebra_h_sources, h_cebra_outputs, [Path(__file__)],
                           score_support="Saved TRAIN-fitted decoder scores on held-out TEST; five seeds per embedding dimension")

    h_soft_sources, h_soft_outputs = render_soft_h()
    write_panel_provenance("h", "Soft held-out decoding: Position X,Y, Direction, and Active/Passive",
                           h_soft_sources, h_soft_outputs, [Path(__file__)],
                           score_support="150 saved results: 3 tasks × 2 architectures × 5 dimensions × 5 seeds")

    # 3i CEBRA: use the already saved English figure and result; no decoding is rerun.
    i_cebra_stem = PROJECT / "results/PAPER_FIGURE_DELIVERY/Figure_3/FIGURE3I_LOCAL_REPLICATION"
    i_cebra_outputs = copy_pair(i_cebra_stem, OUT / "i" / "Figure_3i_CEBRA_Position_XY_16D_seed42")
    i_cebra_result = PROJECT / "results/PAPER_FIGURE_DELIVERY/Figure_3/FIGURE3I_RESULT.json"
    cebra_i_record = json.loads(i_cebra_result.read_text(encoding="utf-8"))
    cebra_i_base = CEBRA_H_EMBED / "position" / "dim_16" / "seed_42"
    cebra_i_sources = [i_cebra_stem.with_suffix(".png"), i_cebra_stem.with_suffix(".pdf"), i_cebra_result]
    cebra_i_sources.extend([cebra_i_base / split / "embedding_and_trials.npz" for split in ("train", "test")])
    write_panel_provenance("i", "CEBRA held-out Position X,Y decoding",
                           cebra_i_sources, i_cebra_outputs, [PROJECT / "cebra_replica/monkey/plot_figure3hi.py"],
                           dimension=cebra_i_record["dimension"], seed=cebra_i_record["seed"],
                           score=cebra_i_record["position_R2"], decoder="Saved CEBRA kNN result; not rerun")

    i_soft_sources, i_soft_outputs, i_soft_details = render_soft_i()
    write_panel_provenance("i", "Soft held-out Position X,Y decoding",
                           i_soft_sources, i_soft_outputs, [Path(__file__)], **i_soft_details)

    index = {
        "title": "Monkey Figure 3 — canonical rebuild",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_folder": rel(OUT),
        "previous_dated_folder_preserved": rel(OLD),
        "training_or_refitting_performed": False,
        "panel_folders": list("abcdefghi"),
        "english_plot_text": True,
        "soft_3b_g_fit_validation": {"complete": len(found), "expected": 14, "missing": missing},
        "soft_3h_result_validation": {"complete": 150, "expected": 150},
        "shared_colors": {
            "direction": "Same HSV palette in CEBRA and Soft trajectory plots",
            "soft_architecture": ARCH_COLORS,
        },
        "panel_provenance": {p: rel(OUT / p / "PROVENANCE.json") for p in "abcdefghi"},
    }
    (OUT / "FIGURE_3_INDEX.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Monkey Figure 3\n\n"
        "Canonical English-language rebuild. Each panel folder (`a`–`i`) contains the plots and a `PROVENANCE.json` with source paths and SHA-256 hashes.\n\n"
        "All panels reuse completed local fits, embeddings, predictions, and metrics. No encoder, embedding, or decoder was refit.\n\n"
        "Panel 3i Soft uses the saved 16D, seed 42 held-out predictions and plots the minimum test trial ID for both architectures.\n",
        encoding="utf-8")
    print(f"Complete: {OUT}")
    print("Panels: a–i; Soft descriptive fits 14/14; Soft 3h metrics 150/150; no fit/refit performed.")


if __name__ == "__main__":
    if sys.argv[1:] == ["--panel", "h"]:
        rebuild_panel_h()
    else:
        main()
