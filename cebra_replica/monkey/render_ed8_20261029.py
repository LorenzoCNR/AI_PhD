"""Rebuild Extended Data Figure 8 from frozen local/reference artifacts.

This renderer is read-only with respect to all fit outputs. It never starts,
resumes, or edits training. Re-run it after the ED8 Soft campaign finishes to
add panels from newly completed, hash-verified manifests.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
from pathlib import Path

import h5py
import cebra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/figura_ext_8_2026-10-29"
CEBRA = ROOT / "results/monkey_extended8_paper_v1"
SOFT = ROOT / "results/monkey_ed8_soft_matched_v1"
DELIVERY = ROOT / "results/PAPER_FIGURE_DELIVERY/Extended_Data/ED8"
H5 = ROOT / "cebra_replica/monkey/official_reference/EDFigure8.h5"
PANEL_NAMES = {"a": "Direction embedding", "b": "Position embedding",
               "c": "Time embedding", "d": "Direction decoding over time",
               "e": "Training loss", "f": "Held-out decoding"}
DIMENSIONS = (3, 4, 8, 16, 32)
SEEDS = (42, 43, 44, 45, 46)
SOFT_TIME_BINS = np.arange(5, 596)
POSITION_CMAP = "seismic"
POSITION_LIMITS = (-15, 15)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_figure(fig, panel: str, stem: str) -> list[Path]:
    folder = OUT / panel
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / f"{stem}.{ext}" for ext in ("png", "pdf")]
    fig.savefig(paths[0], dpi=220, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def verify_artifacts(folder: Path, manifest: dict) -> list[str]:
    errors = []
    for name, expected in manifest.get("artifacts", {}).items():
        path = folder / name
        if not path.is_file():
            errors.append(f"{path.relative_to(ROOT)} (missing)")
        elif sha256(path) != expected:
            errors.append(f"{path.relative_to(ROOT)} (SHA-256 mismatch)")
    return errors


def load_cebra_embedding(mode: str):
    folder = CEBRA / "stage02_embedding_mse" / mode / "seed_42"
    path = folder / "all_trials.npz"
    manifest_path = folder / "EMBEDDING.json"
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing CEBRA {mode} embedding or manifest: {folder}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256(path) != manifest["artifact_sha256"]:
        raise ValueError(f"CEBRA embedding hash mismatch: {path}")
    with np.load(path) as z:
        data = {key: z[key].copy() for key in z.files}
    if data["embedding_raw"].ndim != 2 or data["embedding_raw"].shape[1] != 2:
        raise ValueError(f"Expected 2D CEBRA embedding: {path}")
    if not data["valid_mask"].all():
        raise ValueError(f"Unexpected invalid rows in full-recording CEBRA embedding: {path}")
    return data, path, manifest_path


def load_soft_embedding(job: dict):
    fields = "+".join(sorted(job["fields"]))
    folder = (SOFT / "fits" / job["scope"] / fields /
              ("shuffle_" + str(job["shuffle"])) / job["architecture"] /
              f"dim_{job['dimension']}" / f"seed_{job['seed']}")
    manifest_path = folder / "manifest.json"
    embedding_path = folder / "embedding_full.npz"
    if not manifest_path.is_file() or not embedding_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("actual_updates") != 10000 or verify_artifacts(folder, manifest):
            return None
        if "embedding_full.npz" not in manifest.get("artifacts", {}):
            return None
        with np.load(embedding_path) as z:
            data = {key: z[key].copy() for key in z.files}
        if data["embedding"].shape != (len(data["trial_id"]), 2):
            return None
        if data["valid_mask"].shape != (len(data["trial_id"]),):
            return None
        return data, folder, manifest_path
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def add_latent_axes(ax) -> None:
    ax.set_xlabel("Latent dimension 1")
    ax.set_ylabel("Latent dimension 2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def direction_trajectories(data: dict, embedding_key: str):
    z = data[embedding_key]
    trial_id = data["trial_id"]
    time_id = data["time_id"]
    direction = data["direction"].astype(int)
    by_direction = {d: [] for d in range(8)}
    for trial in np.unique(trial_id):
        ix = np.flatnonzero(trial_id == trial)
        ix = ix[np.argsort(time_id[ix])]
        labels = np.unique(direction[ix])
        if len(ix) == 600 and len(labels) == 1 and 0 <= labels[0] < 8:
            by_direction[int(labels[0])].append(z[ix])
    if any(not by_direction[d] for d in range(8)):
        raise ValueError("Each of the eight directions must have full 600-bin trials")
    return {d: np.mean(np.stack(trials), axis=0) for d, trials in by_direction.items()}


def render_direction_embedding(data: dict, embedding_key: str, title: str,
                               panel: str, stem: str):
    z = data[embedding_key]
    colors = plt.cm.hsv(np.arange(8) / 8)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7))
    points = axes[0].scatter(z[:, 0], z[:, 1], c=data["direction"], cmap="hsv",
                             norm=BoundaryNorm(np.arange(-0.5, 8.5, 1), 256),
                             s=0.08, rasterized=True)
    cbar = fig.colorbar(points, ax=axes[0], ticks=np.arange(8), fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels([str(i) for i in range(1, 9)])
    cbar.set_label("Movement direction")
    axes[0].set_title("All active-trial samples")
    add_latent_axes(axes[0])

    means = direction_trajectories(data, embedding_key)
    for direction, trajectory in means.items():
        axes[1].scatter(trajectory[:, 0], trajectory[:, 1], s=2.1,
                        color=colors[direction], rasterized=True)
    handles = [Line2D([0], [0], color=colors[d], marker="o", lw=0, label=str(d + 1))
               for d in range(8)]
    axes[1].legend(handles=handles, title="Direction", ncol=2, frameon=False,
                   fontsize=8, title_fontsize=9, loc="best")
    axes[1].set_title("Trajectories averaged by direction")
    add_latent_axes(axes[1])
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return save_figure(fig, panel, stem)


def render_position_embedding(data: dict, embedding_key: str, title: str,
                              panel: str, stem: str):
    z = data[embedding_key]
    pos = data["position_xy"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7))
    for idx, (ax, label) in enumerate(zip(axes, ("Hand X position", "Hand Y position"))):
        points = ax.scatter(z[:, 0], z[:, 1], c=pos[:, idx], cmap=POSITION_CMAP,
                            vmin=POSITION_LIMITS[0], vmax=POSITION_LIMITS[1],
                            s=0.08, rasterized=True)
        bar = fig.colorbar(points, ax=ax, fraction=0.046, pad=0.04,
                           ticks=np.linspace(-15, 15, 7))
        bar.set_label("Position (cm)")
        ax.set_title(label)
        add_latent_axes(ax)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return save_figure(fig, panel, stem)


def render_time_embedding(data: dict, embedding_key: str, title: str,
                           panel: str, stem: str):
    z = data[embedding_key]
    pos = data["position_xy"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7))
    for idx, (ax, label) in enumerate(zip(axes, ("Colored by hand X position", "Colored by hand Y position"))):
        points = ax.scatter(z[:, 0], z[:, 1], c=pos[:, idx], cmap=POSITION_CMAP,
                            vmin=POSITION_LIMITS[0], vmax=POSITION_LIMITS[1],
                            s=0.08, rasterized=True)
        bar = fig.colorbar(points, ax=ax, fraction=0.046, pad=0.04,
                           ticks=np.linspace(-15, 15, 7))
        bar.set_label("Position (cm)")
        ax.set_title(label)
        add_latent_axes(ax)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return save_figure(fig, panel, stem)


def soft_jobs():
    plan_path = SOFT / "PLAN.json"
    if not plan_path.is_file():
        return [], plan_path
    return json.loads(plan_path.read_text(encoding="utf-8"))["jobs"], plan_path


def campaign_status(jobs: list[dict]):
    rows = []
    completed = {}
    for job in jobs:
        artifact = load_soft_embedding(job) if job["panel"] in "abc" else None
        fields = "+".join(sorted(job["fields"]))
        folder = (SOFT / "fits" / job["scope"] / fields /
                  ("shuffle_" + str(job["shuffle"])) / job["architecture"] /
                  f"dim_{job['dimension']}" / f"seed_{job['seed']}")
        manifest_path = folder / "manifest.json"
        manifest = None
        valid = False
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                valid = manifest.get("actual_updates") == 10000 and not verify_artifacts(folder, manifest)
            except (OSError, ValueError, json.JSONDecodeError):
                valid = False
        key = (job["panel"], job["architecture"])
        completed[key] = completed.get(key, 0) + int(valid)
        rows.append({"panel": job["panel"], "scope": job["scope"], "fields": job["fields"],
                     "shuffle": job["shuffle"], "architecture": job["architecture"],
                     "dimension": job["dimension"], "seed": job["seed"],
                     "fit_path": str(folder.relative_to(ROOT)), "manifest_exists": manifest_path.is_file(),
                     "fit_complete_and_hash_valid": valid,
                     "embedding_available": artifact is not None})
    return rows, completed


def soft_job_folder(job: dict) -> Path:
    fields = "+".join(sorted(job["fields"]))
    return (SOFT / "fits" / job["scope"] / fields /
            ("shuffle_" + str(job["shuffle"])) / job["architecture"] /
            f"dim_{job['dimension']}" / f"seed_{job['seed']}")


def render_soft_embedding_jobs(jobs: list[dict], inputs: list[Path], outputs: list[Path]):
    output_rows = []
    for job in jobs:
        if job["panel"] not in "abc":
            continue
        loaded = load_soft_embedding(job)
        if loaded is None:
            continue
        data, folder, manifest_path = loaded
        inputs.extend([manifest_path, folder / "embedding_full.npz"])
        arch = job["architecture"].upper()
        if job["panel"] == "a":
            title = f"Soft {arch} (Stable loss) | Direction label | 2D"
            paths = render_direction_embedding(data, "embedding", title, "a",
                                               f"ED8_A_Soft_{arch}_Direction_2D")
        elif job["panel"] == "b":
            title = f"Soft {arch} (Stable loss) | Position X,Y labels | 2D"
            paths = render_position_embedding(data, "embedding", title, "b",
                                              f"ED8_B_Soft_{arch}_Position_2D")
        else:
            title = f"Soft {arch} (Stable loss) | Time-only objective | 2D"
            paths = render_time_embedding(data, "embedding", title, "c",
                                          f"ED8_C_Soft_{arch}_Time_2D")
        outputs.extend(paths)
        output_rows.append({"panel": job["panel"], "architecture": job["architecture"],
                            "fit_manifest": str(manifest_path.relative_to(ROOT)),
                            "embedding": str((folder / "embedding_full.npz").relative_to(ROOT)),
                            "rendered": True})
    return output_rows


def render_cebra_embeddings(inputs: list[Path], outputs: list[Path]):
    specs = {
        "a": ("target", "CEBRA (InfoMSE) | Direction labels | 2D",
              render_direction_embedding, "ED8_A_CEBRA_InfoMSE_Direction_2D"),
        "b": ("position", "CEBRA (InfoMSE) | Position X,Y labels | 2D",
              render_position_embedding, "ED8_B_CEBRA_InfoMSE_Position_2D"),
        "c": ("time", "CEBRA-Time (InfoMSE) | Time objective | 2D",
              render_time_embedding, "ED8_C_CEBRA_Time_InfoMSE_2D"),
    }
    for panel, (mode, title, renderer, stem) in specs.items():
        data, path, manifest = load_cebra_embedding(mode)
        inputs.extend([path, manifest])
        generated = renderer(data, "embedding_raw", title, panel, stem)
        outputs.extend(generated)


def soft_direction_timecourse(inputs: list[Path]):
    """Decode direction per time bin from completed Figure 3h Soft fits only."""
    from sklearn.metrics import accuracy_score

    source_root = ROOT / "results/monkey_fig3h_soft_direction_v1"
    rows = []
    parents = []
    expected_times = SOFT_TIME_BINS
    for architecture in ("cnn", "transformer"):
        for seed in SEEDS:
            fit_folder = source_root / "stage01_fit" / architecture / "dim_4" / f"seed_{seed}"
            embed_folder = source_root / "stage02_embedding" / architecture / "dim_4" / f"seed_{seed}"
            fit_manifest_path = fit_folder / "manifest.json"
            embed_manifest_path = embed_folder / "manifest.json"
            fit_manifest = json.loads(fit_manifest_path.read_text(encoding="utf-8"))
            embed_manifest = json.loads(embed_manifest_path.read_text(encoding="utf-8"))
            config = fit_manifest["config"]
            if (config.get("version") != "soft_v1_stable" or config.get("condition") != "direction"
                    or config.get("output_dimension") != 4 or config.get("seed") != seed
                    or config.get("embedding_temperature") != 1.0
                    or config.get("metadata_temperature") != 0.5
                    or config.get("max_updates") != 10000):
                raise ValueError(f"Incompatible completed Soft direction fit: {fit_manifest_path}")
            if fit_manifest.get("status") not in (None, "complete", "completed"):
                raise ValueError(f"Soft direction fit is not complete: {fit_manifest_path}")
            checkpoint = fit_folder / "checkpoint.pt"
            if (fit_manifest.get("actual_updates") != 10000 or not checkpoint.is_file()
                    or sha256(checkpoint) != fit_manifest.get("checkpoint_sha256")):
                raise ValueError(f"Soft direction checkpoint is missing or not verified: {checkpoint}")
            inputs.append(fit_manifest_path)
            parent_paths = {}
            arrays = {}
            for split in ("train", "test"):
                path = embed_folder / f"{split}_embedding.npz"
                if not path.is_file() or sha256(path) != embed_manifest["files"][split]:
                    raise ValueError(f"Missing or changed Soft {split} embedding: {path}")
                with np.load(path, allow_pickle=False) as archive:
                    arrays[split] = {key: archive[key].copy() for key in archive.files}
                parent_paths[split] = path
                inputs.append(path)
            inputs.append(embed_manifest_path)
            train, test = arrays["train"], arrays["test"]
            train_mask = np.asarray(train["valid_mask"], dtype=bool)
            test_mask = np.asarray(test["valid_mask"], dtype=bool)
            train_trial = train["trial_id"][train_mask]
            test_trial = test["trial_id"][test_mask]
            if np.intersect1d(np.unique(train_trial), np.unique(test_trial)).size:
                raise ValueError("Soft direction train/test trial overlap")
            train_time = train["time_id"][train_mask]
            test_time = test["time_id"][test_mask]
            if not np.array_equal(np.unique(train_time), expected_times) or not np.array_equal(np.unique(test_time), expected_times):
                raise ValueError("Unexpected Soft per-time-bin support")
            for time_bin in expected_times:
                train_rows = train_mask & (train["time_id"] == time_bin)
                test_rows = test_mask & (test["time_id"] == time_bin)
                decoder = cebra.KNNDecoder(n_neighbors=36, metric="cosine")
                decoder.fit(train["embedding_raw"][train_rows], train["direction"][train_rows])
                predicted = decoder.predict(test["embedding_raw"][test_rows])
                accuracy = 100 * accuracy_score(test["direction"][test_rows], predicted)
                rows.append({"method": f"Soft-{architecture.upper()}", "training_seed": seed,
                             "time_bin": int(time_bin), "time_from_onset_ms": int(time_bin - 100),
                             "accuracy_percent": float(accuracy),
                             "train_trials": int(len(np.unique(train["trial_id"][train_rows]))),
                             "test_trials": int(len(np.unique(test["trial_id"][test_rows]))),
                             "dimension": 4, "fit_condition": "Direction"})
            parents.append({"architecture": architecture, "seed": seed, "fit_manifest": str(fit_manifest_path.relative_to(ROOT)),
                            "fit_manifest_sha256": sha256(fit_manifest_path),
                            "embedding_manifest": str(embed_manifest_path.relative_to(ROOT)),
                            "train_embedding_sha256": sha256(parent_paths["train"]),
                            "test_embedding_sha256": sha256(parent_paths["test"])})
    return rows, parents


def read_cached_soft_direction_timecourse(inputs: list[Path]):
    csv_path = OUT / "d" / "ED8_D_Soft_Direction_Decoding_Over_Time.csv"
    provenance_path = OUT / "d" / "ED8_D_Soft_Direction_Decoding_Provenance.json"
    if not csv_path.is_file() or not provenance_path.is_file():
        return None
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (provenance.get("decoder") != "cebra.KNNDecoder, cosine, k=36, fitted separately at each time bin"
            or provenance.get("csv_sha256") != sha256(csv_path)
            or provenance.get("seeds") != list(SEEDS)
            or provenance.get("fit_dimension") != 4):
        return None
    source_root = ROOT / "results/monkey_fig3h_soft_direction_v1"
    for parent in provenance.get("parents", []):
        fit_path = ROOT / parent["fit_manifest"]
        embed_path = ROOT / parent["embedding_manifest"]
        arch, seed = parent["architecture"], parent["seed"]
        fit_folder = source_root / "stage01_fit" / arch / "dim_4" / f"seed_{seed}"
        embed_folder = source_root / "stage02_embedding" / arch / "dim_4" / f"seed_{seed}"
        if (not fit_path.is_file() or sha256(fit_path) != parent["fit_manifest_sha256"]
                or not embed_path.is_file()):
            return None
        fit_record = json.loads(fit_path.read_text(encoding="utf-8"))
        checkpoint = fit_path.parent / "checkpoint.pt"
        config = fit_record.get("config", {})
        if (fit_record.get("actual_updates") != 10000 or not checkpoint.is_file()
                or sha256(checkpoint) != fit_record.get("checkpoint_sha256")
                or config.get("condition") != "direction"
                or config.get("output_dimension") != 4
                or config.get("embedding_temperature") != 1.0
                or config.get("metadata_temperature") != 0.5):
            return None
        for split in ("train", "test"):
            path = embed_folder / f"{split}_embedding.npz"
            if not path.is_file() or sha256(path) != parent[f"{split}_embedding_sha256"]:
                return None
        inputs.extend([fit_path, embed_path,
                       embed_folder / "train_embedding.npz", embed_folder / "test_embedding.npz"])
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    expected = 2 * len(SEEDS) * len(SOFT_TIME_BINS)
    if len(rows) != expected:
        return None
    inputs.extend([csv_path, provenance_path])
    return rows, provenance.get("parents", []), csv_path, provenance_path


def render_panel_d(inputs: list[Path], outputs: list[Path]):
    csv_path = DELIVERY / "DIRECTION_ACCURACY_BY_TIME.csv"
    provenance_path = DELIVERY / "ED8_DIRECTION_TIMECOURSE_PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if sha256(csv_path) != provenance["csv_sha256"]:
        raise ValueError("Direction time-course CSV does not match its provenance")
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    soft_csv = OUT / "d" / "ED8_D_Soft_Direction_Decoding_Over_Time.csv"
    soft_csv.parent.mkdir(parents=True, exist_ok=True)
    soft_provenance_path = OUT / "d" / "ED8_D_Soft_Direction_Decoding_Provenance.json"
    cached = read_cached_soft_direction_timecourse(inputs)
    if cached is None:
        soft_rows, soft_parents = soft_direction_timecourse(inputs)
        with soft_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(soft_rows[0]))
            writer.writeheader()
            writer.writerows(soft_rows)
        soft_provenance = {
            "method": "evaluation from already completed held-out Soft embeddings; no encoder training",
            "source_campaign": "results/monkey_fig3h_soft_direction_v1",
            "fit_condition": "direction categorical",
            "fit_dimension": 4,
            "fit_split": "TRAIN; active trials",
            "score_split": "TEST; trial-disjoint",
            "seeds": list(SEEDS),
            "architectures": ["cnn", "transformer"],
            "decoder": "cebra.KNNDecoder, cosine, k=36, fitted separately at each time bin",
            "time_bins": [5, 595],
            "time_from_movement_onset_ms": [-95, 495],
            "parents": soft_parents,
            "csv_sha256": sha256(soft_csv),
        }
        write_json(soft_provenance_path, soft_provenance)
        inputs.extend([soft_csv, soft_provenance_path])
    else:
        soft_rows, soft_parents, soft_csv, soft_provenance_path = cached
    outputs.extend([soft_csv, soft_provenance_path])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    styles = [
        ("CEBRA-Behavior", "CEBRA (Direction; mean of 5 seeds)", "black", 1.0),
        ("conv-piVAE", "conv-piVAE (Direction; seed 42)", "#555555", 0.78),
        ("Ridge classifier", "Ridge classifier", "#888888", 0.58),
    ]
    for method, label, color, alpha in styles:
        grouped = {}
        for row in rows:
            if row["method"] == method:
                grouped.setdefault(row["training_seed"], []).append(row)
        series = [sorted(values, key=lambda row: int(row["time_bin"]))
                  for values in grouped.values()]
        if not series or any(len(values) != 600 for values in series):
            raise ValueError(f"Incomplete direction time-course for {method}")
        x = np.asarray([int(row["time_from_onset_ms"]) for row in series[0]])
        y = np.mean([[float(row["accuracy_percent"]) for row in values]
                     for values in series], axis=0)
        ax.plot(x, y, label=label, color=color, alpha=alpha, lw=2)
    for method, label, color in (
        ("Soft-CNN", "Soft CNN (Direction; mean of 5 seeds)", "#2166ac"),
        ("Soft-TRANSFORMER", "Soft Transformer (Direction; mean of 5 seeds)", "#d95f02"),
    ):
        grouped = {}
        for row in soft_rows:
            if row["method"] == method:
                grouped.setdefault(row["training_seed"], []).append(row)
        curves = [sorted(values, key=lambda row: int(row["time_bin"])) for values in grouped.values()]
        if len(curves) != 5 or any(len(curve) != len(SOFT_TIME_BINS) for curve in curves):
            raise ValueError(f"Incomplete Soft direction time-course: {method}")
        x = np.asarray([int(row["time_from_onset_ms"]) for row in curves[0]])
        y = np.mean([[float(row["accuracy_percent"]) for row in curve] for curve in curves], axis=0)
        ax.plot(x, y, label=label, color=color, lw=1.8)
    ax.set_xlabel("Time from movement onset (ms)")
    ax.set_ylabel("Direction accuracy (%)")
    ax.set_title("Direction decoding across time | Held-out trials")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    outputs.extend(save_figure(fig, "d", "ED8_D_Direction_Decoding_Over_Time"))

    # Additional direct comparison: keep the published/reference methods
    # separate from the locally trained Soft encoders, with identical axes.
    fig_compare, axes_compare = plt.subplots(1, 2, figsize=(12.5, 4.8),
                                             sharex=True, sharey=True)
    ax_rest, ax_soft = axes_compare
    for method, label, color, alpha in styles:
        grouped = {}
        for row in rows:
            if row["method"] == method:
                grouped.setdefault(row["training_seed"], []).append(row)
        series = [sorted(values, key=lambda row: int(row["time_bin"]))
                  for values in grouped.values()]
        x = np.asarray([int(row["time_from_onset_ms"]) for row in series[0]])
        y = np.mean([[float(row["accuracy_percent"]) for row in values]
                     for values in series], axis=0)
        ax_rest.plot(x, y, label=label, color=color, alpha=alpha, lw=2)
    for method, label, color in (
        ("Soft-CNN", "Soft CNN (Direction; mean of 5 seeds)", "#2166ac"),
        ("Soft-TRANSFORMER", "Soft Transformer (Direction; mean of 5 seeds)", "#d95f02"),
    ):
        grouped = {}
        for row in soft_rows:
            if row["method"] == method:
                grouped.setdefault(row["training_seed"], []).append(row)
        curves = [sorted(values, key=lambda row: int(row["time_bin"]))
                  for values in grouped.values()]
        x = np.asarray([int(row["time_from_onset_ms"]) for row in curves[0]])
        y = np.mean([[float(row["accuracy_percent"]) for row in curve]
                     for curve in curves], axis=0)
        ax_soft.plot(x, y, label=label, color=color, lw=1.8)
    ax_rest.set_title("CEBRA, conv-piVAE and Ridge")
    ax_soft.set_title("Soft CNN and Soft Transformer")
    for ax in axes_compare:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig_compare.supxlabel("Time from movement onset (ms)")
    fig_compare.supylabel("Direction accuracy (%)")
    fig_compare.suptitle("Direction decoding across time | Held-out trials")
    fig_compare.tight_layout(rect=(0.035, 0.02, 1, 0.92))
    outputs.extend(save_figure(fig_compare, "d", "ED8_D_Direction_Decoding_Rest_vs_Soft"))
    inputs.extend([csv_path, provenance_path])
    return {"cebra_local_status": "complete", "soft_status": "complete_from_reused_figure3h_embeddings",
            "soft_fit_source": "results/monkey_fig3h_soft_direction_v1, active train/test, direction labels, 4D, seeds 42-46",
            "soft_decoder_evaluation": str(soft_csv.relative_to(ROOT)),
            "soft_decoder_provenance": str(soft_provenance_path.relative_to(ROOT)),
            "soft_note": "This decoding evaluation reuses completed Figure 3h encoders; it is separate from the 136-cell ED8 campaign."}


def render_panel_e_reference(inputs: list[Path], outputs: list[Path]):
    with h5py.File(H5, "r") as h5:
        values = pickle.loads(bytes(h5["data/block0_values"][0]))
        index = [value.decode("utf-8") for value in h5["data/axis1"][:]]
        columns = [value.decode("utf-8") for value in h5["data/axis0"][:]]
    data = pd.DataFrame(values, index=index, columns=columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.4),
                             gridspec_kw={"width_ratios": [1.35, 1, 1]})
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.31, top=0.86, wspace=0.24)
    all_labels = ["Direction", "Position", "Active/passive condition",
                  "Position + direction", "Active/passive + position",
                  "Active/passive + direction", "Active/passive + position + direction"]
    all_keys = ["target", "pos", "ap", "pos_target", "ap_pos", "ap_target", "ap_pos_target"]
    all_colors = [plt.cm.tab10(i) for i in range(7)]
    for key, label, color in zip(all_keys, all_labels, all_colors):
        y = np.asarray(data.loc[key, "all_trial_loss"], dtype=float)
        axes[0].plot(savgol_filter(y, 9, 7), label=label, color=color, lw=1.25)
    for key, label, color in zip(("target_shuffle", "pos_shuffle", "ap_shuffle"),
                                 ("Direction shuffled", "Position shuffled", "Active/passive shuffled"),
                                 all_colors[:3]):
        y = np.asarray(data.loc[key, "all_trial_loss"], dtype=float)
        axes[0].plot(savgol_filter(y, 9, 7), label=label, color=color, alpha=0.48, lw=1.1)
    axes[0].set_title("InfoNCE | All trials")
    axes[0].set_ylabel("Loss")
    axes[0].set_xlabel("Training steps")
    axes[0].legend(frameon=False, fontsize=7.6, ncol=2, loc="upper center",
                   bbox_to_anchor=(0.5, -0.18))

    def active_panel(ax, group: str, loss_name: str):
        labels = ("Direction", "Position", "Position + direction")
        keys = ("target", "pos", "pos_target")
        colors_here = (plt.cm.tab10(0), plt.cm.tab10(1), plt.cm.tab10(3))
        for key, label, color in zip(keys, labels, colors_here):
            y = np.asarray(data.loc[key, group], dtype=float)
            ax.plot(savgol_filter(y, 9, 7), label=label, color=color, lw=1.25)
        shuffle_suffix = "shuffle" if loss_name == "InfoNCE" else "shuffled"
        for key, label, color in (
            (f"target_{shuffle_suffix}", "Direction shuffled", colors_here[0]),
            (f"pos_{shuffle_suffix}", "Position shuffled", colors_here[1]),
        ):
            y = np.asarray(data.loc[key, group], dtype=float)
            ax.plot(savgol_filter(y, 9, 7), label=label, color=color, alpha=0.48, lw=1.1)
        ax.set_title(f"{loss_name} | Active trials")
        ax.set_xlabel("Training steps")
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.18))

    active_panel(axes[1], "active_trial_loss", "InfoNCE")
    active_panel(axes[2], "active_mse_loss", "InfoMSE")
    for ax in axes:
        ax.set_xlim(0, 10000)
        ax.grid(True, alpha=0.16)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Extended Data Figure 8 | Published CEBRA loss traces", fontsize=15)
    fig.subplots_adjust(top=0.86, bottom=0.31)
    outputs.extend(save_figure(fig, "e", "ED8_E_Published_CEBRA_Loss_Reference"))
    inputs.append(H5)


def render_soft_losses(jobs: list[dict], inputs: list[Path], outputs: list[Path]):
    statuses = []
    labels = {
        ("direction",): "Direction",
        ("position",): "Position",
        ("ap",): "Active/passive condition",
        ("direction", "position"): "Position + direction",
        ("ap", "position"): "Active/passive + position",
        ("ap", "direction"): "Active/passive + direction",
        ("ap", "direction", "position"): "Active/passive + position + direction",
    }
    scopes = ("all_full", "active_full")
    for arch in ("cnn", "transformer"):
        for scope in scopes:
            selected = [j for j in jobs if j["panel"] == "e" and j["architecture"] == arch and j["scope"] == scope]
            loaded = []
            missing = []
            for job in selected:
                folder = soft_job_folder(job)
                manifest_path = folder / "manifest.json"
                history_path = folder / "training_history.csv"
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest.get("actual_updates") != 10000 or verify_artifacts(folder, manifest):
                        raise ValueError("manifest incomplete or hash mismatch")
                    if "training_history.csv" not in manifest.get("artifacts", {}):
                        raise ValueError("training history absent from manifest")
                    with history_path.open(newline="", encoding="utf-8-sig") as stream:
                        history = list(csv.DictReader(stream))
                    if len(history) != 10000:
                        raise ValueError("training history is not 10,000 rows")
                    loaded.append((job, history, manifest_path, history_path))
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    missing.append(str(manifest_path.relative_to(ROOT)))
            expected = len(selected)
            status = {"architecture": arch, "scope": scope, "complete_fits": len(loaded),
                      "planned_fits": expected, "status": "complete" if expected and len(loaded) == expected else "pending",
                      "missing_manifests_or_histories": missing}
            statuses.append(status)
            if not expected or len(loaded) != expected:
                continue
            fig, ax = plt.subplots(figsize=(9.5, 5.6))
            unique_conditions = sorted({tuple(sorted(j["fields"])) for j in selected})
            color_map = {condition: plt.cm.tab10(i % 10) for i, condition in enumerate(unique_conditions)}
            for job, history, manifest_path, history_path in loaded:
                condition = tuple(sorted(job["fields"]))
                label = labels.get(condition, " + ".join(condition))
                if job["shuffle"]:
                    label += " shuffled"
                x = np.asarray([int(row["update"]) for row in history])
                y = np.asarray([float(row["train_loss"]) for row in history])
                ax.plot(x, y, color=color_map[condition], lw=1.0,
                        alpha=0.48 if job["shuffle"] else 0.9,
                        linestyle="--" if job["shuffle"] else "-", label=label)
                inputs.extend([manifest_path, history_path])
            scope_label = "All trials" if scope == "all_full" else "Active trials"
            ax.set_xlabel("Training updates")
            ax.set_ylabel("Soft training loss")
            ax.set_title(f"Soft {arch.upper()} (Stable loss) | {scope_label} | 32D")
            ax.set_xlim(0, 10000)
            ax.grid(True, alpha=0.16)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(frameon=False, fontsize=8, ncol=2)
            fig.tight_layout()
            stem = f"ED8_E_Soft_{arch.upper()}_{scope}_32D"
            outputs.extend(save_figure(fig, "e", stem))
    return statuses


def render_panel_f_cebra(inputs: list[Path], outputs: list[Path]):
    csv_path = DELIVERY / "ED8_AP_DECODING_SEEDS.csv"
    provenance_path = DELIVERY / "ED8_AP_DECODING_PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 8.5), sharex=True)
    for ax, task, title, ylabel in (
        (axes[0], "ap_pos", "Position decoding | Fit labels: active/passive + position", "Position R² (%)"),
        (axes[1], "ap_direction", "Direction decoding | Fit labels: active/passive + direction", "Accuracy (%)"),
    ):
        for i, dim in enumerate(DIMENSIONS):
            subset = [row for row in rows if row["task"] == task and int(row["dimension"]) == dim
                      and row["split"] == "test"]
            if sorted(int(row["seed"]) for row in subset) != list(SEEDS):
                raise ValueError(f"Expected five test scores for {task}, dimension {dim}")
            vals = np.asarray([float(row["score_percent"]) for row in subset])
            ax.scatter(np.full(5, i), vals, facecolors="none", edgecolors="black", s=34, zorder=3)
            ax.hlines(np.median(vals), i - 0.22, i + 0.22, colors="#e67e22", lw=2.2, zorder=4)
        if task == "ap_direction":
            ax.axhline(100 / 16, color="gray", ls="--", lw=1.2, label="Chance (1/16)")
            ax.legend(frameon=False)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(5), [str(dim) for dim in DIMENSIONS])
        ax.grid(True, axis="y", alpha=0.17)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].set_xlabel("Embedding dimension")
    fig.suptitle("CEBRA held-out decoding | Five seeds (42–46)", fontsize=13)
    fig.tight_layout()
    outputs.extend(save_figure(fig, "f", "ED8_F_CEBRA_Heldout_Decoding"))
    inputs.extend([csv_path, provenance_path])


def render_soft_decoding(jobs: list[dict], inputs: list[Path], outputs: list[Path]):
    statuses = []
    for arch in ("cnn", "transformer"):
        selected = [j for j in jobs if j["panel"] == "f" and j["architecture"] == arch]
        loaded = []
        missing = []
        for job in selected:
            folder = soft_job_folder(job)
            manifest_path, score_path = folder / "manifest.json", folder / "score.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("actual_updates") != 10000 or verify_artifacts(folder, manifest):
                    raise ValueError("manifest incomplete or hash mismatch")
                if "score.json" not in manifest.get("artifacts", {}):
                    raise ValueError("test score missing from manifest")
                score = json.loads(score_path.read_text(encoding="utf-8"))
                loaded.append((job, score, manifest_path, score_path))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                missing.append(str(manifest_path.relative_to(ROOT)))
        status = {"architecture": arch, "complete_fits": len(loaded), "planned_fits": len(selected),
                  "status": "complete" if selected and len(loaded) == len(selected) else "pending",
                  "missing_manifests_or_test_scores": missing}
        statuses.append(status)
        if not selected or len(loaded) != len(selected):
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
        for ax, fields, title, metric in (
            (axes[0], {"ap", "position"}, "Position decoding | Fit labels: active/passive + position", "R2"),
            (axes[1], {"ap", "direction"}, "Direction decoding | Fit labels: active/passive + direction", "accuracy"),
        ):
            for i, dim in enumerate(DIMENSIONS):
                subset = [(job, score) for job, score, _, _ in loaded
                          if set(job["fields"]) == fields and int(job["dimension"]) == dim]
                if sorted(int(job["seed"]) for job, _ in subset) != list(SEEDS):
                    raise ValueError(f"Expected five Soft test scores for {fields}, dimension {dim}")
                vals = np.asarray([float(score["score_percent"]) for _, score in subset])
                ax.scatter(np.full(5, i), vals, facecolors="none", edgecolors="black", s=34, zorder=3)
                ax.hlines(np.median(vals), i - 0.22, i + 0.22, colors="#e67e22", lw=2.2, zorder=4)
            if metric == "accuracy":
                ax.axhline(100 / 16, color="gray", ls="--", lw=1.2, label="Chance (1/16)")
                ax.legend(frameon=False)
                ax.set_ylabel("Accuracy (%)")
            else:
                ax.set_ylabel("Position R² (%)")
            ax.set_title(title, fontsize=9.3)
            ax.set_xticks(range(5), [str(dim) for dim in DIMENSIONS])
            ax.set_xlabel("Embedding dimension")
            ax.grid(True, axis="y", alpha=0.17)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.suptitle(f"Soft {arch.upper()} (Stable loss) | Held-out decoding | Five seeds", fontsize=12)
        fig.tight_layout()
        stem = f"ED8_F_Soft_{arch.upper()}_Heldout_Decoding"
        outputs.extend(save_figure(fig, "f", stem))
        for _, _, manifest_path, score_path in loaded:
            inputs.extend([manifest_path, score_path])
    return statuses


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for panel in PANEL_NAMES:
        (OUT / panel).mkdir(parents=True, exist_ok=True)

    inputs: list[Path] = []
    outputs: list[Path] = []
    soft_plan, plan_path = soft_jobs()
    inputs.extend([plan_path, H5])
    soft_rows, soft_counts = campaign_status(soft_plan)

    render_cebra_embeddings(inputs, outputs)
    soft_embedding_outputs = render_soft_embedding_jobs(soft_plan, inputs, outputs)
    panel_d_status = render_panel_d(inputs, outputs)
    render_panel_e_reference(inputs, outputs)
    soft_loss_status = render_soft_losses(soft_plan, inputs, outputs)
    render_panel_f_cebra(inputs, outputs)
    soft_decoding_status = render_soft_decoding(soft_plan, inputs, outputs)

    statuses = {}
    for panel in "abc":
        selected = [row for row in soft_rows if row["panel"] == panel]
        complete = sum(row["fit_complete_and_hash_valid"] for row in selected)
        expected = len(selected)
        rendered = sum(row["panel"] == panel for row in soft_embedding_outputs)
        statuses[panel] = {
            "cebra_status": "complete",
            "soft_status": "complete" if expected and complete == expected and rendered == expected else "partial",
            "soft_fits_complete": complete, "soft_fits_planned": expected,
            "soft_embeddings_rendered": rendered,
            "missing_fit_manifests": [row["fit_path"] + "/manifest.json" for row in selected
                                       if not row["fit_complete_and_hash_valid"]],
        }
    statuses["d"] = panel_d_status
    statuses["e"] = {"cebra_official_reference_status": "complete_from_local_official_hdf5",
                      "soft_status": "complete" if soft_loss_status and all(s["status"] == "complete" for s in soft_loss_status) else "partial",
                      "soft_fit_groups": soft_loss_status}
    statuses["f"] = {"cebra_status": "complete_from_existing_test_score_csv",
                      "soft_status": "complete" if soft_decoding_status and all(s["status"] == "complete" for s in soft_decoding_status) else "partial",
                      "soft_architectures": soft_decoding_status}
    statuses["soft_campaign"] = {"planned_cells": len(soft_plan),
                                 "complete_hash_verified_cells": sum(row["fit_complete_and_hash_valid"] for row in soft_rows),
                                 "panel_architecture_counts": {f"{p}_{a}": n for (p, a), n in soft_counts.items()},
                                 "plan": str(plan_path.relative_to(ROOT)),
                                 "note": "A panel is marked complete only when its expected manifests and dependent artifacts exist and pass SHA-256 checks."}
    write_json(OUT / "PANEL_STATUS.json", statuses)
    write_json(OUT / "SOFT_FIT_AUDIT.json", {"planned_jobs": len(soft_plan), "jobs": soft_rows})

    input_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(set(inputs)) if path.is_file()}
    output_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(set(outputs)) if path.is_file()}
    provenance = {
        "figure": "Extended Data Figure 8",
        "rendered_on": "2026-10-29",
        "renderer": str(Path(__file__).relative_to(ROOT)),
        "renderer_sha256": sha256(Path(__file__)),
        "training_started_or_modified": False,
        "active_ed8_output_root_read_only": str(SOFT.relative_to(ROOT)),
        "previous_figure_folder_preserved": "results/figura_ext_8_2026-10-28",
        "soft_panel_e_dimension": "32D, experimental choice; not established by the official plotting code",
        "soft_fit_temperature": {"embedding": 1.0, "metadata": 0.5},
        "soft_fits": {"planned": len(soft_plan), "hash_verified_complete": sum(row["fit_complete_and_hash_valid"] for row in soft_rows)},
        "inputs_sha256": input_hashes,
        "outputs_sha256": output_hashes,
        "panel_status_file": "PANEL_STATUS.json",
        "interpretation": [
            "CEBRA panels a-c reuse local full-recording 2D InfoMSE embeddings; Soft panels use completed 2D Stable-loss full-recording fits.",
            "Panel d uses existing held-out local direction-decoding curves; no compatible Soft time-course artifact is in the 136-cell campaign.",
            "Panel e plots the published CEBRA curves from the local official HDF5; Soft loss plots are emitted only after every fit in a scope/architecture group has a verified manifest and 10,000-step history.",
            "Panel f reuses the existing local CEBRA test-score CSV; Soft decoding plots are emitted only after every fit for an architecture has a verified test score.",
            "Soft comparison panels are experimental Stable-loss results and are not represented as InfoMSE or as exact CEBRA replications.",
            "Older results/user_soft_monkey_v1 32D loss curves were excluded: their embedding temperature is 0.1, whereas this ED8 Soft campaign uses 1.0, and their label/scope grid differs."
        ],
        "excluded_older_soft_artifacts": {
            "path": "results/user_soft_monkey_v1",
            "reason": "T_embed=0.1 and label/scope grid differs from the 136-cell ED8 campaign (T_embed=1.0); not pooled into this figure."
        }
    }
    write_json(OUT / "PROVENANCE.json", provenance)
    readme = """# Extended Data Figure 8 — local rebuild (2026-10-29)

Each panel is in its own `a`–`f` folder. Figure titles, axes, and legends are in English.

This rebuild reuses saved local CEBRA embeddings/scores and the official local `EDFigure8.h5` loss traces. It does not launch training and reads the active Soft campaign without changing its files. Soft figures are written only after the corresponding fit manifests and output hashes are present. Run `python -m cebra_replica.monkey.render_ed8_20261029` again after the Soft campaign finishes to incorporate newly completed artifacts.

Panel e uses 32D for Soft solely as an experimental choice; the plotting code does not specify the original fit dimension. The Soft objective is Stable loss (`T_embed=1`, `T_metadata=0.5`), not the CEBRA InfoNCE/InfoMSE objective.

See `PANEL_STATUS.json` for panel-by-panel readiness, `SOFT_FIT_AUDIT.json` for all 136 planned cells, and `PROVENANCE.json` for SHA-256 input/output hashes.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output": str(OUT), "plots_written": len(output_hashes),
                      "soft_complete": statuses["soft_campaign"]["complete_hash_verified_cells"],
                      "soft_planned": len(soft_plan), "panels": statuses}, indent=2))


if __name__ == "__main__":
    main()
