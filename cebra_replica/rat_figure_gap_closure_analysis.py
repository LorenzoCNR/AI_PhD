"""Build missing paper-style rat figures from hash-verified frozen embeddings.

This downstream-only companion writes exclusively beneath
results/rat_figure_gap_closure_v1. It never fits an encoder. The only decoder
fits are the same train-only kNN accessibility probes used in the existing
rat_manifold_scoring.py workflow.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "rat_figure_gap_closure_v1"
DATA_ROOT = ROOT / "data" / "rat_hippocampus"
BASELINE_ROOT = ROOT / "results" / "rat_manifold_baselines_v1"
CEBRA_ROOT = ROOT / "results" / "rat_trial_benchmark_v1"
PIVAE_ROOT = ROOT / "results" / "rat_conv_pivae_v1" / "runs"
RATS = ("achilles", "buddy", "cicero", "gatsby")
FOLDS = (0, 1, 2)
SEEDS = tuple(range(42, 52))
K = 36


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def index_manifests() -> dict[tuple, Path]:
    index = {}
    for path in BASELINE_ROOT.rglob("result.json"):
        m = load_json(path)
        if m.get("status") not in ("complete", "complete_embeddings_only"):
            continue
        seed = "deterministic" if m.get("seed") is None else str(m["seed"])
        key = ("baseline", m["method"].lower(), m["rat"], int(m["fold"]), seed)
        index[key] = path
    for path in CEBRA_ROOT.rglob("result.json"):
        m = load_json(path)
        condition = m.get("condition")
        if condition not in ("time", "position_direction"):
            continue
        display = "CEBRA-Time-32D" if condition == "time" else "CEBRA-Behavior-32D"
        key = ("cebra", display, m["rat"], int(m["fold"]), str(m["seed"]))
        index[key] = path
    for path in PIVAE_ROOT.rglob("run.json"):
        m = load_json(path)
        if m.get("status") != "complete":
            continue
        c = m["config"]
        key = ("pivae", "Conv-piVAE-z_mean", c["rat"], int(c["fold"]), str(c["seed"]))
        # A retry may coexist with an interrupted attempt; prefer any verified complete copy.
        if key not in index or m.get("completed_utc", "") > load_json(index[key])["completed_utc"]:
            index[key] = path
    return index


def verify_artifact(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"Artifact hash mismatch: {path}; {actual} != {expected}")


def support_for(rat: str, fold: int) -> dict[str, dict[str, np.ndarray]]:
    # Reuse the exact splitter/support implementation used by the existing scorer.
    from rat_manifold_scoring import support_for as _support_for
    return _support_for(rat, fold)


def load_fit(index: dict[tuple, Path], kind: str, method: str, rat: str,
             fold: int, seed: str, split: str, variant: str = "raw") -> dict:
    key = (kind, method, rat, fold, str(seed))
    manifest_path = index.get(key)
    if manifest_path is None:
        raise KeyError(f"Missing fit: {key}")
    manifest = load_json(manifest_path)
    path = (manifest_path.parent / f"embedding_{split}.npz" if kind == "pivae"
            else manifest_path.parent / f"{split}_embedding.npz")
    if kind == "baseline":
        entry = manifest["embeddings"][split]
        verify_artifact(path, entry["sha256"])
        embed_key = "embedding_raw"
    elif kind == "cebra":
        entry = manifest["embeddings"][split]
        verify_artifact(path, entry["embedding_sha256"])
        embed_key = "embedding"
    else:
        embed_key = "z_mean_raw" if variant == "raw" else "post_mean_behavior_informed_raw"
        entry_key = f"embedding_{split}_sha256"
        verify_artifact(path, manifest["artifact_hashes"][entry_key])
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    if embed_key not in arrays:
        raise RuntimeError(f"Embedding field {embed_key} absent: {path}")
    embedding = arrays[embed_key]
    source_id = arrays["source_id"].astype(np.int64)
    if embedding.ndim != 2 or len(embedding) != len(source_id):
        raise RuntimeError(f"Invalid embedding shape/content: {path}, {embedding.shape}")
    support = support_for(rat, fold)[split]
    support_ids = support["source_id"].astype(np.int64)
    if set(source_id.tolist()) != set(support_ids.tolist()):
        raise RuntimeError(f"Source-row support mismatch: {key}/{split}")
    by_id = {int(row): i for i, row in enumerate(source_id)}
    order = np.asarray([by_id[int(row)] for row in support_ids], dtype=np.int64)
    embedding = embedding[order]
    source_id = source_id[order]
    if kind == "baseline":
        behavior = arrays["behavior"][order]
        valid = arrays["valid_mask"][order].astype(bool)
        if not np.array_equal(valid, support["valid_mask"]):
            raise RuntimeError(f"Baseline validity mask disagrees with frozen support: {path}")
    elif kind == "pivae":
        behavior = arrays["u"][order]
        valid = support["valid_mask"].astype(bool)
    else:
        behavior = arrays["behavior"][order]
        valid = support["valid_mask"].astype(bool)
    if not np.isfinite(embedding[valid]).all():
        raise RuntimeError(f"Non-finite values on valid embedding rows: {path}")
    if kind == "baseline" and not np.isnan(embedding[~valid]).all():
        raise RuntimeError(f"Expected NaN only on masked baseline boundary rows: {path}")
    if not np.array_equal(behavior, support["behavior"]):
        raise RuntimeError(f"Behavior does not match frozen row IDs: {path}")
    return {"embedding": embedding, "behavior": behavior, "source_id": source_id,
            "valid_mask": valid, "manifest": manifest, "manifest_path": manifest_path,
            "embedding_path": path, "embedding_sha256": sha256(path)}


def method_specs(index: dict[tuple, Path]) -> list[dict]:
    specs = [
        {"kind": "cebra", "method": "CEBRA-Behavior-32D", "display": "CEBRA-Behavior", "seed": "42"},
        {"kind": "cebra", "method": "CEBRA-Time-32D", "display": "CEBRA-Time", "seed": "42"},
        {"kind": "pivae", "method": "Conv-piVAE-z_mean", "variant": "raw",
         "display": "Conv-piVAE (no test labels)", "seed": "42"},
        {"kind": "pivae", "method": "Conv-piVAE-z_mean", "variant": "posterior",
         "display": "Conv-piVAE (test labels)", "seed": "42"},
        {"kind": "baseline", "method": "pca", "display": "PCA", "seed": "deterministic"},
        {"kind": "baseline", "method": "umap", "display": "UMAP", "seed": "42"},
        {"kind": "baseline", "method": "opentsne", "display": "t-SNE", "seed": "42"},
    ]
    for spec in specs:
        spec["key"] = spec["kind"], spec["method"]
        spec["available"] = all((spec["kind"], spec["method"], rat, 0, spec["seed"]) in index for rat in RATS)
    return specs


def plot_embedding_panels(index: dict[tuple, Path]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = [s for s in method_specs(index) if s["available"]]
    fig = plt.figure(figsize=(3.7 * len(specs), 3.15 * len(RATS)), dpi=220)
    index_rows = []
    parents = []
    for row, rat in enumerate(RATS):
        for col, spec in enumerate(specs):
            variant = spec.get("variant", "raw")
            data = load_fit(index, spec["kind"], spec["method"], rat, 0,
                            spec["seed"], "test", variant)
            mask = data["valid_mask"]
            emb = data["embedding"][mask]
            beh = data["behavior"][mask]
            if spec["kind"] != "cebra":
                if emb.shape[1] > 2:
                    emb = emb[:, :2]
                ax = fig.add_subplot(len(RATS), len(specs), row * len(specs) + col + 1)
                right = beh[:, 1] > 0.5
                left = beh[:, 2] > 0.5
                ax.scatter(emb[right, 1], emb[right, 0], c=beh[right, 0], cmap="viridis", s=0.5,
                           linewidths=0, alpha=1, rasterized=True)
                ax.scatter(emb[left, 1], emb[left, 0], c=beh[left, 0], cmap="cool", s=0.5,
                           linewidths=0, alpha=1, rasterized=True)
                ax.set_aspect("equal", adjustable="datalim")
            else:
                emb = emb[:, :3]
                ax = fig.add_subplot(len(RATS), len(specs), row * len(specs) + col + 1,
                                     projection="3d")
                right = beh[:, 1] > 0.5
                left = beh[:, 2] > 0.5
                ax.scatter(*emb[left].T, c=beh[left, 0], cmap="cool", s=0.5,
                           linewidths=0, alpha=1, rasterized=True)
                ax.scatter(*emb[right].T, c=beh[right, 0], cmap="viridis", s=0.5,
                           linewidths=0, alpha=1, rasterized=True)
            if row == 0:
                dim = data["embedding"].shape[1]
                shown = 3 if spec["kind"] == "cebra" else min(2, dim)
                ax.set_title(f"{spec['display']}\n{shown} plotted / {dim}D fit", fontsize=10)
            if col == 0:
                ax.text2D(-0.16, 0.5, rat.capitalize(), transform=ax.transAxes,
                          va="center", ha="right", fontsize=10) if spec["kind"] == "cebra" else ax.set_ylabel(rat.capitalize())
            ax.set_axis_off()
            index_rows.append({"rat": rat, "method": spec["display"], "seed": spec["seed"],
                               "fold": 0, "split": "test", "rows": int(mask.sum()),
                               "fit_dimension": int(data["embedding"].shape[1]),
                               "plotted_dimension": 3 if spec["kind"] == "cebra" else min(2, int(data["embedding"].shape[1])),
                               "parent_manifest": data["manifest_path"].relative_to(ROOT).as_posix(),
                               "parent_manifest_sha256": sha256(data["manifest_path"]),
                               "embedding_sha256": data["embedding_sha256"],
                               "notes": ("same behavior-trained fit; q(z|x) inference without test-time labels" if variant == "raw"
                                         else "same behavior-trained fit; q(z|x,u) is the explicit test-label condition; not used for decoding") if spec["kind"] == "pivae"
                                         else "held-out test rows; position-colored; first three CEBRA coordinates or 2D baseline"})
            parents.append({"manifest": data["manifest_path"].relative_to(ROOT).as_posix(),
                            "sha256": sha256(data["manifest_path"])})
    fig.suptitle("Rat embedding comparison (held-out test; paper-style dimensionality)", y=0.995)
    fig.text(0.5, 0.008, "Points colored by position; CEBRA uses first 3 coordinates, other methods 2D. CEBRA panel directions use separate color maps.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0.02, 0.025, 1, 0.97))
    out = OUT / "figure1d"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"FIGURE1D_RAT_METHOD_EMBEDDINGS.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    atomic_csv(out / "FIGURE1D_PANEL_INDEX.csv", index_rows, list(index_rows[0]))
    atomic_json(out / "PROVENANCE.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)), "parents": parents,
        "source": "frozen CEBRA, PCA, UMAP, openTSNE, and Conv-piVAE embeddings",
        "split": "fold 0 test; common valid row support verified by source_id",
        "figure_interpretation": "qualitative panel analogue; CEBRA first 3 dimensions, other methods 2D; exact training settings differ from the paper",
        "not_included": ["AutoLFADS (user previously deferred)"],
        "pivae_label_conditions": "q(z|x) without test-time labels and q(z|x,u) with test-time labels are two frozen inference outputs from the same behavior-trained conv-piVAE fit; both are included as separate conditions for Figure 1d/1e consistency, but q(z|x,u) is excluded from decoding to avoid using test targets as input"})


def plot_embedding_per_rat(index: dict[tuple, Path]) -> None:
    """Full-width Figure 1d layout, one rat per file, from frozen embeddings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = [s for s in method_specs(index) if s["available"]]
    out = OUT / "figure1d"
    out.mkdir(parents=True, exist_ok=True)
    for rat in RATS:
        fig = plt.figure(figsize=(30, 5), dpi=300)
        for i, spec in enumerate(specs):
            data = load_fit(index, spec["kind"], spec["method"], rat, 0,
                            spec["seed"], "test", spec.get("variant", "raw"))
            mask = data["valid_mask"]
            emb = data["embedding"][mask]
            beh = data["behavior"][mask]
            right = beh[:, 1] == 1
            left = beh[:, 2] == 1
            if spec["kind"] == "cebra":
                ax = fig.add_subplot(1, len(specs), i + 1, projection="3d")
                ax.scatter(*emb[left, :3].T, c=beh[left, 0], cmap="cool", s=5,
                           linewidths=0, rasterized=True)
                ax.scatter(*emb[right, :3].T, c=beh[right, 0], cmap="viridis", s=5,
                           linewidths=0, rasterized=True)
            else:
                ax = fig.add_subplot(1, len(specs), i + 1)
                ax.scatter(emb[right, 1], emb[right, 0], c=beh[right, 0], cmap="viridis",
                           s=5, linewidths=0, rasterized=True)
                ax.scatter(emb[left, 1], emb[left, 0], c=beh[left, 0], cmap="cool",
                           s=5, linewidths=0, rasterized=True)
            ax.axis("off")
            title = {"Conv-piVAE (no test labels)": "conv-piVAE\nq(z|x)",
                     "Conv-piVAE (test labels)": "conv-piVAE\nq(z|x,u)"}.get(
                         spec["display"], spec["display"])
            ax.set_title(title, fontsize=12)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"FIGURE1D_{rat.upper()}_METHOD_EMBEDDINGS.{ext}",
                        dpi=300 if ext == "png" else None, bbox_inches="tight",
                        transparent=False, facecolor="white")
        plt.close(fig)


def plot_figure1c_data_overview() -> None:
    """Render a raw-data context panel from the official local rat recordings."""
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out = OUT / "figure1c"
    out.mkdir(parents=True, exist_ok=True)
    n_samples = 1000
    sample_period_s = 0.025
    raw_by_rat = {}
    parent_hashes = {}
    source_lengths = {}
    for rat in RATS:
        source = DATA_ROOT / f"{rat}.jl"
        parent_hashes[rat] = sha256(source)
        record = joblib.load(source)
        spikes = np.asarray(record["spikes"])
        behavior = np.asarray(record["position"])
        if spikes.ndim != 2 or behavior.ndim != 2 or behavior.shape[1] != 3:
            raise RuntimeError(f"Unexpected raw data shape for {rat}: {spikes.shape}, {behavior.shape}")
        if len(spikes) != len(behavior) or len(spikes) < n_samples:
            raise RuntimeError(f"Raw neural/behavior support mismatch for {rat}")
        direction = behavior[:n_samples, 1:3]
        if not np.allclose(direction.sum(axis=1), 1.0):
            raise RuntimeError(f"Expected one-hot right/left direction labels for {rat}")
        raw_by_rat[rat] = (spikes[:n_samples], behavior[:n_samples])
        source_lengths[rat] = int(len(spikes))

    vmax = max(1.0, float(np.percentile(np.concatenate([x.ravel() for x, _ in raw_by_rat.values()]), 99.5)))
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), dpi=220,
                             gridspec_kw={"width_ratios": [1.55, 1.0]})
    image = None
    summary_rows = []
    for row, rat in enumerate(RATS):
        spikes, behavior = raw_by_rat[rat]
        time_s = np.arange(n_samples) * sample_period_s
        activity_ax, position_ax = axes[row]
        image = activity_ax.imshow(
            spikes.T, aspect="auto", origin="lower", cmap="gray_r", vmin=0, vmax=vmax,
            extent=(0, n_samples * sample_period_s, 0, spikes.shape[1]), interpolation="nearest")
        activity_ax.set_ylabel(f"{rat.capitalize()}\nunit # ({spikes.shape[1]})")
        activity_ax.spines["top"].set_visible(False)
        activity_ax.spines["right"].set_visible(False)
        right = behavior[:, 1] == 1
        left = behavior[:, 2] == 1
        position_ax.plot(time_s, behavior[:, 0], color="#b8b8b8", linewidth=0.35, zorder=1)
        position_ax.scatter(time_s[right], behavior[right, 0], s=2.0, color="#4169b2", label="Right", zorder=2)
        position_ax.scatter(time_s[left], behavior[left, 0], s=2.0, color="#cf6b35", label="Left", zorder=2)
        position_ax.set_ylim(0, 1.6)
        position_ax.set_ylabel("Position (m)")
        position_ax.spines["top"].set_visible(False)
        position_ax.spines["right"].set_visible(False)
        if row == 0:
            activity_ax.set_title("Binned neural activity")
            position_ax.set_title("Position, colored by direction")
        if row == len(RATS) - 1:
            activity_ax.set_xlabel("Time (s)")
            position_ax.set_xlabel("Time (s)")
        else:
            activity_ax.set_xticklabels([])
            position_ax.set_xticklabels([])
        summary_rows.append({"rat": rat, "neurons": int(spikes.shape[1]), "source_time_bins": source_lengths[rat],
                             "displayed_time_bins": n_samples, "displayed_duration_s": n_samples * sample_period_s,
                             "spike_value_min": float(spikes.min()), "spike_value_max": float(spikes.max()),
                             "right_label_bins": int(right.sum()), "left_label_bins": int(left.sum()),
                             "source_file_sha256": parent_hashes[rat]})
    fig.subplots_adjust(left=0.10, right=0.91, top=0.93, bottom=0.09, hspace=0.35, wspace=0.32)
    cax = fig.add_axes([0.93, 0.22, 0.015, 0.58])
    fig.colorbar(image, cax=cax, label="Spike counts per 25-ms bin")
    fig.legend(handles=[Line2D([0], [0], marker="o", linestyle="", color="#4169b2", label="Right"),
                        Line2D([0], [0], marker="o", linestyle="", color="#cf6b35", label="Left")],
               loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=2, frameon=False)
    fig.suptitle("Rat hippocampus data overview (Figure 1c-style context)", y=1.01)
    fig.text(0.5, 0.015, "First 1,000 samples (25 s) from each source recording; raw descriptive data, not learned embeddings or held-out performance.",
             ha="center", fontsize=9)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"FIGURE1C_RAT_DATA_OVERVIEW.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    atomic_csv(out / "FIGURE1C_DATA_SUMMARY.csv", summary_rows, list(summary_rows[0]))
    atomic_json(out / "PROVENANCE.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)),
        "parent_source_hashes": parent_hashes,
        "input_semantics": "Official rat-hippocampus data: spike counts in 25-ms bins; behavior columns are position, right, left",
        "display": {"samples_per_rat": n_samples, "duration_seconds": n_samples * sample_period_s,
                    "rats": list(RATS), "neural_transform": "none", "position_transform": "none"},
        "scope": "descriptive Figure 1c-style data-context analogue; no fitting, embedding, metric, or model-selection step",
        "limitations": ["illustrative 25-second excerpt, not a full-recording visualization",
                        "data-context panel only; not a bitwise recreation of paper artwork"]})


def add_pivae_scores(index: dict[tuple, Path]) -> list[dict]:
    from sklearn.metrics import balanced_accuracy_score, r2_score
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    rows = []
    seeds = sorted({int(k[4]) for k in index if k[0] == "pivae"})
    for rat in RATS:
        for fold in FOLDS:
            for seed in seeds:
                key = ("pivae", "Conv-piVAE-z_mean", rat, fold, str(seed))
                if key not in index:
                    continue
                train = load_fit(index, "pivae", "Conv-piVAE-z_mean", rat, fold, str(seed), "train")
                position = KNeighborsRegressor(n_neighbors=K, metric="cosine")
                direction = KNeighborsClassifier(n_neighbors=K, metric="cosine")
                tr = train["valid_mask"]
                position.fit(train["embedding"][tr], train["behavior"][tr, 0])
                direction.fit(train["embedding"][tr], train["behavior"][tr, 1].astype(np.int64))
                for split in ("valid", "test"):
                    sample = load_fit(index, "pivae", "Conv-piVAE-z_mean", rat, fold, str(seed), split)
                    m = sample["valid_mask"]
                    x, y = sample["embedding"][m], sample["behavior"][m]
                    pred_pos = position.predict(x)
                    pred_dir = direction.predict(x)
                    rows.append({"method": "Conv-piVAE-z_mean", "condition": "behavior-trained; q(z|x) inference",
                                 "embedding_dim": int(sample["embedding"].shape[1]), "embedding_variant": "z_mean_raw",
                                 "rat": rat, "fold": fold, "seed": seed, "split": split,
                                 "n_samples_scored": int(m.sum()), "k_neighbors": K, "distance": "cosine",
                                 "position_median_absolute_error_cm": float(np.median(np.abs(pred_pos-y[:,0]))*100),
                                 "position_r2": float(r2_score(y[:,0], pred_pos)),
                                 "direction_balanced_accuracy": float(balanced_accuracy_score(y[:,1].astype(np.int64), pred_dir)),
                                 "parent_result_path": index[key].relative_to(ROOT).as_posix(),
                                 "parent_result_sha256": sha256(index[key]),
                                 "parent_embedding_sha256": sha256(sample["embedding_path"]),
                                 "support_policy": "same frozen 250 ms test/valid centers; no test labels passed to embedding inference"})
    return rows


def aggregate_scores(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["rat"], str(row["seed"]), row["split"], row["metric"])].append(float(row["value"]))
    result = []
    for key, values in groups.items():
        result.append({"method": key[0], "rat": key[1], "seed": key[2], "split": key[3], "metric": key[4],
                       "n_folds": len(values), "mean_across_folds": mean(values),
                       "sd_across_folds": stdev(values) if len(values) > 1 else ""})
    return result


def summarize_score_long(long_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    run_rows = []
    for row in long_rows:
        for metric in ("position_median_absolute_error_cm", "position_r2", "direction_balanced_accuracy"):
            run_rows.append({"method": row["method"], "rat": row["rat"], "seed": str(row["seed"]),
                             "fold": int(row["fold"]), "split": row["split"], "metric": metric,
                             "value": float(row[metric])})
    result = aggregate_scores(run_rows)
    return run_rows, result


def plot_metric_grid(seed_rows: list[dict], output_dir: Path, filename: str,
                     metrics: tuple[str, ...], title: str,
                     method_subset: tuple[str, ...] | None = None,
                     footnote: str | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_methods = ("CEBRA-Behavior-32D", "Conv-piVAE-z_mean", "Conv-piVAE-test-labels",
                   "CEBRA-Time-32D", "PCA", "UMAP", "OPENTSNE")
    methods = method_subset or all_methods
    names = {"CEBRA-Behavior-32D": "CEBRA-Behavior", "Conv-piVAE-z_mean": "Conv-piVAE no test labels",
             "Conv-piVAE-test-labels": "Conv-piVAE with test labels",
             "CEBRA-Time-32D": "CEBRA-Time", "PCA": "PCA", "UMAP": "UMAP", "OPENTSNE": "t-SNE"}
    labels = {"position_median_absolute_error_cm": "Median absolute position error (cm)",
              "position_r2": "Position $R^2$", "direction_balanced_accuracy": "Direction balanced accuracy",
              "held_out_cross_rat_consistency_r2": "Held-out cross-rat consistency $R^2$",
              "held_out_across_run_consistency_r2": "Held-out across-run consistency $R^2$"}
    fig, axes = plt.subplots(len(RATS), len(metrics), figsize=(4.1 * len(metrics), 2.7 * len(RATS)), squeeze=False, dpi=220)
    colors = {"CEBRA-Behavior-32D": "#7353ba", "Conv-piVAE-z_mean": "#e07a23",
              "Conv-piVAE-test-labels": "#d04f3c",
              "CEBRA-Time-32D": "#248f8f", "PCA": "#777777", "UMAP": "#3478bf", "OPENTSNE": "#bd4f77"}
    for ri, rat in enumerate(RATS):
        for ci, metric in enumerate(metrics):
            ax = axes[ri, ci]
            for yi, method in enumerate(methods):
                value_key = "mean_across_folds" if seed_rows and "mean_across_folds" in seed_rows[0] else "value"
                vals = [float(r[value_key]) for r in seed_rows
                        if r["rat"] == rat and r.get("split", "test") == "test"
                        and r["metric"] == metric and r["method"] == method]
                if not vals:
                    continue
                ax.scatter(vals, np.full(len(vals), yi), s=13, color=colors[method], alpha=0.74,
                           edgecolors="white", linewidths=0.25, zorder=3)
                med = median(vals)
                ax.plot([med, med], [yi-0.22, yi+0.22], color="#e88d25", linewidth=2.5, zorder=4)
            ax.set_yticks(range(len(methods)), [names[m] for m in methods] if ci == 0 else [])
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
    fig.suptitle(title, y=0.995)
    default_footnote = "Dots are individual training-seed means across available folds; orange tick is the median. Test-only; no model selection."
    fig.text(0.5, 0.003, footnote or default_footnote, ha="center", fontsize=9)
    fig.tight_layout(rect=(0.01, 0.025, 1, 0.97))
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"{filename}.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)


def draw_trajectory(index: dict[tuple, Path]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KNeighborsRegressor

    out = OUT / "figure2d"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=220, sharey=False)
    rows = []
    parents = []
    for ax, rat in zip(axes.flat, RATS):
        traces = {}
        for cond, display in (("position_direction", "CEBRA-Behavior"),
                              ("position_direction_shuffled", "CEBRA-Shuffle")):
            manifest_path = ROOT / "results" / "rat_figure2c_3d_v1" / rat / cond / "seed_42" / "result.json"
            manifest = load_json(manifest_path)
            emb_path = manifest_path.parent / "all_embedding.npz"
            verify_artifact(emb_path, manifest["artifacts"]["all_embedding.npz"])
            with np.load(emb_path, allow_pickle=False) as z:
                data = {k: z[k] for k in z.files}
            split = manifest["split"]
            train_ids = np.arange(*split["train"], dtype=np.int64)
            test_ids = np.arange(*split["test"], dtype=np.int64)
            valid = data["valid_mask"].astype(bool)
            reg = KNeighborsRegressor(n_neighbors=K, metric="cosine")
            reg.fit(data["embedding"][train_ids][valid[train_ids]], data["behavior"][train_ids][valid[train_ids], 0])
            pred = reg.predict(data["embedding"][test_ids])
            test_valid = valid[test_ids]
            # Official Figure 2d uses test rows 320:1020, 25-ms bins, and
            # a 17.5-second display; this may cross rat-trial boundaries.
            idx = np.arange(320, 1020, dtype=np.int64)
            if len(test_ids) < 1020 or not np.all(test_valid[idx]):
                raise RuntimeError(f"Official Figure 2d test support unavailable for {rat}/{cond}")
            traces[display] = (0.025 * np.arange(len(idx)),
                               data["behavior"][test_ids][idx, 0] * 100,
                               pred[idx] * 100)
            parents.append({"path": manifest_path.relative_to(ROOT).as_posix(), "sha256": sha256(manifest_path),
                            "embedding_sha256": sha256(emb_path)})
        # Both conditions use the same held-out trial and time support.
        if not np.array_equal(traces["CEBRA-Behavior"][0], traces["CEBRA-Shuffle"][0]):
            raise RuntimeError(f"Behavior/shuffle trajectory support mismatch for {rat}")
        t, truth, pred_b = traces["CEBRA-Behavior"]
        _, _, pred_s = traces["CEBRA-Shuffle"]
        ax.plot(t[::2], truth[::2], color="black", lw=2, label="Ground Truth")
        ax.plot(t[::2], pred_b[::2], color="#6235E0", lw=2, label="CEBRA-Behavior")
        ax.plot(t[::10] + 0.05, pred_s[::10], color="gray", lw=2, ls="--", label="CEBRA-Shuffle")
        ax.set_title(rat.capitalize())
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position [cm]")
        ax.set_xlim(-1, 17.5)
        ax.set_xticks(np.linspace(0, 17.5, 8))
        ax.set_yticks(np.linspace(0, 160, 5))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        rows.append({"rat": rat, "condition": "position_direction vs position_direction_shuffled",
                     "split": "contiguous suffix test; fixed test rows 320:1020, crossing trial boundaries", "n_points": len(t),
                     "duration_s": float(t[-1]-t[0]) if len(t)>1 else 0.0,
                     "metric_model": "kNN regressor, k=36, cosine; fit on valid train rows only"})
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=3, frameon=False)
    fig.suptitle("Figure 2d-style held-out trajectory decoding", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"FIGURE2D_HELD_OUT_TRAJECTORIES.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    atomic_csv(out / "FIGURE2D_TRAJECTORY_SUPPORT.csv", rows, list(rows[0]))
    atomic_json(out / "TRAJECTORY_PROVENANCE.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)), "parents": parents,
        "scope": "descriptive visualization from frozen 3D encoders; train-only kNN; test rows not used for selection"})


def consistency_outputs(index: dict[tuple, Path]) -> tuple[list[dict], list[dict], list[dict]]:
    from cebra.integrations.sklearn.metrics import consistency_score

    specs = [
        ("baseline", "pca", "PCA"), ("baseline", "umap", "UMAP"),
        ("baseline", "opentsne", "t-SNE"), ("cebra", "CEBRA-Behavior-32D", "CEBRA-Behavior"),
        ("cebra", "CEBRA-Time-32D", "CEBRA-Time"),
        ("pivae", "Conv-piVAE-z_mean", "Conv-piVAE no test labels", "raw"),
        ("pivae", "Conv-piVAE-z_mean", "Conv-piVAE with test labels", "posterior"),
    ]
    cross_rows, run_rows, dec_rows = [], [], []
    for entry in specs:
        kind, method, display = entry[:3]
        variant = entry[3] if len(entry) > 3 else "raw"
        seeds_for = sorted({k[4] for k in index if k[0] == kind and k[1] == method},
                           key=lambda v: -1 if v == "deterministic" else int(v))
        for fold in FOLDS:
            for seed in seeds_for:
                keys = [(kind, method, rat, fold, seed) for rat in RATS]
                if not all(k in index for k in keys):
                    continue
                data = [load_fit(index, kind, method, rat, fold, seed, "test", variant) for rat in RATS]
                embeddings = [x["embedding"][x["valid_mask"]] for x in data]
                labels = [x["behavior"][x["valid_mask"], 0] for x in data]
                scores, pairs, _ = consistency_score(embeddings, between="datasets", labels=labels,
                                                      dataset_ids=list(RATS), num_discretization_bins=100)
                for score, pair in zip(scores, pairs):
                    cross_rows.append({"method": display, "fold": fold, "seed": seed,
                                       "source_rat": str(pair[0]), "target_rat": str(pair[1]),
                                       "held_out_behavior_aligned_r2": float(score),
                                       "n_source_rows": len(labels[RATS.index(pair[0])]),
                                       "n_target_rows": len(labels[RATS.index(pair[1])])})
            if kind == "baseline" and method == "pca":
                continue
            for rat in RATS:
                seeds = [s for s in seeds_for if (kind, method, rat, fold, s) in index]
                if len(seeds) < 2:
                    continue
                data = [load_fit(index, kind, method, rat, fold, s, "test", variant) for s in seeds]
                masks = [x["valid_mask"] for x in data]
                ids = [x["source_id"][m] for x, m in zip(data, masks)]
                if not all(np.array_equal(ids[0], q) for q in ids[1:]):
                    raise RuntimeError(f"Different test supports across seeds: {display}/{rat}/fold{fold}")
                embeddings = [x["embedding"][m] for x, m in zip(data, masks)]
                scores, pairs, _ = consistency_score(embeddings, between="runs")
                per_seed = defaultdict(list)
                for score, pair in zip(scores, pairs):
                    per_seed[seeds[int(pair[0])]].append(float(score))
                for seed, values in per_seed.items():
                    run_rows.append({"method": display, "rat": rat, "fold": fold, "seed": seed,
                                     "held_out_across_run_r2": mean(values),
                                     "n_other_seed_comparisons": len(values)})
    # Average folds within each training seed, preserving each seed as one point.
    cross_groups = defaultdict(list)
    for row in cross_rows:
        cross_groups[(row["method"], row["seed"], row["source_rat"], row["target_rat"])].append(row["held_out_behavior_aligned_r2"])
    cross_seed = [{"method": key[0], "seed": key[1], "source_rat": key[2], "target_rat": key[3],
                   "n_folds": len(values), "mean_across_folds": mean(values)}
                  for key, values in cross_groups.items()]
    run_groups = defaultdict(list)
    for row in run_rows:
        run_groups[(row["method"], row["rat"], row["seed"])].append(row["held_out_across_run_r2"])
    run_seed = [{"method": key[0], "rat": key[1], "seed": key[2], "n_folds": len(values),
                 "mean_across_folds": mean(values)} for key, values in run_groups.items()]
    # Decoder output uses existing metric definitions, then averages folds per seed.
    metric_path = OUT / "figure2d_metrics" / "RAT_METHOD_DECODING_LONG.csv"
    if metric_path.is_file():
        with metric_path.open(newline="", encoding="utf-8-sig") as stream:
            base_decoder_long = list(csv.DictReader(stream))
        pivae_decoder_long = add_pivae_scores(index)
        decoder_long = base_decoder_long + pivae_decoder_long
        # Preserve the original non-PiVAE table and publish an explicit
        # complete table, rather than leaving the PiVAE scores only in memory.
        all_methods_path = OUT / "figure2d_metrics" / "RAT_METHOD_DECODING_ALL_METHODS_LONG.csv"
        fields = list(dict.fromkeys(key for row in decoder_long for key in row))
        atomic_csv(all_methods_path, decoder_long, fields)
        pivae_manifests = sorted({(r["parent_result_path"], r["parent_result_sha256"])
                                  for r in pivae_decoder_long})
        pivae_embeddings = sorted({(r["parent_result_path"], r["split"], r["parent_embedding_sha256"])
                                   for r in pivae_decoder_long})
        atomic_json(OUT / "figure2d_metrics" / "RAT_METHOD_DECODING_ALL_METHODS_PROVENANCE.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": sha256(Path(__file__)),
            "source_non_pivae_table": metric_path.relative_to(ROOT).as_posix(),
            "source_non_pivae_table_sha256": sha256(metric_path),
            "source_non_pivae_rows": len(base_decoder_long),
            "pivae_scoring_rows_added": len(pivae_decoder_long),
            "combined_rows": len(decoder_long),
            "pivae_unique_fits": len(pivae_manifests),
            "pivae_source_manifests": [{"path": p, "sha256": h} for p, h in pivae_manifests],
            "pivae_embedding_hashes": [{"run_manifest": p, "split": s, "sha256": h}
                                        for p, s, h in pivae_embeddings],
            "metrics": ["position_median_absolute_error_cm", "position_r2", "direction_balanced_accuracy"],
            "protocol": "k=36 cosine kNN probes fitted only on the frozen train split; scored on the same valid/test supports; q(z|x) inference only",
            "test_firewall": "test behavior labels are used only as scoring targets; never passed to encoder inference or probe fitting",
            "q_z_x_u_policy": "excluded from decoding because test-time behavior-conditioned inference would leak targets",
            "no_encoder_training": True,
        })
        _, dec_rows = summarize_score_long(decoder_long)
    return cross_seed, run_seed, dec_rows


def plot_consistency(cross_seed: list[dict], run_seed: list[dict], dec_seed: list[dict], plots_only: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    out = OUT / "figure1e"
    out.mkdir(parents=True, exist_ok=True)
    methods = [method for method in ("CEBRA-Behavior", "CEBRA-Time",
               "Conv-piVAE with test labels", "Conv-piVAE no test labels",
               "t-SNE", "UMAP", "PCA")
               if any(row["method"] == method for row in cross_seed)]
    fig, axes = plt.subplots(ncols=len(methods) + 1, nrows=1,
                            figsize=(1.7 * (len(methods) + 1), 1.4),
                            gridspec_kw={"width_ratios": [1] * len(methods) + [0.08]}, dpi=600)
    mat_rows = []
    for ax, method in zip(axes[:-1], methods):
        matrix = np.full((len(RATS), len(RATS)), np.nan)
        for i, source in enumerate(RATS):
            for j, target in enumerate(RATS):
                vals = [float(r["mean_across_folds"]) for r in cross_seed
                        if r["method"] == method and r["source_rat"] == source and r["target_rat"] == target]
                if vals:
                    matrix[i, j] = mean(vals)
                    mat_rows.append({"method": method, "source_rat": source, "target_rat": target,
                                     "n_seed_replicates": len(vals), "mean_r2": mean(vals),
                                     "sd_across_seed_means": stdev(vals) if len(vals)>1 else ""})
        sns.heatmap(data=np.minimum(matrix * 100, 99), vmin=20, vmax=100,
                    xticklabels=[], yticklabels=[],
                    cmap=sns.color_palette("gray_r", as_cmap=True), annot=True,
                    annot_kws={"fontsize": 12}, fmt=".0f", ax=ax,
                    cbar=ax is axes[-2], cbar_ax=axes[-1] if ax is axes[-2] else None)
        ax.axis("off")
        titles = {"CEBRA-Behavior": "CEBRA-Behavior", "CEBRA-Time": "CEBRA-Time",
                  "Conv-piVAE with test labels": "conv-piVAE\nw/labels",
                  "Conv-piVAE no test labels": "conv-piVAE",
                  "t-SNE": "tSNE", "UMAP": "UMAP", "PCA": "PCA"}
        ax.set_title(f"{titles[method]} ({100*np.nanmean(matrix):.1f})", fontsize=8)
    fig.subplots_adjust(wspace=0.5)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"FIGURE1E_HELD_OUT_CROSS_RAT_CONSISTENCY.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    if plots_only:
        return
    atomic_csv(out / "CROSS_RAT_CONSISTENCY_BY_SEED.csv", cross_seed, list(cross_seed[0]))
    atomic_csv(out / "CROSS_RAT_CONSISTENCY_SUMMARY.csv", mat_rows, list(mat_rows[0]))
    # Extended Data Fig. 4 analogue. Report only metric definitions we actually computed.
    ed_rows = []
    name_map = {"CEBRA-Behavior": "CEBRA-Behavior-32D", "CEBRA-Time": "CEBRA-Time-32D",
                "t-SNE": "OPENTSNE", "Conv-piVAE no test labels": "Conv-piVAE-z_mean",
                "Conv-piVAE with test labels": "Conv-piVAE-test-labels"}
    by_source_seed = defaultdict(list)
    for r in cross_seed:
        by_source_seed[(r["method"], r["seed"], r["source_rat"])].append(float(r["mean_across_folds"]))
    for (method, seed, rat), values in by_source_seed.items():
        ed_rows.append({"rat": rat, "method": name_map.get(method, method), "seed": seed,
                        "metric": "held_out_cross_rat_consistency_r2", "value": mean(values)})
    for r in run_seed:
        ed_rows.append({"rat": r["rat"], "method": name_map.get(r["method"], r["method"]), "seed": r["seed"],
                        "metric": "held_out_across_run_consistency_r2", "value": float(r["mean_across_folds"])})
    for r in dec_seed:
        metric = r["metric"]
        if (r.get("split", "test") == "test"
                and metric in ("position_r2", "position_median_absolute_error_cm", "direction_balanced_accuracy")):
            ed_rows.append({"rat": r["rat"], "method": name_map.get(r["method"], r["method"]), "seed": r["seed"],
                            "metric": metric, "value": float(r["mean_across_folds"])})
    ed_path = OUT / "extended_data_figure4" / "RAT_ED4_ANALOG_METRICS_SEED_LEVEL.csv"
    atomic_csv(ed_path, ed_rows, list(ed_rows[0]))
    metric_order = ("held_out_cross_rat_consistency_r2", "held_out_across_run_consistency_r2",
                    "position_r2", "direction_balanced_accuracy", "position_median_absolute_error_cm")
    plot_metric_grid(ed_rows, OUT / "extended_data_figure4", "FIGURE_ED4_RAT_BENCHMARK_ANALOG",
                     metric_order, "Extended Data Fig. 4-style rat benchmark (held-out analogue)")
    atomic_csv(OUT / "figure2d" / "FIGURE2D_METHOD_METRICS_SEED_LEVEL.csv", dec_seed,
               list(dec_seed[0]) if dec_seed else ["method", "rat", "seed", "split", "metric", "n_folds", "mean_across_folds", "sd_across_folds"])
    plot_metric_grid(dec_seed, OUT / "figure2d", "FIGURE2D_METHOD_COMPARISON_TEST",
                     ("position_median_absolute_error_cm", "position_r2", "direction_balanced_accuracy"),
                     "Figure 2d-style method decoding comparison (held-out test)",
                     method_subset=("CEBRA-Behavior-32D", "Conv-piVAE-z_mean", "CEBRA-Time-32D",
                                    "PCA", "UMAP", "OPENTSNE"),
                     footnote="Dots: test score per training seed, averaged across folds; orange: median. Hyperparameters/models were not chosen from these test scores. q(z|x,u) is excluded because test-time labels would reveal the decoding target.")
    atomic_json(OUT / "figure1e" / "PROVENANCE.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)), "metric_implementation": "CEBRA 0.6.1 consistency_score",
        "cross_rat": "held-out test embeddings; continuous position labels; 100 discretization bins; directed R² after label alignment",
        "across_run": "held-out test embeddings; pairwise directed linear-regression R²; average comparisons per seed, then folds",
        "decoding": "same frozen test/valid supports and train-only kNN k=36 cosine as RAT_METHOD_DECODING_LONG.csv; no outcome used for selection",
        "limitations": ["these are held-out analogues, not a bitwise reproduction of paper's training-consistency panel",
                        "no test_total_r2 composite was substituted: position R², direction balanced accuracy, and position error are reported separately",
                        "PCA is deterministic, so across-run consistency is not applicable",
                        "Conv-piVAE no-test-labels is q(z|x) from a behavior-trained fit, not an unsupervised training condition",
                        "test-label-informed q(z|x,u) is shown as a separate consistency condition but excluded from decoding"]})
    atomic_json(OUT / "extended_data_figure4" / "PROVENANCE.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)), "metrics": list(metric_order),
        "panel_semantics": "held-out analogue; actual separate position/direction metrics, not paper's test_total_r2 composite",
        "points": "individual seed means across available folds; all 120 canonical Conv-piVAE configurations are represented after checkpoint-only embedding recovery",
        "label_conditioned_output": "q(z|x,u) is included for consistency only; excluded from decoding because held-out behavior labels would leak targets"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure1c", action="store_true", help="Build raw rat-data context panel")
    parser.add_argument("--embeddings", action="store_true", help="Build Figure 1d-style embedding panel")
    parser.add_argument("--trajectory", action="store_true", help="Build Figure 2d-style held-out trajectory")
    parser.add_argument("--consistency", action="store_true", help="Build Figure 1e and Extended Data Fig. 4 analogues")
    parser.add_argument("--render-existing", action="store_true", help="Re-render plots from existing result tables without recomputing metrics")
    parser.add_argument("--all", action="store_true", help="Run all downstream analyses")
    args = parser.parse_args()
    if not any((args.figure1c, args.embeddings, args.trajectory, args.consistency, args.render_existing, args.all)):
        parser.error("Select --figure1c, --embeddings, --trajectory, --consistency, or --all")
    if args.figure1c or args.all:
        plot_figure1c_data_overview()
    index = index_manifests()
    if args.render_existing:
        def read_rows(path: Path) -> list[dict]:
            with path.open(newline="", encoding="utf-8-sig") as stream:
                return list(csv.DictReader(stream))
        cross_seed = read_rows(OUT / "figure1e" / "CROSS_RAT_CONSISTENCY_BY_SEED.csv")
        for row in cross_seed:
            row["seed"] = "deterministic" if row["seed"] == "deterministic" else int(row["seed"])
            row["n_folds"] = int(row["n_folds"])
            row["mean_across_folds"] = float(row["mean_across_folds"])
        ed_existing = read_rows(OUT / "extended_data_figure4" / "RAT_ED4_ANALOG_METRICS_SEED_LEVEL.csv")
        run_seed = [{"method": r["method"], "rat": r["rat"], "seed": r["seed"],
                     "mean_across_folds": float(r["value"]), "n_folds": 1}
                    for r in ed_existing if r["metric"] == "held_out_across_run_consistency_r2"]
        dec_seed = read_rows(OUT / "figure2d" / "FIGURE2D_METHOD_METRICS_SEED_LEVEL.csv")
        for row in dec_seed:
            row["mean_across_folds"] = float(row["mean_across_folds"])
            row["seed"] = str(row["seed"])
        plot_consistency(cross_seed, run_seed, dec_seed, plots_only=True)
        plot_metric_grid(dec_seed, OUT / "figure2d", "FIGURE2D_METHOD_COMPARISON_TEST",
                         ("position_median_absolute_error_cm", "position_r2", "direction_balanced_accuracy"),
                         "Figure 2d-style method decoding comparison (held-out test)",
                         method_subset=("CEBRA-Behavior-32D", "Conv-piVAE-z_mean", "CEBRA-Time-32D",
                                        "PCA", "UMAP", "OPENTSNE"),
                         footnote="Dots: test score per training seed, averaged across folds; orange: median. Hyperparameters/models were not chosen from these test scores. q(z|x,u) is excluded because test-time labels would reveal the decoding target.")
        return
    if args.all or args.embeddings:
        plot_embedding_panels(index)
        plot_embedding_per_rat(index)
    if args.all or args.trajectory:
        draw_trajectory(index)
    if args.all or args.consistency:
        cross_seed, run_seed, dec_seed = consistency_outputs(index)
        if not cross_seed:
            raise RuntimeError("No complete cross-rat replicates available")
        plot_consistency(cross_seed, run_seed, dec_seed)


if __name__ == "__main__":
    main()
