"""Recover missing rat figure outputs from frozen fits; fit only 2b hybrid if requested.

This is an additive, versioned branch. It never edits or replaces the source fit
artifacts or older figures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "rat_figure_gap_closure_v1"
RATS = ("achilles", "buddy", "cicero", "gatsby")
FIG2C_CONDITIONS = (
    "position", "direction", "position_direction",
    "position_shuffled", "direction_shuffled", "position_direction_shuffled",
)
LOSS_COLORS = {
    "position": "deepskyblue", "direction": "deepskyblue",
    "position_direction": "deepskyblue", "position_shuffled": "gray",
    "direction_shuffled": "gray", "position_direction_shuffled": "gray",
}
LOSS_ALPHAS = {
    "position": 0.3, "direction": 0.6, "position_direction": 1.0,
    "position_shuffled": 0.3, "direction_shuffled": 0.6,
    "position_direction_shuffled": 1.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def extract_figure2c_loss() -> None:
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fit_root = ROOT / "results" / "rat_figure2c_3d_v1"
    out = OUT / "figure2c_loss"
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    summaries: list[dict] = []
    parents: list[dict] = []
    histories: dict[tuple[str, str], np.ndarray] = {}
    for rat in RATS:
        for condition in FIG2C_CONDITIONS:
            run = fit_root / rat / condition / "seed_42"
            manifest_path = run / "result.json"
            checkpoint_path = run / "checkpoint.pt"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            checkpoint_hash = sha256(checkpoint_path)
            expected_hash = manifest.get("artifacts", {}).get("checkpoint.pt")
            if checkpoint_hash != expected_hash:
                raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint_path}")
            if (manifest.get("rat") != rat or manifest.get("condition") != condition
                    or manifest.get("seed") != 42 or manifest.get("output_dimension") != 3
                    or manifest.get("max_iterations") != 10000):
                raise RuntimeError(f"Unexpected native Figure 2c fit metadata: {manifest_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            loss = checkpoint.get("state_dict", {}).get("loss")
            if loss is None:
                raise RuntimeError(f"Training loss absent from checkpoint: {checkpoint_path}")
            loss = np.asarray(loss.detach().cpu() if hasattr(loss, "detach") else loss,
                              dtype=np.float64).reshape(-1)
            if len(loss) != 10000 or not np.isfinite(loss).all():
                raise RuntimeError(f"Expected 10,000 finite loss values: {checkpoint_path}")
            histories[(rat, condition)] = loss
            result_hash = sha256(manifest_path)
            parents.append({"rat": rat, "condition": condition,
                            "result_json": manifest_path.relative_to(ROOT).as_posix(),
                            "result_json_sha256": result_hash,
                            "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
                            "checkpoint_sha256": checkpoint_hash})
            for step, value in enumerate(loss, start=1):
                rows.append({"rat": rat, "condition": condition, "seed": 42,
                             "optimizer_update": step, "training_objective": float(value),
                             "checkpoint_sha256": checkpoint_hash,
                             "result_json_sha256": result_hash})
            min_index = int(np.argmin(loss))
            summaries.append({"rat": rat, "condition": condition, "seed": 42,
                              "updates": len(loss), "initial_loss": float(loss[0]),
                              "final_loss": float(loss[-1]), "minimum_loss": float(loss[min_index]),
                              "minimum_update": min_index + 1,
                              "final_minus_initial": float(loss[-1] - loss[0]),
                              "checkpoint_sha256": checkpoint_hash})

    long_path = out / "FIGURE2C_LOSS_LONG.csv"
    summary_path = out / "FIGURE2C_LOSS_SUMMARY.csv"
    atomic_csv(long_path, rows, list(rows[0]))
    atomic_csv(summary_path, summaries, list(summaries[0]))
    steps = np.arange(1, 10001)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, dpi=240)
    for ax, rat in zip(axes.flat, RATS):
        for condition in FIG2C_CONDITIONS:
            label = condition.replace("_", " ")
            ax.plot(steps, histories[(rat, condition)], lw=0.9,
                    color=LOSS_COLORS[condition], alpha=LOSS_ALPHAS[condition], label=label)
        ax.set_title(rat.capitalize())
        ax.set_ylabel("InfoNCE Loss")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Optimizer update")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle("Figure 2c training histories recovered from frozen checkpoints")
    fig.tight_layout(rect=(0, 0, 0.82, 0.96))
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(out / f"FIGURE2C_LOSS_ALL_RATS.{suffix}",
                    dpi=300 if suffix == "png" else None, transparent=False, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    for rat in RATS:
        fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=240)
        for condition in FIG2C_CONDITIONS:
            ax.plot(steps, histories[(rat, condition)], lw=1.2,
                    color=LOSS_COLORS[condition], alpha=LOSS_ALPHAS[condition],
                    label=condition.replace("_", " "))
        ax.set(title=f"Figure 2c training history: {rat.capitalize()}",
               xlabel="Iterations", ylabel="InfoNCE Loss")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, ncol=2, fontsize=8)
        fig.tight_layout()
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(out / f"FIGURE2C_LOSS_{rat.upper()}.{suffix}",
                        dpi=300 if suffix == "png" else None, transparent=False, facecolor="white", bbox_inches="tight")
        plt.close(fig)
    prov = {"created_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": sha256(Path(__file__)), "source_fit_branch": fit_root.relative_to(ROOT).as_posix(),
            "operation": "checkpoint history extraction only; no fit or optimizer update",
            "history_key": "checkpoint.pt/state_dict/loss", "expected_updates_per_fit": 10000,
            "completed_histories": len(histories), "long_rows": len(rows), "parents": parents,
            "outputs": {p.name: sha256(p) for p in (long_path, summary_path)
                        if p.is_file()}}
    atomic_json(out / "PROVENANCE.json", prov)


def render_figure2c_gatsby() -> None:
    sys.path.insert(0, str(ROOT / "cebra_replica"))
    import plot_figure2c_3d_cebra_style as renderer
    renderer.OUT = OUT / "figure2c_embedding"
    renderer.OUT.mkdir(parents=True, exist_ok=True)
    renderer.make_figure("gatsby", 42)
    old = ROOT / "results" / "rat_figure2c_3d_cebra_style_v1"
    rows = []
    for rat in RATS:
        if rat == "gatsby":
            stem = f"figure2c_3d_{rat}_seed42"
            base = renderer.OUT / stem
        else:
            stem = f"figure2c_3d_{rat}_seed42"
            base = old / stem
        rows.append({"rat": rat, "seed": 42, "status": "GENERATED_FROM_FROZEN_EMBEDDINGS" if rat == "gatsby" else "REUSED_EXISTING_FIGURE",
                     "png_path": base.with_suffix(".png").relative_to(ROOT).as_posix(),
                     "png_sha256": sha256(base.with_suffix(".png")),
                     "svg_path": base.with_suffix(".svg").relative_to(ROOT).as_posix(),
                     "svg_sha256": sha256(base.with_suffix(".svg"))})
    atomic_csv(OUT / "FIGURE2C_PANEL_INDEX.csv", rows, list(rows[0]))


def load_npz_verified(path: Path, expected_hash: str) -> dict[str, np.ndarray]:
    if sha256(path) != expected_hash:
        raise RuntimeError(f"Embedding hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def fit_figure2b_hybrid(seed: int = 42, max_iterations: int = 5000) -> None:
    """Fit only the missing official-style 5D CEBRA-Hybrid model, one per rat."""
    import cebra
    import joblib
    import torch

    fit_root = OUT / "figure2b_fits"
    fit_root.mkdir(parents=True, exist_ok=True)
    for rat in RATS:
        source = ROOT / "data" / "rat_hippocampus" / f"{rat}.jl"
        raw = joblib.load(source)
        neural = np.asarray(raw["spikes"], dtype=np.float32)
        behavior = np.asarray(raw["position"], dtype=np.float32)
        if neural.ndim != 2 or behavior.ndim != 2 or behavior.shape[1] < 1 or len(neural) != len(behavior):
            raise RuntimeError(f"Unexpected source arrays for {rat}: {neural.shape}, {behavior.shape}")
        split_at = int(0.8 * len(neural))
        run = fit_root / rat / "hybrid_position_time_5d" / f"seed_{seed}"
        manifest_path = run / "result.json"
        expected = {"rat": rat, "condition": "hybrid_position_time_5d", "seed": seed,
                    "max_iterations": max_iterations, "output_dimension": 5,
                    "source_sha256": sha256(source), "script_sha256": sha256(Path(__file__))}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if any(manifest.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f"Existing hybrid fit provenance mismatch: {manifest_path}")
            for name, digest in manifest["artifacts"].items():
                if sha256(run / name) != digest:
                    raise RuntimeError(f"Existing hybrid fit artifact hash mismatch: {run / name}")
            print(f"REUSE {run}", flush=True)
            continue
        if run.exists() and any(run.iterdir()):
            raise RuntimeError(f"Refusing to overwrite partial hybrid fit: {run}")
        run.mkdir(parents=True, exist_ok=False)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        params = {"model_architecture": "offset10-model", "batch_size": 512,
                  "learning_rate": 3e-4, "temperature": 1.0, "output_dimension": 5,
                  "max_iterations": max_iterations, "distance": "cosine",
                  "conditional": "time_delta", "device": "cuda_if_available",
                  "verbose": False, "time_offsets": 10, "hybrid": True}
        model = cebra.CEBRA(**params)
        started = time.perf_counter()
        model.fit(neural[:split_at], behavior[:split_at, 0])
        fit_seconds = time.perf_counter() - started
        checkpoint_path = run / "checkpoint.pt"
        model.save(checkpoint_path)
        artifacts = {checkpoint_path.name: sha256(checkpoint_path)}
        boundaries = np.r_[0, np.flatnonzero(behavior[1:, 1] != behavior[:-1, 1])[1::2] + 1, len(behavior)] if behavior.shape[1] > 1 else np.array([0, len(behavior)])
        trial_id = np.searchsorted(boundaries[1:], np.arange(len(behavior)), side="right").astype(np.int64)
        time_id = np.arange(len(behavior), dtype=np.int64) - boundaries[trial_id]
        for split, indices in (("train", np.arange(split_at)),
                               ("test", np.arange(split_at, len(neural))),
                               ("all", np.arange(len(neural)))):
            embedding = np.asarray(model.transform(neural[indices]), dtype=np.float32)
            if embedding.shape != (len(indices), 5) or not np.isfinite(embedding).all():
                raise RuntimeError(f"Hybrid {split} embedding invariant failed: {rat}, {embedding.shape}")
            path = run / f"{split}_embedding.npz"
            np.savez_compressed(path, embedding=embedding, behavior=behavior[indices],
                                source_id=indices, trial_id=trial_id[indices], time_id=time_id[indices],
                                valid_mask=np.ones(len(indices), dtype=bool), rat=np.array(rat),
                                condition=np.array("hybrid_position_time_5d"), seed=np.array(seed),
                                split=np.array(split))
            artifacts[path.name] = sha256(path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        history = checkpoint.get("state_dict", {}).get("loss")
        if history is None:
            history = np.empty(0, dtype=np.float32)
        else:
            history = np.asarray(history.detach().cpu() if hasattr(history, "detach") else history,
                                 dtype=np.float32).reshape(-1)
        hist_path = run / "training_history.csv"
        atomic_csv(hist_path, [{"optimizer_update": i + 1, "loss": float(v)}
                               for i, v in enumerate(history)], ["optimizer_update", "loss"])
        artifacts[hist_path.name] = sha256(hist_path)
        manifest = {**expected, "parameters": params,
                    "split": {"train": [0, split_at], "test": [split_at, len(neural)]},
                    "fit_seconds": fit_seconds, "training_history_updates": len(history),
                    "artifacts": artifacts, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "cebra_version": getattr(cebra, "__version__", "unknown"),
                    "torch_version": torch.__version__,
                    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                    "scope": "missing Figure 2b hybrid model; contiguous 80/20 train/test; no test outcomes used for fitting/selection",
                    "official_reference_settings": "CEBRA Figure 2 hippocampus demo: 5,000 iterations, offset10-model, batch 512, lr 3e-4, temperature 1, cosine, offset 10; output dimension 5 is the Figure 2b caption's hybrid visualization dimension"}
        atomic_json(manifest_path, manifest)
        print(f"COMPLETE 2b hybrid {rat}, {fit_seconds:.1f}s", flush=True)


def render_figure2b(seed: int = 42) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    out = OUT / "figure2b"
    out.mkdir(parents=True, exist_ok=True)
    pos_root = ROOT / "results" / "rat_figure2c_3d_v1"
    cebra_root = ROOT / "results" / "rat_trial_benchmark_v1"
    hybrid_root = OUT / "figure2b_fits"
    parent_entries = []
    index_rows = []
    for rat in RATS:
        src = ROOT / "data" / "rat_hippocampus" / f"{rat}.jl"
        import joblib
        behavior_full = np.asarray(joblib.load(src)["position"], dtype=np.float32)
        panels = []
        # Existing native 3D position hypothesis and shuffled control.
        for condition, title in (("position", "Position hypothesis"),
                                 ("position_shuffled", "Position shuffled")):
            run = pos_root / rat / condition / f"seed_{seed}"
            manifest_path = run / "result.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            arrays = load_npz_verified(run / "all_embedding.npz", manifest["artifacts"]["all_embedding.npz"])
            panels.append((title, arrays["embedding"][:, :3], arrays["behavior"], arrays["source_id"],
                           "native 3D; 10,000 updates", (0, 1, 2)))
            parent_entries.append({"path": manifest_path.relative_to(ROOT).as_posix(), "sha256": sha256(manifest_path)})
        # Reuse the frozen 32D CEBRA-Time trial-wise model at fold 0, seed 42; show its first 3 coordinates.
        time_run = cebra_root / rat / "fold_0" / "time" / f"seed_{seed}"
        time_manifest_path = time_run / "result.json"
        time_manifest = json.loads(time_manifest_path.read_text(encoding="utf-8"))
        time_parts = []
        for split in ("train", "valid", "test"):
            path = time_run / f"{split}_embedding.npz"
            expected_hash = time_manifest["embeddings"][split]["embedding_sha256"]
            part = load_npz_verified(path, expected_hash)
            time_parts.append(part)
        time_emb = np.concatenate([x["embedding"] for x in time_parts])
        time_behavior = np.concatenate([x["behavior"] for x in time_parts])
        time_ids = np.concatenate([x["source_id"] for x in time_parts])
        order = np.argsort(time_ids)
        if len(np.unique(time_ids)) != len(time_ids) or not np.array_equal(time_ids[order], np.arange(len(behavior_full))):
            raise RuntimeError(f"CEBRA-Time split embedding support is incomplete or duplicated: {rat}")
        if not np.array_equal(time_behavior[order], behavior_full):
            raise RuntimeError(f"CEBRA-Time behavior does not align to source data: {rat}")
        panels.append(("CEBRA-Time", time_emb[order, :3], time_behavior[order], time_ids[order],
                       "32D fit; first 3 coordinates shown", (0, 1, 2)))
        parent_entries.append({"path": time_manifest_path.relative_to(ROOT).as_posix(), "sha256": sha256(time_manifest_path)})
        # Newly fitted 5D CEBRA-Hybrid; display the first 3 dimensions using official axis order.
        hybrid_run = hybrid_root / rat / "hybrid_position_time_5d" / f"seed_{seed}"
        hybrid_manifest_path = hybrid_run / "result.json"
        hybrid_manifest = json.loads(hybrid_manifest_path.read_text(encoding="utf-8"))
        hybrid = load_npz_verified(hybrid_run / "all_embedding.npz", hybrid_manifest["artifacts"]["all_embedding.npz"])
        panels.append(("CEBRA-Hybrid", hybrid["embedding"][:, :5], hybrid["behavior"], hybrid["source_id"],
                       "5D fit; first 3 coordinates shown", (1, 2, 0)))
        parent_entries.append({"path": hybrid_manifest_path.relative_to(ROOT).as_posix(), "sha256": sha256(hybrid_manifest_path)})
        # Ensure all displayed panels have the same row/label support.
        expected = np.arange(len(behavior_full))
        for title, emb, labels, ids, _dim, _order in panels:
            ix = np.argsort(ids)
            if not np.array_equal(ids[ix], expected) or len(labels) != len(emb) or not np.array_equal(labels[ix], behavior_full):
                raise RuntimeError(f"Figure 2b support/label mismatch for {rat}: {title}")
        fig = plt.figure(figsize=(20, 5.8), dpi=240)
        for i, (title, embedding, labels, ids, dim_note, order_dims) in enumerate(panels):
            ax = fig.add_subplot(1, 4, i + 1, projection="3d")
            sorted_order = np.argsort(ids)
            embedding = embedding[sorted_order]
            labels = labels[sorted_order]
            left = labels[:, 2] == 1
            right = labels[:, 1] == 1
            a, b, c = order_dims
            ax.scatter(embedding[left, a], embedding[left, b], embedding[left, c],
                       c=labels[left, 0], cmap="cool", s=0.5, rasterized=True)
            ax.scatter(embedding[right, a], embedding[right, b], embedding[right, c],
                       c=labels[right, 0], cmap="viridis", s=0.5, rasterized=True)
            ax.axis("off")
            ax.set_title(title, fontsize=13)
            ax.text2D(0.02, 0.02, dim_note, transform=ax.transAxes, fontsize=7, alpha=0.8)
        fig.suptitle(f"Figure 2b-style latent visualizations — {rat.capitalize()} (seed {seed})")
        fig.tight_layout(rect=(0, 0.16, 1, 0.94))
        # The two categorical columns select separate position color maps.
        left = behavior_full[:, 2] == 1
        right = behavior_full[:, 1] == 1
        for bounds, cmap, label, selection in (
            ([0.25, 0.07, 0.20, 0.022], "viridis", "Label 1 (right): position [cm]", right),
            ([0.55, 0.07, 0.20, 0.022], "cool", "Label 2 (left): position [cm]", left),
        ):
            values = behavior_full[selection, 0] * 100
            bar = fig.colorbar(ScalarMappable(norm=Normalize(vmin=float(values.min()),
                                                         vmax=float(values.max())), cmap=cmap),
                               cax=fig.add_axes(bounds), orientation="horizontal")
            bar.set_label(label, fontsize=9)
        base = out / f"FIGURE2B_{rat.upper()}_seed{seed}"
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(base.with_suffix(f".{suffix}"), dpi=300 if suffix == "png" else None,
                        transparent=False, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        index_rows.append({"rat": rat, "seed": seed, "figure_png": base.with_suffix(".png").relative_to(ROOT).as_posix(),
                           "panel_count": 4, "position_fit": "frozen native 3D; 10k updates",
                           "shuffle_fit": "frozen native 3D; 10k updates",
                           "time_fit": "frozen CEBRA Time; 32D; first 3 coordinates shown",
                           "hybrid_fit": hybrid_manifest_path.relative_to(ROOT).as_posix(),
                           "interpretation": "qualitative analog; dimensions/training duration differ across panels"})
    atomic_csv(out / "FIGURE2B_PANEL_INDEX.csv", index_rows, list(index_rows[0]))
    atomic_json(out / "PROVENANCE.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)), "official_reference": "https://cebra.ai/docs/cebra-figures/figures/Figure2.html",
        "parents": parent_entries, "figures": {p.name: sha256(p) for p in out.glob("FIGURE2B_*.png")},
        "scope": "descriptive Figure 2b-style panels; no test metric/model selection; existing frozen fits reused where available"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-2b-hybrid", action="store_true")
    parser.add_argument("--render-2b", action="store_true")
    parser.add_argument("--render-2c", action="store_true")
    parser.add_argument("--extract-2c-loss", action="store_true")
    args = parser.parse_args()
    if not any((args.fit_2b_hybrid, args.render_2b, args.render_2c, args.extract_2c_loss)):
        parser.error("Choose at least one explicit action")
    if args.extract_2c_loss:
        extract_figure2c_loss()
    if args.render_2c:
        render_figure2c_gatsby()
    if args.fit_2b_hybrid:
        fit_figure2b_hybrid()
    if args.render_2b:
        render_figure2b()


if __name__ == "__main__":
    main()
