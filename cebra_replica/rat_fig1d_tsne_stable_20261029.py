"""Stable, provenance-rich full-recording t-SNE for rat Figure 1D."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RATS = ("achilles", "buddy", "cicero", "gatsby")
OUT_ROOT = ROOT / "results" / "figura_1_2026-10-29" / "d"
SEED = 42

PARAMS = {
    "n_components": 2,
    "metric": "cosine",
    "perplexity": 30,
    "initialization": "pca",
    "learning_rate": "auto",
    "early_exaggeration": 12,
    "early_exaggeration_iter": 250,
    "n_iter": 1000,
    "n_jobs": 1,
    "random_state": SEED,
    "callbacks_every_iters": 50,
    "verbose": False,
}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_input(rat: str):
    source = ROOT / "data" / "rat_hippocampus" / f"{rat}.jl"
    raw = joblib.load(source)
    spikes = np.asarray(raw["spikes"], dtype=np.float32)
    behavior = np.asarray(raw["position"], dtype=np.float32)
    if spikes.ndim != 2 or behavior.shape != (len(spikes), 3):
        raise ValueError(f"Unexpected {rat} input shapes: spikes={spikes.shape}, position={behavior.shape}")

    # One sample per complete 250 ms window; 25 ms bins, 225 ms overlap.
    source_id = np.arange(5, len(spikes) - 4, dtype=np.int64)
    offsets = np.arange(-5, 5, dtype=np.int64)
    window_input = spikes[source_id[:, None] + offsets[None, :]]
    window_input = window_input.transpose(0, 2, 1).reshape(len(source_id), -1)
    if not np.isfinite(window_input).all():
        raise FloatingPointError(f"Non-finite neural input in {rat}")

    # Row normalization preserves cosine distances. A zero vector has undefined
    # cosine; represent each zero with its own orthogonal unit vector. This gives
    # distance 1 to every other row, matching the finite zero-vector convention
    # used by scikit-learn cosine distance, without feeding NaNs to t-SNE.
    norms = np.linalg.norm(window_input, axis=1)
    zero_rows = np.flatnonzero(norms <= np.finfo(np.float32).eps)
    fit_input = np.zeros((len(source_id), window_input.shape[1] + len(zero_rows)), dtype=np.float32)
    nonzero_rows = np.flatnonzero(norms > np.finfo(np.float32).eps)
    fit_input[nonzero_rows, :window_input.shape[1]] = (
        window_input[nonzero_rows] / norms[nonzero_rows, None]
    )
    if len(zero_rows):
        fit_input[zero_rows, window_input.shape[1] + np.arange(len(zero_rows))] = 1.0
    if not np.isfinite(fit_input).all():
        raise FloatingPointError(f"Non-finite cosine input after zero-vector handling in {rat}")
    if len(source_id) <= 3 * PARAMS["perplexity"]:
        raise ValueError(f"{rat} has too few rows for perplexity={PARAMS['perplexity']}: {len(source_id)}")

    return source, window_input, fit_input, behavior[source_id], source_id, len(zero_rows)


def next_attempt_dir(rat: str) -> Path:
    rat_dir = OUT_ROOT / rat
    rat_dir.mkdir(parents=True, exist_ok=True)
    attempts = []
    for path in rat_dir.glob("tSNE_attempt_*"):
        try:
            attempts.append(int(path.name.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    attempt = max(attempts, default=0) + 1
    return rat_dir / f"tSNE_attempt_{attempt:03d}"


def find_compatible_complete(rat: str, config: dict) -> Path | None:
    rat_dir = OUT_ROOT / rat
    if not rat_dir.exists():
        return None
    for folder in sorted(rat_dir.glob("tSNE_attempt_*")):
        manifest_path = folder / "manifest.json"
        fit_path = folder / "fit_data.npz"
        embedding_path = folder / "embedding.npy"
        if not manifest_path.is_file() or not fit_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("status") == "complete" and manifest.get("config") == config
                    and manifest.get("fit_data_sha256") == sha_file(fit_path)
                    and manifest.get("embedding_sha256") == sha_file(embedding_path)
                    and manifest.get("plot_png_sha256") == sha_file(folder / "tSNE.png")
                    and manifest.get("plot_pdf_sha256") == sha_file(folder / "tSNE.pdf")):
                return folder
        except (OSError, ValueError, KeyError):
            continue
    return None


def render(embedding: np.ndarray, behavior: np.ndarray, rat: str, folder: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    position = behavior[:, 0]
    norm = Normalize(vmin=float(position.min()), vmax=float(position.max()))
    fig = plt.figure(figsize=(7.2, 5.4), facecolor="white")
    ax = fig.add_axes([0.06, 0.08, 0.76, 0.84])
    for label, column, cmap in (("Right", 1, "viridis"), ("Left", 2, "cool")):
        selected = behavior[:, column] == 1
        ax.scatter(embedding[selected, 0], embedding[selected, 1],
                   c=position[selected], cmap=cmap, norm=norm,
                   s=1.0, linewidths=0, rasterized=True)
    ax.set_axis_off()
    ax.set_title(f"t-SNE — {rat.capitalize()}", fontsize=13, pad=10)
    for bounds, cmap, label, side in (
        ([0.87, 0.19, 0.025, 0.61], "cool", "Left — position", "left"),
        ([0.93, 0.19, 0.025, 0.61], "viridis", "Right — position", "right"),
    ):
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes(bounds),
                           orientation="vertical")
        bar.set_ticks([])
        bar.set_label(label, fontsize=9, labelpad=4)
        bar.ax.yaxis.set_label_position(side)
    fig.savefig(folder / "tSNE.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(folder / "tSNE.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_one(rat: str, dry_run: bool = False):
    source, window_input, fit_input, behavior, source_id, zero_count = build_input(rat)
    config = {
        "figure": "Figure 1D",
        "rat": rat,
        "fit_scope": "full recording; all complete 10-bin windows",
        "source": f"data/rat_hippocampus/{rat}.jl",
        "source_sha256": sha_file(source),
        "window_bins": 10,
        "bin_width_ms": 25,
        "overlap_ms": 225,
        "stride_bins": 1,
        "window_flattening": "channel-major; (sample, neuron, time) then flatten",
        "source_rows": int(len(joblib.load(source)["spikes"])),
        "fit_rows": int(len(source_id)),
        "n_neurons": int(window_input.shape[1] // 10),
        "base_input_features": int(window_input.shape[1]),
        "fit_input_features": int(fit_input.shape[1]),
        "zero_window_count": int(zero_count),
        "zero_window_policy": "each zero vector mapped to a unique orthogonal unit vector; finite cosine distance 1 to all distinct rows",
        "cosine_input_preprocessing": "L2-normalize each nonzero window; preserves cosine distances",
        "label_use_in_fit": "none; labels are saved for coloring only",
        "label_columns": ["position", "right", "left"],
        "seed": SEED,
        "fit_input_sha256": sha_array(fit_input),
        "source_id_sha256": sha_array(source_id),
        "params": PARAMS,
    }
    completed = find_compatible_complete(rat, config)
    if completed:
        print(f"REUSE {rat}: {completed}", flush=True)
        return
    if dry_run:
        print(json.dumps({"config": config, "output_parent": str(OUT_ROOT / rat)}, indent=2), flush=True)
        return

    folder = next_attempt_dir(rat)
    folder.mkdir(parents=False, exist_ok=False)
    fit_path = folder / "fit_data.npz"
    np.savez_compressed(fit_path, fit_input=fit_input, original_window_input=window_input,
                        behavior=behavior, source_id=source_id)
    atomic_json(folder / "manifest.json", {
        "status": "running", "pid": __import__("os").getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "fit_data_path": fit_path.relative_to(ROOT).as_posix(),
        "fit_data_sha256": sha_file(fit_path),
        "script_sha256": sha_file(Path(__file__)),
    })
    progress_path = folder / "progress.json"
    started = time.perf_counter()
    atomic_json(progress_path, {"status": "running", "iteration": 0,
                                "total_iterations": PARAMS["n_iter"],
                                "updated_utc": datetime.now(timezone.utc).isoformat()})

    try:
        import openTSNE
        from openTSNE import TSNE

        def callback(iteration, error, _embedding):
            atomic_json(progress_path, {
                "status": "running", "iteration": int(iteration),
                "total_iterations": PARAMS["n_iter"],
                "kl_divergence": float(error),
                "elapsed_seconds": time.perf_counter() - started,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            })
            return False

        model = TSNE(**PARAMS, callbacks=callback)
        embedding = np.asarray(model.fit(fit_input), dtype=np.float32)
        fit_seconds = time.perf_counter() - started
        if embedding.shape != (len(source_id), 2) or not np.isfinite(embedding).all():
            raise FloatingPointError(
                f"Invalid t-SNE output for {rat}: shape={embedding.shape}, "
                f"finite={int(np.isfinite(embedding).sum())}/{embedding.size}"
            )
        np.save(folder / "embedding.npy", embedding, allow_pickle=False)
        render(embedding, behavior, rat, folder)
        manifest = {
            "status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
            "config": config, "resolved_parameters": model.get_params(),
            "openTSNE_version": openTSNE.__version__, "fit_seconds": fit_seconds,
            "fit_data_path": fit_path.relative_to(ROOT).as_posix(),
            "fit_data_sha256": sha_file(fit_path),
            "fit_input_sha256": sha_array(fit_input),
            "embedding_path": (folder / "embedding.npy").relative_to(ROOT).as_posix(),
            "embedding_sha256": sha_file(folder / "embedding.npy"),
            "plot_png": (folder / "tSNE.png").relative_to(ROOT).as_posix(),
            "plot_png_sha256": sha_file(folder / "tSNE.png"),
            "plot_pdf": (folder / "tSNE.pdf").relative_to(ROOT).as_posix(),
            "plot_pdf_sha256": sha_file(folder / "tSNE.pdf"),
            "script_sha256": sha_file(Path(__file__)),
            "full_recording_rows": int(config["source_rows"]),
            "fit_rows": int(len(source_id)),
            "plotted_rows": int(len(embedding)),
        }
        atomic_json(folder / "manifest.json", manifest)
        atomic_json(progress_path, {"status": "complete", "iteration": PARAMS["n_iter"],
                                   "total_iterations": PARAMS["n_iter"],
                                   "fit_seconds": fit_seconds,
                                   "updated_utc": datetime.now(timezone.utc).isoformat()})
        print(f"COMPLETE {rat} rows={len(embedding)} seconds={fit_seconds:.1f} path={folder}", flush=True)
    except Exception as error:
        failure = {
            "status": "failed", "failed_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__, "error": str(error), "config": config,
            "fit_data_path": fit_path.relative_to(ROOT).as_posix(),
            "fit_data_sha256": sha_file(fit_path),
        }
        atomic_json(folder / "failure.json", failure)
        atomic_json(progress_path, {"status": "failed", "error_type": type(error).__name__,
                                   "error": str(error),
                                   "updated_utc": datetime.now(timezone.utc).isoformat()})
        atomic_json(folder / "manifest.json", failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rat", choices=RATS, action="append",
                        help="select rat; repeat for multiple rats; default is all four")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for rat in args.rat or RATS:
        try:
            run_one(rat, dry_run=args.dry_run)
        except Exception as error:
            print(f"FAILED {rat}: {type(error).__name__}: {error}", flush=True)


if __name__ == "__main__":
    main()
