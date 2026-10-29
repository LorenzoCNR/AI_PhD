"""Render full-recording Figure 1d / ED3 panels from frozen embeddings."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]
ED3 = ROOT / "results" / "rat_paper_replication_2023_v1" / "ED3"
SOFT = ROOT / "results" / "user_soft_rat_fullrecording_stable3d_v1"
OUT = ROOT / "results" / "rat_paper_fig1d_ed3_v1"
RATS = ("achilles", "buddy", "cicero", "gatsby")
SEED = 42


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def read_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key].copy() for key in stored.files}


def read_manifest(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing fit manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verified_parent(embedding_path: Path, manifest_path: Path) -> dict:
    manifest = read_manifest(manifest_path)
    expected = manifest.get("embedding_sha256")
    if not embedding_path.is_file() or not expected or sha(embedding_path) != expected:
        raise RuntimeError(f"Embedding/manifest SHA-256 mismatch: {embedding_path}")
    return {"embedding": embedding_path.relative_to(ROOT).as_posix(),
            "embedding_sha256": expected,
            "manifest": manifest_path.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha(manifest_path)}


def validate_panel(panel: dict, raw_behavior: np.ndarray, rat: str) -> dict:
    embedding = panel["embedding"]
    behavior = panel["behavior"]
    source_id = panel["source_id"].astype(np.int64, copy=False)
    mask = panel.get("valid_mask")
    if mask is None:
        mask = np.ones(len(source_id), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    if embedding.ndim != 2 or len(embedding) != len(source_id) or len(behavior) != len(source_id):
        raise ValueError(f"Inconsistent arrays for {rat}/{panel['method']}")
    if behavior.shape != (len(source_id), 3) or mask.shape != source_id.shape:
        raise ValueError(f"Invalid behavior/mask shape for {rat}/{panel['method']}")
    if not np.isfinite(embedding).all() or not np.isfinite(behavior).all():
        raise FloatingPointError(f"Non-finite arrays for {rat}/{panel['method']}")
    if (source_id < 0).any() or (source_id >= len(raw_behavior)).any():
        raise ValueError(f"Out-of-range source ids for {rat}/{panel['method']}")
    if not np.allclose(behavior, raw_behavior[source_id], rtol=0, atol=1e-6):
        raise ValueError(f"Saved behavior does not align with raw labels for {rat}/{panel['method']}")
    if not mask.any():
        raise ValueError(f"Empty plotted support for {rat}/{panel['method']}")
    panel["source_id"] = source_id
    panel["valid_mask"] = mask
    return panel


def paper_panels(rat: str, raw_behavior: np.ndarray) -> list[dict]:
    panels = []
    specs = (
        ("CEBRA-Behaviour", "cebra", "behavior", "embedding", None, 3),
        ("conv-piVAE with labels", "pivae", "", "post_mean_behavior_informed_raw", "valid_mask", 2),
        ("conv-piVAE without labels", "pivae", "", "z_mean_raw", "valid_mask", 2),
        ("CEBRA-Time", "cebra", "time", "embedding", None, 3),
        ("t-SNE", "manifold", "tsne", "embedding", None, 2),
        ("UMAP", "manifold", "umap", "embedding", None, 2),
    )
    pivae_data = None
    pivae_parent = None
    for title, family, method, key, mask_key, plotted_dim in specs:
        if family == "cebra":
            base = ED3 / "cebra" / rat / method / f"seed_{SEED}"
            embedding_path, manifest_path = base / "embedding.npz", base / "manifest.json"
            bundle = read_npz(embedding_path)
            manifest = read_manifest(manifest_path)
            fit_dim = int(manifest["config"]["params"]["output_dimension"])
            if fit_dim != 3 or plotted_dim != 3:
                raise ValueError(f"CEBRA must be a native 3D fit: {rat}/{method}")
            if not np.allclose(np.linalg.norm(bundle[key], axis=1), 1.0, atol=2e-5):
                raise ValueError(f"CEBRA output is not on the unit sphere: {rat}/{method}")
            parent = verified_parent(embedding_path, manifest_path)
            panel = dict(method=title, embedding=bundle[key], behavior=bundle["behavior"],
                         source_id=bundle["source_id"], fit_dimension=fit_dim,
                         fit_rows=len(bundle[key]), plotted_dimensions=plotted_dim,
                         fit_scope="full recording", normalization="unit L2 norm",
                         parent=parent, support_note="all saved CEBRA rows",
                         fit_config=manifest.get("config"),
                         library_version=manifest.get("config", {}).get("cebra_version"))
        elif family == "pivae":
            if pivae_data is None:
                base = ED3 / "pivae_full_recording" / rat / f"seed_{SEED}"
                embedding_path, manifest_path = base / "embedding_full.npz", base / "manifest.json"
                pivae_data = read_npz(embedding_path)
                pivae_parent = verified_parent(embedding_path, manifest_path)
            panel = dict(method=title, embedding=pivae_data[key],
                         behavior=pivae_data["behavior"], source_id=pivae_data["source_id"],
                         valid_mask=pivae_data[mask_key], fit_dimension=2,
                         fit_rows=len(pivae_data[key]), plotted_dimensions=plotted_dim,
                         fit_scope="full recording; valid 10-bin windows plotted",
                         normalization="raw piVAE posterior mean; no L2 normalization",
                         parent=pivae_parent,
                         support_note=("q(z|x), no behavior supplied at inference" if key == "z_mean_raw"
                                       else "q(z|x,u), behavior-informed inference"),
                         fit_config=read_manifest(manifest_path).get("config"),
                         library_version=read_manifest(manifest_path).get("versions", {}))
        else:
            base = ED3 / "manifold" / method / rat / "window10_flat" / f"seed_{SEED}"
            embedding_path, manifest_path = base / "embedding.npz", base / "manifest.json"
            bundle = read_npz(embedding_path)
            manifest = read_manifest(manifest_path)
            if manifest.get("status", "complete") != "complete":
                raise RuntimeError(f"Manifold fit is not complete: {manifest_path}")
            parent = verified_parent(embedding_path, manifest_path)
            panel = dict(method=title, embedding=bundle[key], behavior=bundle["behavior"],
                         source_id=bundle["source_id"], fit_dimension=2,
                         fit_rows=len(bundle[key]), plotted_dimensions=plotted_dim,
                         fit_scope="full-recording 10-bin windows; fit and plot support coincide",
                         normalization="raw manifold coordinates; no extra normalization",
                         parent=parent, support_note="window centers with full local input window",
                         fit_config=manifest.get("config"),
                         resolved_parameters=manifest.get("resolved_parameters", manifest.get("config", {}).get("params")),
                         library_version=manifest.get("library_version"))
        if panel["embedding"].shape[1] != panel["fit_dimension"]:
            raise ValueError(f"Native fit dimension mismatch for {rat}/{title}")
        panels.append(validate_panel(panel, raw_behavior, rat))
    return panels


def soft_panels(rat: str, raw_behavior: np.ndarray) -> list[dict]:
    panels = []
    for condition, condition_label in (("behavior_only", "Position + Direction"),
                                       ("joint", "Time + Behavior")):
        for architecture, architecture_label in (("cnn", "CNN"),
                                                  ("transformer", "Transformer")):
            base = SOFT / rat / condition / architecture / f"seed_{SEED}"
            embedding_path = base / "embedding_full.npz"
            manifest_path = base / "manifest.json"
            bundle = read_npz(embedding_path)
            manifest = read_manifest(manifest_path)
            if manifest.get("status") not in (None, "complete"):
                raise RuntimeError(f"Soft fit is not complete: {manifest_path}")
            if int(manifest["config"]["output_dimension"]) != 3:
                raise ValueError(f"Soft fit must be native 3D: {rat}/{condition}/{architecture}")
            embedding = bundle["embedding_unit"]
            if not np.allclose(np.linalg.norm(embedding, axis=1), 1.0, atol=2e-5):
                raise ValueError(f"Soft unit output has incorrect norms: {rat}/{condition}/{architecture}")
            parent = verified_parent(embedding_path, manifest_path)
            panels.append(validate_panel(dict(
                method=f"Soft {architecture_label} — {condition_label}",
                embedding=embedding, behavior=bundle["behavior"], source_id=bundle["source_id"],
                valid_mask=bundle["valid_mask"], fit_dimension=3, fit_rows=len(embedding),
                plotted_dimensions=3, fit_scope="full recording; valid 10-bin windows plotted",
                normalization="saved embedding_unit; L2 norm 1", parent=parent,
                support_note=f"condition={condition}; native 3D; all three coordinates",
                fit_config=manifest.get("config"), library_version=manifest.get("cebra_version"),
            ), raw_behavior, rat))
    return panels


def draw_scatter(ax, panel: dict, position_norm: Normalize, point_size: float) -> None:
    embedding = panel["embedding"][panel["valid_mask"]]
    behavior = panel["behavior"][panel["valid_mask"]]
    is_3d = panel["plotted_dimensions"] == 3
    for direction, column, cmap in (("left", 2, "cool"), ("right", 1, "viridis")):
        selected = behavior[:, column] == 1
        if is_3d:
            ax.scatter(embedding[selected, 0], embedding[selected, 1], embedding[selected, 2],
                       c=behavior[selected, 0], cmap=cmap, norm=position_norm,
                       s=point_size, linewidths=0, rasterized=True, depthshade=False)
        else:
            ax.scatter(embedding[selected, 0], embedding[selected, 1],
                       c=behavior[selected, 0], cmap=cmap, norm=position_norm,
                       s=point_size, linewidths=0, rasterized=True)
    ax.set_title(panel["method"] + f"\n{panel['plotted_dimensions']}D native",
                 fontsize=9, pad=5)
    ax.set_facecolor("white")
    if is_3d:
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_zlim(-1.05, 1.05)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22, azim=38)
        ax.set_xlabel("Latent 1", fontsize=7, labelpad=0)
        ax.set_ylabel("Latent 2", fontsize=7, labelpad=0)
        ax.set_zlabel("Latent 3", fontsize=7, labelpad=0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor("white")
            axis.line.set_color("#cccccc")
    else:
        ax.set_axis_off()


def panel_record(panel: dict) -> dict:
    ids = panel["source_id"][panel["valid_mask"]]
    return dict(method=panel["method"], fit_dimension=panel["fit_dimension"],
                plotted_coordinates=list(range(panel["plotted_dimensions"])),
                plotted_dimensions=panel["plotted_dimensions"], fit_rows=panel["fit_rows"],
                plotted_rows=int(panel["valid_mask"].sum()), fit_scope=panel["fit_scope"],
                normalization=panel["normalization"], support_note=panel["support_note"],
                fit_config=panel.get("fit_config"),
                resolved_parameters=panel.get("resolved_parameters"),
                library_version=panel.get("library_version"),
                first_source_id=int(ids.min()), last_source_id=int(ids.max()),
                source_ids_sha256=array_sha(ids), parents=[panel["parent"]])


def save_figure(fig, target: Path, metadata: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    outputs = {extension: target.with_suffix("." + extension)
               for extension in ("png", "pdf", "svg")}
    provenance_path = target.with_name(target.name + "_PROVENANCE.json")
    existing = [str(path) for path in [*outputs.values(), provenance_path] if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite immutable figure output: " + ", ".join(existing))
    for extension, path in outputs.items():
        fig.savefig(path, dpi=300 if extension == "png" else None,
                    bbox_inches="tight", facecolor="white")
    metadata.update(created_utc=datetime.now(timezone.utc).isoformat(),
                    script_sha256=sha(Path(__file__)),
                    matplotlib_version=matplotlib.__version__,
                    source_dataset={"path": f"data/rat_hippocampus/{metadata['rat']}.jl",
                                    "sha256": sha(ROOT / "data" / "rat_hippocampus" / f"{metadata['rat']}.jl")},
                    files={extension: sha(path) for extension, path in outputs.items()})
    temporary = provenance_path.with_suffix(provenance_path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(provenance_path)


def render_paper(rat: str, raw_behavior: np.ndarray, figure_name: str, target: Path) -> None:
    panels = paper_panels(rat, raw_behavior)
    vmin, vmax = float(raw_behavior[:, 0].min()), float(raw_behavior[:, 0].max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    fig = plt.figure(figsize=(23, 5.5), facecolor="white")
    grid = fig.add_gridspec(1, 6, left=0.025, right=0.93, top=0.82, bottom=0.08,
                            wspace=0.13)
    fig.suptitle(f"{rat.capitalize()} — {figure_name} — full recording, seed {SEED}",
                 y=0.965, fontsize=13)
    plotted_counts = []
    for index, panel in enumerate(panels):
        ax = fig.add_subplot(grid[0, index], projection="3d" if panel["plotted_dimensions"] == 3 else None)
        draw_scatter(ax, panel, norm, point_size=0.55)
        plotted_counts.append(int(panel["valid_mask"].sum()))
    fig.text(0.48, 0.025,
             "Position shown separately for left (cool) and right (viridis); all saved rows plotted unless a valid-window mask is present.",
             ha="center", fontsize=8)
    for bounds, cmap, label, side in (([0.945, 0.21, 0.012, 0.52], "cool", "Left — position", "left"),
                                      ([0.965, 0.21, 0.012, 0.52], "viridis", "Right — position", "right")):
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes(bounds),
                           orientation="vertical")
        bar.set_ticks([])
        bar.set_label(label, fontsize=8, labelpad=4)
        bar.ax.yaxis.set_label_position(side)
    save_figure(fig, target, dict(panel=figure_name, rat=rat, seed=SEED,
                                  support="full-recording fit; source-id alignment; no point subsampling",
                                  color_key="behavior[:,0] position; columns 1/2 select right/left",
                                  panel_index=[panel_record(panel) for panel in panels]))
    plt.close(fig)


def render_soft(rat: str, raw_behavior: np.ndarray, target: Path) -> None:
    panels = soft_panels(rat, raw_behavior)
    norm = Normalize(vmin=float(raw_behavior[:, 0].min()), vmax=float(raw_behavior[:, 0].max()))
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    grid = fig.add_gridspec(2, 2, left=0.04, right=0.91, top=0.88, bottom=0.07,
                            hspace=0.18, wspace=0.1)
    fig.suptitle(f"{rat.capitalize()} — Soft 3D extension — full-recording fits, seed {SEED}",
                 y=0.97, fontsize=13)
    for index, panel in enumerate(panels):
        ax = fig.add_subplot(grid[index // 2, index % 2], projection="3d")
        draw_scatter(ax, panel, norm, point_size=0.7)
    for bounds, cmap, label, side in (([0.925, 0.22, 0.014, 0.5], "cool", "Left — position", "left"),
                                      ([0.95, 0.22, 0.014, 0.5], "viridis", "Right — position", "right")):
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes(bounds),
                           orientation="vertical")
        bar.set_ticks([])
        bar.set_label(label, fontsize=8, labelpad=4)
        bar.ax.yaxis.set_label_position(side)
    save_figure(fig, target, dict(panel="Soft 3D extension; not an original paper panel",
                                  rat=rat, seed=SEED,
                                  support="full-recording fit; valid-window plot mask; no point subsampling",
                                  color_key="behavior[:,0] position; columns 1/2 select right/left",
                                  panel_index=[panel_record(panel) for panel in panels]))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", action="store_true", help="render Figure 1d Achilles and ED3 for all rats")
    parser.add_argument("--soft", action="store_true", help="render the separate Soft 3D extension")
    parser.add_argument("--rat", choices=RATS, help="restrict rendering to one rat")
    args = parser.parse_args()
    if not args.paper and not args.soft:
        parser.error("choose --paper and/or --soft")
    selected = (args.rat,) if args.rat else RATS
    for rat in selected:
        source = ROOT / "data" / "rat_hippocampus" / f"{rat}.jl"
        raw = joblib.load(source)
        raw_behavior = np.asarray(raw["position"], dtype=np.float32)
        if raw_behavior.ndim != 2 or raw_behavior.shape[1] != 3:
            raise ValueError(f"Unexpected behavior labels for {rat}: {raw_behavior.shape}")
        if args.paper and (not args.rat or rat == "achilles"):
            render_paper(rat, raw_behavior, "Figure 1d paper-configuration comparison",
                         OUT / "Figure_1d" / "FIGURE1D_ACHILLES_PAPER_REPLICA")
        if args.paper:
            render_paper(rat, raw_behavior, "Extended Data Figure 3",
                         OUT / "Extended_Data" / "ED3" / rat.upper() / f"ED3_{rat.upper()}_PAPER_REPLICA")
        if args.soft:
            render_soft(rat, raw_behavior,
                        OUT / "Soft_3D_extension" / f"SOFT_3D_{rat.upper()}_FULL_RECORDING")


if __name__ == "__main__":
    main()
