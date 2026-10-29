"""Render the rat Figure 1 / ED3 panels from saved embeddings only.

This script never fits models or transforms new data. It validates the saved
manifests through rat_rebuild_1d_ed3_20261027.read_panel before plotting.
"""
from __future__ import annotations

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
from matplotlib.ticker import MaxNLocator

import rat_rebuild_1d_ed3_20261027 as source


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATE_TAG = datetime.now().strftime("%Y-%m-%d")
FIG1 = RESULTS / f"figura_1_{DATE_TAG}"
FIG1_SOFT = RESULTS / f"figura_1_soft_{DATE_TAG}"
ED3 = RESULTS / "extended_data_3_data"
ED3_SOFT = RESULTS / "extended_data_3_soft_data"
RATS = ("achilles", "buddy", "cicero", "gatsby")
RAT_PANEL = {"achilles": "A", "buddy": "B", "cicero": "C", "gatsby": "D"}
POSITION_SCALE = (0.0, 1.6)
POSITION_TICKS = (0.0, 0.5, 1.0, 1.5)
COLORS = {"Left": "cool", "Right": "viridis"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def paper_specs():
    specs = list(source.PAPER)
    title, *rest = specs[0]
    specs[0] = (title.replace("CEBRA Behaviour", "CEBRA"), *rest)
    return specs


def load_panels(rat: str, raw: np.ndarray, data_hash: str, specs, campaign):
    panels, records = [], []
    for spec in specs:
        try:
            panel = source.read_panel(rat, raw, data_hash, spec, campaign)
            record = {"status": "verified", **panel["record"]}
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            detail = str(exc).replace("|", "/")
            panel = {"title": spec[0], "error": detail}
            record = {"status": "unavailable", "rat": rat, "seed": 42,
                      "title": spec[0].replace("\n", " "), "reason": detail,
                      "fit_dimension": spec[4], "plotted_dimensions": spec[4]}
        panel["dimension"] = spec[4]
        panels.append(panel)
        records.append(record)
    return panels, records


def draw(panels, rat: str, heading: str, target: Path, full_recording=True, overwrite=False):
    if not overwrite:
        for suffix in ("png", "pdf"):
            if target.with_suffix("." + suffix).exists():
                raise FileExistsError(target.with_suffix("." + suffix))
    fig = plt.figure(figsize=(17, 10), facecolor="white")
    grid = fig.add_gridspec(2, 3, left=.045, right=.875, top=.87, bottom=.10,
                            wspace=.22, hspace=.34)
    norm = Normalize(vmin=POSITION_SCALE[0], vmax=POSITION_SCALE[1])
    for i, panel in enumerate(panels):
        dimension = panel["dimension"]
        ax = fig.add_subplot(grid[i // 3, i % 3], projection="3d" if dimension == 3 else None)
        if "error" in panel:
            ax.set_axis_off()
        else:
            z, behavior, mask = panel["z"], panel["behavior"], panel["mask"]
            for column, cmap in ((2, COLORS["Left"]), (1, COLORS["Right"])):
                selected = mask & (behavior[:, column] == 1)
                kwargs = dict(c=behavior[selected, 0], cmap=cmap, norm=norm,
                              s=.7, linewidths=0, rasterized=True)
                if dimension == 3:
                    kwargs["depthshade"] = False
                ax.scatter(*(z[selected, j] for j in range(dimension)), **kwargs)
            labels = [f"Latent {j + 1}" for j in range(dimension)]
            ax.set_xlabel(labels[0], labelpad=2)
            ax.set_ylabel(labels[1], labelpad=2)
            if dimension == 3:
                ax.set_zlabel(labels[2], labelpad=2)
                ax.view_init(elev=22, azim=38)
                ax.set_box_aspect((1, 1, 1))
            for axis in (ax.xaxis, ax.yaxis, *([ax.zaxis] if dimension == 3 else [])):
                axis.set_major_locator(MaxNLocator(4))
            ax.tick_params(labelsize=7, pad=0)
            ax.set_title(panel["title"], fontsize=9)
    for x, name in ((.895, "Left"), (.952, "Right")):
        colorbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=COLORS[name]),
            cax=fig.add_axes([x, .22, .014, .53]), ticks=POSITION_TICKS)
        colorbar.ax.yaxis.set_ticks_position("right" if name == "Right" else "left")
        colorbar.ax.tick_params(labelsize=7, pad=2)
        fig.text(x + .007, .19, name, ha="center", va="top", fontsize=8)
    fig.suptitle(f"{rat.capitalize()} — {heading}", fontsize=13)
    target.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(target.with_suffix("." + suffix), dpi=220,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_provenance(directory: Path, heading: str, records, outputs, data_files):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "renderer": relative(Path(__file__).resolve()),
        "renderer_sha256": sha256(Path(__file__).resolve()),
        "heading": heading,
        "fit_scope": "saved fits only; no training or transform",
        "position_scale_m": list(POSITION_SCALE),
        "position_ticks_m": list(POSITION_TICKS),
        "direction_colors": COLORS,
        "data_files": {relative(p): sha256(p) for p in data_files},
        "panels": records,
        "outputs": {relative(p): sha256(p) for p in outputs},
    }
    path = directory / "PROVENANCE.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    campaign_path = source.SOFT / "CAMPAIGN_STATE.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    data_files = [campaign_path]
    all_ed3_reference, all_ed3_soft = [], []
    for rat in RATS:
        raw_path = ROOT / "data" / "rat_hippocampus" / f"{rat}.jl"
        raw = np.asarray(joblib.load(raw_path)["position"], dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
            raise ValueError(f"invalid position data: {raw_path}")
        data_hash = sha256(raw_path)
        data_files.append(raw_path)
        ref_panels, ref_records = load_panels(rat, raw, data_hash, paper_specs(), campaign)
        soft_panels, soft_records = load_panels(rat, raw, data_hash, source.SOFT_SPECS, campaign)
        all_ed3_reference.extend(ref_records)
        all_ed3_soft.extend(soft_records)
        panel = RAT_PANEL[rat]
        ref_target = ED3 / panel / f"ED3_{rat.upper()}_PAPER_METHODS"
        soft_target = ED3_SOFT / panel / f"ED3_{rat.upper()}_SOFT_STABLE"
        draw(ref_panels, rat, "Extended Data Figure 3 — paper methods", ref_target)
        draw(soft_panels, rat, "Extended Data Figure 3 — Soft Stable", soft_target)
        save_provenance(ref_target.parent, f"ED3 {rat} paper methods", ref_records,
                        [ref_target.with_suffix(".png"), ref_target.with_suffix(".pdf")], [raw_path])
        save_provenance(soft_target.parent, f"ED3 {rat} Soft Stable", soft_records,
                        [soft_target.with_suffix(".png"), soft_target.with_suffix(".pdf")], [raw_path, campaign_path])
        if rat == "achilles":
            one_d = FIG1 / "D" / "FIGURE_1D_ACHILLES_PAPER_METHODS"
            one_d_soft = FIG1_SOFT / "D" / "FIGURE_1D_ACHILLES_SOFT_STABLE"
            draw(ref_panels, rat, "Figure 1D — paper methods", one_d)
            draw(soft_panels, rat, "Figure 1D — Soft Stable", one_d_soft)
            save_provenance(one_d.parent, "Figure 1D Achilles paper methods", ref_records,
                            [one_d.with_suffix(".png"), one_d.with_suffix(".pdf")], [raw_path])
            save_provenance(one_d_soft.parent, "Figure 1D Achilles Soft Stable", soft_records,
                            [one_d_soft.with_suffix(".png"), one_d_soft.with_suffix(".pdf")], [raw_path, campaign_path])
    print(json.dumps({
        "status": "rendered from existing artifacts; no fits run",
        "ED3_reference_verified": sum(r["status"] == "verified" for r in all_ed3_reference),
        "ED3_reference_unavailable": sum(r["status"] == "unavailable" for r in all_ed3_reference),
        "ED3_soft_verified": sum(r["status"] == "verified" for r in all_ed3_soft),
        "ED3_soft_unavailable": sum(r["status"] == "unavailable" for r in all_ed3_soft),
        "figure_1d_and_ed3_outputs": [relative(ED3), relative(ED3_SOFT), relative(FIG1 / "D"), relative(FIG1_SOFT / "D")],
    }, indent=2))


if __name__ == "__main__":
    main()
