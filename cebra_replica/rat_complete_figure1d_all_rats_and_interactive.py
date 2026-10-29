"""Complete Figure 1D static rat panels and Achilles interactive views from saved fits only."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import to_hex

import rat_rebuild_1d_ed3_20261027 as source
import rat_render_figura1_saved as static_renderer


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATE_TAG = datetime.now().strftime("%Y-%m-%d")
PAPER_ROOT = RESULTS / f"figura_1_{DATE_TAG}"
SOFT_ROOT = RESULTS / f"figura_1_soft_{DATE_TAG}"
INTERACTIVE = PAPER_ROOT / "D" / "interactive"
RATS = ("buddy", "cicero", "gatsby")
RAT_FOLDER = {"buddy": "B", "cicero": "C", "gatsby": "D"}
POSITION_TICKS = [0.0, 0.5, 1.0, 1.5]
POSITION_RANGE = [0.0, 1.6]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def plot_2d(panel: dict, output: Path) -> dict:
    if "error" in panel:
        raise ValueError(f"Cannot plot unavailable panel: {panel['title']}: {panel['error']}")
    if output.exists():
        existing = output.read_text(encoding="utf-8", errors="replace")
        if "Plotly.newPlot" not in existing:
            raise RuntimeError(f"Existing HTML is not a valid Plotly figure: {output}")
    else:
        z = panel["z"]
        behavior = panel["behavior"]
        mask = panel["mask"]
        figure = go.Figure()
        for name, col, cmap in (("Left", 2, "cool"), ("Right", 1, "viridis")):
            selected = mask & (behavior[:, col] == 1)
            cm = matplotlib.colormaps[cmap]
            colorscale = [[i / 255, to_hex(cm(i / 255))] for i in range(256)]
            figure.add_trace(go.Scatter(
                x=z[selected, 0], y=z[selected, 1], mode="markers", name=name,
                marker={
                    "size": 3,
                    "color": behavior[selected, 0],
                    "colorscale": colorscale,
                    "cmin": POSITION_RANGE[0],
                    "cmax": POSITION_RANGE[1],
                    "showscale": True,
                    "colorbar": {
                        "title": {"text": f"{name}<br>Position (m)"},
                        "tickmode": "array", "tickvals": POSITION_TICKS,
                        "x": 1.02 if name == "Left" else 1.14,
                        "len": 0.72, "thickness": 12,
                    },
                },
                customdata=behavior[selected, 0],
                hovertemplate=(
                    f"{name}<br>Position: %{{customdata:.3f}} m"
                    "<br>Latent 1: %{x:.4f}<br>Latent 2: %{y:.4f}<extra></extra>"
                ),
            ))
        figure.update_layout(
            title=f"Achilles — {panel['title']}<br><sup>Full-recording fit · seed 42 · native 2D output</sup>",
            xaxis_title="Latent 1", yaxis_title="Latent 2",
            width=1150, height=820,
            margin={"l": 40, "r": 210, "t": 100, "b": 50},
            showlegend=True,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.write_html(str(output), full_html=True, include_plotlyjs=True,
                          config={"responsive": True, "displaylogo": False})
    if "Plotly.newPlot" not in output.read_text(encoding="utf-8", errors="replace"):
        raise RuntimeError(f"Plotly HTML validation failed: {output}")
    return {
        "status": "verified", "title": panel["title"], "dimension": 2,
        "html": relative(output), "html_sha256": sha256(output),
        "embedding": panel["record"]["embedding"],
        "embedding_sha256": panel["record"]["embedding_sha256"],
        "plotted_rows": panel["record"]["plotted_rows"],
        "native_embedding": True,
    }


def main() -> None:
    campaign_path = source.SOFT / "CAMPAIGN_STATE.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    all_static = {"paper": [], "soft": []}

    for rat in RATS:
        data_path = ROOT / "data" / "rat_hippocampus" / f"{rat}.jl"
        raw = np.asarray(joblib.load(data_path)["position"], dtype=np.float32)
        data_hash = sha256(data_path)
        folder_name = RAT_FOLDER[rat]
        paper_panels, paper_records = static_renderer.load_panels(
            rat, raw, data_hash, static_renderer.paper_specs(), campaign)
        soft_panels, soft_records = static_renderer.load_panels(
            rat, raw, data_hash, source.SOFT_SPECS, campaign)

        paper_dir = PAPER_ROOT / "D" / "other_rats" / folder_name
        paper_target = paper_dir / f"FIGURE_1D_{rat.upper()}_PAPER_METHODS"
        static_renderer.draw(paper_panels, rat, "Figure 1D — paper methods", paper_target)
        static_renderer.save_provenance(
            paper_dir, f"Figure 1D {rat} — paper methods", paper_records,
            [paper_target.with_suffix(".png"), paper_target.with_suffix(".pdf")],
            [data_path],
        )
        all_static["paper"].append({
            "rat": rat, "folder": relative(paper_dir),
            "verified_panels": sum(r.get("status") == "verified" for r in paper_records),
            "unavailable_panels": sum(r.get("status") == "unavailable" for r in paper_records),
            "provenance": relative(paper_dir / "PROVENANCE.json"),
        })

        soft_dir = SOFT_ROOT / "D" / "other_rats" / folder_name
        soft_target = soft_dir / f"FIGURE_1D_{rat.upper()}_SOFT_STABLE"
        static_renderer.draw(soft_panels, rat, "Figure 1D — Soft Stable (separate extension)", soft_target)
        static_renderer.save_provenance(
            soft_dir, f"Figure 1D {rat} — Soft Stable", soft_records,
            [soft_target.with_suffix(".png"), soft_target.with_suffix(".pdf")],
            [data_path, campaign_path],
        )
        all_static["soft"].append({
            "rat": rat, "folder": relative(soft_dir),
            "verified_panels": sum(r.get("status") == "verified" for r in soft_records),
            "unavailable_panels": sum(r.get("status") == "unavailable" for r in soft_records),
            "provenance": relative(soft_dir / "PROVENANCE.json"),
        })

    for parent, key in ((PAPER_ROOT / "D", "paper"), (SOFT_ROOT / "D", "soft")):
        parent.mkdir(parents=True, exist_ok=True)
        index = parent / "OTHER_RATS_PROVENANCE.json"
        index.write_text(json.dumps({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "figure": "1D", "scope": "static figures for non-Achilles rats",
            "training_performed": False, "records": all_static[key],
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Add interactive 2D versions of every available Achilles paper-method fit.
    achilles_path = ROOT / "data" / "rat_hippocampus" / "achilles.jl"
    achilles = np.asarray(joblib.load(achilles_path)["position"], dtype=np.float32)
    specs = static_renderer.paper_specs()
    panels, _ = static_renderer.load_panels(
        "achilles", achilles, sha256(achilles_path), specs, campaign)
    INTERACTIVE.mkdir(parents=True, exist_ok=True)
    dynamic_records = []
    for index, filename in (
        (1, "PIVAE_Achilles_Position_Direction_2D.html"),
        (2, "PIVAE_Achilles_No_Labels_at_Inference_2D.html"),
        (5, "UMAP_Achilles_2D.html"),
    ):
        panel = panels[index]
        output = INTERACTIVE / filename
        if "error" in panel:
            dynamic_records.append({"status": "unavailable", "title": panel["title"],
                                    "reason": panel["error"], "dimension": 2})
        else:
            dynamic_records.append(plot_2d(panel, output))

    for filename, title, dimension in (
        ("CEBRA_Achilles_Position_Direction_3D.html", "CEBRA Position + Direction", 3),
        ("CEBRA_Achilles_Time_3D.html", "CEBRA Time", 3),
    ):
        path = INTERACTIVE / filename
        dynamic_records.append({
            "status": "verified" if path.is_file() else "unavailable", "title": title,
            "dimension": dimension, "html": relative(path) if path.is_file() else None,
            "html_sha256": sha256(path) if path.is_file() else None,
            "reason": None if path.is_file() else "interactive HTML not present",
        })

    soft_interactive = SOFT_ROOT / "D" / "interactive"
    for condition, label in (
        ("time_direction_position", "Time + Position + Direction"),
        ("time_only", "Time"), ("direction_position", "Position + Direction"),
    ):
        for architecture in ("cnn", "transformer"):
            filename = f"Achilles_SoftStable_{architecture}_{condition}_3D.html"
            path = soft_interactive / filename
            dynamic_records.append({
                "status": "verified" if path.is_file() else "unavailable",
                "title": f"Soft Stable {architecture.upper()} — {label}",
                "dimension": 3, "html": relative(path) if path.is_file() else None,
                "html_sha256": sha256(path) if path.is_file() else None,
                "reason": None if path.is_file() else "interactive HTML not present",
            })
    tsne_path = (source.ED3 / "manifold" / "tsne" / "achilles" / "window10_flat"
                 / "seed_42" / "embedding.npz")
    dynamic_records.append({
        "status": "unavailable", "title": "t-SNE", "dimension": 2,
        "reason": "No verified saved embedding; no fit launched.",
        "expected_embedding": relative(tsne_path),
    })
    manifest = INTERACTIVE / "ALL_MODELS_PROVENANCE.json"
    manifest.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "figure": "1D", "rat": "achilles", "date_tag": DATE_TAG,
        "training_performed": False,
        "color_encoding": "position (m), range 0–1.6, ticks 0, 0.5, 1.0, 1.5; direction Left=cool, Right=viridis",
        "records": dynamic_records,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "date_tag": DATE_TAG, "training_performed": False,
        "static_other_rats": all_static,
        "achilles_interactive_verified": sum(r.get("status") == "verified" for r in dynamic_records),
        "achilles_interactive_unavailable": sum(r.get("status") == "unavailable" for r in dynamic_records),
        "interactive_manifest": relative(manifest),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
