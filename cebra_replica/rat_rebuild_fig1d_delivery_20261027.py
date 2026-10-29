"""Rebuild Figure 1D paper and Soft plots from verified saved embeddings only."""
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
import rat_rebuild_1d_position_scale_20261027 as cebra_renderer
import rat_figure1_soft_achilles_interactive as soft_renderer


ROOT = Path(__file__).resolve().parents[1]
FIG1 = ROOT / "results" / f"figura_1_{datetime.now().strftime('%Y-%m-%d')}"
PANEL_D = FIG1 / "d"
RATS = ("achilles", "buddy", "cicero", "gatsby")
POSITION_RANGE = (0.0, 1.6)
POSITION_TICKS = (0.0, 0.5, 1.0, 1.5)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_panels(rat: str, raw: np.ndarray, data_path: Path, campaign: dict, specs):
    panels, records = [], []
    data_hash = sha256(data_path)
    for spec in specs:
        try:
            panel = source.read_panel(rat, raw, data_hash, spec, campaign)
            record = {"status": "verified", **panel["record"]}
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            panel = {"title": spec[0], "error": str(exc)}
            record = {
                "status": "unavailable", "rat": rat, "title": spec[0].replace("\n", " "),
                "expected_dimension": spec[4], "reason": str(exc),
            }
        panel["dimension"] = spec[4]
        panels.append(panel)
        records.append(record)
    return panels, records


def render_static(rat: str, rat_dir: Path, raw: np.ndarray, data_path: Path,
                  campaign: dict, paper_specs, soft_specs):
    paper_panels, paper_records = load_panels(rat, raw, data_path, campaign, paper_specs)
    soft_panels, soft_records = load_panels(rat, raw, data_path, campaign, soft_specs)
    paper_target = rat_dir / f"FIGURE_1D_{rat.upper()}_PAPER_METHODS"
    soft_target = rat_dir / f"FIGURE_1D_{rat.upper()}_SOFT"
    static_renderer.draw(paper_panels, rat, "Figure 1D", paper_target, overwrite=True)
    static_renderer.draw(soft_panels, rat, "Figure 1D — Soft", soft_target, overwrite=True)
    return {
        "paper": paper_records,
        "soft": soft_records,
        "outputs": [paper_target.with_suffix(".png"), paper_target.with_suffix(".pdf"),
                    soft_target.with_suffix(".png"), soft_target.with_suffix(".pdf")],
    }


def render_2d(panel: dict, target: Path) -> dict:
    z = panel["z"]
    behavior = panel["behavior"]
    mask = panel["mask"]
    figure = go.Figure()
    for name, column, cmap, x in (("Left", 2, "cool", 1.02), ("Right", 1, "viridis", 1.15)):
        selected = mask & (behavior[:, column] == 1)
        color_map = matplotlib.colormaps[cmap]
        colorscale = [[index / 255, to_hex(color_map(index / 255))] for index in range(256)]
        figure.add_trace(go.Scatter(
            x=z[selected, 0], y=z[selected, 1], mode="markers", name=name,
            marker={
                "size": 3, "color": behavior[selected, 0], "colorscale": colorscale,
                "cmin": POSITION_RANGE[0], "cmax": POSITION_RANGE[1], "showscale": True,
                "colorbar": {
                    "title": {"text": name, "side": "bottom"},
                    "tickmode": "array", "tickvals": POSITION_TICKS,
                    "x": x, "len": 0.72, "thickness": 12,
                },
            },
            customdata=behavior[selected, 0],
            hovertemplate=(f"{name}<br>Position: %{{customdata:.3f}} m"
                           "<br>Latent 1: %{x:.4f}<br>Latent 2: %{y:.4f}<extra></extra>"),
        ))
    short_title = panel["title"].replace("\n", " — ")
    figure.update_layout(
        title=f"Achilles — {short_title}",
        xaxis_title="Latent 1", yaxis_title="Latent 2",
        width=1150, height=820,
        margin={"l": 40, "r": 230, "t": 75, "b": 45},
        showlegend=True,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(target), full_html=True, include_plotlyjs=True,
                      config={"responsive": True, "displaylogo": False})
    html = target.read_text(encoding="utf-8", errors="replace")
    if "Plotly.newPlot" not in html or target.stat().st_size < 100_000:
        raise RuntimeError(f"Invalid interactive Plotly output: {target}")
    return {
        "status": "verified", "title": short_title,
        "html": relative(target), "html_sha256": sha256(target),
        "embedding": panel["record"]["embedding"],
        "embedding_sha256": panel["record"]["embedding_sha256"],
        "plotted_rows": panel["record"]["plotted_rows"],
    }


def render_achilles_interactive(rat_dir: Path, raw: np.ndarray, data_path: Path,
                                campaign: dict, paper_specs, soft_specs):
    interactive_dir = rat_dir / "interactive"
    interactive_dir.mkdir(parents=True, exist_ok=True)
    paper_panels, _ = load_panels("achilles", raw, data_path, campaign, paper_specs)
    records = []

    for index, model_name, filename in (
        (0, "CEBRA — Position + Direction", "CEBRA_Achilles_Position_Direction.html"),
        (3, "CEBRA — Time", "CEBRA_Achilles_Time.html"),
    ):
        panel = paper_panels[index]
        if "error" in panel:
            records.append({"status": "unavailable", "title": model_name, "reason": panel["error"]})
            continue
        target = interactive_dir / filename
        counts = cebra_renderer.interactive(panel, "achilles", target, overwrite=True)
        records.append({"status": "verified", "title": model_name, "html": relative(target),
                        "html_sha256": sha256(target), "plotted_rows_by_direction": counts,
                        "embedding_sha256": panel["record"]["embedding_sha256"]})

    for index, title, filename in (
        (1, "conv-piVAE — Position + Direction (q(z|x,u))",
         "PIVAE_Achilles_Position_Direction_q_z_x_u.html"),
        (2, "conv-piVAE — Position + Direction (q(z|x))",
         "PIVAE_Achilles_Position_Direction_q_z_x.html"),
        (4, "t-SNE", "TSNE_Achilles.html"),
        (5, "UMAP", "UMAP_Achilles.html"),
    ):
        panel = paper_panels[index]
        if "error" in panel:
            record = {"status": "unavailable", "title": title, "reason": panel["error"]}
            records.append(record)
        else:
            records.append(render_2d(panel, interactive_dir / filename))

    soft_renderer.OUT = interactive_dir
    soft_records = []
    for spec in soft_specs:
        _, _, (condition, architecture), _, _ = spec
        labels = next(label for key, label in soft_renderer.CONDITIONS.items() if key == condition)
        architecture_label = {"cnn": "CNN", "transformer": "Transformer"}[architecture]
        data_path_soft = (soft_renderer.SOURCE / condition / architecture / "seed_42" / "embedding_full.npz")
        record = soft_renderer.make_plot(data_path_soft, condition, architecture, labels, overwrite=True)
        soft_records.append(record)
        records.append({"status": "verified", "title": f"Soft {architecture_label} — {labels}",
                        "html": record["html"], "html_sha256": record["html_sha256"]})

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "figure": "1D", "rat": "achilles", "training_performed": False,
        "color_encoding": {"position_unit": "m", "range": list(POSITION_RANGE),
                           "ticks": list(POSITION_TICKS), "Left": "cool", "Right": "viridis"},
        "records": records,
    }
    manifest_path = interactive_dir / "ALL_MODELS_PROVENANCE.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    soft_manifest_path = interactive_dir / "PROVENANCE_SOFT.json"
    soft_manifest_path.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rat": "achilles", "training_performed": False,
        "position_scale_m": {"min": 0.0, "max": 1.6, "ticks": list(POSITION_TICKS)},
        "records": soft_records,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return records, manifest_path


def main():
    campaign_path = source.SOFT / "CAMPAIGN_STATE.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    paper_specs = static_renderer.paper_specs()
    soft_specs = source.SOFT_SPECS
    PANEL_D.mkdir(parents=True, exist_ok=True)
    rat_status = {}
    for rat in RATS:
        data_path = ROOT / "data" / "rat_hippocampus" / f"{rat}.jl"
        raw = np.asarray(joblib.load(data_path)["position"], dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
            raise ValueError(f"Invalid position data for {rat}: {data_path}")
        rat_dir = PANEL_D / rat.capitalize()
        rat_dir.mkdir(parents=True, exist_ok=True)
        rendered = render_static(rat, rat_dir, raw, data_path, campaign, paper_specs, soft_specs)
        combined_records = {key: value for key, value in rendered.items() if key != "outputs"}
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "figure": "1D", "rat": rat, "training_performed": False,
            "source_data": relative(data_path), "source_data_sha256": sha256(data_path),
            "campaign_state": relative(campaign_path), "campaign_state_sha256": sha256(campaign_path),
            "panels": combined_records,
            "outputs": {relative(path): sha256(path) for path in rendered["outputs"]},
        }
        (rat_dir / "PROVENANCE.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        rat_status[rat] = {
            "folder": relative(rat_dir),
            "paper_available": sum(item["status"] == "verified" for item in rendered["paper"]),
            "paper_total": len(rendered["paper"]),
            "soft_available": sum(item["status"] == "verified" for item in rendered["soft"]),
            "soft_total": len(rendered["soft"]),
        }

    achilles_path = ROOT / "data" / "rat_hippocampus" / "achilles.jl"
    achilles_raw = np.asarray(joblib.load(achilles_path)["position"], dtype=np.float32)
    dynamic_records, dynamic_manifest = render_achilles_interactive(
        PANEL_D / "Achilles", achilles_raw, achilles_path, campaign, paper_specs, soft_specs)
    panel_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "figure": "1D", "panel": "d", "training_performed": False,
        "rat_folders": rat_status,
        "achilles_interactive_manifest": relative(dynamic_manifest),
        "achilles_interactive_verified": sum(item["status"] == "verified" for item in dynamic_records),
        "achilles_interactive_unavailable": sum(item["status"] == "unavailable" for item in dynamic_records),
    }
    (PANEL_D / "PROVENANCE.json").write_text(
        json.dumps(panel_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    index = """# Figura 1D — output

Cartella del pannello `d`, con una sottocartella per ratto. Ogni ratto contiene le tavole statiche Paper e Soft e un `PROVENANCE.json`.

| Ratto | Paper statico | Soft statico |
|---|---|---|
"""
    for rat in RATS:
        name = rat.capitalize()
        index += f"| {name} | `{name}/FIGURE_1D_{rat.upper()}_PAPER_METHODS.png/.pdf` | `{name}/FIGURE_1D_{rat.upper()}_SOFT.png/.pdf` |\n"
    index += "\nGli HTML ruotabili di Achilles sono in `Achilles/interactive/`. Titoli: modello ed etichette usate; Left/Right sotto le barre colore.\n"
    index += "\nSpecifiche complete Paper e Soft: [`SPECIFICHE_MODELLI.md`](SPECIFICHE_MODELLI.md).\n"
    (PANEL_D / "INDICE.md").write_text(index, encoding="utf-8")
    print(json.dumps({
        "panel_folder": relative(PANEL_D), "rat_status": rat_status,
        "achilles_interactive_verified": sum(item["status"] == "verified" for item in dynamic_records),
        "achilles_interactive_unavailable": sum(item["status"] == "unavailable" for item in dynamic_records),
        "training_performed": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
