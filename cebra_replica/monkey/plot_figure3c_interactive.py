"""Create interactive, native-3D Figure 3c views from frozen embeddings only."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "results" / "figura_3_2026-10-29" / "c"
CEBRA = PROJECT / "results/monkey_figure3_paper_v1/stage02_embedding/figure3c_active_position_3d/seed_42"
SOFT = PROJECT / "results/monkey_fig3_soft_descriptive_matched_v1/c_position_active_3d"
POSITION_LIMITS = (-15.0, 15.0)
POSITION_TICKS = tuple(float(x) for x in np.linspace(-15, 15, 7))
POSITION_COLORSCALE = "RdBu_r"  # low=blue, zero=white, high=red, matching the static panels


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT).as_posix()


def provenance_source(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": rel(path), "sha256": sha256(path)}


def load_sources() -> dict[str, dict]:
    result: dict[str, dict] = {}

    cebra_npz = CEBRA / "all_trials.npz"
    cebra_prov = CEBRA / "EMBEDDING_PROVENANCE.json"
    with np.load(cebra_npz, allow_pickle=False) as data:
        result["CEBRA"] = {
            "embedding": data["embedding_raw"].copy(),
            "position_xy": data["position_xy"].copy(),
            "trial_id": data["trial_id"].copy(),
            "time_id": data["time_id"].copy(),
            "sources": [cebra_npz, cebra_prov],
        }

    for arch in ("cnn", "transformer"):
        folder = SOFT / arch / "seed_42"
        embedding_path = folder / "embedding.npz"
        manifest_path = folder / "manifest.json"
        checkpoint_path = folder / "checkpoint.pt"
        history_path = folder / "training_history.csv"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sha256(checkpoint_path) != manifest["checkpoint_sha256"]:
            raise ValueError(f"Soft checkpoint hash mismatch: {checkpoint_path}")
        if sha256(embedding_path) != manifest["embedding_sha256"]:
            raise ValueError(f"Soft embedding hash mismatch: {embedding_path}")
        if manifest["config"]["output_dimension"] != 3 or manifest["config"]["target"] != "position":
            raise ValueError(f"Unexpected Soft Figure 3c fit configuration: {manifest_path}")
        with np.load(embedding_path, allow_pickle=False) as data:
            result[f"Soft {arch.upper()}"] = {
                "embedding": data["embedding"].copy(),
                "position_xy": data["position_xy"].copy(),
                "trial_id": data["trial_id"].copy(),
                "time_id": data["time_id"].copy(),
                "sources": [embedding_path, manifest_path, checkpoint_path, history_path],
            }

    reference = result["CEBRA"]
    expected_rows = len(reference["embedding"])
    if reference["embedding"].shape != (expected_rows, 3):
        raise ValueError("CEBRA embedding is not three-dimensional")
    for name, item in result.items():
        if item["embedding"].shape != (expected_rows, 3):
            raise ValueError(f"Unexpected {name} embedding shape: {item['embedding'].shape}")
        if item["position_xy"].shape != (expected_rows, 2):
            raise ValueError(f"Unexpected {name} position shape: {item['position_xy'].shape}")
        if not np.isfinite(item["embedding"]).all() or not np.isfinite(item["position_xy"]).all():
            raise ValueError(f"Non-finite values in {name} source")
        if not np.array_equal(item["time_id"], reference["time_id"]):
            raise ValueError(f"{name} rows do not align with CEBRA time indices")
        if not np.allclose(item["position_xy"], reference["position_xy"], rtol=0, atol=1e-6):
            raise ValueError(f"{name} position labels do not align with CEBRA rows")
    return result


def build_figure(name: str, item: dict) -> go.Figure:
    embedding = item["embedding"]
    position = item["position_xy"]
    customdata = np.column_stack((position, item["trial_id"], item["time_id"]))
    trace = go.Scatter3d(
        x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2],
        mode="markers",
        marker={
            "size": 1.7,
            "opacity": 0.58,
            "color": position[:, 0],
            "colorscale": POSITION_COLORSCALE,
            "cmin": POSITION_LIMITS[0],
            "cmax": POSITION_LIMITS[1],
            "showscale": True,
            "colorbar": {
                "title": {"text": "X position (cm)"},
                "tickmode": "array",
                "tickvals": POSITION_TICKS,
                "len": 0.78,
            },
        },
        customdata=customdata,
        hovertemplate=(
            "Embedding 1: %{x:.3f}<br>Embedding 2: %{y:.3f}<br>Embedding 3: %{z:.3f}"
            "<br>X position: %{customdata[0]:.2f} cm"
            "<br>Y position: %{customdata[1]:.2f} cm"
            "<br>Trial: %{customdata[2]}<br>Time bin: %{customdata[3]}<extra></extra>"
        ),
        name=name,
    )
    fig = go.Figure(data=[trace])
    fig.update_layout(
        title={"text": f"Figure 3c — {name} — Position X,Y", "x": 0.5},
        scene={
            "xaxis_title": "Embedding coordinate 1",
            "yaxis_title": "Embedding coordinate 2",
            "zaxis_title": "Embedding coordinate 3",
            "aspectmode": "data",
            "xaxis": {"showgrid": True, "zeroline": False},
            "yaxis": {"showgrid": True, "zeroline": False},
            "zaxis": {"showgrid": True, "zeroline": False},
        },
        updatemenus=[{
            "type": "buttons",
            "direction": "right",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.12,
            "yanchor": "top",
            "buttons": [
                {
                    "label": "Color: X position",
                    "method": "restyle",
                    "args": [{
                        "marker.color": [position[:, 0]],
                        "marker.colorbar.title.text": "X position (cm)",
                    }, [0]],
                },
                {
                    "label": "Color: Y position",
                    "method": "restyle",
                    "args": [{
                        "marker.color": [position[:, 1]],
                        "marker.colorbar.title.text": "Y position (cm)",
                    }, [0]],
                },
            ],
        }],
        annotations=[{
            "text": "Full active-trial set · 115,800 time bins · drag to rotate; scroll to zoom",
            "xref": "paper", "yref": "paper", "x": 0.5, "y": -0.08,
            "showarrow": False, "font": {"size": 12},
        }],
        margin={"l": 0, "r": 0, "t": 95, "b": 50},
        template="plotly_white",
    )
    return fig


def write_panel_provenance(presentations: list[dict]) -> None:
    path = OUT / "PROVENANCE.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("figure") != "3c" or record.get("training_or_refitting_performed") is not False:
        raise ValueError(f"Unexpected panel provenance: {path}")
    for presentation in presentations:
        matching = [i for i, old in enumerate(record["presentations"])
                    if old.get("description") == presentation["description"]]
        if matching:
            record["presentations"][matching[0]] = presentation
        else:
            record["presentations"].append(presentation)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources = load_sources()
    presentations = []
    outputs = []
    for name, item in sources.items():
        safe_name = name.replace(" ", "_")
        output = OUT / f"Figure_3c_{safe_name}_Position_XY_interactive.html"
        fig = build_figure(name, item)
        pio.write_html(
            fig,
            file=str(output),
            full_html=True,
            include_plotlyjs=True,
            auto_open=False,
            config={"displaylogo": False, "scrollZoom": True, "responsive": True},
        )
        outputs.append(output)
        presentations.append({
            "description": f"Interactive native-3D embedding: {name}; color selectable by X or Y position",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "sources": [provenance_source(path) for path in item["sources"]],
            "renderers": [provenance_source(Path(__file__))],
            "outputs": [provenance_source(output)],
            "plotted_support": "All 115,800 active-recording time bins; no retraining or embedding transform",
            "embedding_dimension": 3,
            "color_scale": {"limits_cm": POSITION_LIMITS, "ticks_cm": POSITION_TICKS},
            "interactive_controls": ["3D camera rotate/zoom", "X-position or Y-position color toggle", "hover trial/time/position"],
        })
    write_panel_provenance(presentations)
    index_path = OUT.parent / "FIGURE_3_INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["interactive_panel_outputs"] = {
        "3c": [rel(path) for path in outputs],
    }
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "outputs": [rel(path) for path in outputs]}, indent=2))


if __name__ == "__main__":
    main()
