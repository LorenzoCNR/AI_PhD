"""Create rotatable Figure 1D views for existing Achilles Soft fits only."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import plotly
import plotly.graph_objects as go
import matplotlib
from matplotlib.colors import to_hex


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "user_soft_rat_fullrecording_t1_3conditions_seed42_v1" / "achilles"
DATE_TAG = datetime.now().strftime("%Y-%m-%d")
OUT = ROOT / "results" / f"figura_1_{DATE_TAG}" / "d" / "Achilles" / "interactive"
TICKS = [0.0, 0.5, 1.0, 1.5]
TICK_TEXT = ["0", "0.5", "1.0", "1.5"]
CONDITIONS = {
    "time_only": "Time",
    "direction_position": "Position + Direction",
    "time_direction_position": "Time + Position + Direction",
}
ARCHITECTURES = ("cnn", "transformer")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def make_plot(data_path: Path, condition: str, architecture: str, labels: str, overwrite=False) -> dict:
    with np.load(data_path, allow_pickle=False) as saved:
        required = {"embedding", "behavior", "source_id", "valid_mask", "rat", "condition", "architecture", "seed", "fit_scope"}
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"{data_path}: missing {sorted(missing)}")
        z = np.asarray(saved["embedding"], dtype=np.float32)
        behavior = np.asarray(saved["behavior"], dtype=np.float32)
        source_id = np.asarray(saved["source_id"])
        valid = np.asarray(saved["valid_mask"], dtype=bool)
        meta = {
            "rat": str(saved["rat"].item()),
            "condition": str(saved["condition"].item()),
            "architecture": str(saved["architecture"].item()),
            "seed": int(saved["seed"].item()),
            "fit_scope": str(saved["fit_scope"].item()),
        }

    if meta != {"rat": "achilles", "condition": condition, "architecture": architecture, "seed": 42, "fit_scope": "full_recording"}:
        raise ValueError(f"Fit identity/scope mismatch: {data_path} -> {meta}")
    if z.ndim != 2 or z.shape[1] != 3 or behavior.shape != (len(z), 3) or source_id.shape != (len(z),) or valid.shape != (len(z),):
        raise ValueError(f"Invalid aligned arrays in {data_path}")
    if not valid.any() or not np.isfinite(z[valid]).all() or not np.isfinite(behavior[valid]).all():
        raise ValueError(f"Invalid plotted values in {data_path}")
    if behavior[valid, 0].min() < -1e-6 or behavior[valid, 0].max() > 1.6 + 1e-6:
        raise ValueError(f"Position outside [0, 1.6] m in {data_path}")
    left = valid & (behavior[:, 2] == 1)
    right = valid & (behavior[:, 1] == 1)
    if np.any(left & right) or int(left.sum() + right.sum()) != int(valid.sum()):
        raise ValueError(f"Direction labels do not partition valid rows in {data_path}")

    fig = go.Figure()
    for name, select, cmap, colorbar_x in (
        ("Left", left, "cool", 1.02),
        ("Right", right, "viridis", 1.13),
    ):
        cmap_object = matplotlib.colormaps[cmap]
        colorscale = [[index / 255, to_hex(cmap_object(index / 255))] for index in range(256)]
        fig.add_trace(go.Scatter3d(
            x=z[select, 0], y=z[select, 1], z=z[select, 2],
            mode="markers", name=name,
            marker={
                "size": 2,
                "color": behavior[select, 0],
                "colorscale": colorscale,
                "cmin": 0.0,
                "cmax": 1.6,
                "opacity": 1.0,
                "showscale": True,
                "colorbar": {
                    "title": {"text": name, "side": "bottom"},
                    "x": colorbar_x,
                    "len": 0.66,
                    "thickness": 12,
                    "tickmode": "array",
                    "tickvals": TICKS,
                    "ticktext": TICK_TEXT,
                },
            },
            hovertemplate=(
                f"{name}<br>Position: %{{marker.color:.3f}} m"
                "<br>Latent 1: %{x:.3f}<br>Latent 2: %{y:.3f}"
                "<br>Latent 3: %{z:.3f}<extra></extra>"
            ),
        ))

    architecture_label = {"cnn": "CNN", "transformer": "Transformer"}[architecture]
    title = f"Soft {architecture_label} — {labels}"
    fig.update_layout(
        title=f"Achilles — {title}",
        scene={
            "xaxis_title": "Latent 1",
            "yaxis_title": "Latent 2",
            "zaxis_title": "Latent 3",
            "aspectmode": "cube",
        },
        width=1150,
        height=820,
        margin={"l": 0, "r": 190, "t": 105, "b": 0},
        showlegend=True,
        legend={"x": 0.02, "y": 0.98},
    )
    for trace in fig.data:
        if list(trace.marker.colorbar.tickvals) != TICKS or trace.marker.cmin != 0 or trace.marker.cmax != 1.6:
            raise AssertionError("Color scale configuration mismatch")

    OUT.mkdir(parents=True, exist_ok=True)
    condition_slug = {
        "time_only": "Time",
        "direction_position": "Position_Direction",
        "time_direction_position": "Time_Position_Direction",
    }[condition]
    target = OUT / f"Soft_Achilles_{architecture_label}_{condition_slug}.html"
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {target}")
    fig.write_html(str(target), full_html=True, include_plotlyjs=True,
                   config={"responsive": True, "displaylogo": False})
    if target.stat().st_size < 100_000 or "Plotly.newPlot" not in target.read_text(encoding="utf-8"):
        raise RuntimeError(f"Invalid interactive HTML: {target}")

    return {
        **meta,
        "labels_used_in_fit": labels,
        "native_dimension": 3,
        "plotted_dimension": 3,
        "representation": "embedding (saved model output)",
        "row_support": "valid_mask true; 10-bin window interior",
        "recording_rows": int(len(z)),
        "plotted_rows": int(valid.sum()),
        "left_rows": int(left.sum()),
        "right_rows": int(right.sum()),
        "source_npz": str(data_path.relative_to(ROOT)),
        "source_npz_sha256": sha256(data_path),
        "html": str(target.relative_to(ROOT)),
        "html_sha256": sha256(target),
        "output_dimension": 3,
        "position_unit": "m",
        "colorbar_range_m": [0, 1.6],
        "colorbar_ticks_m": TICKS,
    }


def main() -> None:
    records = []
    for condition, labels in CONDITIONS.items():
        for architecture in ARCHITECTURES:
            path = SOURCE / condition / architecture / "seed_42" / "embedding_full.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            records.append(make_plot(path, condition, architecture, labels))

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__).resolve()),
        "plotly_version": plotly.__version__,
        "training_performed": False,
        "position_scale": {"unit": "m", "min": 0.0, "max": 1.6, "ticks": TICKS},
        "conditions": CONDITIONS,
        "architectures": list(ARCHITECTURES),
        "records": records,
    }
    out_manifest = OUT / "MANIFEST_SOFT_ACHILLES_3D.json"
    out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    readme = OUT / "README.md"
    readme.write_text(
        "# Soft — Achilles, Figura 1D, viste 3D ruotabili\n\n"
        "Sei HTML interattivi ricavati dagli embedding Stable già salvati: tre condizioni di etichette × CNN/Transformer. "
        "Ogni fit usa l'intero recording, seed 42, output nativo 3D; il grafico mostra soltanto le righe `valid_mask` "
        "(centri con finestra temporale valida). Colore = posizione in metri, scala comune 0–1,6 con tacche 0, 0,5, 1,0, 1,5; "
        "Left usa `cool`, Right usa `viridis`. I grafici si ruotano trascinando il mouse. Nessun training è stato eseguito.\n\n"
        "La cartella è separata dai pannelli CEBRA e dalle figure statiche. Provenienza e hash sono nel manifest.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "interactive_html": len(records), "manifest": str(out_manifest)}, indent=2))


if __name__ == "__main__":
    main()
