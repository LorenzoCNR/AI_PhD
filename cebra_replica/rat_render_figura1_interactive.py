"""Generate rotatable Achilles 3D views from saved embeddings only."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

import rat_rebuild_1d_ed3_20261027 as source
import rat_rebuild_1d_position_scale_20261027 as cebra_renderer
import rat_figure1_soft_achilles_interactive as soft_renderer


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATE_TAG = datetime.now().strftime("%Y-%m-%d")
CEBRA_OUT = RESULTS / f"figura_1_{DATE_TAG}" / "D" / "interactive"
SOFT_OUT = RESULTS / f"figura_1_soft_{DATE_TAG}" / "D" / "interactive"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main():
    rat = "achilles"
    source_path = ROOT / "data" / "rat_hippocampus" / f"{rat}.jl"
    raw = np.asarray(joblib.load(source_path)["position"], dtype=np.float32)
    campaign = json.loads((source.SOFT / "CAMPAIGN_STATE.json").read_text(encoding="utf-8"))
    specs = list(source.PAPER)
    title, *rest = specs[0]
    specs[0] = (title.replace("CEBRA Behaviour", "CEBRA"), *rest)
    panels = []
    for spec in specs:
        try:
            panel = source.read_panel(rat, raw, sha256(source_path), spec, campaign)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            panel = {"title": spec[0], "error": str(exc)}
        panel["dimension"] = spec[4]
        panels.append(panel)

    CEBRA_OUT.mkdir(parents=True, exist_ok=True)
    cebra_records = []
    for panel, slug in ((panels[0], "CEBRA_Achilles_Position_Direction_3D"),
                        (panels[3], "CEBRA_Achilles_Time_3D")):
        if "error" in panel:
            cebra_records.append({"status": "unavailable", "panel": slug, "reason": panel["error"]})
            continue
        target = CEBRA_OUT / f"{slug}.html"
        counts = cebra_renderer.interactive(panel, rat, target)
        cebra_records.append({"status": "verified", "panel": slug, "html": relative(target),
                              "html_sha256": sha256(target), "embedding": panel["record"]["embedding"],
                              "embedding_sha256": panel["record"]["embedding_sha256"],
                              "plotted_rows_by_direction": counts, "plotted_dimension": 3})
    cebra_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "renderer": relative(Path(__file__).resolve()),
        "renderer_sha256": sha256(Path(__file__).resolve()),
        "source_renderer": relative(Path(cebra_renderer.__file__).resolve()),
        "training_performed": False,
        "rat": rat,
        "position_scale_m": [0, 1.6],
        "position_ticks_m": [0, 0.5, 1.0, 1.5],
        "direction_colors": {"Left": "cool", "Right": "viridis"},
        "records": cebra_records,
    }
    (CEBRA_OUT / "PROVENANCE.json").write_text(
        json.dumps(cebra_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    soft_renderer.OUT = SOFT_OUT
    soft_records = []
    for condition, labels in soft_renderer.CONDITIONS.items():
        for architecture in soft_renderer.ARCHITECTURES:
            data_path = soft_renderer.SOURCE / condition / architecture / "seed_42" / "embedding_full.npz"
            soft_records.append(soft_renderer.make_plot(data_path, condition, architecture, labels))
    # Normalize a few mojibake sequences in the inherited HTML title strings.
    for html_path in SOFT_OUT.glob("*.html"):
        html = html_path.read_text(encoding="utf-8")
        html = html.replace("â€”", "—").replace("Â·", "·").replace("Ã—", "×")
        html_path.write_text(html, encoding="utf-8")
    for record in soft_records:
        record["html_sha256"] = sha256(ROOT / record["html"])
    soft_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "renderer": relative(Path(__file__).resolve()),
        "renderer_sha256": sha256(Path(__file__).resolve()),
        "soft_renderer": relative(Path(soft_renderer.__file__).resolve()),
        "training_performed": False,
        "rat": rat,
        "fit_scope": "full recording; plot uses saved valid_mask rows",
        "native_dimension": 3,
        "position_scale_m": [0, 1.6],
        "position_ticks_m": [0, 0.5, 1.0, 1.5],
        "direction_colors": {"Left": "cool", "Right": "viridis"},
        "records": soft_records,
    }
    (SOFT_OUT / "PROVENANCE.json").write_text(
        json.dumps(soft_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"cebra_interactive": sum(r["status"] == "verified" for r in cebra_records),
                      "soft_interactive": len(soft_records), "training_performed": False,
                      "cebra_output": relative(CEBRA_OUT), "soft_output": relative(SOFT_OUT)}, indent=2))


if __name__ == "__main__":
    main()
