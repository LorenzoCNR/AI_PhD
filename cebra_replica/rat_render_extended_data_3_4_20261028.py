"""Render rat ED3/ED4 from the approved saved embeddings and metric tables."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np

import rat_rebuild_1d_ed3_20261027 as embeddings
import rat_render_figura1_saved as figure1

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ED3 = RESULTS / "figura_ext_3_2026-10-28"
ED4 = RESULTS / "figura_ext_4_2026-10-28"
METRICS = RESULTS / "rat_rebuild_20261027" / "Figure_1E_ED4"
RATS = (("a", "achilles"), ("b", "buddy"), ("c", "cicero"), ("d", "gatsby"))
PAPER = (
    ("CEBRA behavior (position + direction)", "CEBRA — Position + Direction"),
    ("CEBRA time", "CEBRA — Time"),
    ("conv-piVAE q(z|x,u)", "conv-piVAE — Position + Direction — q(z|x,u)"),
    ("conv-piVAE q(z|x)", "conv-piVAE — Position + Direction — q(z|x)"),
    ("UMAP", "UMAP"), ("t-SNE", "t-SNE"), ("PCA", "PCA"),
)
SOFT = tuple(
    (f"Soft {arch} {condition}", f"Soft {label} — {title}")
    for condition, title in (("time_direction_position", "Time + Position + Direction"),
                             ("time_only", "Time"),
                             ("direction_position", "Position + Direction"))
    for arch, label in (("cnn", "CNN"), ("transformer", "Transformer"))
)
METRIC_SPECS = (
    ("train_cross_subject_r2", "Consistency across subjects", "R² (%)"),
    ("train_run_r2", "Consistency across runs", "R² (%)"),
    ("test_total_r2", "Decoding: Position + Direction", "R² (%)"),
    ("test_position_error_m", "Decoding: Position error", "Error (cm)"),
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def save(fig, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    if any(path.exists() for path in paths):
        raise FileExistsError(f"Refusing to overwrite {stem}")
    for path in paths:
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def render_ed3(rat: str, panels: list, stem: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.subplots_adjust(left=.06, right=.88, top=.90, bottom=.10, wspace=.35, hspace=.38)
    norm = Normalize(0, 1.6)
    for ax, panel in zip(axes.flat, panels):
        ax.set_title(panel["title"], fontsize=10)
        if "error" in panel:
            ax.text(.5, .5, "Non disponibile", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        z, labels, valid = panel["z"], panel["behavior"], panel["mask"]
        for index, color in ((2, "cool"), (1, "viridis")):
            mask = valid & (labels[:, index] == 1)
            ax.scatter(z[mask, 0], z[mask, 1], c=labels[mask, 0], cmap=color,
                       norm=norm, s=.7, linewidths=0, rasterized=True)
        ax.set_xlabel("Latent 1")
        ax.set_ylabel("Latent 2")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(labelsize=8)
    for x, label, color in ((.901, "Left", "cool"), (.956, "Right", "viridis")):
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=color),
                           cax=fig.add_axes([x, .23, .014, .52]), ticks=[0, .5, 1, 1.5])
        bar.ax.yaxis.set_ticks_position("left" if label == "Left" else "right")
        bar.ax.tick_params(labelsize=8, pad=2)
        fig.text(x + .007, .20, label, ha="center", va="top", fontsize=9)
    fig.suptitle(f"{rat.capitalize()} — Extended Data Figure 3", fontsize=14)
    return save(fig, stem)


def build_ed3() -> None:
    campaign_path = embeddings.SOFT / "CAMPAIGN_STATE.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    summary = {}
    for letter, rat in RATS:
        rat_dir = ED3 / letter / rat.capitalize()
        raw_path = ROOT / "data" / "rat_hippocampus" / f"{rat}.jl"
        raw = np.asarray(joblib.load(raw_path)["position"], dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
            raise ValueError(f"Invalid behavior data: {raw_path}")
        records = {}
        for family, specs in (("Paper", figure1.paper_specs()), ("Soft", embeddings.SOFT_SPECS)):
            panels, items = figure1.load_panels(rat, raw, sha(raw_path), specs, campaign)
            for item in items:
                if item["status"] == "verified":
                    item["fit_dimension"] = item.get("fit_dimension")
                    item["plotted_dimensions"] = 2
                    item["coordinates"] = [1, 2]
            paths = render_ed3(rat, panels, rat_dir / f"ED3_{rat.upper()}_{family.upper()}")
            records[family] = {"models": items, "outputs": {rel(p): sha(p) for p in paths}}
        write_json(rat_dir / "PROVENANCE.json", {
            "figure": "Extended Data 3", "folder_letter": letter, "rat": rat,
            "training_performed": False, "source_data": rel(raw_path), "source_data_sha256": sha(raw_path),
            "soft_campaign": rel(campaign_path), "soft_campaign_sha256": sha(campaign_path),
            "coordinates_plotted": [1, 2], "groups": records,
        })
        summary[rat] = {family: {"available": sum(m["status"] == "verified" for m in data["models"]),
                                 "total": len(data["models"])} for family, data in records.items()}
        print(f"ED3 {rat}: {summary[rat]}", flush=True)
    write_json(ED3 / "PROVENANCE.json", {"figure": "Extended Data 3", "training_performed": False,
               "renderer": rel(Path(__file__)), "renderer_sha256": sha(Path(__file__)), "rats": summary})
    (ED3 / "RIFERIMENTI.md").write_text("""# Extended Data Figure 3 — riferimenti

Le cartelle `a/Achilles`, `b/Buddy`, `c/Cicero`, `d/Gatsby` identificano i ratti; le lettere sono organizzative. Ogni cartella contiene le tavole Paper e Soft, PNG e PDF, e `PROVENANCE.json` con percorsi e hash dei singoli embedding.

La figura usa gli stessi fit full-recording seed 42 di Figura 1D: CEBRA e Soft nativamente 3D, piVAE e UMAP 2D. Qui si visualizzano le coordinate 1 e 2, come nel notebook ufficiale ED3; per Soft si usa la stessa vista. Barre posizione a destra: Left e Right. Nessun training o nuova trasformazione.

- Renderer: `cebra_replica/rat_render_extended_data_3_4_20261028.py`; lettura e validazione degli embedding: `cebra_replica/rat_rebuild_1d_ed3_20261027.py` e `cebra_replica/rat_render_figura1_saved.py`.
- CEBRA: `results/rat_paper_replication_2023_v1/ED3/cebra/<rat>/{behavior,time}/seed_42/`.
- conv-piVAE, stesso fit con due uscite: `results/rat_paper_replication_2023_v1/ED3/pivae_full_recording/<rat>/seed_42/`.
- Soft Stable: `results/user_soft_rat_fullrecording_t1_3conditions_seed42_v1/<rat>/<condition>/<architecture>/seed_42/`.
- UMAP/t-SNE: `results/rat_paper_replication_2023_v1/ED3/manifold/`. UMAP disponibile per Achilles; t-SNE non disponibile nei fit attesi. AutoLFADS, presente nel paper, non è disponibile nelle stime locali usate.
- Codice ufficiale ED3: https://cebra.ai/docs/cebra-figures/figures/ExtendedDataFigure3.html
""", encoding="utf-8")


def read_ed4():
    audit = json.loads((METRICS / "support_audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise RuntimeError("ED4 support audit did not pass")
    with (METRICS / "ed4_plot_values_v3.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def render_ed4(rat: str, methods: tuple, rows: list, stem: Path) -> tuple[list[Path], list[dict]]:
    fig, axes = plt.subplots(1, 4, figsize=(18, max(5.5, len(methods) * .62)), sharey=True)
    plt.subplots_adjust(left=.29, right=.98, top=.86, bottom=.16, wspace=.35)
    missing = []
    for ax, (metric, title, xlabel) in zip(axes, METRIC_SPECS):
        for i, (method, label) in enumerate(methods):
            selected = [row for row in rows if row["rat"] == rat and row["method"] == method and row["metric"] == metric]
            values = [float(row["value"]) for row in selected]
            if values:
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite ED4 metric: {rat}/{method}/{metric}")
                jitter = np.linspace(-.10, .10, len(values)) if len(values) > 1 else [0]
                ax.scatter(values, np.full(len(values), i) + jitter, s=24, facecolors="none",
                           edgecolors="black", linewidths=.8, zorder=3)
                ax.scatter([np.median(values)], [i], marker="|", s=200, color="#e88525", zorder=4)
            else:
                ax.text(.02, i, "n/a", transform=ax.get_yaxis_transform(), fontsize=8, va="center")
                missing.append({"method": method, "metric": metric})
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=.18)
        ax.set_yticks(range(len(methods)), [item[1] for item in methods], fontsize=9)
        ax.tick_params(axis="y", length=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_ylim(len(methods) - .5, -.5)
    fig.suptitle(f"{rat.capitalize()} — Extended Data Figure 4", fontsize=14)
    return save(fig, stem), missing


def build_ed4() -> None:
    rows = read_ed4()
    expected_metrics = {entry[0] for entry in METRIC_SPECS}
    if not rows or not expected_metrics.issubset({row["metric"] for row in rows}):
        raise ValueError("Missing ED4 metrics")
    inputs = ("ed4_plot_values_v3.csv", "train_cross_subject_consistency.csv", "train_run_consistency.csv",
              "heldout_decoding.csv", "support_audit.json", "manifest.json")
    source_hashes = {rel(METRICS / name): sha(METRICS / name) for name in inputs}
    for letter, rat in RATS:
        rat_dir = ED4 / letter / rat.capitalize()
        groups = {}
        for name, methods in (("Paper", PAPER), ("Soft", SOFT)):
            paths, missing = render_ed4(rat, methods, rows, rat_dir / f"ED4_{rat.upper()}_{name.upper()}")
            groups[name] = {"methods": [m[0] for m in methods], "missing_metric_cells": missing,
                            "outputs": {rel(path): sha(path) for path in paths}}
        write_json(rat_dir / "PROVENANCE.json", {"figure": "Extended Data 4", "rat": rat,
                   "folder_letter": letter, "training_performed": False, "seeds": list(range(42, 48)),
                   "folds": [0, 1, 2], "input_tables": source_hashes, "groups": groups})
        print(f"ED4 {rat}: Paper and Soft", flush=True)
    write_json(ED4 / "PROVENANCE.json", {"figure": "Extended Data 4", "training_performed": False,
               "renderer": rel(Path(__file__)), "renderer_sha256": sha(Path(__file__)),
               "input_tables": source_hashes})
    (ED4 / "RIFERIMENTI.md").write_text("""# Extended Data Figure 4 — riferimenti

Le cartelle `a/Achilles`, `b/Buddy`, `c/Cicero`, `d/Gatsby` identificano i ratti; le lettere sono organizzative. Ogni cartella contiene Paper e Soft separati in PNG/PDF e un manifest con hash e celle non disponibili.

Le quattro colonne rappresentano consistenza train tra soggetti, consistenza train tra run, decoding test congiunto Position + Direction R², ed errore mediano di posizione test. Cerchi: valori dei singoli seed, mediati sui tre fold; linea arancione: mediana. PCA è deterministica e ha un solo valore, non sei run indipendenti. Le celle n/a non sono sostituite: ad esempio q(z|x,u) usa le etichette in inferenza e non è valutato come decoding senza etichette; PCA non ha consistenza across-run indipendente.

- Renderer: `cebra_replica/rat_render_extended_data_3_4_20261028.py`.
- Tabelle e audit: `results/rat_rebuild_20261027/Figure_1E_ED4/ed4_plot_values_v3.csv`, `train_cross_subject_consistency.csv`, `train_run_consistency.csv`, `heldout_decoding.csv`, `support_audit.json`, `manifest.json`.
- Generazione metriche: `cebra_replica/rat_rebuild_1e_ed4_metrics.py`; aggregazione originale dei valori: `cebra_replica/rat_rebuild_1e_ed4_plot.py`.
- Fit CEBRA/piVAE/Soft Stable: `results/rat_paper_fig1e_ed4_6seed_v1/{cebra,pivae,soft}/`; embedding Soft esportati: `results/rat_rebuild_20261027/Figure_1E_ED4/embeddings/soft/`; baseline: `results/rat_manifold_baselines_v1/`.
- Sei seed 42–47 e tre fold, non i dieci seed del paper. Il t-SNE riutilizzato usa perplexity 30. Le matrici di consistenza sono su train; il decoding è su test held-out. La procedura di aggregazione locale non è dichiarata riproduzione bit-per-bit del paper.
- Codice ufficiale ED4: https://cebra.ai/docs/cebra-figures/figures/ExtendedDataFigure4.html
""", encoding="utf-8")


def main() -> None:
    for destination in (ED3, ED4):
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing output directory: {destination}")
    print(f"Starting {datetime.now(timezone.utc).isoformat()} using saved artifacts", flush=True)
    build_ed3()
    build_ed4()
    print(json.dumps({"status": "complete", "ed3": rel(ED3), "ed4": rel(ED4),
                      "training_performed": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()
