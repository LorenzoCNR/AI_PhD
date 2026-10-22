"""Rebuild only rat panels 1D, 2C, 2D and Extended Data 4 from frozen artifacts.

No model fitting is performed. Outputs are first written to a staging directory;
--promote replaces only the obsolete figure files in the four requested panels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cebra_replica"))
from rat_soft_stable_figures import RATS, SOURCES, DISPLAY, embedding, csv_rows, sha
from rat_figure_gap_closure_analysis import index_manifests, load_fit, method_specs

DELIVERY = ROOT / "results" / "RAT_FIGURES"
STAGE = DELIVERY / ".stage_rat_final_figures_20261015"
PAPER = DELIVERY / "Reference_components"
BASE_LOSS = ROOT / "results" / "rat_figure_gap_closure_v1" / "figure2c_loss" / "FIGURE2C_LOSS_LONG.csv"
TABLES = DELIVERY / "Tables"
STAMP = datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_figure(fig, base: Path, panel: str, parents: list[dict], details: dict) -> None:
    import matplotlib.pyplot as plt
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(base.with_suffix("." + ext), dpi=300 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    record = {"created_utc": STAMP, "panel": panel, "script_sha256": digest(Path(__file__)),
              "parents": parents, "files": {e: digest(base.with_suffix("." + e)) for e in ("png", "pdf")}}
    record.update(details)
    base.with_name(base.name + "_PROVENANCE.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def final_figure1d(index: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    specs = [s for s in method_specs(index) if s["available"] and
             s["display"] != "Conv-piVAE (test labels)"]
    if sum(s["kind"] == "pivae" for s in specs) != 1:
        raise RuntimeError("Figure 1D must contain exactly one Conv-piVAE output: q(z|x)")
    expected = {"CEBRA-Behavior", "CEBRA-Time", "Conv-piVAE (no test labels)", "PCA", "UMAP", "t-SNE"}
    if not expected.issubset({s["display"] for s in specs}):
        raise RuntimeError("A frozen reference embedding is missing for Figure 1D")
    soft_keys = [("CNN", "joint"), ("CNN", "time_only"),
                 ("Transformer", "joint"), ("Transformer", "time_only")]
    rows = []
    for rat in RATS:
        reference = []
        parents = []
        for spec in specs:
            d = load_fit(index, spec["kind"], spec["method"], rat, 0, spec["seed"], "test",
                         spec.get("variant", "raw"))
            reference.append((spec["display"], d["embedding"][d["valid_mask"]],
                              d["behavior"][d["valid_mask"]], d["source_id"][d["valid_mask"]],
                              d["manifest_path"], d["embedding_sha256"]))
        soft = []
        for key in soft_keys:
            item = embedding(*key, rat, seed=42)
            if item is None:
                continue
            arr, parent = item
            valid = arr["valid_mask"].astype(bool)
            soft.append((DISPLAY[key].replace("\n", " "), arr["embedding_raw"][valid],
                         arr["behavior"][valid], arr["source_id"][valid],
                         ROOT / parent["manifest"], parent["sha256"]))
        if len(soft) != 4:
            raise RuntimeError(f"Expected all four frozen Soft embeddings for {rat}, found {len(soft)}")
        all_data = reference + soft
        support = all_data[0][3]
        behavior = all_data[0][2]
        for label, emb, beh, ids, manifest, emb_hash in all_data:
            if not np.array_equal(ids, support) or not np.array_equal(beh, behavior):
                raise RuntimeError(f"Held-out rows/behavior differ across Figure 1D methods: {rat}/{label}")
            parents.append({"path": manifest.relative_to(ROOT).as_posix(), "sha256": digest(manifest),
                            "embedding_sha256": emb_hash})
            rows.append({"rat": rat, "method": label, "split": "held_out_test", "seed": 42,
                         "fold": 0, "n_samples": len(ids), "fit_dimension": int(emb.shape[1]),
                         "plotted_dimensions": 2, "representation": "embedding_raw or q(z|x)"})
        pos = behavior[:, 0]
        norm = Normalize(float(np.nanmin(pos)), float(np.nanmax(pos)))
        fig = plt.figure(figsize=(22, 8.4), dpi=300)
        gs = fig.add_gridspec(2, 12, left=0.035, right=0.99, top=0.94, bottom=0.14,
                              hspace=0.18, wspace=0.08)
        for i, (label, emb, beh, *_rest) in enumerate(reference):
            ax = fig.add_subplot(gs[0, 2*i:2*i+2])
            left, right = beh[:, 2] == 1, beh[:, 1] == 1
            ax.scatter(emb[left, 0], emb[left, 1], c=pos[left], cmap="cool", norm=norm, s=2, linewidths=0,
                       rasterized=True)
            ax.scatter(emb[right, 0], emb[right, 1], c=pos[right], cmap="viridis", norm=norm, s=2, linewidths=0,
                       rasterized=True)
            ax.set_title("Conv-piVAE q(z|x)" if label == "Conv-piVAE (no test labels)" else label, fontsize=10)
            ax.set_axis_off()
        for i, (label, emb, beh, *_rest) in enumerate(soft):
            ax = fig.add_subplot(gs[1, 2+2*i:2+2*(i+1)])
            left, right = beh[:, 2] == 1, beh[:, 1] == 1
            ax.scatter(emb[left, 0], emb[left, 1], c=pos[left], cmap="cool", norm=norm, s=2, linewidths=0,
                       rasterized=True)
            ax.scatter(emb[right, 0], emb[right, 1], c=pos[right], cmap="viridis", norm=norm, s=2, linewidths=0,
                       rasterized=True)
            ax.set_title(label, fontsize=10, color="red")
            ax.set_axis_off()
        for bounds, cmap, label in (([0.36, 0.065, 0.13, 0.018], "cool", "Left"),
                                    ([0.53, 0.065, 0.13, 0.018], "viridis", "Right")):
            cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes(bounds), orientation="horizontal")
            cb.set_label(label, fontsize=9)
            cb.set_ticks([])
        base = STAGE / "Figure_1" / "1D" / f"FIGURE1D_{rat.upper()}"
        write_figure(fig, base, "1D", parents, {"rat": rat, "split": "held-out test",
             "pivae": "single behavior-trained Conv-piVAE output q(z|x); behavior not supplied at inference",
             "plot": "first two embedding coordinates; legend labels are Left and Right only"})
    write_csv(STAGE / "Figure_1" / "1D" / "FIGURE1D_PANEL_INDEX.csv", rows)


def final_figure2c() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cebra_rows = csv_rows(BASE_LOSS)
    conditions = tuple(dict.fromkeys(r["condition"] for r in cebra_rows))
    required = {"position", "direction", "position_direction", "position_shuffled",
                "direction_shuffled", "position_direction_shuffled"}
    if not required.issubset(set(conditions)):
        raise RuntimeError(f"Figure 2C reference loss conditions incomplete: {conditions}")
    names = {"position": "Position", "direction": "Direction", "position_direction": "Position + direction",
             "position_shuffled": "Position shuffle", "direction_shuffled": "Direction shuffle",
             "position_direction_shuffled": "Position + direction shuffle"}
    colors = {"position": "#4477AA", "direction": "#EE6677", "position_direction": "#228833",
              "position_shuffled": "#AABBDD", "direction_shuffled": "#EEAABB",
              "position_direction_shuffled": "#99CCAA"}
    all_rows = []
    parents = [{"path": BASE_LOSS.relative_to(ROOT).as_posix(), "sha256": digest(BASE_LOSS)}]
    ordered_conditions = ("position", "direction", "position_direction", "position_shuffled",
                         "direction_shuffled", "position_direction_shuffled")
    for rat in RATS:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4), dpi=250)
        for condition in ordered_conditions:
            selected = [r for r in cebra_rows if r["rat"] == rat and r["condition"] == condition]
            selected.sort(key=lambda r: int(r["optimizer_update"]))
            if not selected:
                raise RuntimeError(f"Missing CEBRA loss history: {rat}/{condition}")
            axes[0].plot([int(r["optimizer_update"]) for r in selected],
                         [float(r["training_objective"]) for r in selected],
                         color=colors[condition], lw=1.15, label=names[condition])
            for r in selected:
                all_rows.append({"rat": rat, "method": "CEBRA reference", "condition": condition,
                                 "seed": r["seed"], "update": r["optimizer_update"],
                                 "split": "training", "loss": r["training_objective"]})
        axes[0].set_title("CEBRA reference (including shuffle)")
        axes[0].set_ylabel("Training objective")
        axes[0].set_xlabel("Optimizer update")
        axes[0].legend(frameon=False, fontsize=7)
        for key, source in SOURCES.items():
            path = source / "stage01_fit" / rat / "fold_0" / "soft_v1_stable" / "seed_42" / "manifest.json"
            if not path.is_file() and key == ("Transformer", "joint"):
                path = ROOT / "results" / "user_soft_rat_transformer_v1" / "stage01_fit" / rat / "fold_0" / "soft_v1_stable" / "seed_42" / "manifest.json"
            if not path.is_file():
                raise RuntimeError(f"Missing frozen Soft history: {path}")
            doc = json.loads(path.read_text(encoding="utf-8"))
            history = doc.get("history", [])
            if not history or "train_loss" not in history[0]:
                raise RuntimeError(f"Soft training history is absent/incomplete: {path}")
            updates = [int(h["update"]) for h in history]
            losses = [float(h["train_loss"]) for h in history]
            axes[1].plot(updates, losses, lw=1.25, label=DISPLAY[key].replace("\n", " "))
            for u, loss in zip(updates, losses):
                all_rows.append({"rat": rat, "method": DISPLAY[key].replace("\n", " "),
                                 "condition": key[1], "architecture": key[0], "seed": 42,
                                 "fold": 0, "update": u, "split": "training", "loss": loss})
            parents.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)})
        axes[1].set_title("Soft models")
        axes[1].set_xlabel("Optimizer update")
        axes[1].legend(frameon=False, fontsize=8)
        for ax in axes:
            ax.grid(alpha=0.22)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        base = STAGE / "Figure_2" / "2C" / f"FIGURE2C_{rat.upper()}"
        write_figure(fig, base, "2C", parents, {"rat": rat, "split": "training histories",
             "comparison": "reference and Soft losses are in separate axes; objective scales are not equated",
             "reference_conditions": sorted(required), "soft_seed": 42, "soft_fold": 0})
    write_csv(STAGE / "Figure_2" / "2C" / "FIGURE2C_TRAINING_LOSSES_LONG.csv", all_rows)


def final_figure2d_and_ed4() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline_path = PAPER / "Figure_2" / "Figure_2D" / "FIGURE2D_METHOD_METRICS_SEED_LEVEL.csv"
    soft_path = TABLES / "SOFT_DECODING_BY_SEED.csv"
    baseline = [r for r in csv_rows(baseline_path) if r.get("split") == "test" and
                r.get("method") != "Conv-piVAE-test-labels"]
    soft = [r for r in csv_rows(soft_path) if r.get("representation") == "embedding_raw" and r.get("split") == "test"]
    ref_methods = ("CEBRA-Behavior-32D", "Conv-piVAE-z_mean", "CEBRA-Time-32D", "PCA", "UMAP", "OPENTSNE")
    display = {"CEBRA-Behavior-32D": "CEBRA-Behavior", "Conv-piVAE-z_mean": "Conv-piVAE q(z|x)",
               "CEBRA-Time-32D": "CEBRA-Time", "PCA": "PCA", "UMAP": "UMAP", "OPENTSNE": "t-SNE"}
    base_rows = []
    for r in baseline:
        if r["method"] not in ref_methods:
            continue
        base_rows.append({**r, "method_display": display[r["method"]], "decoder": "kNN"})
    soft_rows = []
    for r in soft:
        if r.get("metric") not in ("position_median_absolute_error_cm", "position_r2", "direction_balanced_accuracy"):
            continue
        arch, cond, decoder = r["architecture"], r["condition"], r["decoder"]
        soft_rows.append({**r, "method_display": f"{r['method']} {cond} {decoder}", "seed": r["seed"]})
    decoding = base_rows + soft_rows
    if not decoding:
        raise RuntimeError("No held-out decoding rows available")
    write_csv(STAGE / "Figure_2" / "2D" / "FIGURE2D_METRICS_BY_SEED.csv", decoding)
    metrics = ("position_median_absolute_error_cm", "position_r2", "direction_balanced_accuracy")
    labels = {"position_median_absolute_error_cm": "Median absolute position error (cm)",
              "position_r2": "Position R²", "direction_balanced_accuracy": "Direction balanced accuracy"}
    fig, axes = plt.subplots(4, 3, figsize=(18, 13), squeeze=False, dpi=230)
    for ri, rat in enumerate(RATS):
        for ci, metric in enumerate(metrics):
            ax = axes[ri, ci]
            methods = list(dict.fromkeys(r["method_display"] for r in decoding if r["rat"] == rat and r["metric"] == metric))
            for yi, method in enumerate(methods):
                vals = [float(r["mean_across_folds"]) for r in decoding if r["rat"] == rat and
                        r["metric"] == metric and r["method_display"] == method]
                is_soft = method.startswith("Soft")
                ax.scatter(vals, np.full(len(vals), yi), s=15, color="#c62828" if is_soft else "black",
                           alpha=0.78, edgecolors="white", linewidths=.25, zorder=3)
                if vals:
                    med = float(np.median(vals))
                    ax.plot([med, med], [yi-.22, yi+.22], color="#e88d25", lw=2.4, zorder=4)
            if ci == 0:
                ax.set_yticks(range(len(methods)), methods, fontsize=7)
                for tick, label in zip(ax.get_yticklabels(), methods):
                    if label.startswith("Soft"):
                        tick.set_color("#c62828")
            else:
                ax.set_yticks(range(len(methods)), [])
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if ri == 0:
                ax.set_title(labels[metric])
            if ci == 0:
                ax.set_ylabel(rat.capitalize())
            if ri == 3:
                ax.set_xlabel(labels[metric])
    fig.tight_layout()
    write_figure(fig, STAGE / "Figure_2" / "2D" / "FIGURE2D_METHOD_COMPARISON", "2D",
                 [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in (baseline_path, soft_path)],
                 {"split": "held-out test", "metrics": list(metrics),
                  "decoders": "reference kNN; Soft kNN + Ridge for position, kNN classifier for direction",
                  "test_role": "evaluation only; no selection"})

    ref_ed_path = DELIVERY / "Extended_Data" / "ED4" / "REFERENCE_METRICS_BY_SEED.csv"
    consistency_metrics = ("held_out_cross_rat_consistency_r2", "held_out_across_run_consistency_r2")
    ref_ed = [r for r in csv_rows(ref_ed_path) if r.get("method") != "Conv-piVAE-test-labels"
              and r.get("metric") in consistency_metrics]
    ed_rows = [{**r, "method_display": display.get(r["method"], r["method"])} for r in ref_ed]
    cross = [r for r in csv_rows(TABLES / "SOFT_CROSS_RAT_CONSISTENCY_BY_SEED.csv")
             if r.get("representation") == "embedding_raw" and r.get("source_rat") != r.get("target_rat")]
    grouped = {}
    for r in cross:
        key = (r["method"], r["architecture"], r["condition"], r["source_rat"], r["seed"])
        grouped.setdefault(key, []).append(float(r["mean_across_folds"]))
    for (method, arch, cond, rat, seed), values in grouped.items():
        ed_rows.append({"rat": rat, "method": method, "method_display": f"{method} {cond}", "seed": seed,
                        "metric": "held_out_cross_rat_consistency_r2", "value": float(np.mean(values)),
                        "split": "test"})
    across = [r for r in csv_rows(TABLES / "SOFT_ACROSS_RUN_CONSISTENCY_BY_SEED.csv")
              if r.get("representation") == "embedding_raw"]
    for r in across:
        ed_rows.append({"rat": r["rat"], "method": r["method"],
                        "method_display": f"{r['method']} {r['condition']}", "seed": r["source_seed"],
                        "metric": "held_out_across_run_consistency_r2", "value": r["mean_across_folds"],
                        "split": "test"})
    ed_metrics = consistency_metrics
    write_csv(STAGE / "Extended_Data" / "ED4" / "ED4_CONSISTENCY_BY_SEED.csv", ed_rows)
    fig, axes = plt.subplots(4, len(ed_metrics), figsize=(13, 12), squeeze=False, dpi=220)
    ed_labels = {"held_out_cross_rat_consistency_r2": "Cross-rat consistency R²",
                 "held_out_across_run_consistency_r2": "Across-run consistency R²"}
    for ri, rat in enumerate(RATS):
        for ci, metric in enumerate(ed_metrics):
            ax = axes[ri, ci]
            methods = list(dict.fromkeys(r["method_display"] for r in ed_rows if r["rat"] == rat and r["metric"] == metric))
            for yi, method in enumerate(methods):
                vals = [float(r["value"]) for r in ed_rows if r["rat"] == rat and r["metric"] == metric and r["method_display"] == method]
                if not vals:
                    continue
                soft_method = method.startswith("Soft")
                ax.scatter(vals, np.full(len(vals), yi), s=13, color="#c62828" if soft_method else "black",
                           alpha=.78, edgecolors="white", linewidths=.25, zorder=3)
                med = float(np.median(vals))
                ax.plot([med, med], [yi-.22, yi+.22], color="#e88d25", lw=2.4, zorder=4)
            if ci == 0:
                ax.set_yticks(range(len(methods)), methods, fontsize=6)
                for tick, label in zip(ax.get_yticklabels(), methods):
                    if label.startswith("Soft"):
                        tick.set_color("#c62828")
            else:
                ax.set_yticks(range(len(methods)), [])
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if ri == 0:
                ax.set_title(ed_labels[metric], fontsize=9)
            if ci == 0:
                ax.set_ylabel(rat.capitalize())
            if ri == 3:
                ax.set_xlabel(ed_labels[metric])
    fig.tight_layout()
    write_figure(fig, STAGE / "Extended_Data" / "ED4" / "ED4_METHOD_COMPARISON", "ED4",
                 [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in
                  (ref_ed_path, TABLES / "SOFT_CROSS_RAT_CONSISTENCY_BY_SEED.csv",
                   TABLES / "SOFT_ACROSS_RUN_CONSISTENCY_BY_SEED.csv", soft_path)],
                 {"split": "held-out test", "metrics": list(ed_metrics),
                  "interpretation": "consistency comparison only: cross-rat and across-run; no decoding metrics"})


def promote() -> None:
    expected = []
    for rat in RATS:
        expected += [STAGE / "Figure_1" / "1D" / f"FIGURE1D_{rat.upper()}.{e}" for e in ("png", "pdf")]
        expected += [STAGE / "Figure_2" / "2C" / f"FIGURE2C_{rat.upper()}.{e}" for e in ("png", "pdf")]
    expected += [STAGE / "Figure_2" / "2D" / f"FIGURE2D_METHOD_COMPARISON.{e}" for e in ("png", "pdf")]
    expected += [STAGE / "Extended_Data" / "ED4" / f"ED4_METHOD_COMPARISON.{e}" for e in ("png", "pdf")]
    for p in expected:
        if not p.is_file() or p.stat().st_size < 1000:
            raise RuntimeError(f"Staged figure missing/invalid; old outputs preserved: {p}")
    cleanup = {
        DELIVERY / "Figure_1" / "1D": lambda p: p.suffix.lower() in (".png", ".pdf", ".svg") or p.name.endswith("_PROVENANCE.json"),
        DELIVERY / "Figure_2" / "2C": lambda p: p.suffix.lower() in (".png", ".pdf", ".svg") or (p.name.startswith("FIGURE2C_") and p.name.endswith("_PROVENANCE.json")),
        DELIVERY / "Figure_2" / "2D": lambda p: p.name.startswith(("FIGURE2D_REFERENCE_METHOD_COMPARISON", "FIGURE2D_SOFT_DECODING", "FIGURE2D_METHOD_COMPARISON")) and (p.suffix.lower() in (".png", ".pdf", ".svg") or p.name.endswith("PROVENANCE.json")),
        DELIVERY / "Extended_Data" / "ED4": lambda p: (p.suffix.lower() in (".png", ".pdf", ".svg") and not p.name.startswith("ED4_METHOD_COMPARISON")) or (p.name.endswith("PROVENANCE.json") and p.name not in ("REFERENCE_PROVENANCE.json", "SOFT_METRICS_PROVENANCE.json")),
    }
    targets = [(STAGE / "Figure_1" / "1D", DELIVERY / "Figure_1" / "1D"),
               (STAGE / "Figure_2" / "2C", DELIVERY / "Figure_2" / "2C"),
               (STAGE / "Figure_2" / "2D", DELIVERY / "Figure_2" / "2D"),
               (STAGE / "Extended_Data" / "ED4", DELIVERY / "Extended_Data" / "ED4")]
    for _src, dst in targets:
        dst.mkdir(parents=True, exist_ok=True)
        for old in list(dst.iterdir()):
            if old.is_file() and cleanup[dst](old):
                old.unlink()
    for src, dst in targets:
        for file in src.iterdir():
            if file.is_file():
                shutil.move(str(file), str(dst / file.name))
    shutil.rmtree(STAGE)


def refresh_provenance() -> None:
    bases = [DELIVERY / "Figure_1" / "1D" / f"FIGURE1D_{rat.upper()}" for rat in RATS]
    bases += [DELIVERY / "Figure_2" / "2C" / f"FIGURE2C_{rat.upper()}" for rat in RATS]
    bases += [DELIVERY / "Figure_2" / "2D" / "FIGURE2D_METHOD_COMPARISON",
              DELIVERY / "Extended_Data" / "ED4" / "ED4_METHOD_COMPARISON"]
    script_hash = digest(Path(__file__))
    for base in bases:
        provenance = base.with_name(base.name + "_PROVENANCE.json")
        if not provenance.is_file():
            raise FileNotFoundError(provenance)
        record = json.loads(provenance.read_text(encoding="utf-8"))
        for ext, expected in record["files"].items():
            if digest(base.with_suffix("." + ext)) != expected:
                raise RuntimeError(f"Final figure hash mismatch: {base}.{ext}")
        record["script_sha256"] = script_hash
        record["provenance_refreshed_utc"] = datetime.now(timezone.utc).isoformat()
        provenance.write_text(json.dumps(record, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote", action="store_true", help="After building/validating, remove obsolete plot files and promote the staged replacements")
    parser.add_argument("--refresh-provenance", action="store_true", help="Verify final figure hashes and refresh their source-script hash")
    args = parser.parse_args()
    if args.refresh_provenance:
        refresh_provenance()
        print("Verified all final PNG/PDF hashes and refreshed provenance source hashes.")
        return
    if args.promote:
        promote()
        print(json.dumps({"status": "promoted", "staging": str(STAGE),
                          "training_performed": False, "time_utc": STAMP}, indent=2))
        return
    if STAGE.exists():
        raise FileExistsError(f"Staging directory already exists; inspect before continuing: {STAGE}")
    STAGE.mkdir(parents=True)
    index = index_manifests()
    final_figure1d(index)
    final_figure2c()
    final_figure2d_and_ed4()
    print(json.dumps({"status": "promoted" if args.promote else "staged",
                      "staging": str(STAGE), "figures": 4 + 4 + 2 + 2,
                      "training_performed": False, "time_utc": STAMP}, indent=2))


if __name__ == "__main__":
    main()
