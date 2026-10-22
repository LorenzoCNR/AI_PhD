"""Split the crowded rat ED3a display; frozen held-out embeddings only."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rat_soft_stable_ed3 import load_panel, sha256, relative
from rat_figure_gap_closure_analysis import index_manifests, load_fit, RATS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/RAT_FIGURES/.revision_20261021/Extended_Data/ED3"
REFERENCES = (
    ("CEBRA-Behavior", "cebra", "CEBRA-Behavior-32D", "raw", "42"),
    ("CEBRA-Time", "cebra", "CEBRA-Time-32D", "raw", "42"),
    ("conv-piVAE", "pivae", "Conv-piVAE-z_mean", "raw", "42"),
    ("PCA", "baseline", "pca", "raw", "deterministic"),
    ("UMAP", "baseline", "umap", "raw", "42"),
    ("tSNE", "baseline", "opentsne", "raw", "42"),
)
SOFT = tuple((f"Soft {arch} - {condition}", "soft", f"{arch}-{condition}", "raw", "42")
             for arch in ("CNN", "Transformer") for condition in ("joint", "time", "behavior"))


def panel(index, spec, rat):
    """Load native embeddings and verify parent hashes before any plotting."""
    if spec[1] != "soft":
        d = load_fit(index, spec[1], spec[2], rat, 0, spec[4], "test", spec[3])
        mask = d["valid_mask"].astype(bool)
        return {"embedding": d["embedding"][mask], "position": d["behavior"][mask, 0],
                "direction": d["behavior"][mask, 1], "source_id": d["source_id"][mask],
                "parent_path": relative(d["embedding_path"]),
                "parent_sha256": d["embedding_sha256"]}
    if not spec[2].endswith("-behavior"):
        return load_panel(index, spec, rat, "embedding_raw")
    arch = "cnn" if spec[2].startswith("CNN") else "transformer"
    base = ROOT / "results/user_soft_rat_behavior_only_v1" / arch / "stage02_embedding" / rat / "fold_0/soft_v1_stable/seed_42"
    path = base / "test_embedding.npz"
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    if sha256(path) != manifest["embeddings"]["test"]["embedding_sha256"]:
        raise ValueError(f"Parent embedding checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as z:
        mask = z["valid_mask"].astype(bool)
        for key in ("embedding_raw", "behavior", "source_id", "trial_id", "time_id"):
            if len(z[key]) != len(mask):
                raise ValueError(f"Cardinality mismatch: {key}/{path}")
        return {"embedding": z["embedding_raw"][mask], "position": z["behavior"][mask, 0],
                "direction": z["behavior"][mask, 1], "source_id": z["source_id"][mask],
                "parent_path": relative(path), "parent_sha256": sha256(path)}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    OUT.mkdir(parents=True, exist_ok=True)
    issue = OUT / "ISSUES_BEFORE_CHANGE.json"
    if not issue.exists():
        issue.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(),
            "issue": "ED3a crowded 10 methods by four rats, missing behavior-only Soft",
            "location": "cebra_replica/rat_soft_stable_ed3.py:render",
            "evidence": "40 axes in one 12 by 30 inch figure per representation",
            "severity": "REPORTING ONLY", "scientific_impact": "Readability; no metric or fit changes",
            "required_action": "Plot regeneration only",
            "fix": "One rat and representation per group; two by three readable panels; six reference and six Soft methods; only q(z|x) conv-piVAE",
            "raw_unit_fix": "Normalize each full native embedding row for unit display, before selecting displayed coordinates; old reference rows were native in both exports"}, indent=2) + "\n", encoding="utf-8")
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "text.color": "black", "axes.labelcolor": "black", "font.size": 11})
    index = index_manifests()
    rows = []
    outputs = []
    for rat in RATS:
        loaded = {spec[0]: panel(index, spec, rat) for spec in REFERENCES + SOFT}
        support = next(iter(loaded.values()))["source_id"]
        for name, d in loaded.items():
            if not np.array_equal(support, d["source_id"]):
                raise ValueError(f"Different held-out source support: {rat}/{name}")
        position = next(iter(loaded.values()))["position"]
        norm = Normalize(float(position.min()), float(position.max()))
        for representation in ("RAW", "UNIT"):
            for group, specs in (("REFERENCE", REFERENCES), ("SOFT", SOFT)):
                fig, axes = plt.subplots(2, 3, figsize=(11, 7), dpi=220)
                for ax, spec in zip(axes.flat, specs):
                    d = loaded[spec[0]]
                    emb = np.asarray(d["embedding"], dtype=float)
                    if emb.ndim != 2 or emb.shape[1] < 2 or not np.isfinite(emb).all():
                        raise ValueError(f"Invalid native embedding {rat}/{spec[0]}")
                    zero_norms = 0
                    if representation == "UNIT":
                        norms = np.linalg.norm(emb, axis=1, keepdims=True)
                        zero_norms = int(np.sum(norms[:, 0] == 0))
                        emb = np.divide(emb, norms, out=np.zeros_like(emb), where=norms > 0)
                    for right, cmap in ((False, "cool"), (True, "viridis")):
                        keep = (d["direction"] > 0) == right
                        ax.scatter(emb[keep, 0], emb[keep, 1], c=d["position"][keep], cmap=cmap,
                                   norm=norm, s=4, alpha=.75, linewidths=0, rasterized=True)
                    ax.set_title(spec[0], color="red" if group == "SOFT" else "black")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    rows.append({"rat": rat, "group": group, "method": spec[0], "seed": spec[4],
                        "fold": 0, "split": "test", "representation": representation,
                        "fit_dimension": d["embedding"].shape[1], "plotted_dimension": 2,
                        "n_rows": len(emb), "zero_norm_rows": zero_norms,
                        "parent_embedding": d["parent_path"], "parent_sha256": d["parent_sha256"]})
                fig.suptitle(f"{rat.capitalize()} - {representation.lower()}", fontsize=14)
                fig.subplots_adjust(top=.89, bottom=.17, wspace=.18, hspace=.25)
                for left, cmap, label in ((.26, "cool", "Left"), (.58, "viridis", "Right")):
                    cax = fig.add_axes([left, .075, .16, .022])
                    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
                    cb.set_ticks([])
                    cb.set_label(label)
                name = f"ED3A_{rat.upper()}_{representation}_{group}"
                for ext in ("png", "pdf"):
                    dest = OUT / f"{name}.{ext}"
                    fig.savefig(dest, dpi=240, bbox_inches="tight", facecolor="white", transparent=False)
                    outputs.append({"path": relative(dest), "sha256": sha256(dest)})
                plt.close(fig)
    table = OUT / "ED3_PANEL_INDEX.csv"
    with table.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "PROVENANCE.json").write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)), "split": "frozen fold0 held-out test", "seed": 42,
        "selection": "Prespecified illustrative seed42 fold0, no selection by outcome",
        "layout": "rat x representation x reference/Soft; 2 rows by 3 columns per figure",
        "embedding_display": "First two native coordinates; unit normalized over all fitted dimensions before display",
        "conv_pivae": "Behavior-trained fit, q(z|x) inference without test labels only",
        "point_count_policy": "Every common valid held-out source row; no subsampling",
        "scientific_metrics_modified": False, "training_performed": False,
        "panel_count": len(rows), "files": outputs, "panel_index_sha256": sha256(table)}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "panels": len(rows), "png": len(outputs)//2, "pdf": len(outputs)//2}))


if __name__ == "__main__":
    main()
