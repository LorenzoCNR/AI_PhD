"""Reconstruct Figure 3h/i panels from locally fitted frozen held-out models."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cebra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

from export_figure3h import ROOT as EMBEDDING_ROOT
from fit_figure3h import DIMENSIONS, PROJECT, SEEDS, sha256
from score_figure3h import K, ROOT as SCORE_ROOT


ROOT = PROJECT / "results" / "monkey_figure3_paper_v1" / "stage03_figures" / "figure3hi"
TASKS = ("position", "direction", "ap")


def panel_h() -> None:
    rows = []
    for task in TASKS:
        per_dimension = []
        for dimension in DIMENSIONS:
            values = []
            for seed in SEEDS:
                path = SCORE_ROOT / task / f"dim_{dimension}" / f"seed_{seed}" / "DECODING_RESULT.json"
                if not path.exists():
                    raise FileNotFoundError(f"Figure 3h requires all 75 scores: {path}")
                record = json.loads(path.read_text(encoding="utf-8"))
                values.append(float(record["score"]))
                rows.append({"task": task, "dimension": dimension, "seed": seed,
                             "score": values[-1], "metric": record["metric"],
                             "result_sha256": sha256(path)})
            per_dimension.append(values)
        if len(per_dimension) != 5:
            raise ValueError(f"Incomplete task: {task}")
    plt.rcParams.update({"font.size": 13, "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, axes = plt.subplots(3, 1, figsize=(5, 20))
    titles = ("Position decoding (active)", "Direction decoding (active)", "AP decoding")
    units = ("$R^2$ (%)", "Acc(%)", "Acc(%)")
    paper_ylims = ((50, 100), (50, 80), (50, 100))
    for axis, task, title, unit, paper_ylim in zip(axes, TASKS, titles, units, paper_ylims):
        values_by_dim = [[row["score"] for row in rows if row["task"] == task and row["dimension"] == dim]
                         for dim in DIMENSIONS]
        for index, values in enumerate(values_by_dim):
            axis.scatter(np.full(len(values), index), values, s=55, facecolors="none", edgecolors="black", zorder=3)
            axis.hlines(np.median(values), index - 0.27, index + 0.27, color="orange", lw=2.5, zorder=4)
        observed = [v for group in values_by_dim for v in group]
        low, high = paper_ylim
        if min(observed) < low or max(observed) > high:
            padding = max(2.0, (max(observed) - min(observed)) * 0.08)
            low, high = min(low, min(observed) - padding), max(high, max(observed) + padding)
        axis.set_ylim(low, high)
        axis.set_xticks(range(len(DIMENSIONS)), [str(d) for d in DIMENSIONS])
        axis.set_xlabel("Dimensions")
        axis.set_ylabel(unit)
        axis.set_title(title, pad=22)
        axis.spines["right"].set_visible(False)
        axis.spines["top"].set_visible(False)
    fig.subplots_adjust(hspace=0.8)
    ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "FIGURE3H_LOCAL_REPLICATION.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(ROOT / "FIGURE3H_LOCAL_REPLICATION.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    (ROOT / "FIGURE3H_PROVENANCE.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "https://cebra.ai/docs/cebra-figures/figures/Figure3.html",
        "score_records": rows, "plot_script_sha256": sha256(Path(__file__)),
        "presentation": "five black seed points and orange median per dimension, per Figure 3h",
        "deviation": "axis limits expand only if needed to keep all local results visible",
    }, indent=2), encoding="utf-8")


def panel_i() -> None:
    seed, dimension, task = 42, 16, "position"
    branch = EMBEDDING_ROOT / task / f"dim_{dimension}" / f"seed_{seed}"
    train_path = branch / "train" / "embedding_and_trials.npz"
    test_path = branch / "test" / "embedding_and_trials.npz"
    for path in (train_path, test_path):
        record = json.loads((path.parent / "EMBEDDING_PROVENANCE.json").read_text(encoding="utf-8"))
        if sha256(path) != record["embedding_sha256"]:
            raise ValueError(f"Embedding hash mismatch: {path}")
    with np.load(train_path) as archive:
        x_train = np.asarray(archive["embedding_raw"], dtype=np.float32)
        y_train = np.asarray(archive["position_xy"], dtype=np.float32)
        train_ids = np.unique(archive["trial_id"])
    with np.load(test_path) as archive:
        x_test = np.asarray(archive["embedding_raw"], dtype=np.float32)
        y_test = np.asarray(archive["position_xy"], dtype=np.float32)
        test_ids = np.unique(archive["trial_id"])
    if len(test_ids) != 26 or np.intersect1d(train_ids, test_ids).size:
        raise ValueError("Figure 3i held-out split mismatch")
    decoder = cebra.KNNDecoder(n_neighbors=K, metric="cosine")
    decoder.fit(x_train, y_train)
    predicted = np.asarray(decoder.predict(x_test), dtype=np.float32)
    r2 = float(r2_score(y_test, predicted))
    ROOT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test[:, 0], y_test[:, 1], s=1, color="gray")
    ax.scatter(predicted[:, 0], predicted[:, 1], s=1, color="red")
    ax.axis("off")
    fig.savefig(ROOT / "FIGURE3I_LOCAL_REPLICATION.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(ROOT / "FIGURE3I_LOCAL_REPLICATION.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    (ROOT / "FIGURE3I_RESULT.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": task, "dimension": dimension, "seed": seed,
        "test_trial_count": len(test_ids), "test_row_count": len(x_test),
        "position_R2": r2, "paper_position_R2_reported": 0.88,
        "decoder": "cebra.KNNDecoder", "k": K, "distance": "cosine",
        "train_embedding_sha256": sha256(train_path), "test_embedding_sha256": sha256(test_path),
        "plot_script_sha256": sha256(Path(__file__)),
        "seed_selection": "seed 42 fixed before test scoring; not selected by test performance",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    panel_i()
    panel_h()
