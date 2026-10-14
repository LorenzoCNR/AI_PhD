"""Render the held-out AP position/direction decoding panels of ED8."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fit_figure3h import DIMENSIONS, PROJECT, SEEDS


SCORES = PROJECT / "results/monkey_extended8_paper_v1/stage03_decoding_heldout"
OUTPUT = PROJECT / "results/monkey_extended8_paper_v1/stage03_figures/heldout_decoding"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    tasks = ("ap_pos", "ap_direction")
    titles = ("Position decoding (trained with A/P + pos)",
              "Direction decoding (trained with A/P + direction)")
    units = ("$R^2$ (%)", "ACC (%)")
    ylims = ((50, 80), (0, 60))
    rows, parent_hashes = [], {}
    score_arrays = {}
    for task in tasks:
        by_dimension = []
        for dimension in DIMENSIONS:
            values = []
            for seed in SEEDS:
                path = SCORES / task / f"dim_{dimension}" / f"seed_{seed}" / "DECODING_RESULT.json"
                record = json.loads(path.read_text(encoding="utf-8"))
                expected_metric = "R2_percent" if task == "ap_pos" else "accuracy_percent"
                if record["metric"] != expected_metric or record["split_score"] != "test":
                    raise ValueError(f"Unexpected ED8 score definition: {path}")
                values.append(float(record["score"]))
                rows.append({"task": task, "dimension": dimension, "seed": seed,
                             "score": float(record["score"]), "metric": record["metric"],
                             "score_sha256": sha256(path)})
                parent_hashes[f"{task}/d{dimension}/s{seed}"] = sha256(path)
            by_dimension.append(values)
        score_arrays[task] = by_dimension

    OUTPUT.mkdir(parents=True, exist_ok=True)
    png, pdf = OUTPUT / "EXTENDED8_HELDOUT_AP_DECODING.png", OUTPUT / "EXTENDED8_HELDOUT_AP_DECODING.pdf"
    provenance = OUTPUT / "PROVENANCE.json"
    if provenance.exists():
        old = json.loads(provenance.read_text(encoding="utf-8"))
        if old["score_sha256"] != parent_hashes or sha256(png) != old["figure_sha256"][png.name] or sha256(pdf) != old["figure_sha256"][pdf.name]:
            raise ValueError("Existing ED8 decoding figure provenance mismatch")
        print("REUSE ED8 AP decoding figure")
        return
    if png.exists() or pdf.exists():
        raise FileExistsError("Partial ED8 decoding figure exists")
    with plt.rc_context({"font.size": 13, "figure.facecolor": "white", "axes.facecolor": "white"}):
        fig, axes = plt.subplots(2, 1, figsize=(5, 15))
        fig.subplots_adjust(hspace=0.8)
        for index, (axis, task, title, unit, ylim) in enumerate(zip(axes, tasks, titles, units, ylims)):
            for dim_index, (dimension, values) in enumerate(zip(DIMENSIONS, score_arrays[task])):
                axis.scatter(np.full(len(values), dim_index), values, s=55,
                             facecolors="none", edgecolors="black", zorder=3)
                axis.hlines(np.median(values), dim_index - 0.27, dim_index + 0.27,
                            color="orange", lw=2.5, zorder=4)
            axis.set_ylim(*ylim)
            axis.set_xlabel("Dimensions")
            axis.set_ylabel(unit)
            axis.set_title(title, y=1.1)
            axis.set_xticks(np.arange(len(DIMENSIONS)), [str(d) for d in DIMENSIONS])
            if index == 1:
                axis.axhline(100 / 16, lw=2, ls="dashed", color="gray")
            axis.spines["right"].set_visible(False)
            axis.spines["top"].set_visible(False)
        fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    provenance.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(),
                                      "script_sha256": sha256(Path(__file__)),
                                      "score_sha256": parent_hashes,
                                      "figure_sha256": {path.name: sha256(path) for path in (png, pdf)},
                                      "seeds": list(SEEDS), "dimensions": list(DIMENSIONS),
                                      "split": "train-fit/test-score",
                                      "decoder": "CEBRA KNN k=36 cosine",
                                      "presentation": "open black seed points and orange median as official ED8 style"}, indent=2) + "\n", encoding="utf-8")
    print(f"COMPLETE {OUTPUT}")


if __name__ == "__main__":
    main()
