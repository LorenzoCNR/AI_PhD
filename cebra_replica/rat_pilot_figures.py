"""Descriptive figures from frozen Achilles/fold-0/seed-42 scoring artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "rat_trial_scoring_v1" / "achilles" / "fold_0"
OUT = ROOT / "results" / "rat_pilot_figures_v1" / "achilles" / "fold_0" / "seed_42"
CONDITIONS = (
    "time", "position", "direction", "position_direction",
    "position_shuffled", "direction_shuffled", "position_direction_shuffled",
)
LABELS = ("Time", "Position", "Direction", "Position + direction",
          "Position shuffled", "Direction shuffled", "Both shuffled")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    parents = {}
    for condition, label in zip(CONDITIONS, LABELS):
        path = SOURCE / condition / "seed_42" / "scores.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        test = next(row for row in data["metrics"] if row["split"] == "test")
        if test["n_samples"] != 1903:
            raise AssertionError(f"Unexpected test support: {condition}")
        rows.append({"condition": condition, "label": label,
                     "position_error_cm": test["position_median_absolute_error_cm"],
                     "position_r2": test["position_r2"],
                     "direction_balanced_accuracy": test["direction_balanced_accuracy"]})
        parents[condition] = {"path": str(path), "sha256": sha256(path)}

    csv_path = OUT / "PILOT_HELD_OUT_METRICS.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    colors = ["#4477aa"] * 4 + ["#cc6677"] * 3
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), layout="constrained")
    metrics = (
        ("position_error_cm", "Median position error (cm)", "Lower is better"),
        ("position_r2", "Position $R^2$", "Higher is better"),
        ("direction_balanced_accuracy", "Direction balanced accuracy", "Higher is better"),
    )
    for axis, (key, title, direction) in zip(axes, metrics):
        values = [row[key] for row in rows]
        axis.bar(x, values, color=colors, width=0.74)
        axis.set_title(title)
        axis.set_ylabel(direction)
        axis.set_xticks(x, LABELS, rotation=55, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        if key == "direction_balanced_accuracy":
            axis.axhline(0.5, color="#333333", linestyle="--", linewidth=1,
                         label="Balanced-chance reference")
            axis.legend(fontsize=7, loc="lower left")
    fig.suptitle("Achilles — held-out trial split, fold 0, seed 42\n"
                 "Preliminary descriptive results; one rat / one fold / one seed", fontsize=12)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"pilot_held_out_comparison.{extension}", dpi=240)
    plt.close(fig)

    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "script_sha256": sha256(Path(__file__)), "parents": parents,
                  "scope": "Descriptive pilot only; no new fitting or model selection",
                  "test_support": "Achilles fold 0, 1903 rows, seed 42",
                  "metric_caution": "Position-only R2 is not necessarily the paper's mixed R2; "
                                    "shuffled direction is not an empirical chance null."}
    (OUT / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
