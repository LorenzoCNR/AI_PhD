"""Compare frozen pilot scores with published rat results without equating protocols."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results" / "rat_reports_v1" / "RAT_DECODING_RESULTS_LONG.csv"
OUT = ROOT / "results" / "rat_reports_v1"
PAPER_URL = "https://cebra.ai/docs/cebra-figures/figures/Figure2.html"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    pilot = [row for row in source if row["fold"] == "0" and row["seed"] == "42"
             and row["split"] == "test"]
    if len(pilot) != 28 or len({(r["rat"], r["condition"]) for r in pilot}) != 28:
        raise RuntimeError("Four-rat pilot is incomplete; comparison is not written")
    rows = []
    for rat in ("achilles", "buddy", "cicero", "gatsby"):
        for condition in ("position_direction", "position_direction_shuffled"):
            ours = next(r for r in pilot if r["rat"] == rat and r["condition"] == condition)
            paper_error = (5.8 if condition == "position_direction" else 44.7) if rat == "achilles" else ""
            paper_joint_r2 = (0.7335 if condition == "position_direction" else -0.499) if rat == "achilles" else ""
            rows.append({"rat": rat, "condition": condition,
                         "our_trial_held_out_position_error_cm": ours["position_median_absolute_error_cm"],
                         "our_trial_held_out_position_only_r2": ours["position_r2"],
                         "our_trial_held_out_direction_bacc": ours["direction_balanced_accuracy"],
                         "paper_figure2d_position_error_cm": paper_error,
                         "paper_figure2d_joint_position_direction_r2": paper_joint_r2,
                         "directly_comparable": "NO",
                         "reason": "Our one seed/one complete-trial split differs from paper multi-run/multi-split; R2 targets differ"})
    target = OUT / "RAT_PILOT_VS_PAPER_FIG2D.csv"
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "script_sha256": sha256(Path(__file__)), "source_csv": str(SOURCE),
                  "source_csv_sha256": sha256(SOURCE), "paper_source": PAPER_URL,
                  "paper_numbers_scope": "Figure 2d first rat only, ten runs averaged over three splits",
                  "comparability": "Descriptive scale comparison only; not a direct replication estimate",
                  "rows": len(rows)}
    (OUT / "RAT_PILOT_VS_PAPER_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"{target}: {len(rows)} comparison rows")


if __name__ == "__main__":
    main()
