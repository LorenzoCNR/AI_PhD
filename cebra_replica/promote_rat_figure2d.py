"""Promote only the regenerated two-metric rat Figure 2D and its tables."""
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from rat_revision_common import DELIVERY, ROOT, STAGE, sha


PANEL = "Figure_2/2D"
SOURCE = STAGE / PANEL
TARGET = DELIVERY / PANEL
RATS = ("ACHILLES", "BUDDY", "CICERO", "GATSBY")
FIGURE_FILES = [
    f"FIGURE2D_{rat}_DECODING.{ext}"
    for rat in RATS for ext in ("png", "pdf")
] + [f"FIGURE2D_{rat}_DECODING_PROVENANCE.json" for rat in RATS]
TABLE_FILES = [
    "FIGURE2D_METRICS_COVERAGE.csv",
    "FIGURE2D_METRICS_FOLD_RECORDS.csv",
    "FIGURE2D_METRICS_SEED_RECORDS.csv",
]
FILES = FIGURE_FILES + TABLE_FILES
MANIFEST = DELIVERY / "RAT_FIGURE2D_ACCURACY_UPDATE_20261022.json"


def validate_sources() -> dict:
    for name in FILES:
        src = SOURCE / name
        if not src.is_file() or SOURCE.resolve() not in src.resolve().parents:
            raise RuntimeError(f"Missing/out-of-scope staged file: {src}")
    for rat in RATS:
        image = SOURCE / f"FIGURE2D_{rat}_DECODING.png"
        pdf = image.with_suffix(".pdf")
        with Image.open(image) as im:
            if min(im.size) < 200 or ("A" in im.getbands() and im.getchannel("A").getextrema() != (255, 255)):
                raise RuntimeError(f"Invalid staged image: {image}")
            rgb = im.convert("RGB")
            if rgb.getpixel((0, 0)) != (255, 255, 255) or rgb.resize((128, 128)).getextrema() == ((255, 255),) * 3:
                raise RuntimeError(f"Blank/nonwhite staged image: {image}")
        if pdf.stat().st_size < 1000:
            raise RuntimeError(f"Missing/empty staged PDF: {pdf}")
        record = json.loads((SOURCE / f"FIGURE2D_{rat}_DECODING_PROVENANCE.json").read_text(encoding="utf-8"))
        if record["output_hashes"]["png"] != sha(image) or record["output_hashes"]["pdf"] != sha(pdf):
            raise RuntimeError(f"Staged figure provenance hash mismatch: {rat}")

    with (SOURCE / "FIGURE2D_METRICS_COVERAGE.csv").open(encoding="utf-8-sig", newline="") as stream:
        coverage = list(csv.DictReader(stream))
    expected = {("position_median_absolute_error_cm", "knn"): 48,
                ("position_median_absolute_error_cm", "ridge"): 48,
                ("direction_accuracy", "knn"): 48}
    observed = {}
    for row in coverage:
        key = (row["metric"], row["decoder"])
        observed[key] = observed.get(key, 0) + (row["status"] == "AVAILABLE")
        if row["status"] != "AVAILABLE":
            raise RuntimeError(f"Unexpected missing Figure 2D score: {row}")
    if observed != expected:
        raise RuntimeError(f"Unexpected Figure 2D metric coverage: {observed}")
    with (SOURCE / "FIGURE2D_METRICS_FOLD_RECORDS.csv").open(encoding="utf-8-sig", newline="") as stream:
        fold_records = list(csv.DictReader(stream))
    baseline_methods = {"conv-piVAE", "PCA", "UMAP", "tSNE"}
    baseline_accuracy = [row for row in fold_records
                         if row["method"] in baseline_methods and row["decoder"] == "knn"]
    if len(baseline_accuracy) != 228 or any(not row["direction_accuracy"] for row in baseline_accuracy):
        raise RuntimeError(f"Incomplete frozen-baseline direction accuracy: {len(baseline_accuracy)} rows")
    max_bacc_difference = max(float(row["direction_balanced_accuracy_abs_difference"])
                              for row in baseline_accuracy)
    return {"coverage_rows": len(coverage),
            "available_cells": {f"{metric}|{decoder}": count for (metric, decoder), count in observed.items()},
            "frozen_baseline_direction_accuracy_rows": len(baseline_accuracy),
            "max_abs_difference_historical_balanced_accuracy_check": max_bacc_difference}


def main() -> None:
    if MANIFEST.exists():
        raise FileExistsError(f"Refusing to overwrite promotion record: {MANIFEST}")
    summary = validate_sources()
    replacements = []
    for name in FILES:
        src, dst = SOURCE / name, TARGET / name
        if TARGET.resolve() not in dst.resolve().parents:
            raise RuntimeError(f"Out-of-scope destination: {dst}")
        if not dst.is_file():
            raise FileNotFoundError(f"Expected existing Figure 2D deliverable: {dst}")
        replacements.append({"path": dst.relative_to(DELIVERY).as_posix(),
                             "old_sha256": sha(dst), "new_sha256": sha(src)})

    TARGET.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy2(SOURCE / name, TARGET / name)
        if sha(SOURCE / name) != sha(TARGET / name):
            raise RuntimeError(f"Copied file hash mismatch: {name}")
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE",
        "scope": PANEL,
        "replaced_files": replacements,
        "figure_count": 4,
        "figure_columns": ["Median absolute position error (cm)", "Direction accuracy"],
        "ridge_reference_position_fold_scores": 228,
        "encoder_retrained": False,
        "validation": summary,
        "source_stage": SOURCE.relative_to(ROOT).as_posix(),
        "note": "Only the four Figure 2D plot triplets and three Figure 2D metric CSVs were replaced; trajectories and other panels were untouched. The direction panel uses ordinary kNN accuracy. Frozen-baseline accuracies were recomputed from saved embeddings; historical balanced-accuracy checks are retained in the fold table.",
    }
    MANIFEST.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "files_replaced": len(FILES),
                      "manifest": MANIFEST.relative_to(ROOT).as_posix(), **summary}, indent=2))


if __name__ == "__main__":
    main()
