"""Validate and replace only the active rat Figure 2B plot deliverables."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from rat_revision_common import DELIVERY, ROOT, STAGE, sha

RATS = ("achilles", "buddy", "cicero", "gatsby")
OUT = STAGE / "Figure_2" / "2B"
TARGET = DELIVERY / "Figure_2" / "2B"


def expected_bases():
    return {
        f"FIGURE2B_{model}_{rat.upper()}"
        for rat in RATS
        for model in ("CEBRA", "SOFT_CNN", "SOFT_TRANSFORMER")
    }


def validate_stage():
    if not OUT.is_dir() or OUT.resolve().parent != (STAGE / "Figure_2").resolve():
        raise RuntimeError(f"Unexpected or missing stage directory: {OUT}")
    bases = expected_bases()
    for base in bases:
        png, pdf = OUT / f"{base}.png", OUT / f"{base}.pdf"
        provenance = OUT / f"{base}_PROVENANCE.json"
        if not png.is_file() or not pdf.is_file() or not provenance.is_file():
            raise RuntimeError(f"Incomplete staged figure pair: {base}")
        if png.stat().st_size < 10_000 or pdf.stat().st_size < 10_000:
            raise RuntimeError(f"Unexpectedly small staged figure: {base}")
        with Image.open(png) as image:
            if min(image.size) < 500:
                raise RuntimeError(f"Staged image resolution too small: {base}")
            if image.convert("RGB").getpixel((0, 0)) != (255, 255, 255):
                raise RuntimeError(f"Staged image background is not white: {base}")
        record = json.loads(provenance.read_text(encoding="utf-8"))
        if record.get("encoder_retrained") is not False:
            raise RuntimeError(f"Staged provenance does not confirm frozen fits: {base}")
        if record.get("output_hashes", {}).get("png") != sha(png):
            raise RuntimeError(f"Staged PNG hash mismatch: {base}")
        if record.get("output_hashes", {}).get("pdf") != sha(pdf):
            raise RuntimeError(f"Staged PDF hash mismatch: {base}")

    expected_indexes = {
        "FIGURE2B_CEBRA_PANEL_INDEX.csv": 16,
        "FIGURE2B_SOFT_PANEL_INDEX.csv": 48,
    }
    for name, expected_count in expected_indexes.items():
        path = OUT / name
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != expected_count:
            raise RuntimeError(f"Unexpected panel count in {name}: {len(rows)}")
    allowed = {f"{base}.{ext}" for base in bases for ext in ("png", "pdf")}
    allowed |= {f"{base}_PROVENANCE.json" for base in bases}
    allowed |= set(expected_indexes)
    actual = {p.name for p in OUT.iterdir() if p.is_file()}
    if actual != allowed:
        raise RuntimeError(f"Unexpected files in stage; missing={allowed-actual}, extra={actual-allowed}")
    return sorted(OUT.iterdir())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    staged = validate_stage()
    if not TARGET.is_dir() or TARGET.resolve().parent != (DELIVERY / "Figure_2").resolve():
        raise RuntimeError(f"Unexpected or missing active target: {TARGET}")
    old_files = sorted(p for p in TARGET.iterdir() if p.is_file())
    unexpected = [p.name for p in old_files if not (
        p.suffix.lower() in {".png", ".pdf", ".csv", ".json"}
        and p.name.startswith("FIGURE2B_")
    )]
    if unexpected:
        raise RuntimeError(f"Refusing to remove non-Figure-2B files: {unexpected}")
    removed = [dict(path=p.name, sha256=sha(p), bytes=p.stat().st_size) for p in old_files]
    new = [dict(path=p.name, sha256=sha(p), bytes=p.stat().st_size) for p in staged]
    summary = dict(
        status="DRY_RUN" if not args.execute else "READY_TO_REPLACE",
        target=str(TARGET.resolve()), stage=str(OUT.resolve()),
        figures=len(expected_bases()), encoder_fits=0,
        removed_old_files=removed, promoted_files=new,
        untouched=["Figure_2/2C", "all other figures and model artifacts"],
    )
    if not args.execute:
        print(json.dumps(summary, indent=2))
        return

    for path in old_files:
        path.unlink()
    for source in staged:
        destination = TARGET / source.name
        shutil.move(str(source), str(destination))
    summary["status"] = "COMPLETE"
    summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest = TARGET / "FIGURE2B_REPLACEMENT_20261022.json"
    summary["manifest"] = manifest.name
    manifest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
