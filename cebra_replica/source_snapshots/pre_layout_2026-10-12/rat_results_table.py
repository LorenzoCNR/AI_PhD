"""Build a rolling, source-linked table from immutable rat scoring JSON files."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCORES = ROOT / "results" / "rat_trial_scoring_v1"
OUT = ROOT / "results" / "rat_reports_v1"
FIELDS = ("rat", "fold", "condition", "seed", "split", "n_samples",
          "position_median_absolute_error_cm", "position_r2",
          "direction_balanced_accuracy", "score_json_sha256", "fit_manifest_sha256")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows = []
    parents = {}
    for path in sorted(SCORES.rglob("scores.json")):
        score = json.loads(path.read_text(encoding="utf-8"))
        fit_manifest = Path(score["source_fit_manifest"])
        if not fit_manifest.exists() or sha256(fit_manifest) != score["source_fit_manifest_sha256"]:
            raise AssertionError(f"Fit provenance mismatch: {path}")
        parent_hash = sha256(path)
        parents[str(path)] = parent_hash
        for metric in score["metrics"]:
            if metric["split"] not in ("valid", "test"):
                raise AssertionError(f"Unexpected scoring split: {path}")
            rows.append({**metric, "score_json_sha256": parent_hash,
                         "fit_manifest_sha256": score["source_fit_manifest_sha256"]})
    rows.sort(key=lambda x: (x["rat"], x["fold"], x["seed"], x["condition"], x["split"]))
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "RAT_DECODING_RESULTS_LONG.csv"
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    provenance = {"updated_utc": datetime.now(timezone.utc).isoformat(),
                  "script_sha256": sha256(Path(__file__)), "score_parents": parents,
                  "rows": len(rows), "completed_scored_fits": len(parents),
                  "scope": "Rolling aggregation of immutable decoder results; no new metrics calculated."}
    (OUT / "RAT_DECODING_RESULTS_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"{target}: {len(rows)} rows, {len(parents)} scored fits")


if __name__ == "__main__":
    main()
