"""Resumable 4-rat x 3-fold x 10-seed trial-wise CEBRA campaign."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RATS = ("achilles", "buddy", "cicero", "gatsby")
CONDITIONS = ("time", "position", "direction", "position_direction",
              "position_shuffled", "direction_shuffled", "position_direction_shuffled")
FOLDS = (0, 1, 2)
SEEDS = tuple(range(42, 52))
UPDATES = 10_000
OUTPUT = ROOT / "results" / "rat_trial_benchmark_v1"
CAMPAIGN = ROOT / "results" / "rat_campaign_v1"
BENCHMARK = ROOT / "cebra_replica" / "rat_benchmark.py"
LEGACY_BENCHMARK = (ROOT / "cebra_replica" / "source_snapshots" /
                    "pre_layout_2026-10-12" / "rat_benchmark.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Existing immutable fits were produced by the exact pre-relocation source.
# Accept that archived hash as well as the current path-only-adjusted runner.
BENCHMARK_SHA256S = {sha256(BENCHMARK), sha256(LEGACY_BENCHMARK)}


def trials():
    # Visit every rat within each seed/fold before adding more replications;
    # this changes execution order only, not the planned configurations.
    for seed, fold, rat, condition in itertools.product(SEEDS, FOLDS, RATS, CONDITIONS):
        yield rat, fold, seed, condition


def trial_dir(rat: str, fold: int, seed: int, condition: str) -> Path:
    return OUTPUT / rat / f"fold_{fold}" / condition / f"seed_{seed}"


def verify_complete(rat: str, fold: int, seed: int, condition: str) -> bool:
    folder = trial_dir(rat, fold, seed, condition)
    manifest = folder / "result.json"
    if not manifest.exists():
        if folder.exists() and any(folder.iterdir()):
            raise RuntimeError(f"Partial fit requires inspection: {folder}")
        return False
    result = json.loads(manifest.read_text(encoding="utf-8"))
    required = {"rat": rat, "fold": fold, "seed": seed, "condition": condition,
                "max_iterations": UPDATES,
                "source_sha256": sha256(ROOT / "data" / "rat_hippocampus" / f"{rat}.jl")}
    if any(result.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"Completed fit provenance mismatch: {manifest}")
    if result.get("script_sha256") not in BENCHMARK_SHA256S:
        raise RuntimeError(f"Completed fit runner hash is not the current or archived source: {manifest}")
    checkpoint = folder / "checkpoint.pt"
    if not checkpoint.exists() or sha256(checkpoint) != result.get("checkpoint_sha256"):
        raise RuntimeError(f"Checkpoint missing or changed: {checkpoint}")
    for split in ("train", "valid", "test"):
        path = folder / f"{split}_embedding.npz"
        if not path.exists() or sha256(path) != result["embeddings"][split]["embedding_sha256"]:
            raise RuntimeError(f"Embedding missing or changed: {path}")
    return True


def save_state(state: dict) -> None:
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    path = CAMPAIGN / "campaign_state.json"
    staging = path.with_suffix(".tmp")
    staging.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    staging.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Launch fits; otherwise read-only plan/status")
    parser.add_argument("--max-new-fits", type=int, default=None,
                        help="Stop after this many new fits, for qualification or scheduled blocks")
    args = parser.parse_args()
    complete = 0
    for rat, fold, seed, condition in trials():
        complete += verify_complete(rat, fold, seed, condition)
    planned = len(RATS) * len(FOLDS) * len(SEEDS) * len(CONDITIONS)
    print(json.dumps({"planned": planned, "completed_verified": complete,
                      "remaining": planned - complete, "rats": RATS, "folds": FOLDS,
                      "seeds": SEEDS, "conditions": CONDITIONS,
                      "updates_per_fit": UPDATES}, indent=2), flush=True)
    if not args.execute:
        return
    state = {"created_utc": datetime.now(timezone.utc).isoformat(),
             "runner_sha256": sha256(BENCHMARK),
             "campaign_sha256": sha256(Path(__file__)), "planned": planned,
             "completed_verified_at_start": complete, "completed_now": 0,
             "last_trial": None, "status": "running"}
    save_state(state)
    for rat, fold, seed, condition in trials():
        if verify_complete(rat, fold, seed, condition):
            continue
        if args.max_new_fits is not None and state["completed_now"] >= args.max_new_fits:
            state["status"] = "stopped_at_requested_limit"
            save_state(state)
            return
        state["last_trial"] = dict(rat=rat, fold=fold, seed=seed, condition=condition)
        state["last_started_utc"] = datetime.now(timezone.utc).isoformat()
        save_state(state)
        command = [sys.executable, str(BENCHMARK), "--rat", rat,
                   "--fold", str(fold), "--condition", condition, "--seed", str(seed),
                   "--max-iterations", str(UPDATES)]
        print(f"CAMPAIGN FIT {complete + state['completed_now'] + 1}/{planned}: "
              f"{rat}/fold_{fold}/{condition}/seed_{seed}", flush=True)
        process = subprocess.run(command, cwd=ROOT, check=False)
        if process.returncode != 0 or not verify_complete(rat, fold, seed, condition):
            state["status"] = "failed"
            state["returncode"] = process.returncode
            save_state(state)
            raise RuntimeError(f"Fit failed or incomplete: {state['last_trial']}")
        state["completed_now"] += 1
        state["last_completed_utc"] = datetime.now(timezone.utc).isoformat()
        save_state(state)
    state["status"] = "complete"
    save_state(state)


if __name__ == "__main__":
    main()
