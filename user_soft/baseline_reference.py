"""Hash-qualified reuse of frozen joint-behavior/time rat reference fits."""
from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import config_hash, eligible_centers, load_rows, sha256

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {"position_direction": "CEBRA-Behavior", "time": "CEBRA-Time"}
RATS = ("achilles", "buddy", "cicero", "gatsby")


def now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _output(output):
    output = Path(output).resolve()
    results = (ROOT / "results").resolve()
    if results not in output.parents:
        raise ValueError("New study output must be under AI_PhD/results")
    for name in ("rat_trial_benchmark_v1", "rat_trial_scoring_v1", "PAPER_FIGURE_DELIVERY"):
        protected = (results / name).resolve()
        if output == protected or protected in output.parents:
            raise ValueError("Existing reference artifacts are protected")
    return output


def verify_reference(output, rat, fold, seed, condition):
    """Check provenance and source equivalence only; compute no scientific score."""
    output = _output(output)
    if rat not in RATS or fold not in range(3) or seed not in range(42, 52) or condition not in CONDITIONS:
        raise ValueError("Reference identity outside fixed campaign")
    run = ROOT / "results" / "rat_trial_benchmark_v1" / rat / f"fold_{fold}" / condition / f"seed_{seed}"
    manifest_path = run / "result.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = ROOT / "data" / "rat_hippocampus" / f"{rat}.jl"
    expected = dict(rat=rat, fold=fold, seed=seed, condition=condition,
                    max_iterations=10000, source_sha256=sha256(source), cebra_version="0.6.1")
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Frozen reference identity/source/version mismatch: {run}")
    scripts = (ROOT / "cebra_replica" / "rat_benchmark.py",
               ROOT / "cebra_replica" / "source_snapshots" / "pre_layout_2026-10-12" / "rat_benchmark.py")
    script_hashes = {str(path): sha256(path) for path in scripts}
    if manifest.get("script_sha256") not in script_hashes.values():
        raise ValueError("Unknown frozen fit script hash")
    params = manifest["parameters"]
    expected_params = dict(model_architecture="offset10-model", batch_size=512,
                           learning_rate=3e-4, output_dimension=32, max_iterations=10000,
                           distance="cosine", time_offsets=10,
                           conditional="time" if condition == "time" else "time_delta",
                           temperature=1.12 if condition == "time" else 1.0)
    if any(params.get(key) != value for key, value in expected_params.items()):
        raise ValueError("Frozen reference numerical configuration mismatch")
    checkpoint = run / "checkpoint.pt"
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != manifest["checkpoint_sha256"]:
        raise ValueError("Frozen reference checkpoint hash mismatch")
    data_folder = output / "stage00_data" / rat / f"fold_{fold}"
    data_manifest = data_folder / "manifest.json"
    prepared = json.loads(data_manifest.read_text(encoding="utf-8"))
    if prepared["parent_source_sha256"] != expected["source_sha256"]:
        raise ValueError("Prepared source and reference source differ")
    entries = {}
    for split in ("train", "valid", "test"):
        data_path = data_folder / f"{split}.npz"
        if sha256(data_path) != prepared["splits"][split]["sha256"]:
            raise ValueError(f"Prepared cache content hash mismatch: {split}")
        rows = load_rows(data_path)
        path = run / f"{split}_embedding.npz"
        actual = sha256(path)
        if actual != manifest["embeddings"][split]["embedding_sha256"]:
            raise ValueError(f"Reference embedding hash mismatch: {split}")
        with np.load(path, allow_pickle=False) as archive:
            if not np.array_equal(archive["source_id"], rows.source_id):
                raise ValueError(f"Reference source-row ordering differs: {split}")
            if not np.array_equal(archive["behavior"], rows.behavior):
                raise ValueError(f"Reference behavior differs from frozen source: {split}")
            shape = archive["embedding"].shape
            if shape != (len(rows.source_id), 32) or list(shape) != manifest["embeddings"][split]["shape"]:
                raise ValueError(f"Reference embedding cardinality differs: {split}")
        entries[split] = dict(path=str(path), embedding_sha256=actual,
                              prepared_data_sha256=prepared["splits"][split]["sha256"],
                              rows=len(rows.source_id), row_ids_exact=True, behavior_exact=True,
                              shape=list(shape))
    proof = dict(created_utc=now(), config=expected, parameters=params,
                 parent_source_path=str(source), parent_manifest_path=str(manifest_path),
                 parent_manifest_sha256=sha256(manifest_path),
                 checkpoint_sha256=checkpoint_hash, accepted_script_hashes=script_hashes,
                 source_data_manifest_sha256=sha256(data_manifest),
                 verifier_source_sha256=sha256(Path(__file__)), embeddings=entries,
                 verification_only=True, scientific_scores_computed=False,
                 test_access="source/metadata equivalence and content hashes only; no scientific outcome evaluated")
    proof["signature"] = config_hash({key: value for key, value in proof.items() if key != "created_utc"})
    path = output / "stage00_reference_verification" / rat / f"fold_{fold}" / CONDITIONS[condition] / f"seed_{seed}" / "proof.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["signature"] != proof["signature"]:
            raise ValueError("Reference proof dependencies changed; new version required")
        proof = old
    else:
        _write_json(path, proof)
    return run, data_folder, proof


def evaluate_reference(output, rat, fold, seed, condition):
    """Authorized downstream stage: export metadata, then score without encoder fit."""
    from .downstream import _LazyArchive, score_arrays
    output = _output(output)
    run, data_folder, proof = verify_reference(output, rat, fold, seed, condition)
    method = CONDITIONS[condition]
    key = Path(rat) / f"fold_{fold}" / method / f"seed_{seed}"
    destination = output / "stage02_reference" / key
    path = destination / "manifest.json"
    signature = config_hash(dict(proof=proof["signature"], exporter_source=sha256(Path(__file__))))
    config = dict(rat=rat, fold=fold, seed=seed, version=method, condition=condition,
                  architecture="offset10-model", output_dimension=32, batch_size=512,
                  learning_rate=3e-4, max_updates=10000, input_policy="legacy_concat",
                  reused_reference=True, cebra_version="0.6.1")
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["signature"] != signature:
            raise ValueError("Reference export dependency mismatch")
        for split, entry in record["embeddings"].items():
            if sha256(destination / f"{split}_embedding.npz") != entry["embedding_sha256"]:
                raise ValueError("Exported reference embedding hash mismatch")
    else:
        if destination.exists() and any(destination.iterdir()):
            raise ValueError("Incomplete reference export requires inspection")
        destination.mkdir(parents=True, exist_ok=True)
        record = dict(signature=signature, created_utc=now(), config=config, seed=seed,
                      rat=rat, fold=fold, version=method, parent_verification_signature=proof["signature"],
                      parent_checkpoint_sha256=proof["checkpoint_sha256"],
                      parent_manifest_sha256=proof["parent_manifest_sha256"],
                      raw_semantics="native normalized official CNN output, not pre-normalization activations",
                      embeddings={})
        for split in ("train", "valid", "test"):
            rows = load_rows(data_folder / f"{split}.npz")
            with np.load(run / f"{split}_embedding.npz", allow_pickle=False) as archive:
                raw = archive["embedding"].copy()
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            if not np.isfinite(raw).all() or np.any(norms <= 0):
                raise ValueError("Invalid frozen native embedding")
            unit = raw / norms
            mask = np.zeros(len(raw), dtype=bool)
            mask[eligible_centers(rows)] = True
            file = destination / f"{split}_embedding.npz"
            np.savez_compressed(file, embedding=raw, embedding_raw=raw, embedding_unit=unit,
                                behavior=rows.behavior, source_id=rows.source_id, trial_id=rows.trial_id,
                                time_id=rows.time_id, session_id=rows.session_id, valid_mask=mask,
                                split=split, rat=rat, fold=fold, seed=seed,
                                model="offset10-model", objective=method)
            record["embeddings"][split] = dict(embedding_sha256=sha256(file), shape=list(raw.shape),
                                                valid_rows=int(mask.sum()),
                                                parent_embedding_sha256=proof["embeddings"][split]["embedding_sha256"])
        _write_json(path, record)
    metric_path = output / "stage03_reference" / key / "metrics.json"
    metric_signature = config_hash(dict(parent=sha256(path), scorer=sha256(Path(__file__).parent / "downstream.py")))
    if metric_path.exists():
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        if metrics["signature"] != metric_signature:
            raise ValueError("Reference decoder dependency mismatch; new downstream version required")
    else:
        metrics = dict(signature=metric_signature, config=config, created_utc=now(),
                       parent_embedding_manifest_sha256=sha256(path),
                       metrics=score_arrays({split: _LazyArchive(destination / f"{split}_embedding.npz")
                                             for split in ("train", "valid", "test")}))
        _write_json(metric_path, metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--evaluate", action="store_true", help="Only after loss/fit technical qualification gates pass")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "user_soft_rat_v1")
    parser.add_argument("--rat", choices=RATS)
    parser.add_argument("--fold", type=int, choices=range(3))
    parser.add_argument("--seed", type=int, choices=range(42, 52))
    parser.add_argument("--condition", choices=tuple(CONDITIONS))
    args = parser.parse_args()
    rats = (args.rat,) if args.rat else RATS
    folds = (args.fold,) if args.fold is not None else range(3)
    seeds = (args.seed,) if args.seed is not None else range(42, 52)
    conditions = (args.condition,) if args.condition else tuple(CONDITIONS)
    count = 0
    for rat, fold, seed, condition in itertools.product(rats, folds, seeds, conditions):
        function = evaluate_reference if args.evaluate else verify_reference
        function(args.output, rat, fold, seed, condition)
        count += 1
    print(json.dumps(dict(mode="evaluate" if args.evaluate else "verify_only", completed=count,
                         encoder_fits=0, output=str(args.output)), indent=2))


if __name__ == "__main__":
    main()
