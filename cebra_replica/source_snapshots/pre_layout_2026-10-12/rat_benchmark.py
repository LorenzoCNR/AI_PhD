"""Immutable, trial-split rat CEBRA fit and embedding export.

This is the first transferable benchmark cell, not a literal Figure 2 demo run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
CONDITIONS = ("time", "position", "direction", "position_direction",
              "position_shuffled", "direction_shuffled", "position_direction_shuffled")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_ids(dataset) -> np.ndarray:
    return np.concatenate([np.arange(s.start, s.stop, dtype=np.int64)
                           for s in dataset.selected_indices])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rat", choices=("achilles", "buddy", "cicero", "gatsby"), default="achilles")
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--condition", choices=CONDITIONS, default="position_direction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from cebra.datasets.hippocampus import SingleRatTrialSplitDataset

    splits = {name: SingleRatTrialSplitDataset(name=args.rat, split_no=args.fold,
                                                split=name, root=ROOT / "data")
              for name in ("train", "valid", "test")}
    ids = {name: source_ids(dataset) for name, dataset in splits.items()}
    for name, dataset in splits.items():
        if len(ids[name]) != len(dataset.neural) or len(dataset.neural) != len(dataset.continuous_index):
            raise AssertionError(f"Cardinality mismatch: {name}")
    if any(np.intersect1d(ids[a], ids[b]).size for a, b in
           (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise AssertionError("Trial split has overlapping source rows")
    source = ROOT / "data" / "rat_hippocampus" / f"{args.rat}.jl"
    print(json.dumps({"rat": args.rat, "fold": args.fold, "condition": args.condition,
                      "seed": args.seed, "max_iterations": args.max_iterations,
                      "source_sha256": file_hash(source),
                      "splits": {name: {"rows": len(ids[name]),
                                        "trials": len(splits[name].selected_indices)}
                                 for name in splits}}, indent=2), flush=True)
    if args.dry_run:
        return

    import cebra

    out = ROOT / "results" / "rat_trial_benchmark_v1" / args.rat / f"fold_{args.fold}" / args.condition / f"seed_{args.seed}"
    result_path = out / "result.json"
    checkpoint = out / "checkpoint.pt"
    expected = {"rat": args.rat, "fold": args.fold, "condition": args.condition,
                "seed": args.seed, "max_iterations": args.max_iterations,
                "source_sha256": file_hash(source), "script_sha256": file_hash(Path(__file__))}
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if all(result.get(k) == v for k, v in expected.items()) and checkpoint.exists() and all(
                (out / f"{s}_embedding.npz").exists() for s in splits):
            print(f"REUSE {out}", flush=True)
            return
        raise RuntimeError(f"Existing result differs or is incomplete: {out}")
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Partial directory needs inspection before retry: {out}")
    out.mkdir(parents=True, exist_ok=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    params = dict(model_architecture="offset10-model", batch_size=512,
                  learning_rate=3e-4, temperature=1.12 if args.condition == "time" else 1.0,
                  output_dimension=32, max_iterations=args.max_iterations, distance="cosine",
                  conditional="time" if args.condition == "time" else "time_delta",
                  device="cuda_if_available", verbose=False, time_offsets=10)
    train = splits["train"]
    neural = train.neural.numpy()
    behavior = train.continuous_index.numpy()
    rng = np.random.default_rng(args.seed + 10_000)
    if args.condition == "time":
        label = None
    elif args.condition.startswith("position_direction"):
        label = behavior
    elif args.condition.startswith("position"):
        label = behavior[:, 0]
    else:
        label = behavior[:, 1]
    if "shuffled" in args.condition:
        label = label[rng.permutation(len(label))]
    model = cebra.CEBRA(**params)
    started = time.perf_counter()
    print(f"START FIT {out}", flush=True)
    if label is None:
        model.fit(neural)
    else:
        model.fit(neural, label)
    fit_seconds = time.perf_counter() - started
    model.save(checkpoint)
    outputs = {}
    for name, dataset in splits.items():
        embedding = np.asarray(model.transform(dataset.neural.numpy()), dtype=np.float32)
        if embedding.shape != (len(dataset.neural), params["output_dimension"]) or not np.isfinite(embedding).all():
            raise AssertionError(f"Invalid embedding: {name}, {embedding.shape}")
        path = out / f"{name}_embedding.npz"
        np.savez_compressed(path, embedding=embedding,
                            behavior=dataset.continuous_index.numpy(), source_id=ids[name],
                            split=np.array(name), rat=np.array(args.rat), fold=np.array(args.fold),
                            condition=np.array(args.condition), seed=np.array(args.seed))
        outputs[name] = {"embedding_sha256": file_hash(path), "shape": list(embedding.shape)}
    result = dict(**expected, created_utc=datetime.now(timezone.utc).isoformat(),
                  parameters=params, fit_seconds=fit_seconds,
                  checkpoint_sha256=file_hash(checkpoint), embeddings=outputs,
                  cebra_version=getattr(cebra, "__version__", "unknown"),
                  torch_version=torch.__version__,
                  device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                  scope="trial_held_out_benchmark; no test metric used for model selection")
    result_path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"COMPLETE FIT {out} in {fit_seconds:.1f} s", flush=True)


if __name__ == "__main__":
    main()
