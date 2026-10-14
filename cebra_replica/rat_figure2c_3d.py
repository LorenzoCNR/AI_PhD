"""Six 3D rat embedding conditions corresponding to CEBRA Figure 2c."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RATS = ("achilles", "buddy", "cicero", "gatsby")
CONDITIONS = ("position", "direction", "position_direction",
              "position_shuffled", "direction_shuffled", "position_direction_shuffled")
OUT = ROOT / "results" / "rat_figure2c_3d_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_metadata(position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    changes = np.where(position[1:, 1] != position[:-1, 1])[0]
    boundaries = np.append(np.insert(changes[1::2], 0, 0), len(position))
    trial_id = np.searchsorted(boundaries[1:], np.arange(len(position)), side="right").astype(np.int64)
    time_id = np.arange(len(position), dtype=np.int64) - boundaries[trial_id]
    return trial_id, time_id


def fit_one(rat: str, condition: str, seed: int, max_iterations: int) -> None:
    import cebra

    source = ROOT / "data" / "rat_hippocampus" / f"{rat}.jl"
    raw = joblib.load(source)
    neural = np.asarray(raw["spikes"], dtype=np.float32)
    behavior = np.asarray(raw["position"], dtype=np.float32)
    if neural.shape[0] != behavior.shape[0]:
        raise AssertionError("Neural/behavior cardinality mismatch")
    split_at = int(0.8 * len(neural))
    if split_at < 512 or len(neural) - split_at < 1:
        raise AssertionError("Insufficient train or test observations")
    out = OUT / rat / condition / f"seed_{seed}"
    manifest_path = out / "result.json"
    expected = {"rat": rat, "condition": condition, "seed": seed,
                "max_iterations": max_iterations, "output_dimension": 3,
                "source_sha256": sha256(source), "script_sha256": sha256(Path(__file__))}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Provenance mismatch: {manifest_path}")
        for name, digest in manifest["artifacts"].items():
            if sha256(out / name) != digest:
                raise RuntimeError(f"Artifact hash mismatch: {out / name}")
        print(f"REUSE {out}", flush=True)
        return
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Partial result requires inspection: {out}")
    out.mkdir(parents=True, exist_ok=False)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    label = behavior[:split_at]
    if condition.startswith("position_direction"):
        label = label.copy()
    elif condition.startswith("position"):
        label = label[:, 0].copy()
    else:
        label = label[:, 1].copy()
    if "shuffled" in condition:
        label = label[np.random.default_rng(seed + 10_000).permutation(len(label))]

    parameters = {"model_architecture": "offset10-model", "batch_size": 512,
                  "learning_rate": 3e-4, "temperature": 1.0,
                  "output_dimension": 3, "max_iterations": max_iterations,
                  "distance": "cosine", "conditional": "time_delta",
                  "device": "cuda_if_available", "verbose": False, "time_offsets": 10}
    model = cebra.CEBRA(**parameters)
    started = time.perf_counter()
    model.fit(neural[:split_at], label)
    fit_seconds = time.perf_counter() - started
    model.save(out / "checkpoint.pt")
    trial_id, time_id = source_metadata(behavior)
    artifacts = {"checkpoint.pt": sha256(out / "checkpoint.pt")}
    for name, indices in (("train", np.arange(split_at)),
                          ("test", np.arange(split_at, len(neural))),
                          ("all", np.arange(len(neural)))):
        embedding = np.asarray(model.transform(neural[indices]), dtype=np.float32)
        if embedding.shape != (len(indices), 3) or not np.isfinite(embedding).all():
            raise AssertionError(f"Invalid {name} embedding: {embedding.shape}")
        path = out / f"{name}_embedding.npz"
        np.savez_compressed(path, embedding=embedding, neural=neural[indices],
                            behavior=behavior[indices], source_id=indices,
                            trial_id=trial_id[indices], time_id=time_id[indices],
                            valid_mask=np.ones(len(indices), dtype=bool),
                            split=np.array(name), rat=np.array(rat),
                            condition=np.array(condition), seed=np.array(seed))
        artifacts[path.name] = sha256(path)
    manifest = dict(**expected, parameters=parameters,
                    split={"train": [0, split_at], "test": [split_at, len(neural)]},
                    fit_seconds=fit_seconds, artifacts=artifacts,
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    cebra_version=getattr(cebra, "__version__", "unknown"),
                    torch_version=torch.__version__,
                    device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                    scope="3D Figure-2c-style visualization; contiguous 80/20 demo split; "
                          "test not used for fitting or model selection")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"COMPLETE {rat}/{condition}/seed_{seed}: {fit_seconds:.1f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rat", choices=RATS, default="achilles")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--condition", choices=CONDITIONS)
    args = parser.parse_args()
    for condition in (args.condition,) if args.condition else CONDITIONS:
        fit_one(args.rat, condition, args.seed, args.max_iterations)


if __name__ == "__main__":
    main()
