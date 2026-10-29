"""Native 3D, full-recording CEBRA fits for the published ED3 settings."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "rat_paper_replication_2023_v1" / "ED3" / "cebra"
RATS = ("achilles", "buddy", "cicero", "gatsby")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rat", choices=RATS, required=True)
    p.add_argument("--condition", choices=("behavior", "time"), required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    import cebra
    from cebra.datasets.hippocampus import SingleRatDataset

    source = ROOT / "data" / "rat_hippocampus" / f"{a.rat}.jl"
    ds = SingleRatDataset(name=a.rat, root=ROOT / "data", download=False)
    params = dict(model_architecture="offset10-model", output_dimension=3,
                  batch_size=512, learning_rate=3e-4, time_offsets=10,
                  temperature=1.0 if a.condition == "behavior" else 2.25,
                  max_iterations=10000 if a.condition == "behavior" else 4000,
                  distance="cosine", conditional="time_delta" if a.condition == "behavior" else "time",
                  device="cuda_if_available", verbose=False)
    config = dict(figure="ED3", rat=a.rat, condition=a.condition, seed=a.seed,
                  params=params, data_sha256=digest(source), script_sha256=digest(Path(__file__)),
                  input_scope="full recording", cebra_version=cebra.__version__)
    dest = OUT / a.rat / a.condition / f"seed_{a.seed}"
    print(json.dumps(dict(config=config, output=str(dest), rows=len(ds)), indent=2), flush=True)
    if a.dry_run:
        return
    if dest.exists():
        manifest = dest / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text())["config"] == config:
            if all((dest / f).exists() for f in ("checkpoint.pt", "embedding.npz")):
                print("REUSE", dest, flush=True)
                return
        raise RuntimeError(f"Existing output requires inspection: {dest}")
    dest.mkdir(parents=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    model = cebra.CEBRA(**params)
    started = time.perf_counter()
    x = ds.neural.numpy()
    model.fit(x, ds.continuous_index.numpy()) if a.condition == "behavior" else model.fit(x)
    model.save(dest / "checkpoint.pt")
    z = np.asarray(model.transform(x), dtype=np.float32)
    if z.shape != (len(ds), 3) or not np.isfinite(z).all():
        raise RuntimeError(f"Invalid embedding: {z.shape}")
    np.savez_compressed(dest / "embedding.npz", embedding=z,
                        behavior=ds.continuous_index.numpy(), source_id=np.arange(len(ds)))
    record = dict(config=config, fit_seconds=time.perf_counter() - started,
                  checkpoint_sha256=digest(dest / "checkpoint.pt"),
                  embedding_sha256=digest(dest / "embedding.npz"),
                  device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    (dest / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print("COMPLETE", dest, flush=True)


if __name__ == "__main__":
    main()
