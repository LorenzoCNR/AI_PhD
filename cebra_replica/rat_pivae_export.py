"""Export trial-split inputs for the vendored conv-piVAE benchmark.

This intentionally uses the same CEBRA SingleRatTrialSplitDataset, offset,
and indexing implementation as the reference conv-piVAE notebook. It does not
fit a model or compute any scientific outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
import cebra
from cebra.datasets.hippocampus import SingleRatTrialSplitDataset


RATS = ("achilles", "buddy", "cicero", "gatsby")
SPLITS = ("train", "valid", "test")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def content_hash(arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        h.update(name.encode("utf-8"))
        h.update(str(value.dtype).encode("ascii"))
        h.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        h.update(value.tobytes())
    return h.hexdigest()


def source_trial_map(position: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror the exact round-trip boundary rule in CEBRA hippocampus.py."""
    direction_change_idx = np.where(position[1:, 1] != position[:-1, 1])[0]
    trial_bounds = np.append(
        np.insert(direction_change_idx[1::2], 0, 0), len(position)
    ).astype(np.int64)
    if np.any(np.diff(trial_bounds) <= 0):
        raise ValueError("Non-increasing trial boundaries in rat source labels")
    source_rows = np.arange(len(position), dtype=np.int64)
    trial_id = np.searchsorted(trial_bounds[1:], source_rows, side="right").astype(np.int32)
    time_id = (source_rows - trial_bounds[trial_id]).astype(np.int32)
    return trial_bounds, trial_id, time_id


def extract_one(data_root: Path, out_root: Path, rat: str, fold: int, split: str,
                source_trial_id: np.ndarray) -> dict:
    dataset = SingleRatTrialSplitDataset(
        name=rat, split_no=fold, split=split, root=data_root
    )
    dataset.offset = cebra.data.datatypes.Offset(5, 5)
    anchors = torch.arange(len(dataset), dtype=torch.long)
    windows = dataset[anchors].detach().cpu().numpy().transpose(0, 2, 1)
    behavior = dataset.index.detach().cpu().numpy().astype(np.float32, copy=False)

    source_ids = np.concatenate([
        np.arange(section.start, section.stop, dtype=np.int64)
        for section in dataset.selected_indices
    ])
    expanded_local_ids = dataset.expand_index(anchors).detach().cpu().numpy().astype(np.int64)
    if len(source_ids) != len(dataset) or len(windows) != len(source_ids):
        raise AssertionError("dataset rows, source IDs, and window cardinalities differ")
    if windows.ndim != 3 or windows.shape[1] != 10 or behavior.shape != (len(windows), 3):
        raise AssertionError(f"unexpected arrays: windows={windows.shape}, behavior={behavior.shape}")

    window_source_ids = source_ids[expanded_local_ids]
    trial_id = source_trial_id[source_ids]
    window_trial_id = source_trial_id[window_source_ids]
    crosses_trial = np.any(window_trial_id != trial_id[:, None], axis=1)
    arrays = {
        "x": windows.astype(np.float32, copy=False),
        "u": behavior,
        "source_id": source_ids,
        "trial_id": trial_id.astype(np.int32, copy=False),
        "time_id": np.zeros(len(source_ids), dtype=np.int32),
        "window_source_id": window_source_ids,
        "window_trial_id": window_trial_id.astype(np.int32, copy=False),
        "crosses_trial_boundary": crosses_trial.astype(np.uint8),
    }
    out_dir = out_root / rat / f"fold_{fold}" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "input.npz"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite immutable input bundle: {target}")
    np.savez_compressed(target, **arrays)
    return {
        "path": str(target),
        "rows": int(len(windows)),
        "window_shape": list(windows.shape),
        "source_ids_sha256": hashlib.sha256(source_ids.tobytes()).hexdigest(),
        "bundle_content_sha256": content_hash(arrays),
        "bundle_file_sha256": sha256(target),
        "cross_trial_window_count": int(crosses_trial.sum()),
        "cross_trial_fraction": float(crosses_trial.mean()) if len(crosses_trial) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    data_root = project_root / "data"
    source_code = project_root / "CEBRA" / "cebra" / "datasets" / "hippocampus.py"
    base_code = project_root / "CEBRA" / "cebra" / "data" / "base.py"
    export_code = Path(__file__).resolve()
    manifest_path = args.output_root / "INPUT_BUNDLE_MANIFEST.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {manifest_path}")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "rats": list(RATS), "folds": [0, 1, 2],
            "splits": list(SPLITS), "time_window_bins": 10,
            "offset_left_bins": 5, "offset_right_bins": 5,
            "sampling_period_ms": 25,
            "indexing": "official local SingleRatTrialSplitDataset.expand_index",
            "trial_boundary_behavior": "preserved as implemented; crossings are counted, not silently filtered",
        },
        "software": {"python": platform.python_version(), "torch": torch.__version__, "cebra": cebra.__version__},
        "source_hashes": {
            "export_script_sha256": sha256(export_code),
            "hippocampus_py_sha256": sha256(source_code),
            "base_py_sha256": sha256(base_code),
        },
        "bundles": [],
    }
    for rat in RATS:
        raw_path = data_root / "rat_hippocampus" / f"{rat}.jl"
        raw = joblib.load(raw_path)
        position = np.asarray(raw["position"])
        trial_bounds, source_trial_id, source_time_id = source_trial_map(position)
        rat_entry = {
            "rat": rat,
            "raw_data_sha256": sha256(raw_path),
            "source_rows": int(len(position)),
            "trial_count": int(len(trial_bounds) - 1),
            "trial_bounds_sha256": hashlib.sha256(trial_bounds.tobytes()).hexdigest(),
            "splits": [],
        }
        # time_id is calculated from the same trial bounds for source rows.
        for fold in (0, 1, 2):
            for split in SPLITS:
                entry = extract_one(data_root, args.output_root, rat, fold, split, source_trial_id)
                # Fill the real within-trial time index for the anchored observations.
                z = np.load(entry["path"])
                arrays = {key: z[key] for key in z.files}
                z.close()
                arrays["time_id"] = source_time_id[arrays["source_id"]].astype(np.int32)
                target = Path(entry["path"])
                target.unlink()
                np.savez_compressed(target, **arrays)
                entry["bundle_content_sha256"] = content_hash(arrays)
                entry["bundle_file_sha256"] = sha256(target)
                rat_entry["splits"].append({"fold": fold, "split": split, **entry})
                print(f"EXPORTED {rat} fold={fold} {split}: {entry['rows']} rows, "
                      f"cross-trial windows={entry['cross_trial_window_count']}", flush=True)
        manifest["bundles"].append(rat_entry)

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"WROTE {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
