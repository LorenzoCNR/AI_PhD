"""Verify the official Area 2 bundles and produce the descriptive Figure 3a panel.

The split mapping is recovered by exact trial-content hashes, never by assuming
that ``all_all`` is ordered train/validation/test. This stage performs no fit.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data" / "monkey_reaching_preload_smth_40"
OUTPUT = PROJECT / "results" / "monkey_figure3_v1" / "stage00_data"
EXPECTED_MD5 = {
    "all_all.jl": "dea556301fa4fafa86e28cf8621cab5a",
    "all_train.jl": "e280e4cd86969e6fd8bfd3a8f402b2fe",
    "all_valid.jl": "8cd25169d31f83ae01b03f7b1b939723",
    "all_test.jl": "25d3ff2c15014db8b8bf2543482ae881",
    "active_all.jl": "c626acea5062122f5a68ef18d3e45e51",
    "active_train.jl": "72a48056691078eee22c36c1992b1d37",
    "active_valid.jl": "dd58eb1e589361b4132f34b22af56b79",
    "active_test.jl": "35b7e060008a8722c536584c4748f2ea",
    "passive_all.jl": "bbb1bc9d8eec583a46f6673470fc98ad",
    "passive_train.jl": "f22e05a69f70e18ba823a0a89162a45c",
    "passive_valid.jl": "2dcc10c27631b95a075eaa2d2297bb4a",
    "passive_test.jl": "42453ae3e4fd27d82d297f78c13cd6b7",
}


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def trial_hash(data: dict, trial: int) -> str:
    length = int(data["trial_len"])
    rows = np.ascontiguousarray(data["spikes"][trial * length : (trial + 1) * length])
    return hashlib.sha256(rows.tobytes()).hexdigest()


def verify_and_map() -> tuple[dict, dict]:
    files = {}
    for name, expected in EXPECTED_MD5.items():
        path = DATA / name
        actual = digest(path, "md5")
        if actual != expected:
            raise ValueError(f"Official checksum mismatch: {name}: {actual} != {expected}")
        files[name] = {"md5": actual, "sha256": digest(path, "sha256")}

    datasets = {name: joblib.load(DATA / name) for name in EXPECTED_MD5}
    complete = datasets["all_all.jl"]
    trial_len = int(complete["trial_len"])
    assert trial_len == 600 and complete["spikes"].shape[1] == 65
    assert complete["spikes"].shape[0] == int(complete["num_trials"]) * trial_len
    keys = {trial_hash(complete, i): i for i in range(int(complete["num_trials"]))}
    if len(keys) != int(complete["num_trials"]):
        raise ValueError("Duplicate spike trial hashes in all_all")

    mapping = {}
    seen = set()
    for split in ("train", "valid", "test"):
        part = datasets[f"all_{split}.jl"]
        indices = []
        for i in range(int(part["num_trials"])):
            key = trial_hash(part, i)
            if key not in keys:
                raise ValueError(f"{split} trial {i} missing from all_all")
            j = keys[key]
            for field in ("spikes", "pos", "vel"):
                start, stop = i * trial_len, (i + 1) * trial_len
                parent_start, parent_stop = j * trial_len, (j + 1) * trial_len
                if not np.array_equal(part[field][start:stop], complete[field][parent_start:parent_stop]):
                    raise ValueError(f"{split} trial {i} {field} mismatch")
            for field in ("passive", "movement_dir", "movement_dir_actpas"):
                if part[field][i] != complete[field][j]:
                    raise ValueError(f"{split} trial {i} {field} mismatch")
            if j in seen:
                raise ValueError(f"Overlapping split assignment for all trial {j}")
            seen.add(j)
            indices.append(j)
        mapping[split] = indices
    if seen != set(range(int(complete["num_trials"]))):
        raise ValueError("Split trial IDs do not partition all_all")

    session_mapping = {}
    for session, passive in (("active", False), ("passive", True)):
        session_mapping[session] = {}
        for split in ("all", "train", "valid", "test"):
            part = datasets[f"{session}_{split}.jl"]
            indices = []
            for i in range(int(part["num_trials"])):
                j = keys.get(trial_hash(part, i))
                if j is None or bool(complete["passive"][j]) != passive:
                    raise ValueError(f"{session}_{split} trial {i} not in matching all_all population")
                start, stop = i * trial_len, (i + 1) * trial_len
                parent_start, parent_stop = j * trial_len, (j + 1) * trial_len
                for field in ("spikes", "pos", "vel"):
                    if not np.array_equal(part[field][start:stop], complete[field][parent_start:parent_stop]):
                        raise ValueError(f"{session}_{split} trial {i} {field} mismatch")
                for field in ("movement_dir", "movement_dir_actpas"):
                    if part[field][i] != complete[field][j]:
                        raise ValueError(f"{session}_{split} trial {i} {field} mismatch")
                indices.append(j)
            session_mapping[session][split] = indices
        all_ids = set(session_mapping[session]["all"])
        split_ids = [set(session_mapping[session][split]) for split in ("train", "valid", "test")]
        if any(split_ids[i] & split_ids[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError(f"{session} split overlap")
        if set.union(*split_ids) != all_ids:
            raise ValueError(f"{session} split coverage mismatch")
        if split_ids[0] != set(mapping["train"]) & all_ids:
            raise ValueError(f"{session} train differs from filtered all_train")

    return complete, {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": digest(Path(__file__), "sha256"),
        "source": "https://github.com/AdaptiveMotorControlLab/CEBRA/blob/main/cebra/datasets/monkey_reaching.py",
        "source_file_hashes": files,
        "trial_length_bins": trial_len,
        "bin_ms": 1,
        "neural_channels": int(complete["spikes"].shape[1]),
        "trial_counts": {name: int(data["num_trials"]) for name, data in datasets.items()},
        "all_trial_indices_by_split": mapping,
        "session_trial_indices_by_split": session_mapping,
        "split_mapping_method": "exact per-trial SHA-256 of spikes, verified against position, velocity and labels",
        "split_warning": "Session-specific valid/test are repartitioned after active/passive filtering and differ from filtered all_valid/all_test; do not mix their held-out definitions.",
        "fit_performed": False,
    }


def plot_figure3a(complete: dict, destination: Path) -> None:
    """Use the official Figure 3a scatter/HSV/size/axis-off visual grammar."""
    length = int(complete["trial_len"])
    pos = np.asarray(complete["pos"])
    passive = np.repeat(np.asarray(complete["passive"], dtype=bool), length)
    direction = np.repeat(np.asarray(complete["movement_dir"], dtype=int), length)
    if not np.isfinite(pos).all() or not np.isin(direction, np.arange(8)).all():
        raise ValueError("Invalid position or direction in Figure 3a data")

    with plt.rc_context({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"}):
        fig = plt.figure(figsize=(8, 4))
        for slot, mask in ((121, ~passive), (122, passive)):
            ax = fig.add_subplot(slot)
            ax.scatter(pos[mask, 0], pos[mask, 1], color=plt.cm.hsv(direction[mask] / 8), s=1)
            ax.axis("off")
        fig.savefig(destination / "FIGURE3A_BEHAVIOR_FROM_OFFICIAL_RAW.png", dpi=300, facecolor="white")
        fig.savefig(destination / "FIGURE3A_BEHAVIOR_FROM_OFFICIAL_RAW.pdf", facecolor="white")
        plt.close(fig)


def main() -> None:
    complete, provenance = verify_and_map()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = OUTPUT / "DATA_AND_SPLIT_PROVENANCE.json"
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old["source_file_hashes"] != provenance["source_file_hashes"] or old["all_trial_indices_by_split"] != provenance["all_trial_indices_by_split"]:
            raise ValueError("Existing immutable manifest conflicts with source data")
    else:
        manifest.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    if not (OUTPUT / "FIGURE3A_BEHAVIOR_FROM_OFFICIAL_RAW.png").exists():
        plot_figure3a(complete, OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "trial_counts": provenance["trial_counts"], "fit_performed": False}))


if __name__ == "__main__":
    main()
