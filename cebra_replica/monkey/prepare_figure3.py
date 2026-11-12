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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data" / "monkey_reaching_preload_smth_40"
OUTPUT = PROJECT / "results" / "monkey_figure3_v1" / "stage00_data"
DELIVERY_PANEL_A = PROJECT / "results" / "figura_3_2026-10-29" / "plot" / "a"
OFFICIAL_FIG3_IMAGE = PROJECT / "cebra_replica" / "monkey" / "official_reference" / "fig3grey_source.png"
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


def plot_figure3a(complete: dict, destination: Path, stem: str) -> None:
    """Plot full-sample active/passive trajectories with the paper's HSV targets."""
    length = int(complete["trial_len"])
    pos = np.asarray(complete["pos"])
    passive = np.repeat(np.asarray(complete["passive"], dtype=bool), length)
    direction = np.repeat(np.asarray(complete["movement_dir"], dtype=int), length)
    if not np.isfinite(pos).all() or not np.isin(direction, np.arange(8)).all():
        raise ValueError("Invalid position or direction in Figure 3a data")

    # Match the official notebook's exact discrete colors: hsv(target / 8).
    cmap = ListedColormap([plt.cm.hsv(i / 8) for i in range(8)])
    norm = BoundaryNorm(np.arange(-0.5, 8.5, 1), cmap.N)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    official_image = plt.imread(OFFICIAL_FIG3_IMAGE)
    task_cartoon = official_image[45:155, 0:145, :].copy()
    # Keep the monkey and manipulandum, but exclude the source figure's labels.
    task_cartoon[108 - 45 :, 100:, 3] = 0
    with plt.rc_context({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"}):
        fig = plt.figure(figsize=(10, 7))
        task_ax = fig.add_axes([0.38, 0.735, 0.24, 0.22])
        task_ax.imshow(task_cartoon, interpolation="nearest")
        task_ax.axis("off")

        axes = (
            fig.add_axes([0.04, 0.19, 0.44, 0.49]),
            fig.add_axes([0.52, 0.19, 0.44, 0.49]),
        )
        for ax, mask, title in zip(axes, (~passive, passive), ("Active", "Passive")):
            ax.scatter(
                pos[mask, 0], pos[mask, 1], c=direction[mask], cmap=cmap,
                norm=norm, s=1, linewidths=0, rasterized=True,
            )
            ax.set_title(title, fontsize=15, pad=10)
            ax.set_aspect("equal", adjustable="datalim")
            ax.axis("off")

        colorbar = fig.colorbar(
            mappable, cax=fig.add_axes([0.20, 0.09, 0.60, 0.035]),
            orientation="horizontal", ticks=np.arange(8),
            boundaries=np.arange(-0.5, 8.5, 1), spacing="uniform",
        )
        colorbar.set_ticklabels([f"{45 * i}°" for i in range(8)])
        colorbar.set_label("Movement direction (°)", labelpad=6)
        colorbar.ax.tick_params(length=0, pad=5)
        destination.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination / f"{stem}.png", dpi=300, facecolor="white", bbox_inches="tight")
        fig.savefig(destination / f"{stem}.pdf", facecolor="white", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    complete, provenance = verify_and_map()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = OUTPUT / "DATA_AND_SPLIT_PROVENANCE.json"
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old["source_file_hashes"] != provenance["source_file_hashes"] or old["all_trial_indices_by_split"] != provenance["all_trial_indices_by_split"]:
            raise ValueError("Existing immutable manifest conflicts with source data")
    provenance["figure3a_render"] = {
        "renderer": str(Path(__file__).relative_to(PROJECT)),
        "renderer_sha256": digest(Path(__file__), "sha256"),
        "target_color_mapping": {
            str(45 * i): {"direction_code": i, "color": f"matplotlib hsv({i}/8)"}
            for i in range(8)
        },
        "angle_convention": "0 degrees along +x; increasing counter-clockwise",
        "panels": ["Active", "Passive"],
        "cartoon_inserted": True,
        "cartoon_source": "https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/04855fee-76f8-47cb-8c2c-ff0277dfc885/fig3grey.png?format=2500w",
        "cartoon_asset": str(OFFICIAL_FIG3_IMAGE.relative_to(PROJECT)),
        "cartoon_asset_sha256": digest(OFFICIAL_FIG3_IMAGE, "sha256"),
        "cartoon_note": "Task cartoon cropped from the official Figure 3 reference image; paper caption credits the cartoon to SciDraw.",
    }
    manifest.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    plot_figure3a(complete, OUTPUT, "FIGURE3A_BEHAVIOR_FROM_OFFICIAL_RAW")
    plot_figure3a(complete, DELIVERY_PANEL_A, "Figure_3a_Active_Passive_Movement")

    delivery_manifest = PROJECT / "results" / "figura_3_2026-10-29" / "processo" / "provenienza" / "a" / "PROVENANCE.json"
    rendered = [
        DELIVERY_PANEL_A / "Figure_3a_Active_Passive_Movement.pdf",
        DELIVERY_PANEL_A / "Figure_3a_Active_Passive_Movement.png",
    ]
    delivery_record = {
        "figure": "3a",
        "training_or_refitting_performed": False,
        "presentations": [{
            "description": "Full-sample hand trajectories, separated into active/passive trials and colored by target direction in degrees.",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "renderer": str(Path(__file__).relative_to(PROJECT)),
            "renderer_sha256": digest(Path(__file__), "sha256"),
            "data_source": "data/monkey_reaching_preload_smth_40/all_all.jl",
            "source_sha256": digest(DATA / "all_all.jl", "sha256"),
            "target_color_mapping": provenance["figure3a_render"]["target_color_mapping"],
            "angle_convention": provenance["figure3a_render"]["angle_convention"],
            "cartoon_inserted": True,
            "cartoon_source": provenance["figure3a_render"]["cartoon_source"],
            "cartoon_asset": provenance["figure3a_render"]["cartoon_asset"],
            "cartoon_asset_sha256": provenance["figure3a_render"]["cartoon_asset_sha256"],
            "cartoon_note": provenance["figure3a_render"]["cartoon_note"],
            "outputs": [
                {"path": str(path.relative_to(PROJECT)), "sha256": digest(path, "sha256")}
                for path in rendered
            ],
        }],
    }
    delivery_manifest.write_text(json.dumps(delivery_record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(DELIVERY_PANEL_A), "trial_counts": provenance["trial_counts"], "fit_performed": False, "angle_labels": [f"{45*i}°" for i in range(8)], "cartoon_inserted": True}))


if __name__ == "__main__":
    main()
