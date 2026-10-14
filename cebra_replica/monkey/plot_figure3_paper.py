"""Render CEBRA portions of Figure 3 from local frozen embeddings.

Uses the authors' panel dimensions, colors, coordinate choices, markers and
labels. Missing comparison methods are never filled from author reference data.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fit_figure3_paper import PROJECT, SEED


ROOT = PROJECT / "results" / "monkey_figure3_paper_v1"
EMBED = ROOT / "stage02_embedding"
FIGURES = ROOT / "stage03_figures"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(name: str) -> dict:
    path = EMBED / name / f"seed_{SEED}" / "all_trials.npz"
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def clear_axis(axis) -> None:
    for dim in (axis.xaxis, axis.yaxis, axis.zaxis):
        dim.set_pane_color((1, 1, 1, 0))
        dim.set_ticks([])
        dim._axinfo["grid"]["color"] = (1, 1, 1, 0)


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    png = FIGURES / f"{name}.png"
    pdf = FIGURES / f"{name}.pdf"
    if png.exists() and pdf.exists():
        plt.close(fig)
        return
    if png.exists() or pdf.exists():
        raise FileExistsError(f"Partial pre-existing figure pair: {name}")
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_3b_cebra_only() -> None:
    names = ("figure3b_e_active_target_4d", "figure3b_active_time_4d")
    specs = (("CEBRA-Behavior", (2, 0)), ("CEBRA-Time", (0, 1)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for axis, name, (title, coords) in zip(axes, names, specs):
        data = load(name)
        trials = data["embedding_raw"].reshape(-1, 600, 4)
        direction = data["direction"].reshape(-1, 600)[:, 0]
        for label in range(8):
            if not np.any(direction == label):
                raise ValueError(f"Missing direction {label} for {name}")
            mean = trials[direction == label].mean(axis=0)
            axis.plot(mean[:, coords[0]], mean[:, coords[1]], color=plt.cm.hsv(label / 8), linewidth=3)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(title, fontsize=20)
    save(fig, "FIGURE3B_CEBRA_ONLY_PAPER_SETTINGS")


def plot_3cd(name: str, title: str, output_name: str) -> None:
    data = load(name)
    features = data["embedding_raw"]
    labels = data["position_xy"]
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(title, fontsize=20)
    for slot, column, text in ((121, 0, "x pos"), (122, 1, "y pos")):
        axis = fig.add_subplot(slot, projection="3d")
        axis.set_title(text)
        points = axis.scatter(features[:, 0], features[:, 1], features[:, 2], c=labels[:, column], cmap="seismic", s=0.05, vmin=-15, vmax=15)
        clear_axis(axis)
        if column == 1:
            bar = plt.colorbar(points, fraction=0.03, pad=0.05, ticks=np.linspace(-15, 15, 7))
            bar.ax.tick_params(labelsize=15)
            bar.ax.set_title("(cm)", fontsize=10)
    save(fig, output_name)


def mean_trajectories(data: dict, passive: bool, mode: str) -> list[np.ndarray]:
    trial_passive = data["passive"].reshape(-1, 600)[:, 0]
    trial_direction = data["direction"].reshape(-1, 600)[:, 0]
    trials = data["embedding_raw"].reshape(-1, 600, 4)
    means = []
    for label in range(8):
        selected = trials[(trial_passive == passive) & (trial_direction == label)]
        if len(selected) == 0:
            raise ValueError(f"Missing direction {label}, passive={passive}")
        mean = selected.mean(axis=0)
        if mode == "global":
            mean = mean / np.linalg.norm(mean)
        else:
            mean = mean / np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
        means.append(mean)
    return means


def plot_3efg(name_active: str, name_passive: str | None, title: str, coords: tuple[int, int, int], mode: str, output_name: str) -> None:
    active = load(name_active)
    passive = load(name_passive) if name_passive else active
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(title, fontsize=20)
    for slot, dataset, is_passive, label in ((121, active, False, "Active"), (122, passive, True, "Passive")):
        axis = fig.add_subplot(slot, projection="3d")
        for direction, trajectory in enumerate(mean_trajectories(dataset, is_passive, mode)):
            color = plt.cm.hsv(direction / 8)
            x, y, z = (trajectory[:, index] for index in coords)
            axis.plot(x, y, z, color=color)
            axis.plot(x[0], y[0], z[0], color=color, marker="o", markersize=10)
            axis.plot(x[-1], y[-1], z[-1], color=color, marker="<", markersize=10)
        clear_axis(axis)
        axis.set_title(label, fontsize=20)
    save(fig, output_name)


def main() -> None:
    expected = (
        "figure3b_e_active_target_4d", "figure3b_active_time_4d",
        "figure3c_active_position_3d", "figure3d_active_time_3d",
        "figure3e_passive_target_4d", "figure3f_all_target_ap_4d",
        "figure3g_all_position_ap_4d",
    )
    for name in expected:
        if not (EMBED / name / f"seed_{SEED}" / "EMBEDDING_PROVENANCE.json").exists():
            raise FileNotFoundError(f"Missing frozen embedding: {name}")
    plot_3b_cebra_only()
    plot_3cd("figure3c_active_position_3d", "CEBRA-Behavior, labels - continuous position", "FIGURE3C_PAPER_SETTINGS")
    plot_3cd("figure3d_active_time_3d", "CEBRA-Time, labels - continuous position", "FIGURE3D_PAPER_SETTINGS")
    plot_3efg("figure3b_e_active_target_4d", "figure3e_passive_target_4d", "Separately trained active/passive trials with direction (1-8) labels", (1, 2, 0), "global", "FIGURE3E_PAPER_SETTINGS")
    plot_3efg("figure3f_all_target_ap_4d", None, "Concurrently trained active/passive trials with direction (1-8) + active/passive trial info labels", (2, 0, 1), "row", "FIGURE3F_PAPER_SETTINGS")
    plot_3efg("figure3g_all_position_ap_4d", None, "Trained with continuous position label", (0, 1, 2), "row", "FIGURE3G_PAPER_SETTINGS")
    status = {
        "figure3a": "complete_in_monkey_figure3_v1_stage00",
        "figure3b": "CEBRA-only; piVAE, autoLFADS, tSNE, UMAP pending",
        "figure3c": "complete_local_fit_descriptive",
        "figure3d": "complete_local_fit_descriptive",
        "figure3e": "complete_local_fit_descriptive",
        "figure3f": "complete_local_fit_descriptive",
        "figure3g": "complete_local_fit_descriptive",
        "figure3h": "pending_5_seed_5_dimension_decoding",
        "figure3i": "pending_decoded_trajectory",
        "note": "Reference HDF5 was not used to render local results. Full-sample descriptive, not held-out.",
    }
    status_path = FIGURES / "PANEL_STATUS.json"
    if not status_path.exists():
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    parent_hashes = {name: sha256(EMBED / name / f"seed_{SEED}" / "all_trials.npz") for name in expected}
    provenance = {
        "script_sha256": sha256(Path(__file__)),
        "reference_plot_source": "https://github.com/AdaptiveMotorControlLab/cebra-figures/blob/main/src/Figure3.py",
        "parent_embedding_sha256": parent_hashes,
        "figure_sha256": {p.name: sha256(p) for p in FIGURES.glob("FIGURE3*.*")},
        "author_reference_results_used_as_local_data": False,
        "seed": SEED,
        "split": "all_descriptive",
    }
    provenance_path = FIGURES / "FIGURE_PROVENANCE.json"
    if provenance_path.exists():
        old = json.loads(provenance_path.read_text(encoding="utf-8"))
        if old["parent_embedding_sha256"] != parent_hashes or old["figure_sha256"] != provenance["figure_sha256"]:
            raise ValueError("Figure provenance mismatch on resume")
    else:
        provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Figures in {FIGURES}")


if __name__ == "__main__":
    main()
