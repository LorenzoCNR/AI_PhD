"""Figure 3b from LOCAL frozen embeddings, official layout without autoLFADS."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "results/monkey_figure3_paper_v1"
FIGURE = ROOT / "stage03_figures/figure3b_comparison"
TRIAL_PROVENANCE = PROJECT / "results/monkey_figure3_v1/stage00_data/DATA_AND_SPLIT_PROVENANCE.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path, key: str, manifest: Path, ordinal_trial_ids: bool,
         active_trial_ids: np.ndarray, trial_length: int,
         expected_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if not path.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"Frozen local embedding missing: {path}")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    expected = record.get("artifact_sha256", record.get("embedding_sha256"))
    if expected and sha256(path) != expected:
        raise ValueError(f"Frozen embedding hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as artifact:
        embedding = np.asarray(artifact[key], dtype=np.float32)
        direction = np.asarray(artifact["direction"]).reshape(-1)
        trial = np.asarray(artifact["trial_id"], dtype=np.int64).reshape(-1)
        time = np.asarray(artifact["time_id"]).reshape(-1)
        valid = np.asarray(artifact["valid_mask"]).reshape(-1)
    if not all(len(array) == len(embedding) for array in (direction, trial, time, valid)):
        raise ValueError("Embedding/trial/time/direction/valid cardinality mismatch")
    if ordinal_trial_ids:
        if np.any(trial < 0) or np.any(trial >= len(active_trial_ids)):
            raise ValueError(f"Ordinal trial ID outside active-session map: {path}")
        trial = active_trial_ids[trial]
    if not np.all(valid):
        raise ValueError("Figure 3b requires complete active trials")
    keys = trial * trial_length + time
    if not np.array_equal(np.sort(keys), expected_keys):
        raise ValueError(f"Trial/time support differs from frozen active trials: {path}")
    order = np.argsort(keys, kind="stable")
    embedding, direction, keys = embedding[order], direction[order], keys[order]
    if not np.all(direction.reshape(-1, trial_length) == direction.reshape(-1, trial_length)[:, :1]):
        raise ValueError("Direction changes within a trial")
    if not np.isfinite(embedding).all():
        raise ValueError("Non-finite embedding")
    return embedding, direction, keys, sha256(path)


def main() -> None:
    cebra_root = ROOT / "stage02_embedding"
    baseline_root = ROOT / "stage02_embedding_baselines"
    specs = [
        ("CEBRA-Behavior", cebra_root / "figure3b_e_active_target_4d/seed_42/all_trials.npz",
         "embedding_raw", cebra_root / "figure3b_e_active_target_4d/seed_42/EMBEDDING_PROVENANCE.json", (2, 0), False),
        ("pi-VAE w/ label", baseline_root / "conv_pivae/seed_42/all_trials.npz",
         "with_label_raw", baseline_root / "conv_pivae/seed_42/EMBEDDING.json", (2, 3), True),
        ("CEBRA-Time", cebra_root / "figure3b_active_time_4d/seed_42/all_trials.npz",
         "embedding_raw", cebra_root / "figure3b_active_time_4d/seed_42/EMBEDDING_PROVENANCE.json", (0, 1), False),
        ("pi-VAE w/o", baseline_root / "conv_pivae/seed_42/all_trials.npz",
         "without_label_raw", baseline_root / "conv_pivae/seed_42/EMBEDDING.json", (2, 3), True),
        ("tSNE", baseline_root / "opentsne/seed_42/all_trials.npz",
         "embedding_raw", baseline_root / "opentsne/seed_42/EMBEDDING.json", (0, 1), True),
        ("UMAP", baseline_root / "umap/seed_42/all_trials.npz",
         "embedding_raw", baseline_root / "umap/seed_42/EMBEDDING.json", (0, 1), True),
    ]
    provenance_path = FIGURE / "PROVENANCE.json"
    png = FIGURE / "FIGURE3B_LOCAL_NO_AUTOLFADS.png"
    pdf = FIGURE / "FIGURE3B_LOCAL_NO_AUTOLFADS.pdf"
    if provenance_path.exists():
        record = json.loads(provenance_path.read_text(encoding="utf-8"))
        if sha256(png) != record["figure_sha256"][png.name] or sha256(pdf) != record["figure_sha256"][pdf.name]:
            raise ValueError("Previously rendered figure changed")
        print(f"REUSE {FIGURE}: figure hashes verified")
        return
    trial_provenance = json.loads(TRIAL_PROVENANCE.read_text(encoding="utf-8"))
    active_trial_ids = np.asarray(
        trial_provenance["session_trial_indices_by_split"]["active"]["all"], dtype=np.int64
    )
    trial_length = int(trial_provenance["trial_length_bins"])
    expected_keys = np.sort(
        np.repeat(active_trial_ids, trial_length) * trial_length
        + np.tile(np.arange(trial_length, dtype=np.int64), len(active_trial_ids))
    )
    loaded = [read(path, key, manifest, ordinal, active_trial_ids, trial_length, expected_keys)
              for _, path, key, manifest, _, ordinal in specs]
    reference_labels = loaded[0][1]
    reference_keys = loaded[0][2]
    if not all(np.array_equal(item[2], reference_keys) for item in loaded):
        raise ValueError("Methods differ in aligned trial/time support")
    if not all(np.array_equal(item[1], reference_labels) for item in loaded):
        raise ValueError("Methods differ in aligned trial-level direction labels")
    FIGURE.mkdir(parents=True, exist_ok=True)
    if png.exists() or pdf.exists():
        raise FileExistsError("Partial Figure 3b output exists")
    with plt.rc_context({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"}):
        figure = plt.figure(figsize=(30 * 6 / 7, 5))
        figure.subplots_adjust(wspace=0, hspace=0)
        for j, ((title, _, _, _, coords, _), (embedding, direction, _, _)) in enumerate(zip(specs, loaded)):
            axis = figure.add_subplot(1, len(specs), j + 1)
            trials = embedding.reshape(-1, trial_length, embedding.shape[1])
            trial_labels = direction.reshape(-1, trial_length)[:, 1]
            for label in range(8):
                selected = trials[trial_labels == label]
                if len(selected) == 0:
                    raise ValueError(f"Missing direction {label} for {title}")
                mean = selected.mean(axis=0)
                axis.plot(mean[:, coords[0]], mean[:, coords[1]],
                          color=plt.cm.hsv(label / 8), linewidth=3)
            for spine in axis.spines.values():
                spine.set_visible(False)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(title, fontsize=20)
        figure.savefig(png, dpi=300, facecolor="white", bbox_inches="tight")
        figure.savefig(pdf, facecolor="white", bbox_inches="tight")
        plt.close(figure)
    record = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "script_sha256": sha256(Path(__file__)),
              "parent_embedding_sha256": {title: result[3] for (title, *_), result in zip(specs, loaded)},
              "trial_mapping_sha256": sha256(TRIAL_PROVENANCE),
              "trial_alignment": "active-session ordinal IDs mapped to original all-recording IDs; sorted and matched by trial_id/time_id",
              "reference_plot_source": "CEBRA cebra-figures/src/Figure3.py",
              "reference_hdf5_used": False, "autoLFADS_included": False,
              "full_sample_descriptive": True,
              "pivae_variants": "two emissions from one direction-prior-trained checkpoint, outputs 0 and 6",
              "baseline_parameter_caveat": "UMAP/openTSNE settings follow the prior rat baseline; exact monkey paper settings were not documented",
              "figure_sha256": {path.name: sha256(path) for path in (png, pdf)}}
    provenance_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"COMPLETE {FIGURE}")


if __name__ == "__main__":
    main()
