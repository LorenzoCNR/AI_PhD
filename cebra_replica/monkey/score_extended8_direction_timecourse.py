"""Held-out time-resolved direction decoding for Extended Data Figure 8."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cebra
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score

from fit_figure3h import PROJECT, SEEDS, sha256


BASE = PROJECT / "results/monkey_figure3_paper_v1/stage02_embedding_heldout/direction/dim_4"
PIVAE = PROJECT / "results/monkey_extended8_paper_v1/stage02_embedding_heldout/conv_pivae"
OUTPUT = PROJECT / "results/monkey_extended8_paper_v1/stage04_direction_timecourse"
K = 36


def load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    provenance_path = path.parent / "EMBEDDING_PROVENANCE.json"
    if not provenance_path.exists():
        provenance_path = path.parent / "EMBEDDING.json"
    record = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected_hash = record.get("embedding_sha256", record.get("artifact_sha256"))
    if expected_hash is None or sha256(path) != expected_hash:
        raise ValueError(f"Embedding hash mismatch: {path}")
    record["verified_embedding_sha256"] = expected_hash
    with np.load(path, allow_pickle=False) as z:
        x = np.asarray(z["embedding_raw"], dtype=np.float32)
        y = np.asarray(z["direction"], dtype=np.int32)
        trial = np.asarray(z["trial_id"], dtype=np.int32)
        time = np.asarray(z["time_id"], dtype=np.int32)
        valid = np.asarray(z["valid_mask"], dtype=bool)
    if not (len(x) == len(y) == len(trial) == len(time) == len(valid)) or not valid.all():
        raise ValueError("Timecourse support cardinality mismatch")
    return x.reshape(-1, 600, x.shape[1]), y.reshape(-1, 600), trial.reshape(-1, 600), record


def score_knn(train_path: Path, test_path: Path, seed_label: str) -> tuple[np.ndarray, dict[str, str]]:
    x_train, y_train, trial_train, train_record = load(train_path)
    x_test, y_test, trial_test, test_record = load(test_path)
    if np.intersect1d(trial_train[:, 0], trial_test[:, 0]).size:
        raise ValueError("Train/test trial overlap")
    curves = np.empty(600, dtype=np.float32)
    for t in range(600):
        model = cebra.KNNDecoder(n_neighbors=K, metric="cosine")
        model.fit(x_train[:, t, :], y_train[:, t])
        curves[t] = 100 * accuracy_score(y_test[:, t], model.predict(x_test[:, t, :]))
    return curves, {"train_embedding_sha256": train_record["verified_embedding_sha256"],
                    "test_embedding_sha256": test_record["verified_embedding_sha256"],
                    "seed": seed_label}


def score_ridge() -> tuple[np.ndarray, dict[str, str]]:
    import joblib
    data_root = PROJECT / "data/monkey_reaching_preload_smth_40"
    train_data = joblib.load(data_root / "active_train.jl")
    test_data = joblib.load(data_root / "active_test.jl")
    x_train = np.asarray(train_data["spikes"], dtype=np.float32).reshape(-1, 600, 65)
    x_test = np.asarray(test_data["spikes"], dtype=np.float32).reshape(-1, 600, 65)
    y_train = np.repeat(np.asarray(train_data["movement_dir"], dtype=np.int32), 600).reshape(-1, 600)
    y_test = np.repeat(np.asarray(test_data["movement_dir"], dtype=np.int32), 600).reshape(-1, 600)
    curves = np.empty(600, dtype=np.float32)
    for t in range(600):
        model = RidgeClassifier(alpha=1.0)
        model.fit(x_train[:, t, :], y_train[:, t])
        curves[t] = 100 * accuracy_score(y_test[:, t], model.predict(x_test[:, t, :]))
    parents = {"active_train_sha256": sha256(data_root / "active_train.jl"),
               "active_test_sha256": sha256(data_root / "active_test.jl")}
    return curves, parents


def read_existing_curves(csv_path: Path) -> dict[str, list[np.ndarray]]:
    """Load and validate an already computed table without refitting decoders."""
    expected = {("CEBRA-Behavior", str(seed)) for seed in SEEDS}
    expected |= {("conv-piVAE", "42"), ("Ridge classifier", "ridge_alpha_1")}
    grouped: dict[tuple[str, str], list[tuple[int, int, float]]] = {}
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["method", "training_seed", "time_bin", "time_from_onset_ms", "accuracy_percent"]:
            raise ValueError("Existing timecourse CSV has unexpected columns")
        for row in reader:
            key = (row["method"], row["training_seed"])
            grouped.setdefault(key, []).append(
                (int(row["time_bin"]), int(row["time_from_onset_ms"]), float(row["accuracy_percent"]))
            )
    if set(grouped) != expected or sum(map(len, grouped.values())) != len(expected) * 600:
        raise ValueError("Existing timecourse CSV is incomplete or has unexpected method/seed rows")
    curves: dict[str, list[np.ndarray]] = {"CEBRA-Behavior": [], "conv-piVAE": [], "Ridge classifier": []}
    for key in sorted(expected, key=lambda item: (item[0], item[1])):
        rows = sorted(grouped[key], key=lambda item: item[0])
        if [row[0] for row in rows] != list(range(600)) or [row[1] for row in rows] != list(range(-100, 500)):
            raise ValueError(f"Unexpected time support in existing CSV for {key}")
        values = np.asarray([row[2] for row in rows], dtype=np.float32)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 100)):
            raise ValueError(f"Invalid accuracies in existing CSV for {key}")
        curves[key[0]].append(values)
    curves["CEBRA-Behavior"] = [curves["CEBRA-Behavior"][SEEDS.index(seed)] for seed in SEEDS]
    return curves


def embedding_parent_hashes(train_path: Path, test_path: Path, seed_label: str) -> dict[str, str]:
    _, _, train_trials, train_record = load(train_path)
    _, _, test_trials, test_record = load(test_path)
    if np.intersect1d(train_trials[:, 0], test_trials[:, 0]).size:
        raise ValueError("Train/test trial overlap while validating existing timecourse table")
    return {"train_embedding_sha256": train_record["verified_embedding_sha256"],
            "test_embedding_sha256": test_record["verified_embedding_sha256"],
            "seed": seed_label}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT / "DIRECTION_ACCURACY_BY_TIME.csv"
    plot_png = OUTPUT / "EXTENDED8_DIRECTION_DECODING_TIMECOURSE.png"
    plot_pdf = OUTPUT / "EXTENDED8_DIRECTION_DECODING_TIMECOURSE.pdf"
    provenance_path = OUTPUT / "PROVENANCE.json"
    if provenance_path.exists():
        prior = json.loads(provenance_path.read_text(encoding="utf-8"))
        if sha256(csv_path) != prior["csv_sha256"] or sha256(plot_png) != prior["figure_sha256"][plot_png.name] or sha256(plot_pdf) != prior["figure_sha256"][plot_pdf.name]:
            raise ValueError("Existing ED8 timecourse output changed")
        print("REUSE ED8 timecourse")
        return
    parents: dict[str, dict[str, str]] = {}
    if csv_path.exists():
        # The prior render failed after the complete CSV was written; preserve and reuse it.
        curves = read_existing_curves(csv_path)
        for seed in SEEDS:
            branch = BASE / f"seed_{seed}"
            parents[f"CEBRA-Behavior/seed_{seed}"] = embedding_parent_hashes(
                branch / "train/embedding_and_trials.npz", branch / "test/embedding_and_trials.npz", str(seed)
            )
        parents["conv-piVAE"] = embedding_parent_hashes(
            PIVAE / "train/seed_42/embedding_and_trials.npz",
            PIVAE / "test/seed_42/embedding_and_trials.npz", "42"
        )
        data_root = PROJECT / "data/monkey_reaching_preload_smth_40"
        parents["Ridge classifier"] = {
            "active_train_sha256": sha256(data_root / "active_train.jl"),
            "active_test_sha256": sha256(data_root / "active_test.jl"),
        }
    else:
        curves = {"CEBRA-Behavior": []}
        for seed in SEEDS:
            branch = BASE / f"seed_{seed}"
            curve, hashes = score_knn(branch / "train/embedding_and_trials.npz",
                                      branch / "test/embedding_and_trials.npz", str(seed))
            curves["CEBRA-Behavior"].append(curve)
            parents[f"CEBRA-Behavior/seed_{seed}"] = hashes
        pi_curve, pi_hashes = score_knn(PIVAE / "train/seed_42/embedding_and_trials.npz",
                                        PIVAE / "test/seed_42/embedding_and_trials.npz", "42")
        curves["conv-piVAE"] = [pi_curve]
        parents["conv-piVAE"] = pi_hashes
        ridge_curve, ridge_hashes = score_ridge()
        curves["Ridge classifier"] = [ridge_curve]
        parents["Ridge classifier"] = ridge_hashes
    mean_curves = {name: np.mean(values, axis=0) for name, values in curves.items()}
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("method", "training_seed", "time_bin", "time_from_onset_ms", "accuracy_percent"))
            for name, items in curves.items():
                for seed_index, curve in enumerate(items):
                    seed = SEEDS[seed_index] if name == "CEBRA-Behavior" else (42 if name == "conv-piVAE" else "ridge_alpha_1")
                    for t, value in enumerate(curve):
                        writer.writerow((name, seed, t, t - 100, float(value)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    styles = (("CEBRA-Behavior", "k", 1.0), ("conv-piVAE", "black", 0.5), ("Ridge classifier", "black", 0.2))
    with plt.rc_context({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, color, alpha in styles:
            ax.plot(mean_curves[name], label=name, color=color, alpha=alpha, lw=5)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.legend(frameon=False, fontsize=12)
        ax.set_ylabel("ACC (%)", fontsize=25)
        ax.set_yticks(np.linspace(20, 100, 5))
        ax.tick_params(axis="y", labelsize=25)
        ax.set_xlabel("Time from movement onset (ms)", fontsize=25)
        ax.set_xticks(np.linspace(0, 600, 7, dtype=int), np.linspace(-100, 500, 7, dtype=int), fontsize=25)
        fig.savefig(plot_png, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(plot_pdf, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    provenance_path.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(),
                                          "script_sha256": sha256(Path(__file__)),
                                          "parents": parents,
                                          "csv_sha256": sha256(csv_path),
                                          "figure_sha256": {path.name: sha256(path) for path in (plot_png, plot_pdf)},
                                          "split": "train-fit/test-score, active trials",
                                          "time_support": "600 bins, 1 ms; x-axis shifted by movement onset (-100 to 499 ms)",
                                          "CEBRA_and_piVAE_decoder": f"KNN cosine k={K}, fit separately at each time bin",
                                          "Ridge_classifier_alpha": 1.0,
                                          "piVAE_representation": "q(z|x), no inference-time direction labels",
                                          "CEBRA_curve": "mean of five seed-specific curves; replicate values retained in CSV"}, indent=2) + "\n", encoding="utf-8")
    print(f"COMPLETE {OUTPUT}")


if __name__ == "__main__":
    main()
