"""Add stable-Soft decoded position trajectories to the rat Figure 2D delivery.

Encoders and embeddings remain frozen. Position probes are fit on saved training
embeddings, Ridge alpha is selected on saved validation embeddings, and only the
fixed held-out test support is plotted. The output replaces the older reference-
only trajectory image after the combined replacement passes integrity checks.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cebra_replica"))
sys.path.insert(0, str(ROOT))

from rat_soft_stable_figures import RATS, SOURCES  # noqa: E402
from rat_figure_gap_closure_analysis import index_manifests, load_fit, verify_artifact  # noqa: E402
from rat_manifold_scoring import support_for  # noqa: E402
from user_soft.downstream import KNN_NEIGHBORS, RIDGE_ALPHA_GRID, _rows  # noqa: E402


DELIVERY = ROOT / "results" / "RAT_FIGURES" / "Figure_2" / "2D"
STAGE = DELIVERY / ".stage_method_trajectories_20261015"
BASELINE_ROOT = ROOT / "results" / "rat_figure2c_3d_v1"
METRIC_ROOTS = {
    ("CNN", "joint"): ROOT / "results" / "user_soft_rat_v1",
    ("CNN", "time_only"): ROOT / "results" / "user_soft_rat_time_only_v1",
    ("Transformer", "joint"): ROOT / "results" / "user_soft_rat_transformer_5seed_v1" / "joint",
    ("Transformer", "time_only"): ROOT / "results" / "user_soft_rat_transformer_5seed_v1" / "time_only",
}
SOFT_SPECS = (
    ("CNN", "joint", "Soft CNN: Time + Behavior", "#d73027"),
    ("CNN", "time_only", "Soft CNN: Time", "#fc8d59"),
    ("Transformer", "joint", "Soft Transformer: Time + Behavior", "#4575b4"),
    ("Transformer", "time_only", "Soft Transformer: Time", "#74add1"),
)
DISPLAY_ROWS = slice(320, 1020)  # Existing Figure 2D 700-bin held-out view.
TIME_BIN_S = 0.025


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No trajectory rows to save: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def soft_source_root(architecture: str, condition: str, rat: str) -> Path:
    source = SOURCES[(architecture, condition)]
    candidate = source / "stage02_embedding" / rat / "fold_0" / "soft_v1_stable" / "seed_42"
    if not (candidate / "test_embedding.npz").is_file() and architecture == "Transformer" and condition == "joint":
        source = ROOT / "results" / "user_soft_rat_transformer_v1"
    return source


def load_soft_bundle(architecture: str, condition: str, rat: str, support: dict) -> tuple[dict, dict]:
    source = soft_source_root(architecture, condition, rat)
    base = source / "stage02_embedding" / rat / "fold_0" / "soft_v1_stable" / "seed_42"
    manifest_path = base / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = {}
    parents = [{"path": manifest_path.relative_to(ROOT).as_posix(), "sha256": digest(manifest_path)}]
    for split in ("train", "valid", "test"):
        path = base / f"{split}_embedding.npz"
        expected = manifest["embeddings"][split]["embedding_sha256"]
        verify_artifact(path, expected)
        with np.load(path, allow_pickle=False) as archive:
            bundle = {key: archive[key].copy() for key in archive.files}
        n = len(bundle["source_id"])
        required = ("embedding_raw", "behavior", "source_id", "valid_mask", "trial_id", "time_id")
        if any(key not in bundle or len(bundle[key]) != n for key in required):
            raise RuntimeError(f"Embedding/metadata cardinality or field error: {path}")
        if bundle["embedding_raw"].ndim != 2 or bundle["behavior"].ndim != 2:
            raise RuntimeError(f"Invalid embedding/behavior dimensions: {path}")
        if bundle["valid_mask"].dtype != np.bool_:
            raise RuntimeError(f"valid_mask is not Boolean: {path}")
        expected_ids = np.asarray(support[split]["source_id"], dtype=np.int64)
        ids = np.asarray(bundle["source_id"], dtype=np.int64)
        if len(np.unique(ids)) != len(ids) or set(ids.tolist()) != set(expected_ids.tolist()):
            raise RuntimeError(f"Soft source-row support differs from frozen {split}: {path}")
        by_id = {int(value): i for i, value in enumerate(ids)}
        ordered = np.asarray([by_id[int(value)] for value in expected_ids], dtype=np.int64)
        if not np.array_equal(bundle["valid_mask"][ordered], support[split]["valid_mask"]):
            raise RuntimeError(f"Soft validity mask differs from frozen {split} support: {path}")
        if not np.array_equal(bundle["behavior"][ordered], support[split]["behavior"]):
            raise RuntimeError(f"Soft behavior labels differ from frozen {split} support: {path}")
        parents.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path),
                        "split": split, "rows": n})
        splits[split] = bundle

    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if np.intersect1d(splits[left]["source_id"], splits[right]["source_id"]).size:
            raise RuntimeError(f"Soft {left}/{right} source rows overlap: {rat}/{architecture}/{condition}")
        if np.intersect1d(splits[left]["trial_id"], splits[right]["trial_id"]).size:
            raise RuntimeError(f"Soft {left}/{right} trials overlap: {rat}/{architecture}/{condition}")

    metric_root = METRIC_ROOTS[(architecture, condition)]
    metric_path = metric_root / "stage03_metrics" / rat / "fold_0" / "soft_v1_stable" / "seed_42" / "metrics.json"
    if not metric_path.is_file() and architecture == "Transformer" and condition == "joint":
        metric_path = ROOT / "results" / "user_soft_rat_transformer_v1" / "stage03_metrics" / rat / "fold_0" / "soft_v1_stable" / "seed_42" / "metrics.json"
    metric_data = json.loads(metric_path.read_text(encoding="utf-8"))
    if metric_data.get("parent_embedding_manifest_sha256") != digest(manifest_path):
        raise RuntimeError(f"Metrics do not belong to this frozen embedding manifest: {metric_path}")
    valid_ridge = [row for row in metric_data["metrics"]
                   if row.get("representation") == "embedding_raw" and row.get("split") == "valid"
                   and row.get("decoder") == "ridge"]
    test_ridge = [row for row in metric_data["metrics"]
                  if row.get("representation") == "embedding_raw" and row.get("split") == "test"
                  and row.get("decoder") == "ridge"]
    if len(valid_ridge) != 1 or len(test_ridge) != 1:
        raise RuntimeError(f"Expected one raw Ridge record per split in {metric_path}")
    selected_alpha = float(valid_ridge[0]["selected_alpha"])
    if float(test_ridge[0]["selected_alpha"]) != selected_alpha:
        raise RuntimeError(f"Stored Ridge alpha differs between validation and test: {metric_path}")
    parents.append({"path": metric_path.relative_to(ROOT).as_posix(), "sha256": digest(metric_path),
                    "selected_alpha_from_validation": selected_alpha})
    return splits, {"manifest": manifest, "manifest_path": manifest_path,
                    "selected_alpha": selected_alpha, "parents": parents}


def load_reference_trajectory(index: dict, rat: str, label: str, all_test_ids: np.ndarray,
                              selected_ids: np.ndarray) -> tuple[np.ndarray, dict]:
    """Decode from the matching frozen CEBRA trial split, not the old row split."""
    from sklearn.neighbors import KNeighborsRegressor

    method = "CEBRA-Behavior-32D" if label == "CEBRA-Behavior" else "CEBRA-Time-32D"
    train = load_fit(index, "cebra", method, rat, 0, "42", "train")
    test = load_fit(index, "cebra", method, rat, 0, "42", "test")
    if not np.array_equal(test["source_id"], all_test_ids):
        raise RuntimeError(f"CEBRA {label} held-out support differs from the shared test rows: {rat}")
    train_mask = train["valid_mask"]
    test_mask = test["valid_mask"]
    model = KNeighborsRegressor(n_neighbors=KNN_NEIGHBORS, metric="cosine")
    model.fit(train["embedding"][train_mask], train["behavior"][train_mask, 0])
    prediction = np.full(len(test_mask), np.nan, dtype=float)
    prediction[test_mask] = model.predict(test["embedding"][test_mask])
    row_map = {int(value): i for i, value in enumerate(test["source_id"])}
    order = np.asarray([row_map[int(value)] for value in selected_ids], dtype=np.int64)
    if not np.all(test_mask[order]):
        raise RuntimeError(f"CEBRA {label} lacks valid rows on the plotted held-out trial: {rat}")
    parent = []
    for record in (train, test):
        parent.append({"path": record["manifest_path"].relative_to(ROOT).as_posix(),
                       "sha256": digest(record["manifest_path"]),
                       "embedding_path": record["embedding_path"].relative_to(ROOT).as_posix(),
                       "embedding_sha256": record["embedding_sha256"], "split": "train" if record is train else "test"})
    return prediction[order], {"label": label, "parents": parent}


def choose_display_trial(support: dict) -> np.ndarray:
    """Select the longest valid held-out trial; tie-break by smallest trial ID."""
    test = support["test"]
    candidates = []
    for trial in np.unique(test["trial_id"]):
        rows = np.flatnonzero((test["trial_id"] == trial) & test["valid_mask"])
        rows = rows[np.argsort(test["time_id"][rows], kind="stable")]
        if len(rows) and np.all(np.diff(test["time_id"][rows]) == 1):
            candidates.append((len(rows), int(trial), rows))
    if not candidates:
        raise RuntimeError("No contiguous valid held-out trial is available for the trajectory figure")
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


def build_figure(index: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    STAGE.mkdir(parents=True, exist_ok=False)
    supports = {rat: support_for(rat, 0) for rat in RATS}
    trial_rows = {rat: choose_display_trial(supports[rat]) for rat in RATS}
    common_n = min(len(rows) for rows in trial_rows.values())
    if common_n < 2:
        raise RuntimeError("Common contiguous held-out trial support is too short to plot")
    fig, axes = plt.subplots(4, 2, figsize=(17, 12), sharex=True, sharey="row", dpi=240)
    parents = []
    long_rows = []
    colors = {"ground_truth": "black", "CEBRA-Behavior": "#6235E0", "CEBRA-Time": "#13a5a5"}
    styles = {"joint": "-", "time_only": "--"}

    for row_idx, rat in enumerate(RATS):
        frozen = supports[rat]["test"]
        chosen_rows = trial_rows[rat][:common_n]
        trial_id = int(frozen["trial_id"][chosen_rows[0]])
        trial_ids = frozen["time_id"][chosen_rows].astype(np.int64)
        if not np.all(frozen["trial_id"][chosen_rows] == trial_id) or not np.all(np.diff(trial_ids) == 1):
            raise RuntimeError(f"Displayed support is not one contiguous held-out trial: {rat}")
        common_ids = np.asarray(frozen["source_id"][chosen_rows], dtype=np.int64)
        all_test_ids = np.asarray(frozen["source_id"], dtype=np.int64)
        true_pos_m = np.asarray(frozen["behavior"][chosen_rows, 0], dtype=float)
        if not np.isfinite(true_pos_m).all():
            raise RuntimeError(f"Nonfinite ground-truth position in held-out segment: {rat}")
        time_s = (trial_ids - trial_ids[0]).astype(float) * TIME_BIN_S
        knn_ax, ridge_ax = axes[row_idx]
        knn_ax.plot(time_s, true_pos_m * 100, color=colors["ground_truth"], lw=1.8,
                    label="Ground Truth", zorder=5)
        ridge_ax.plot(time_s, true_pos_m * 100, color=colors["ground_truth"], lw=1.8,
                      label="Ground Truth", zorder=5)

        for label in ("CEBRA-Behavior", "CEBRA-Time"):
            prediction, reference = load_reference_trajectory(index, rat, label, all_test_ids, common_ids)
            parents.extend(reference["parents"])
            knn_ax.plot(time_s, prediction * 100, color=colors[label], lw=1.2, label=label)
            for source_id, time_value, time_id, truth, decoded in zip(common_ids, time_s, trial_ids[:common_n], true_pos_m, prediction):
                long_rows.append({"rat": rat, "trial_id": trial_id, "method": label, "decoder": "kNN36_cosine",
                                  "architecture": "CEBRA", "condition": label, "seed": 42, "fold": 0,
                                  "split": "held_out_test", "source_id": int(source_id), "time_id": int(time_id),
                                  "time_within_trial_s": float(time_value), "true_position_cm": float(truth * 100),
                                  "predicted_position_cm": float(decoded * 100)})

        for architecture, condition, label, color in SOFT_SPECS:
            splits, provenance = load_soft_bundle(architecture, condition, rat, supports[rat])
            parents.extend(provenance["parents"])
            x_train, y_train = _rows(splits["train"], "embedding_raw")
            x_valid, y_valid = _rows(splits["valid"], "embedding_raw")
            x_test, _ = _rows(splits["test"], "embedding_raw")
            test_mask = splits["test"]["valid_mask"]
            test_positions = np.flatnonzero(test_mask)
            knn = KNeighborsRegressor(n_neighbors=KNN_NEIGHBORS, metric="cosine")
            knn.fit(x_train, y_train[:, 0])
            knn_values = np.full(len(test_mask), np.nan, dtype=float)
            knn_values[test_positions] = knn.predict(x_test)

            best_alpha, best_score = None, -np.inf
            for alpha in RIDGE_ALPHA_GRID:
                candidate = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
                candidate.fit(x_train, y_train[:, 0])
                score = float(candidate.score(x_valid, y_valid[:, 0]))
                if score > best_score:
                    best_alpha, best_score = float(alpha), score
            if best_alpha != provenance["selected_alpha"]:
                raise RuntimeError(f"Validation-selected Ridge alpha differs from frozen score: {rat}/{label}")
            ridge = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
            ridge.fit(x_train, y_train[:, 0])
            ridge_values = np.full(len(test_mask), np.nan, dtype=float)
            ridge_values[test_positions] = ridge.predict(x_test)

            test_source_ids = np.asarray(splits["test"]["source_id"], dtype=np.int64)
            by_id = {int(value): i for i, value in enumerate(test_source_ids)}
            if len(by_id) != len(test_source_ids) or not set(common_ids.tolist()).issubset(by_id):
                raise RuntimeError(f"Soft held-out source IDs do not cover common trajectory: {rat}/{label}")
            order = np.asarray([by_id[int(value)] for value in common_ids], dtype=np.int64)
            if not np.all(test_mask[order]) or not np.array_equal(splits["test"]["behavior"][order, 0], true_pos_m):
                raise RuntimeError(f"Soft test support/labels differ on the plotted trial: {rat}/{label}")
            pred_knn, pred_ridge = knn_values[order], ridge_values[order]
            if not np.isfinite(pred_knn).all() or not np.isfinite(pred_ridge).all():
                raise RuntimeError(f"Nonfinite Soft decoded trajectory: {rat}/{label}")
            line_style = styles[condition]
            knn_ax.plot(time_s, pred_knn * 100, color=color, lw=1.1, ls=line_style, label=label)
            ridge_ax.plot(time_s, pred_ridge * 100, color=color, lw=1.1, ls=line_style, label=label)
            for i, (source_id, time_value, time_id, truth, p_knn, p_ridge) in enumerate(
                    zip(common_ids, time_s, trial_ids[:common_n], true_pos_m, pred_knn, pred_ridge)):
                for decoder, prediction in (("kNN36_cosine", p_knn), ("Ridge_validation_selected", p_ridge)):
                    long_rows.append({"rat": rat, "trial_id": trial_id, "method": label, "decoder": decoder,
                                      "architecture": architecture, "condition": condition,
                                      "seed": 42, "fold": 0, "split": "held_out_test",
                                      "source_id": int(source_id), "time_id": int(time_id),
                                      "time_within_trial_s": float(time_value), "true_position_cm": float(truth * 100),
                                      "predicted_position_cm": float(prediction * 100),
                                      "ridge_alpha": best_alpha if decoder.startswith("Ridge") else ""})

        for ax in (knn_ax, ridge_ax):
            ax.set_xlim(0, float(time_s[-1]))
            ax.set_ylim(0, 160)
            ax.set_yticks(np.linspace(0, 160, 5))
            ax.grid(axis="y", alpha=0.18)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_ylabel(f"{rat.capitalize()}\nPosition (cm)")
        if row_idx == 0:
            knn_ax.set_title("Held-out position decoding — kNN", fontsize=11)
            ridge_ax.set_title("Held-out position decoding — Ridge", fontsize=11)
            knn_ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.48), ncol=2, frameon=False, fontsize=7.5)
            ridge_ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.48), ncol=2, frameon=False, fontsize=7.5)
        if row_idx == len(RATS) - 1:
            knn_ax.set_xlabel("Time within held-out trial (s)")
            ridge_ax.set_xlabel("Time within held-out trial (s)")

    fig.suptitle("Figure 2D — decoded position on common held-out trajectories", y=0.995, fontsize=14)
    fig.text(0.5, 0.01, f"Seed 42, fold 0; longest valid test trial per rat, displayed at common duration ({common_n} bins, {common_n * TIME_BIN_S:.2f} s). Encoders frozen; Ridge alpha selected on validation.",
             ha="center", fontsize=8)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.06, top=0.86, hspace=0.28, wspace=0.18)
    base = STAGE / "FIGURE2D_METHOD_TRAJECTORIES"
    for ext in ("png", "pdf"):
        fig.savefig(base.with_suffix("." + ext), dpi=300 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    table = STAGE / "FIGURE2D_METHOD_TRAJECTORIES_LONG.csv"
    write_csv(table, long_rows)
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "panel": "Figure 2D",
        "script_sha256": digest(Path(__file__)),
        "files": {ext: digest(base.with_suffix("." + ext)) for ext in ("png", "pdf")},
        "table_sha256": digest(table), "parents": parents,
        "split": "held-out test; longest valid contiguous complete trial per rat, truncated to common duration across rats",
        "common_support": "same trial_id/time_id/source_id, behavior labels and valid mask across all methods, asserted before plotting",
        "encoders": "frozen; no encoder retraining",
        "decoders": {"reference": "kNN regressor, k=36, cosine; fitted on valid training rows",
                     "Soft kNN": "kNN regressor, k=36, cosine; fitted on valid training rows",
                     "Soft Ridge": "StandardScaler + Ridge; alpha selected on validation from frozen grid; then refit on train"},
        "selection_firewall": "test used only for plotted predictions; no test-based alpha selection or model selection",
        "seed_fold": "seed 42, fold 0", "rows_in_long_table": len(long_rows),
        "display_rule": "choose longest valid test trial per rat; ties by smallest trial_id; show first common_n consecutive valid bins",
        "displayed_bins_per_rat": common_n,
        "displayed_trials": {rat: int(supports[rat]["test"]["trial_id"][trial_rows[rat][0]]) for rat in RATS},
    }
    (STAGE / "FIGURE2D_METHOD_TRAJECTORIES_PROVENANCE.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def promote() -> None:
    base = STAGE / "FIGURE2D_METHOD_TRAJECTORIES"
    expected = [base.with_suffix(".png"), base.with_suffix(".pdf"),
                STAGE / "FIGURE2D_METHOD_TRAJECTORIES_LONG.csv",
                STAGE / "FIGURE2D_METHOD_TRAJECTORIES_PROVENANCE.json"]
    if any(not path.is_file() or path.stat().st_size < 1000 for path in expected):
        raise RuntimeError("Staged trajectory deliverable is incomplete; existing output preserved")
    provenance = json.loads(expected[3].read_text(encoding="utf-8"))
    for ext, expected_hash in provenance["files"].items():
        if digest(base.with_suffix("." + ext)) != expected_hash:
            raise RuntimeError(f"Staged trajectory image hash mismatch: {ext}")
    DELIVERY.mkdir(parents=True, exist_ok=True)
    for path in expected:
        shutil.move(str(path), str(DELIVERY / path.name))
    # Superseded reference-only view is now included in the combined method figure.
    for name in ("FIGURE2D_REFERENCE_TRAJECTORIES.png", "FIGURE2D_REFERENCE_TRAJECTORIES.pdf",
                 "REFERENCE_TRAJECTORY_PROVENANCE.json"):
        old = DELIVERY / name
        if old.is_file():
            old.unlink()
    shutil.rmtree(STAGE)


def main() -> None:
    if STAGE.exists():
        raise FileExistsError(f"Staging directory already exists; inspect before proceeding: {STAGE}")
    index = index_manifests()
    build_figure(index)
    promote()
    print(json.dumps({"status": "complete", "figure": "FIGURE2D_METHOD_TRAJECTORIES",
                      "rat_count": len(RATS), "soft_methods": 4, "encoder_training": False,
                      "trajectory_rows": 4 * 700 * (2 + 4 * 2),
                      "output_directory": str(DELIVERY)}, indent=2))


if __name__ == "__main__":
    main()
