"""Compare new-sampler CEBRA embeddings with the existing Soft Figure 2c fits.

No models are fit here. Split plots/decoding use the original train/valid/test
supports; full-recording plots are descriptive.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))
from prepare_data import DEFAULT_OUTPUT

SOURCE_ROOT = DEFAULT_OUTPUT
NEW_ROOT = SOURCE_ROOT / "2c_cebra_continuous_trial_safe"
OUTPUT_ROOT = NEW_ROOT / "plot_new_sampling"
SCOPES = ("split", "full")
RATS = ("achilles", "gatsby")
CONDITIONS = ("position", "direction", "position_direction", "position_shuffled",
              "direction_shuffled", "position_direction_shuffled")
CONDITION_LABELS = {
    "position": "Position", "direction": "Direction",
    "position_direction": "Position + Direction",
    "position_shuffled": "Position shuffled",
    "direction_shuffled": "Direction shuffled",
    "position_direction_shuffled": "Position + Direction shuffled",
}
MODELS = (
    ("cebra_cnn", "CEBRA (new sampling)", NEW_ROOT),
    ("soft_stable_cnn", "Soft CNN", SOURCE_ROOT),
    ("soft_stable_transformer", "Soft Transformer", SOURCE_ROOT),
)
MODEL_OUTPUT_IDS = {
    "cebra_cnn": "cebra_cnn",
    "soft_stable_cnn": "soft_cnn",
    "soft_stable_transformer": "soft_transformer",
}
SUPPORTS = {"split": ("train", "test"), "full": ("full",)}
KEYS = ("embedding_unit", "true_position", "true_direction", "source_id", "trial_id", "time_id")
POSITION_NORM = Normalize(vmin=0.0, vmax=1.6)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_embedding(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        missing = set(KEYS) - set(archive.files)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        result = {key: np.asarray(archive[key]) for key in KEYS}
    z = result["embedding_unit"]
    n = len(z)
    if z.ndim != 2 or z.shape != (n, 3) or n == 0 or not np.isfinite(z).all():
        raise ValueError(f"{path}: expected finite unit embeddings shaped (n, 3)")
    if len(np.unique(result["source_id"])) != n:
        raise ValueError(f"{path}: duplicate source IDs")
    for key in ("true_position", "true_direction", "source_id", "trial_id", "time_id"):
        if result[key].ndim != 1 or len(result[key]) != n:
            raise ValueError(f"{path}: {key} is not row-aligned")
    return result


def load_history(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["update", "train_loss"] or len(frame) != 10000:
        raise ValueError(f"{path}: expected 10,000 rows of update,train_loss")
    frame = frame.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"{path}: non-finite training loss")
    if not np.array_equal(frame["update"].to_numpy(), np.arange(1, 10001)):
        raise ValueError(f"{path}: updates do not run from 1 to 10,000")
    return frame


def fit_folder(scope: str, rat: str, model: str, condition: str, root: Path) -> Path:
    return root / "fits" / scope / rat / model / condition


def aligned(reference: dict, other: dict, where: str):
    if not np.array_equal(reference["source_id"], other["source_id"]):
        raise ValueError(f"{where}: source IDs do not align")
    for key in ("trial_id", "time_id", "true_position", "true_direction"):
        if not np.array_equal(reference[key], other[key]):
            raise ValueError(f"{where}: {key} differs across models")


def save_figure(fig, path: Path):
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def embedding_panels(scope: str, rat: str, support: str, loaded: dict, output: Path):
    methods = [item[0] for item in MODELS]
    model_labels = {item[0]: item[1] for item in MODELS}
    direction_values = np.sort(np.unique(loaded[(CONDITIONS[0], methods[0], support)]["true_direction"]))
    if len(direction_values) != 2:
        raise ValueError(f"{scope}/{rat}/{support}: expected two direction labels, got {direction_values}")
    direction_cmap = ListedColormap(["#376B9A", "#C45A4A"])
    direction_norm = BoundaryNorm([-0.5, 0.5, 1.5], direction_cmap.N)
    directory = output / scope / rat
    directory.mkdir(parents=True, exist_ok=True)

    # Save the exact plotted coordinates and labels for review/reproduction.
    for condition in CONDITIONS:
        for model in methods:
            data = loaded[(condition, model, support)]
            frame = pd.DataFrame({
                "source_id": data["source_id"],
                "trial_id": data["trial_id"],
                "time_id": data["time_id"],
                "true_position": data["true_position"],
                "true_direction": data["true_direction"],
                "embedding_1": data["embedding_unit"][:, 0],
                "embedding_2": data["embedding_unit"][:, 1],
                "embedding_3": data["embedding_unit"][:, 2],
            })
            frame.to_csv(directory / f"points_{support}_{condition}_{MODEL_OUTPUT_IDS[model]}.csv", index=False)

    for color_by in ("position", "direction"):
        fig = plt.figure(figsize=(18, 4.2 * len(CONDITIONS)))
        colorbar_mappable = None
        for row, condition in enumerate(CONDITIONS):
            for col, model in enumerate(methods):
                data = loaded[(condition, model, support)]
                z = data["embedding_unit"]
                ax = fig.add_subplot(len(CONDITIONS), len(methods), row * len(methods) + col + 1,
                                     projection="3d")
                if color_by == "position":
                    colorbar_mappable = ax.scatter(*z.T, c=data["true_position"], cmap="viridis",
                                                   norm=POSITION_NORM, s=3, rasterized=True)
                else:
                    label_to_code = {value: i for i, value in enumerate(direction_values)}
                    color = np.asarray([label_to_code[value] for value in data["true_direction"]])
                    colorbar_mappable = ax.scatter(*z.T, c=color, cmap=direction_cmap,
                                                   norm=direction_norm, s=3, rasterized=True)
                ax.set_title(f"{model_labels[model]}\n{CONDITION_LABELS[condition]}")
                ax.set_xlabel("Embedding 1")
                ax.set_ylabel("Embedding 2")
                ax.set_zlabel("Embedding 3")
                ax.tick_params(labelsize=8, pad=1)
        support_label = "held-out test" if support == "test" else support
        fig.suptitle(f"{rat.capitalize()} — {scope}, {support_label}: new CEBRA vs Soft", y=0.97)
        fig.subplots_adjust(left=0.02, right=0.87, top=0.87, bottom=0.06, wspace=0.08, hspace=0.40)
        if color_by == "position":
            bar = fig.colorbar(colorbar_mappable, cax=fig.add_axes([0.90, 0.22, 0.018, 0.55]))
            bar.set_label("True position (m)")
        else:
            bar = fig.colorbar(colorbar_mappable, cax=fig.add_axes([0.90, 0.22, 0.018, 0.55]),
                               ticks=[0, 1])
            bar.ax.set_yticklabels([str(value) for value in direction_values])
            bar.set_label("True direction")
        save_figure(fig, directory / f"embedding_{support}_by_{color_by}")


def linear_cka(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    denom = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    return float(np.linalg.norm(x.T @ y) ** 2 / denom) if denom else float("nan")


def decode_direction(splits):
    train, valid, test = (splits[name] for name in ("train", "valid", "test"))
    classes, y_train = np.unique(train["true_direction"], return_inverse=True)
    if len(classes) != 2:
        raise ValueError("held-out direction decoding requires two train classes")
    grid = np.arange(1, 100, 2)
    max_k = min(int(grid[-1]), len(train["embedding_unit"]))
    candidates = grid[grid <= max_k]
    neighbors = NearestNeighbors(n_neighbors=int(candidates[-1])).fit(train["embedding_unit"])
    valid_neighbors = neighbors.kneighbors(valid["embedding_unit"], return_distance=False)
    cumulative = np.cumsum(y_train[valid_neighbors], axis=1)
    grid_rows = []
    for k in candidates:
        prediction = classes[(cumulative[:, k - 1] * 2 > k).astype(int)]
        grid_rows.append({
            "k": int(k),
            "valid_accuracy": float(accuracy_score(valid["true_direction"], prediction)),
            "valid_balanced_accuracy": float(balanced_accuracy_score(valid["true_direction"], prediction)),
        })
    best = max(grid_rows, key=lambda row: (row["valid_balanced_accuracy"], -row["k"]))
    test_neighbors = neighbors.kneighbors(test["embedding_unit"], n_neighbors=best["k"], return_distance=False)
    prediction = classes[(y_train[test_neighbors].sum(axis=1) * 2 > best["k"]).astype(int)]
    metrics = {
        "selected_k": best["k"],
        "valid_balanced_accuracy": best["valid_balanced_accuracy"],
        "test_accuracy": float(accuracy_score(test["true_direction"], prediction)),
        "test_balanced_accuracy": float(balanced_accuracy_score(test["true_direction"], prediction)),
    }
    predictions = pd.DataFrame({
        "source_id": test["source_id"],
        "trial_id": test["trial_id"],
        "true_direction": test["true_direction"],
        "predicted_direction": prediction,
    })
    return metrics, pd.DataFrame(grid_rows), predictions


def run_one(scope: str, rat: str):
    output = OUTPUT_ROOT / scope / rat
    output.mkdir(parents=True, exist_ok=True)
    supports = ("train", "valid", "test") if scope == "split" else ("full",)
    loaded = {}
    histories = {}
    input_hashes = {}
    for condition in CONDITIONS:
        for model, _, root in MODELS:
            folder = fit_folder(scope, rat, model, condition, root)
            manifest_path = folder / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("scope"), manifest.get("rat"), manifest.get("condition"), manifest.get("model")) != (scope, rat, condition, model):
                raise ValueError(f"fit manifest identity mismatch: {manifest_path}")
            if manifest.get("embedding_dimension") != 3 or manifest.get("updates") != 10000:
                raise ValueError(f"unexpected fit configuration: {manifest_path}")
            artifact_key = f"source_fit/{scope}/{rat}/{MODEL_OUTPUT_IDS[model]}/{condition}"
            input_hashes[f"{artifact_key}/manifest.json"] = sha256(manifest_path)
            histories[(condition, model)] = load_history(folder / "training_history.csv")
            input_hashes[f"{artifact_key}/training_history.csv"] = sha256(folder / "training_history.csv")
            for support in supports:
                path = folder / f"{support}_embedding.npz"
                loaded[(condition, model, support)] = load_embedding(path)
                input_hashes[f"{artifact_key}/{support}_embedding.npz"] = sha256(path)

    # Compare sample-wise only after exact source/time/label identity checks.
    for condition in CONDITIONS:
        for support in supports:
            ref = loaded[(condition, MODELS[0][0], support)]
            for model, _, _ in MODELS[1:]:
                aligned(ref, loaded[(condition, model, support)],
                        f"{scope}/{rat}/{condition}/{support}/{model}")

    for support in SUPPORTS[scope]:
        embedding_panels(scope, rat, support, loaded, OUTPUT_ROOT)

    # Loss curves are in separate method facets with independent y scales:
    # CEBRA InfoNCE and Soft's objective are not numerically interchangeable.
    loss_rows = []
    fig, axes = plt.subplots(len(CONDITIONS), len(MODELS), figsize=(16, 2.4 * len(CONDITIONS)),
                             sharex=True, squeeze=False)
    for r, condition in enumerate(CONDITIONS):
        for c, (model, label, _) in enumerate(MODELS):
            history = histories[(condition, model)]
            ax = axes[r, c]
            ax.plot(history["update"], history["train_loss"], lw=1.1, color="#315C85")
            ax.set_title(f"{label}\n{CONDITION_LABELS[condition]}")
            ax.set_xlabel("Update")
            ax.set_ylabel("Training loss (own scale)")
            ax.grid(alpha=0.2)
            loss_rows.extend({
                "scope": scope, "rat": rat, "condition": condition,
                "model": label, "update": int(row.update), "train_loss": float(row.train_loss),
            } for row in history.itertuples(index=False))
    fig.suptitle(f"{rat.capitalize()} / {scope}: training loss trajectories")
    fig.text(0.5, 0.01, "Each model has its own y scale; absolute values are not directly comparable across objectives.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    save_figure(fig, output / "training_loss_comparison")
    pd.DataFrame(loss_rows).to_csv(output / "training_loss_data.csv", index=False)

    cka_rows = []
    model_pairs = list(itertools.combinations([item[0] for item in MODELS], 2))
    for support in supports:
        for condition in CONDITIONS:
            for model_a, model_b in model_pairs:
                a = loaded[(condition, model_a, support)]
                b = loaded[(condition, model_b, support)]
                cka_rows.append({
                    "scope": scope, "rat": rat, "support": support,
                    "condition": condition,
                    "model_a": MODEL_OUTPUT_IDS[model_a], "model_b": MODEL_OUTPUT_IDS[model_b],
                    "n_shared": len(a["source_id"]),
                    "linear_cka": linear_cka(a["embedding_unit"], b["embedding_unit"]),
                })
    cka_table = pd.DataFrame(cka_rows)
    cka_table.to_csv(output / "cross_model_cka.csv", index=False)
    for support in supports:
        subset = cka_table[cka_table.support == support]
        fig, axes = plt.subplots(len(CONDITIONS), 1, figsize=(10, 2.1 * len(CONDITIONS)), squeeze=False)
        pretty_model = {MODEL_OUTPUT_IDS[model]: label for model, label, _ in MODELS}
        for row, condition in enumerate(CONDITIONS):
            frame = subset[subset.condition == condition]
            labels = [f"{pretty_model[a]} vs {pretty_model[b]}"
                      for a, b in zip(frame.model_a, frame.model_b)]
            bars = axes[row, 0].bar(labels, frame.linear_cka, color=["#4C78A8", "#F58518", "#54A24B"])
            axes[row, 0].set_ylim(0, 1)
            axes[row, 0].set_ylabel("Linear CKA")
            axes[row, 0].set_title(CONDITION_LABELS[condition])
            axes[row, 0].tick_params(axis="x", rotation=10, labelsize=9)
            for bar, value in zip(bars, frame.linear_cka):
                axes[row, 0].text(bar.get_x() + bar.get_width()/2, value + 0.025,
                                  f"{value:.2f}", ha="center", va="bottom", fontsize=9)
        fig.suptitle(f"{rat.capitalize()} / {scope} / {support}: same-condition model similarity")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        save_figure(fig, output / f"cross_model_cka_{support}")

    decoding_rows, grid_rows, prediction_rows = [], [], []
    if scope == "split":
        for condition in CONDITIONS:
            for model, label, _ in MODELS:
                splits = {
                    support: loaded[(condition, model, support)]
                    for support in ("train", "valid", "test")
                }
                metrics, grid, predictions = decode_direction(splits)
                decoding_rows.append({
                    "scope": scope, "rat": rat, "condition": condition,
                    "model": MODEL_OUTPUT_IDS[model], "display_name": label, **metrics,
                })
                grid["condition"] = condition
                grid["model"] = MODEL_OUTPUT_IDS[model]
                grid_rows.extend(grid.to_dict("records"))
                predictions["condition"] = condition
                predictions["model"] = MODEL_OUTPUT_IDS[model]
                prediction_rows.extend(predictions.to_dict("records"))
        decode_table = pd.DataFrame(decoding_rows)
        decode_table.to_csv(output / "heldout_direction_decoding.csv", index=False)
        pd.DataFrame(grid_rows).to_csv(output / "direction_decoding_validation_grid.csv", index=False)
        pd.DataFrame(prediction_rows).to_csv(output / "direction_decoding_test_predictions.csv", index=False)
        fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(3.2 * len(CONDITIONS), 4.5), squeeze=False)
        for i, condition in enumerate(CONDITIONS):
            frame = decode_table[decode_table.condition == condition]
            bars = axes[0, i].bar(frame.display_name, frame.test_balanced_accuracy,
                                  color=["#315C85", "#F28E2B", "#59A14F"])
            axes[0, i].set_ylim(0, 1)
            axes[0, i].set_ylabel("Test balanced accuracy")
            axes[0, i].set_title(CONDITION_LABELS[condition])
            axes[0, i].tick_params(axis="x", rotation=15)
            for bar, value in zip(bars, frame.test_balanced_accuracy):
                axes[0, i].text(bar.get_x() + bar.get_width()/2, value + 0.02,
                                f"{value:.2f}", ha="center", va="bottom", fontsize=9)
        fig.suptitle(f"{rat.capitalize()} / split: held-out direction decoding")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        save_figure(fig, output / "heldout_direction_decoding")
    else:
        decode_table = pd.DataFrame()

    summary = {
        "scope": scope,
        "rat": rat,
        "conditions": list(CONDITIONS),
        "models": [{"id": MODEL_OUTPUT_IDS[model], "label": label} for model, label, _ in MODELS],
        "supports_plotted": list(SUPPORTS[scope]),
        "split_decoding": "train kNN labels; validation selects odd k in 1..99 by balanced accuracy; test is held out",
        "full_scope": "descriptive fit on the entire recording; no held-out decoding is reported",
        "loss_comparison": "method-specific facets with independent y scales; absolute loss values are not directly comparable",
        "cross_model_cka": "same source-aligned support and same condition; geometric similarity, not performance",
        "heldout_direction_decoding": decode_table.to_dict("records"),
        "inputs": input_hashes,
    }
    summary_path = output / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                            encoding="utf-8")
    artifacts = {
        str(path.relative_to(output)): sha256(path)
        for path in output.rglob("*")
        if path.is_file() and path.name != "comparison_complete.json"
    }
    (output / "comparison_complete.json").write_text(
        json.dumps(artifacts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=SCOPES, action="append")
    parser.add_argument("--rat", choices=RATS, action="append")
    args = parser.parse_args(argv)
    results = []
    for scope in (args.scope or SCOPES):
        for rat in (args.rat or RATS):
            results.append(run_one(scope, rat))
            print(f"COMPLETE {scope} {rat}", flush=True)
    index = OUTPUT_ROOT / "COMPARISON_INDEX.md"
    lines = [
        "# Confronti 2c: CEBRA nuovo sampler vs Soft",
        "",
        "CEBRA uses the completed trial-safe time_delta fits. Soft CNN and Soft Transformer reuse the existing fits; no Soft model was retrained.",
        "",
        "Split embeddings show train and test; decoding fits on train, selects k on validation, and reports held-out test. Full embeddings are descriptive. Loss panels use independent scales because objectives differ. CKA measures geometry on source-aligned embeddings, not performance.",
        "",
    ]
    for scope in (args.scope or SCOPES):
        for rat in (args.rat or RATS):
            lines.append(f"- {scope}/{rat}/: embedding comparison by position/direction, training loss, cross-model CKA"
                         + (" and held-out direction decoding" if scope == "split" else ""))
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"INDEX {index}", flush=True)


if __name__ == "__main__":
    main()
