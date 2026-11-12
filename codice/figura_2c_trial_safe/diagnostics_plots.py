"""Diagnostics from saved fits. CLI: --root ROOT --scope split|full --rat RAT
--model MODEL --output DIR --left-label VALUE --right-label VALUE.

History columns are exactly update,train_loss. Split embeddings show train and
held-out test; clustering and CKA use train, or full for full scope. The target
of direction_shuffled is its saved binary pseudo-direction. The odd k=1..99
grid is a NEW explicit choice: the historical exact grid was not recorded.
Existing owned outputs resume only with identical arguments and input hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, accuracy_score, balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors

CONDITIONS = ("position", "direction", "position_direction", "position_shuffled",
              "direction_shuffled", "position_direction_shuffled")
CONDITION_LABELS = {"position": "Position", "direction": "Direction",
                    "position_direction": "Position + Direction",
                    "position_shuffled": "Position (shuffled)",
                    "direction_shuffled": "Direction (shuffled)",
                    "position_direction_shuffled": "Position + Direction (shuffled)"}
# Output IDs and labels are separate from the source fit directory names.
MODEL_OUTPUT_IDS = {"cebra_cnn": "cebra_cnn", "soft_stable_cnn": "soft_cnn",
                    "soft_stable_transformer": "soft_transformer"}
MODEL_LABELS = {"cebra_cnn": "CEBRA CNN", "soft_cnn": "Soft CNN",
                "soft_transformer": "Soft Transformer", "soft_stable_cnn": "Soft CNN",
                "soft_stable_transformer": "Soft Transformer"}
PARTITION_LABELS = {"true_direction": "Real direction", "pseudo_direction": "Shuffled training labels",
                    "kmeans_direction": "K-means groups (original fit)",
                    "kmeans_direction_shuffled": "K-means groups (shuffled fit)"}
KEYS = ("embedding_unit", "true_position", "true_direction", "training_target", "source_id", "trial_id")


def load_embedding(path):
    with np.load(path, allow_pickle=False) as archive:
        missing = set(KEYS) - set(archive.files)
        if missing:
            raise ValueError(f"{path}: missing keys {sorted(missing)}")
        data = {key: np.asarray(archive[key]) for key in KEYS}
    z = data["embedding_unit"]
    if z.ndim != 2 or z.shape[1] < 3 or not len(z):
        raise ValueError(f"{path}: embedding_unit must have shape (n, >=3)")
    n = len(z)
    for key, value in data.items():
        if value.ndim == 0 or len(value) != n:
            raise ValueError(f"{path}: {key} is not row aligned")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"{path}: non-finite {key}")
    for key in ("source_id", "trial_id", "true_position", "true_direction"):
        if data[key].ndim != 1:
            raise ValueError(f"{path}: {key} must be one-dimensional")
    if len(np.unique(data["source_id"])) != n:
        raise ValueError(f"{path}: duplicate source_id")
    return data


def linear_cka(x, y):
    """Centered linear CKA, using feature-space products rather than n by n Gram matrices."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x - x.mean(axis=0); y = y - y.mean(axis=0)
    denominator = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    return float(np.linalg.norm(x.T @ y) ** 2 / denominator) if denominator else np.nan


def _cluster(x):
    if len(x) < 2 or len(np.unique(x, axis=0)) < 2:
        return np.zeros(len(x), dtype=int)
    return KMeans(n_clusters=2, random_state=42, n_init=20).fit_predict(x)


def _silhouette(x, labels):
    if not 1 < len(np.unique(labels)) < len(labels):
        return np.nan
    try:
        return float(silhouette_score(x, labels, sample_size=5000, random_state=42))
    except ValueError:  # A random subsample can contain just one class.
        return np.nan


def decode(splits):
    """One neighbor search for validation and one for test, with train labels only."""
    train, valid, test = (splits[s] for s in ("train", "valid", "test"))
    grid = np.arange(1, 100, 2)
    candidates = grid[grid <= len(train["embedding_unit"])]
    classes, y = np.unique(train["true_direction"], return_inverse=True)
    if len(classes) != 2 or not len(candidates):
        raise ValueError("kNN requires two train direction classes and nonempty train")
    neighbors = NearestNeighbors(n_neighbors=int(candidates[-1])).fit(train["embedding_unit"])
    valid_indices = neighbors.kneighbors(valid["embedding_unit"], return_distance=False)
    cumulative = np.cumsum(y[valid_indices], axis=1)
    rows = []
    for k in grid:
        usable = k <= len(train["embedding_unit"])
        pred = classes[(cumulative[:, k-1] * 2 > k).astype(int)] if usable else None
        rows.append({"k": int(k), "available": bool(usable),
                     "valid_accuracy": float(accuracy_score(valid["true_direction"], pred)) if usable else np.nan,
                     "valid_balanced_accuracy": float(balanced_accuracy_score(valid["true_direction"], pred)) if usable else np.nan})
    best = max((r for r in rows if r["available"]), key=lambda r: (r["valid_balanced_accuracy"], -r["k"]))
    test_indices = neighbors.kneighbors(test["embedding_unit"], n_neighbors=best["k"], return_distance=False)
    pred = classes[(y[test_indices].sum(axis=1) * 2 > best["k"]).astype(int)]
    return {"selected_k": best["k"], "valid_balanced_accuracy": best["valid_balanced_accuracy"],
            "test_accuracy": float(accuracy_score(test["true_direction"], pred)),
            "test_balanced_accuracy": float(balanced_accuracy_score(test["true_direction"], pred))}, rows, pred


def _save(fig, output, name):
    for ext in ("png", "pdf"):
        fig.savefig(output / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _align(a, b):
    ids, ia, ib = np.intersect1d(a["source_id"], b["source_id"], return_indices=True)
    for key in ("trial_id", "true_direction", "true_position"):
        if not np.array_equal(a[key][ia], b[key][ib]):
            raise ValueError(f"shared source_id has inconsistent {key}")
    return ids, ia, ib


def run(root, scope, rat, model, output, left_label, right_label):
    """Create or resume a diagnostic output owned by this module; return its path."""
    root, output = Path(root), Path(output)
    if scope not in ("split", "full") or str(left_label) == str(right_label):
        raise ValueError("invalid scope or direction mapping")
    if output.resolve().is_relative_to((root / "fits").resolve()):
        raise ValueError("output must be outside fits")
    base = root / "fits" / scope / rat / model
    sets = ("train", "valid", "test") if scope == "split" else ("full",)
    reference = "train" if scope == "split" else "full"
    files = [base / c / name for c in CONDITIONS for name in
             ([f"{s}_embedding.npz" for s in sets] + ["training_history.csv"])]
    model_output_id = MODEL_OUTPUT_IDS.get(model, model)
    input_hashes = {}
    for condition in CONDITIONS:
        for name in ([f"{s}_embedding.npz" for s in sets] + ["training_history.csv"]):
            path = base / condition / name
            input_hashes[f"source_fit/{scope}/{rat}/{model_output_id}/{condition}/{name}"] = _hash(path)
    signature = {"root": str(root.resolve()), "scope": scope, "rat": rat, "model": model_output_id,
                 "left_label": str(left_label), "right_label": str(right_label),
                 "script_sha256": _hash(Path(__file__)), "inputs": input_hashes}
    marker = output / "diagnostics_run.json"
    complete = output / "diagnostics_complete.json"
    if output.exists():
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != signature:
            raise ValueError("existing output is unowned or inputs/configuration changed")
        if complete.exists():
            try:
                manifest = json.loads(complete.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                manifest = {}
            if manifest and all((output / p).is_file() and _hash(output / p) == h for p, h in manifest.items()):
                return output
    else:
        output.mkdir(parents=True)
        marker.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    # A missing or invalid completion manifest means this owned run is resumed.
    if complete.exists():
        complete.unlink()
    data, histories = {}, {}
    for condition in CONDITIONS:
        folder = base / condition
        splits = {s: load_embedding(folder / f"{s}_embedding.npz") for s in sets}
        for s, d in splits.items():
            if not set(map(str, np.unique(d["true_direction"]))) <= {str(left_label), str(right_label)}:
                raise ValueError(f"{condition}/{s}: unknown direction label")
            if condition == "direction_shuffled":
                target = d["training_target"]
                if target.ndim == 2 and target.shape[1] == 1:
                    target = target[:, 0]
                if target.ndim != 1 or not np.isin(target, [0, 1]).all():
                    raise ValueError(f"{condition}/{s}: training_target must be binary 0/1")
                d["training_target"] = target
        if set(map(str, np.unique(splits[reference]["true_direction"]))) != {str(left_label), str(right_label)}:
            raise ValueError(f"{condition}/{reference}: both direction classes required")
        if scope == "split":
            for a, b in itertools.combinations(sets, 2):
                for key in ("source_id", "trial_id"):
                    if np.intersect1d(splits[a][key], splits[b][key]).size:
                        raise ValueError(f"{condition}: {key} overlaps between {a} and {b}")
        history = pd.read_csv(folder / "training_history.csv")
        if set(history.columns) != {"update", "train_loss"} or len(history.columns) != 2 or history.empty:
            raise ValueError(f"{folder}: history must contain exactly update,train_loss")
        history = history.apply(pd.to_numeric, errors="raise")
        if not np.isfinite(history.to_numpy()).all():
            raise ValueError(f"{folder}: non-finite training history")
        data[condition], histories[condition] = splits, history

    # Save the precise plotted coordinates and labels for each plotted support.
    for support in (("train", "test") if scope == "split" else ("full",)):
        fig = plt.figure(figsize=(15, 9)); norm = Normalize(0, 1.6)
        for i, condition in enumerate(CONDITIONS):
            d = data[condition][support]; z = d["embedding_unit"]
            frame = pd.DataFrame({k: d[k] for k in ("source_id", "trial_id", "true_position", "true_direction")})
            for j in range(z.shape[1]): frame[f"embedding_{j+1}"] = z[:, j]
            frame.to_csv(output / f"points_{support}_{condition}.csv", index=False)
            ax = fig.add_subplot(2, 3, i+1, projection="3d")
            for label, cmap in ((left_label, "cool"), (right_label, "viridis")):
                mask = d["true_direction"].astype(str) == str(label)
                ax.scatter(*z[mask, :3].T, c=d["true_position"][mask], cmap=cmap, norm=norm, s=2, rasterized=True)
            ax.set(title=CONDITION_LABELS[condition], xlabel="embedding 1", ylabel="embedding 2", zlabel="embedding 3")
        fig.subplots_adjust(right=.85, hspace=.3)
        for j, (name, cmap) in enumerate(((f"Left ({left_label})", "cool"), (f"Right ({right_label})", "viridis"))):
            bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([.88+j*.055, .2, .018, .6]), ticks=[0, .5, 1, 1.5])
            bar.set_label(f"{name}: position (m)")
        title = "test held-out" if support == "test" else support
        fig.suptitle(f"{rat.capitalize()} / {MODEL_LABELS.get(model, model)} — {title}")
        _save(fig, output, f"embedding_{support}")
    fig, ax = plt.subplots(figsize=(9, 5)); loss_rows = []
    for condition, history in histories.items():
        ax.plot(history["update"], history["train_loss"], label=CONDITION_LABELS[condition])
        loss_rows.append(history.assign(condition=condition))
    ax.set(xlabel="update", ylabel="train loss", title=f"{rat.capitalize()} / {MODEL_LABELS.get(model, model)}: training loss")
    ax.legend(); _save(fig, output, "loss")
    pd.concat(loss_rows).to_csv(output / "loss_data.csv", index=False)

    # Both k-means fits use the same intersected, source-ID-aligned observations.
    a, b = (data[c][reference] for c in ("direction", "direction_shuffled"))
    ids, ia, ib = _align(a, b)
    if len(ids) < 2:
        raise ValueError("direction conditions need at least two shared rows")
    za, zb = a["embedding_unit"][ia], b["embedding_unit"][ib]
    true, pseudo = a["true_direction"][ia], b["training_target"][ib]
    ca, cb = _cluster(za), _cluster(zb)
    labels = {"true_direction": true, "pseudo_direction": pseudo, "kmeans_direction": ca, "kmeans_direction_shuffled": cb}
    part = pd.DataFrame({"source_id": ids, "trial_id": a["trial_id"][ia], **labels})
    part.to_csv(output / "partitions.csv", index=False)
    ari = [dict(partition_a=x, partition_b=y, support=reference, n_shared=len(ids), ari=adjusted_rand_score(labels[x], labels[y]))
           for x, y in itertools.combinations(labels, 2)]
    sil = [dict(condition=condition, partition=name, support=reference, n_shared=len(ids),
                silhouette=_silhouette(z, value))
           for condition, z in (("direction", za), ("direction_shuffled", zb)) for name, value in labels.items()]
    pd.DataFrame(ari).to_csv(output / "ari.csv", index=False)
    pd.DataFrame(sil).to_csv(output / "silhouette.csv", index=False)
    fig = plt.figure(figsize=(13, 8))
    for i, (condition, z) in enumerate((("direction", za), ("direction_shuffled", zb))):
        for j, (name, value) in enumerate((("Real direction", true), ("Shuffled training labels", pseudo),
                                           ("K-means groups", ca if i == 0 else cb))):
            ax = fig.add_subplot(2, 3, i*3+j+1, projection="3d")
            _, code = np.unique(value, return_inverse=True)
            ax.scatter(*z[:, :3].T, c=code, cmap="coolwarm", s=2, rasterized=True)
            ax.set(title=name, xlabel="embedding 1", ylabel="embedding 2", zlabel="embedding 3")
    fig.text(.012, .70, "Original Direction fit", rotation=90, ha="left", va="center", fontsize=11)
    fig.text(.012, .28, "Shuffled Direction fit", rotation=90, ha="left", va="center", fontsize=11)
    fig.suptitle(f"{rat.capitalize()} / {MODEL_LABELS.get(model, model)} — {reference}; same source bins\n"
                 "Columns: real direction | shuffled labels | K-means groups", y=.99)
    _save(fig, output, "true_pseudo_kmeans")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, z, condition in zip(axes, (za, zb), ("direction", "direction_shuffled")):
        ax.scatter(z[:, 0], z[:, 1], c=pseudo, cmap="coolwarm", s=2, rasterized=True)
        fit_label = "original labels" if condition == "direction" else "shuffled labels"
        ax.set(title=f"Direction fit ({fit_label})", xlabel="embedding 1", ylabel="embedding 2")
    fig.suptitle(f"{rat.capitalize()} / {MODEL_LABELS.get(model, model)} — color shows shuffled labels")
    _save(fig, output, "comparative_scatter")

    cka = []
    for x, y in itertools.combinations(CONDITIONS, 2):
        dx, dy = data[x][reference], data[y][reference]
        shared, ix, iy = _align(dx, dy)
        cka.append(dict(condition_a=x, condition_b=y, support=reference, n_shared=len(shared),
                        linear_cka=linear_cka(dx["embedding_unit"][ix], dy["embedding_unit"][iy]) if len(shared) >= 2 else np.nan))
    pd.DataFrame(cka).to_csv(output / "cka.csv", index=False)
    decoding = []
    if scope == "split":
        validation, predictions = [], []
        for condition in CONDITIONS:
            metrics, grid, pred = decode(data[condition]); decoding.append(dict(condition=condition, **metrics))
            validation.extend(dict(condition=condition, **r) for r in grid)
            t = data[condition]["test"]
            predictions.extend(dict(condition=condition, source_id=s, trial_id=tr, true_direction=y, predicted_direction=p)
                               for s, tr, y, p in zip(t["source_id"], t["trial_id"], t["true_direction"], pred))
        pd.DataFrame(decoding).to_csv(output / "knn_metrics.csv", index=False)
        pd.DataFrame(validation).to_csv(output / "knn_validation_grid.csv", index=False)
        pd.DataFrame(predictions).to_csv(output / "knn_test_predictions.csv", index=False)
    summary = {"scope": scope, "clustering_support": reference, "n_common_direction_rows": len(ids),
               "knn_grid": "odd integers 1..99; new choice, historical exact grid unrecorded; k<=train size",
               "selection": "validation balanced accuracy, smallest k on ties",
               "silhouette": "sample_size=5000 random_state=42", "kmeans": "2 clusters, n_init=20 random_state=42",
               "pseudo_direction": "direction_shuffled training_target directly, 0/1", "ari": ari, "silhouette_values": sil,
               "cka": cka, "knn": decoding,
               "loss": [dict(condition=c, n_updates=len(h), first_train_loss=float(h.train_loss.iloc[0]),
                             last_train_loss=float(h.train_loss.iloc[-1])) for c, h in histories.items()]}
    def json_safe(value):
        if isinstance(value, dict): return {k: json_safe(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)): return [json_safe(v) for v in value]
        if isinstance(value, np.generic): value = value.item()
        if isinstance(value, float) and not np.isfinite(value): return None
        return value
    (output / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
    artifacts = {p.name: _hash(p) for p in output.iterdir() if p.is_file() and p.name not in (marker.name, complete.name)}
    complete.write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "scope", "rat", "model", "output", "left-label", "right-label"):
        parser.add_argument(f"--{name}", required=True, **({"choices": ("split", "full")} if name == "scope" else {}))
    args = parser.parse_args(argv)
    run(args.root, args.scope, args.rat, args.model, args.output, args.left_label, args.right_label)


if __name__ == "__main__":
    main()
