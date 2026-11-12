"""Compare three completed diagnostics for one rat and scope, without refitting.

CLI: python summary_comparison.py --rat RAT --scope split|full --diagnostic DIR1
DIR2 DIR3 --output NEW_DIRECTORY

ARI and silhouette use train (split) or full, CKA compares conditions within
each model, and split decoding uses held-out test. A support audit compares
saved source/trial IDs; differing supports are shown, never silently pooled.
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
import numpy as np
import pandas as pd

CONDITIONS = ("position", "direction", "position_direction", "position_shuffled",
              "direction_shuffled", "position_direction_shuffled")
PARTITIONS = ("true_direction", "pseudo_direction", "kmeans_direction", "kmeans_direction_shuffled")
# Display names only; internal model IDs retain the implementation/version.
MODEL_LABELS = {"cebra_cnn": "CEBRA CNN", "soft_cnn": "Soft CNN",
                "soft_transformer": "Soft Transformer", "soft_stable_cnn": "Soft CNN",
                "soft_stable_transformer": "Soft Transformer"}
OUTPUT_MODEL_LABELS = {"cebra_cnn": "CEBRA CNN", "soft_cnn": "Soft CNN",
                       "soft_transformer": "Soft Transformer", "soft_stable_cnn": "Soft CNN",
                       "soft_stable_transformer": "Soft Transformer"}
CONDITION_LABELS = {"position": "Position", "direction": "Direction",
                    "position_direction": "Position + Direction",
                    "position_shuffled": "Position (shuffled)",
                    "direction_shuffled": "Direction (shuffled)",
                    "position_direction_shuffled": "Position + Direction (shuffled)"}


def _sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _kind(condition):
    return "shuffled" if condition.endswith("_shuffled") else "original"


def _partition_kind(partition):
    return {"true_direction": "true label", "pseudo_direction": "shuffled target",
            "kmeans_direction": "original fit", "kmeans_direction_shuffled": "shuffled fit"}[partition]


def _read_table(path, columns):
    table = pd.read_csv(path)
    missing = set(columns) - set(table.columns)
    if missing or table.empty:
        raise ValueError(f"{path}: missing columns {sorted(missing)} or empty table")
    return table


def _verify(directory, rat, scope):
    marker = json.loads((directory / "diagnostics_run.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "diagnostics_complete.json").read_text(encoding="utf-8"))
    if marker.get("rat") != rat or marker.get("scope") != scope or not marker.get("model"):
        raise ValueError(f"{directory}: rat, scope or model does not match")
    support = "train" if scope == "split" else "full"
    required = {"ari.csv", "silhouette.csv", "cka.csv", "partitions.csv"}
    required.update(f"points_{support}_{c}.csv" for c in CONDITIONS)
    if scope == "split":
        required.update(("knn_metrics.csv", "knn_test_predictions.csv"))
    if not isinstance(manifest, dict) or not required <= manifest.keys():
        raise ValueError(f"{directory}: incomplete diagnostics manifest")
    for name in required:
        path = directory / name
        if not path.is_file() or _sha256(path) != manifest[name]:
            raise ValueError(f"{path}: missing or does not match diagnostics manifest")
    return marker


def _ids(frame, name):
    if frame[["source_id", "trial_id"]].isna().any().any() or frame.source_id.duplicated().any():
        raise ValueError(f"{name}: missing or duplicate source IDs")
    return frozenset(zip(frame.source_id.astype(str), frame.trial_id.astype(str)))


def _add_metadata(table, model, rat, scope, support):
    table.insert(0, "model", model)
    table.insert(0, "scope", scope)
    table.insert(0, "rat", rat)
    if "support" in table:
        if not table.support.eq(support).all():
            raise ValueError(f"{model}: unexpected support in diagnostic table")
    else:
        table.insert(3, "support", support)
    return table


def _plot(table, category, value, title, xlabel, output, name, models):
    categories = list(dict.fromkeys(table[category].tolist()))
    def display(value):
        if value in CONDITION_LABELS: return CONDITION_LABELS[value]
        if value in PARTITIONS: return _partition_kind(value)
        if " vs " in value:
            a, b = value.split(" vs ", 1)
            return f"{display(a)} vs {display(b)}"
        if " | " in value:
            a, b = value.split(" | ", 1)
            return f"{display(a)} — {display(b)}"
        return value
    fig, ax = plt.subplots(figsize=(11, max(4.5, .42 * len(categories) + 2)))
    positions = np.arange(len(categories))
    colors = ("#0072B2", "#D55E00", "#009E73")
    for index, model in enumerate(models):
        data = table[table.model == model].set_index(category).reindex(categories)
        ax.scatter(data[value], positions + (index - 1)*.19, label=MODEL_LABELS.get(model, model), s=38, color=colors[index])
    ax.set_yticks(positions, [display(category) for category in categories]); ax.invert_yaxis()
    ax.axvline(0, color="0.75", lw=.7); ax.grid(axis="x", alpha=.25)
    ax.set(xlabel=xlabel, title=title); ax.legend(loc="best", title="Model")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(diagnostics, rat, scope, output):
    """Compare three completed diagnostic directories; create a new output directory.

    Returns output path. Inputs are never modified; existing output is rejected.
    All original per-model values and supports are retained in output CSV files.
    """
    paths = [Path(p) for p in diagnostics]
    output = Path(output)
    if scope not in ("split", "full") or len(paths) != 3 or len({p.resolve() for p in paths}) != 3:
        raise ValueError("provide exactly three distinct diagnostic directories and one valid scope")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if any(output.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(output.resolve()) for p in paths):
        raise ValueError("output and diagnostic directories must not contain each other")
    markers = [_verify(path, rat, scope) for path in paths]
    models = [m["model"] for m in markers]
    if len(set(models)) != 3:
        raise ValueError("diagnostic directories must identify three distinct models")
    support = "train" if scope == "split" else "full"
    collected = {"ari": [], "silhouette": [], "cka": [], "decoding": []}
    support_sets = {}
    for path, model in zip(paths, models):
        ari = _read_table(path / "ari.csv", ["partition_a", "partition_b", "support", "n_shared", "ari"])
        if (not set(ari.partition_a).union(ari.partition_b) <= set(PARTITIONS)
                or set(map(frozenset, zip(ari.partition_a, ari.partition_b))) != set(map(frozenset, itertools.combinations(PARTITIONS, 2)))
                or len(ari) != 6):
            raise ValueError(f"{path}: unexpected ARI partition pairs")
        ari["partition_a_type"] = ari.partition_a.map(_partition_kind)
        ari["partition_b_type"] = ari.partition_b.map(_partition_kind)
        ari["comparison"] = ari.partition_a + " vs " + ari.partition_b
        collected["ari"].append(_add_metadata(ari, model, rat, scope, support))
        sil = _read_table(path / "silhouette.csv", ["condition", "partition", "support", "n_shared", "silhouette"])
        if (set(sil.condition) != {"direction", "direction_shuffled"} or set(sil.partition) != set(PARTITIONS)
                or len(sil) != 8 or sil.duplicated(["condition", "partition"]).any()):
            raise ValueError(f"{path}: unexpected silhouette conditions or partitions")
        sil["fit_type"] = sil.condition.map(_kind)
        sil["partition_type"] = sil.partition.map(_partition_kind)
        sil["comparison"] = sil.condition + " | " + sil.partition
        collected["silhouette"].append(_add_metadata(sil, model, rat, scope, support))
        cka = _read_table(path / "cka.csv", ["condition_a", "condition_b", "support", "n_shared", "linear_cka"])
        if (len(cka) != 15 or set(map(frozenset, zip(cka.condition_a, cka.condition_b))) !=
                set(map(frozenset, itertools.combinations(CONDITIONS, 2)))):
            raise ValueError(f"{path}: unexpected CKA condition pairs")
        cka["fit_type_a"] = cka.condition_a.map(_kind)
        cka["fit_type_b"] = cka.condition_b.map(_kind)
        cka["comparison"] = cka.condition_a + " vs " + cka.condition_b
        collected["cka"].append(_add_metadata(cka, model, rat, scope, support))
        partitions = _read_table(path / "partitions.csv", ["source_id", "trial_id", *PARTITIONS])
        support_sets[(model, support, "direction_pair_common")] = _ids(partitions, path / "partitions.csv")
        for condition in CONDITIONS:
            file = path / f"points_{support}_{condition}.csv"
            points = _read_table(file, ["source_id", "trial_id"])
            support_sets[(model, support, condition)] = _ids(points, file)
        if scope == "split":
            knn = _read_table(path / "knn_metrics.csv", ["condition", "selected_k", "valid_balanced_accuracy", "test_accuracy", "test_balanced_accuracy"])
            if len(knn) != 6 or set(knn.condition) != set(CONDITIONS):
                raise ValueError(f"{path}: unexpected decoding conditions")
            knn["fit_type"] = knn.condition.map(_kind)
            knn["training_support"] = "train"; knn["selection_support"] = "valid"
            collected["decoding"].append(_add_metadata(knn, model, rat, scope, "test"))
            pred = _read_table(path / "knn_test_predictions.csv", ["condition", "source_id", "trial_id"])
            if set(pred.condition) != set(CONDITIONS):
                raise ValueError(f"{path}: unexpected prediction conditions")
            for condition in CONDITIONS:
                support_sets[(model, "test", condition)] = _ids(pred[pred.condition == condition], path / "knn_test_predictions.csv")
    audit = []
    for (model_a, model_b) in itertools.combinations(models, 2):
        for _, set_support, condition in [k for k in support_sets if k[0] == model_a]:
            a = support_sets[(model_a, set_support, condition)]; b = support_sets[(model_b, set_support, condition)]
            audit.append(dict(rat=rat, scope=scope, support=set_support, condition=condition,
                              model_a=model_a, model_b=model_b, n_a=len(a), n_b=len(b),
                              n_shared=len(a & b), same_source_trial_pairs=a == b))
    output.mkdir(parents=True, exist_ok=False)
    tables = {name: pd.concat(rows, ignore_index=True) for name, rows in collected.items() if rows}
    for name, table in tables.items():
        exported = table.copy()
        for column in ("model", "model_a", "model_b"):
            if column in exported:
                exported[column] = exported[column].map(lambda value: OUTPUT_MODEL_LABELS.get(value, value))
        exported.to_csv(output / f"{name}.csv", index=False)
    audit_table = pd.DataFrame(audit)
    for column in ("model_a", "model_b"):
        if column in audit_table:
            audit_table[column] = audit_table[column].map(lambda value: OUTPUT_MODEL_LABELS.get(value, value))
    audit_table.to_csv(output / "support_audit.csv", index=False)
    note = "supports differ: see support_audit.csv" if any(not r["same_source_trial_pairs"] for r in audit) else "source/trial supports match across models"
    prefix = f"{rat.capitalize()} | {scope} | {note}"
    _plot(tables["ari"], "comparison", "ari", f"{prefix}\nARI partitions on {support}", "Adjusted Rand index", output, "ari", models)
    _plot(tables["silhouette"], "comparison", "silhouette", f"{prefix}\nSilhouette on {support}: fit | partition", "Silhouette score", output, "silhouette", models)
    _plot(tables["cka"], "comparison", "linear_cka", f"{prefix}\nLinear CKA between conditions within each model ({support})", "Linear CKA", output, "cka", models)
    if scope == "split":
        for metric in ("test_accuracy", "test_balanced_accuracy"):
            _plot(tables["decoding"], "condition", metric, f"{prefix}\nkNN {metric.replace('_', ' ')} | test held-out", metric.replace("_", " "), output, metric, models)
    provenance = {"rat": rat, "scope": scope, "diagnostics": [dict(model=OUTPUT_MODEL_LABELS.get(m, m), path=str(p.resolve()),
                  run_sha256=_sha256(p / "diagnostics_run.json"), complete_sha256=_sha256(p / "diagnostics_complete.json"))
                  for p, m in zip(paths, models)], "support_note": note,
                  "cka_definition": "within-model condition pairs, not cross-model CKA",
                  "knn": "train fit; validation selects k; test held-out" if scope == "split" else "not evaluated for full"}
    (output / "comparison_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rat", required=True)
    parser.add_argument("--scope", required=True, choices=("split", "full"))
    parser.add_argument("--diagnostic", nargs=3, required=True, metavar=("DIR1", "DIR2", "DIR3"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    run(args.diagnostic, args.rat, args.scope, args.output)


if __name__ == "__main__":
    main()
