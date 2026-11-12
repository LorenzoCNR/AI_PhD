"""Insert completed full-recording t-SNE fits into existing paper-only Fig. 1D and ED3."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

import rat_rebuild_1d_ed3_20261027 as embedding_source
import rat_render_figura1_saved as renderer
import rat_render_extended_data_3_4_20261028 as ed3_renderer

ROOT = Path(__file__).resolve().parents[1]
RATS = ("achilles", "buddy", "cicero", "gatsby")
FIG1 = ROOT / "results" / "figura_1_2026-10-27" / "d"
ED3 = ROOT / "results" / "figura_ext_3_2026-10-28"
TSNE = ROOT / "results" / "figura_1_2026-10-29" / "d"
EXPECTED_PARAMS = {
    "n_components": 2,
    "metric": "cosine",
    "perplexity": 30,
    "initialization": "pca",
    "learning_rate": "auto",
    "n_iter": 1000,
    "random_state": 42,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def load_tsne(rat: str, raw: np.ndarray, source_path: Path) -> tuple[dict, dict]:
    source_hash = sha256(source_path)
    candidates = []
    for folder in sorted((TSNE / rat).glob("tSNE_attempt_*")):
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = manifest.get("config", {})
        if (manifest.get("status") == "complete" and config.get("rat") == rat
                and config.get("source_sha256") == source_hash
                and all(config.get("params", {}).get(key) == value
                        for key, value in EXPECTED_PARAMS.items())):
            candidates.append((folder, manifest_path, manifest))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one compatible completed t-SNE fit for {rat}; found {len(candidates)}")

    folder, manifest_path, manifest = candidates[0]
    fit_data_path = folder / "fit_data.npz"
    embedding_path = folder / "embedding.npy"
    if sha256(fit_data_path) != manifest.get("fit_data_sha256"):
        raise ValueError(f"t-SNE fit-data hash mismatch: {fit_data_path}")
    if sha256(embedding_path) != manifest.get("embedding_sha256"):
        raise ValueError(f"t-SNE embedding hash mismatch: {embedding_path}")
    if manifest.get("fit_input_sha256") != manifest["config"].get("fit_input_sha256"):
        raise ValueError(f"t-SNE input hash mismatch in manifest: {manifest_path}")

    with np.load(fit_data_path, allow_pickle=False) as arrays:
        behavior = np.asarray(arrays["behavior"], dtype=np.float32)
        source_id = np.asarray(arrays["source_id"], dtype=np.int64)
    embedding = np.load(embedding_path, allow_pickle=False).astype(np.float32, copy=False)
    expected_ids = np.arange(5, len(raw) - 4, dtype=np.int64)
    if not np.array_equal(source_id, expected_ids):
        raise ValueError(f"t-SNE source rows are not all ordered 10-bin window centers: {rat}")
    if behavior.shape != (len(source_id), 3) or not np.allclose(behavior, raw[source_id], rtol=0, atol=1e-6):
        raise ValueError(f"t-SNE behavior/source identity mismatch: {rat}")
    if embedding.shape != (len(source_id), 2) or not np.isfinite(embedding).all():
        raise ValueError(f"Invalid t-SNE embedding shape or values: {rat}")
    left, right = behavior[:, 2] == 1, behavior[:, 1] == 1
    if np.any(left & right):
        raise ValueError(f"Overlapping Left/Right labels: {rat}")
    plotted = left | right
    if not plotted.any():
        raise ValueError(f"No labeled rows to plot: {rat}")

    spec = next(spec for spec in renderer.paper_specs()
                if spec[1] == "manifold" and spec[2] == "tsne")
    panel = {"title": spec[0], "z": embedding, "behavior": behavior,
             "mask": plotted, "dimension": 2}
    record = {
        "status": "verified", "title": spec[0].replace("\n", " "),
        "family": "manifold", "method": "t-SNE", "rat": rat, "seed": 42,
        "fit_dimension": 2, "plotted_dimensions": 2, "coordinates": [1, 2],
        "fit_scope": "Fit full recording on complete 10-bin windows; plotted rows require a valid Left/Right label.",
        "stored_rows": int(len(source_id)), "valid_mask_rows": int(len(source_id)),
        "plotted_rows": int(plotted.sum()), "excluded_without_direction": int((~plotted).sum()),
        "plotted_source_ids_sha256": hashlib.sha256(
            np.ascontiguousarray(source_id[plotted]).tobytes()).hexdigest(),
        "normalization": "raw t-SNE coordinates", "embedding": relative(embedding_path),
        "embedding_sha256": manifest["embedding_sha256"],
        "fit_data": relative(fit_data_path), "fit_data_sha256": manifest["fit_data_sha256"],
        "manifest": relative(manifest_path), "manifest_sha256": sha256(manifest_path),
        "checkpoint_sha256": None, "source_data_sha256": source_hash,
        "config": manifest["config"], "resolved_parameters": manifest["resolved_parameters"],
        "openTSNE_version": manifest["openTSNE_version"],
        "insertion_script": relative(Path(__file__)),
        "insertion_script_sha256": sha256(Path(__file__)),
    }
    return panel, record


def load_paper_panels(rat: str, raw: np.ndarray, source_path: Path, campaign: dict):
    data_hash = sha256(source_path)
    panels, records = [], []
    for spec in renderer.paper_specs():
        if spec[1] == "manifold" and spec[2] == "tsne":
            panel, record = load_tsne(rat, raw, source_path)
        else:
            try:
                panel = embedding_source.read_panel(rat, raw, data_hash, spec, campaign)
                record = {"status": "verified", **panel["record"]}
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                detail = str(error).replace("|", "/")
                panel = {"title": spec[0], "error": detail}
                record = {"status": "unavailable", "rat": rat, "seed": 42,
                          "title": spec[0].replace("\n", " "), "expected_dimension": spec[4],
                          "reason": detail}
            panel["dimension"] = spec[4]
        panels.append(panel)
        records.append(record)
    return panels, records


def assert_same_non_tsne_data(baseline: list, candidate: list) -> None:
    tsne_index = next(i for i, panel in enumerate(candidate) if panel["title"] == "t-SNE")
    for index, (old, new) in enumerate(zip(baseline, candidate)):
        if index == tsne_index:
            continue
        if old.get("title") != new.get("title") or old.get("error") != new.get("error"):
            raise RuntimeError(f"Non-t-SNE panel status/title changed at index {index}")
        if "error" not in old:
            for key in ("z", "behavior", "mask"):
                if not np.array_equal(old[key], new[key]):
                    raise RuntimeError(f"Non-t-SNE panel data changed: {old['title']}/{key}")


def assert_baseline_matches(original: Path, baseline: Path) -> None:
    if sha256(original) != sha256(baseline):
        raise RuntimeError(f"Current renderer does not reproduce existing panel exactly: {original}")


def classify_existing(original: Path, baseline: Path, candidate: Path) -> bool:
    current_hash = sha256(original)
    if current_hash == sha256(baseline):
        return False
    if current_hash == sha256(candidate):
        return True
    raise RuntimeError(f"Existing paper panel is neither baseline nor this t-SNE insertion: {original}")


def replace_file(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tsne-insertion.tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite temporary file: {temporary}")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def update_figure1d(rat: str, record: dict, output_png: Path, output_pdf: Path) -> None:
    path = FIG1 / rat.capitalize() / "PROVENANCE.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    paper = provenance["panels"]["paper"]
    index = next(i for i, item in enumerate(paper) if item.get("title") == "t-SNE")
    paper[index] = record
    provenance["outputs"][relative(output_png)] = sha256(output_png)
    provenance["outputs"][relative(output_pdf)] = sha256(output_pdf)
    provenance["tsne_insertion_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["tsne_insertion_script"] = relative(Path(__file__))
    provenance["tsne_insertion_script_sha256"] = sha256(Path(__file__))
    write_json(path, provenance)


def update_ed3(rat: str, letter: str, record: dict, output_png: Path, output_pdf: Path) -> None:
    path = ED3 / letter / rat.capitalize() / "PROVENANCE.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    paper = provenance["groups"]["Paper"]["models"]
    index = next(i for i, item in enumerate(paper) if item.get("title") == "t-SNE")
    paper[index] = record
    provenance["groups"]["Paper"]["outputs"][relative(output_png)] = sha256(output_png)
    provenance["groups"]["Paper"]["outputs"][relative(output_pdf)] = sha256(output_pdf)
    provenance["tsne_insertion_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["tsne_insertion_script"] = relative(Path(__file__))
    provenance["tsne_insertion_script_sha256"] = sha256(Path(__file__))
    write_json(path, provenance)


def main() -> None:
    campaign = json.loads((embedding_source.SOFT / "CAMPAIGN_STATE.json").read_text(encoding="utf-8"))
    letter = {"achilles": "a", "buddy": "b", "cicero": "c", "gatsby": "d"}
    summary = {}
    with tempfile.TemporaryDirectory(prefix="fig1d_ed3_tsne_") as temporary:
        preview = Path(temporary)
        staged = []
        for rat in RATS:
            source_path = ROOT / "data" / "rat_hippocampus" / f"{rat}.jl"
            raw = np.asarray(joblib.load(source_path)["position"], dtype=np.float32)
            panels, records = load_paper_panels(rat, raw, source_path, campaign)
            baseline_panels, _ = renderer.load_panels(
                rat, raw, sha256(source_path), renderer.paper_specs(), campaign)
            assert_same_non_tsne_data(baseline_panels, panels)
            fig1_target = FIG1 / rat.capitalize() / f"FIGURE_1D_{rat.upper()}_PAPER_METHODS"
            ed3_target = ED3 / letter[rat] / rat.capitalize() / f"ED3_{rat.upper()}_PAPER"
            fig1_preview = preview / "fig1d" / rat / fig1_target.name
            ed3_preview = preview / "ed3" / rat / ed3_target.name
            fig1_baseline = preview / "baseline_fig1d" / rat / fig1_target.name
            ed3_baseline = preview / "baseline_ed3" / rat / ed3_target.name
            renderer.draw(baseline_panels, rat, "Figure 1D", fig1_baseline, overwrite=False)
            ed3_renderer.render_ed3(rat, baseline_panels, ed3_baseline)
            renderer.draw(panels, rat, "Figure 1D", fig1_preview, overwrite=False)
            ed3_renderer.render_ed3(rat, panels, ed3_preview)
            fig1_updated = classify_existing(fig1_target.with_suffix(".png"),
                                             fig1_baseline.with_suffix(".png"),
                                             fig1_preview.with_suffix(".png"))
            ed3_updated = classify_existing(ed3_target.with_suffix(".png"),
                                            ed3_baseline.with_suffix(".png"),
                                            ed3_preview.with_suffix(".png"))
            if not fig1_updated:
                assert_baseline_matches(fig1_target.with_suffix(".png"), fig1_baseline.with_suffix(".png"))
            if not ed3_updated:
                assert_baseline_matches(ed3_target.with_suffix(".png"), ed3_baseline.with_suffix(".png"))
            staged.append((rat, panels, records, fig1_preview, ed3_preview, fig1_target, ed3_target,
                           fig1_updated, ed3_updated))

        for rat, panels, records, fig1_preview, ed3_preview, fig1_target, ed3_target, fig1_updated, ed3_updated in staged:
            fig1_target.parent.mkdir(parents=True, exist_ok=True)
            ed3_target.parent.mkdir(parents=True, exist_ok=True)
            if not fig1_updated:
                for suffix in (".png", ".pdf"):
                    replace_file(fig1_preview.with_suffix(suffix), fig1_target.with_suffix(suffix))
            if not ed3_updated:
                for suffix in (".png", ".pdf"):
                    replace_file(ed3_preview.with_suffix(suffix), ed3_target.with_suffix(suffix))
            if not fig1_target.with_suffix(".pdf").is_file() or not ed3_target.with_suffix(".pdf").is_file():
                raise FileNotFoundError(f"Paper PDF missing after t-SNE insertion for {rat}")
            record = records[next(i for i, item in enumerate(records) if item.get("title") == "t-SNE")]
            update_figure1d(rat, record, fig1_target.with_suffix(".png"), fig1_target.with_suffix(".pdf"))
            update_ed3(rat, letter[rat], record, ed3_target.with_suffix(".png"), ed3_target.with_suffix(".pdf"))
            summary[rat] = {"tSNE": record["embedding"], "plotted_rows": record["plotted_rows"],
                            "figure1d": relative(fig1_target.with_suffix(".png")),
                            "ed3": relative(ed3_target.with_suffix(".png")),
                            "figure1d_soft_touched": False, "ed3_soft_touched": False}

    root_manifest = FIG1 / "PROVENANCE.json"
    root = json.loads(root_manifest.read_text(encoding="utf-8"))
    for rat in RATS:
        per_rat = json.loads((FIG1 / rat.capitalize() / "PROVENANCE.json").read_text(encoding="utf-8"))
        paper = per_rat["panels"]["paper"]
        root["rat_folders"][rat]["paper_available"] = sum(item.get("status") == "verified" for item in paper)
        root["rat_folders"][rat]["paper_total"] = len(paper)
    root["tsne_insertion_utc"] = datetime.now(timezone.utc).isoformat()
    root["tsne_insertion_script"] = relative(Path(__file__))
    root["tsne_insertion_script_sha256"] = sha256(Path(__file__))
    write_json(root_manifest, root)

    ed3_root_path = ED3 / "PROVENANCE.json"
    ed3_root = json.loads(ed3_root_path.read_text(encoding="utf-8"))
    for rat in RATS:
        per_rat = json.loads((ED3 / letter[rat] / rat.capitalize() / "PROVENANCE.json").read_text(encoding="utf-8"))
        paper = per_rat["groups"]["Paper"]["models"]
        ed3_root["rats"][rat]["Paper"]["available"] = sum(item.get("status") == "verified" for item in paper)
        ed3_root["rats"][rat]["Paper"]["total"] = len(paper)
    ed3_root["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
    ed3_root["last_update_script"] = relative(Path(__file__))
    ed3_root["last_update_script_sha256"] = sha256(Path(__file__))
    write_json(ed3_root_path, ed3_root)

    index = FIG1 / "INDICE.md"
    text = index.read_text(encoding="utf-8")
    note = ("\nLa tavola Paper include ora t-SNE 2D per tutti e quattro i ratti: full recording, "+
            "finestre neurali complete da 10 bin, cosine, perplexity 30, inizializzazione PCA, "+
            "learning rate auto, early exaggeration 12/250, 1.000 iterazioni, seed 42. "+
            "Gli embedding e i parametri esatti "+
            "sono nei `PROVENANCE.json` per ratto. Le tavole Soft non sono state rigenerate.\n")
    if "La tavola Paper include ora t-SNE 2D" not in text:
        index.write_text(text.rstrip() + "\n" + note, encoding="utf-8")

    references = ED3 / "RIFERIMENTI.md"
    text = references.read_text(encoding="utf-8")
    lines = text.splitlines()
    replacement = ("- t-SNE aggiunto per tutti i ratti da `results/figura_1_2026-10-29/d/<rat>/tSNE_attempt_*/`: "+
                   "fit full-recording su finestre da 10 bin, output 2D, cosine, perplexity 30, PCA init, "+
                   "learning rate auto, early exaggeration 12 per 250 iterazioni, 1.000 iterazioni, seed 42; "+
                   "nessuna label comportamentale nel fit. "+
                   "Hash e configurazione nei `PROVENANCE.json` dei pannelli.")
    replaced = False
    for i, line in enumerate(lines):
        if line.startswith("- UMAP/t-SNE:"):
            lines[i] = replacement
            replaced = True
    if not replaced:
        lines.append(replacement)
    references.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
