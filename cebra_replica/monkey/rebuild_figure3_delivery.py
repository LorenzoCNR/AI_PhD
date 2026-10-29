"""Rebuild the Figure 3 delivery folder from frozen local artifacts only.

This script does not fit encoders or decoders. Pass an empty staging directory
with --output; after visual verification, replace the delivery directory with
the staged result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT / "results" / "PAPER_FIGURE_DELIVERY" / "Figure_3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_pair(source_stem: Path, target_stem: Path) -> None:
    for suffix in (".png", ".pdf"):
        source = source_stem.with_suffix(suffix)
        target = target_stem.with_suffix(suffix)
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true",
                        help="Continue the task-owned staging directory after removing obsolete held-out B-G plots")
    args = parser.parse_args()
    out = args.output.resolve()
    if out != DEFAULT_OUTPUT.resolve() and PROJECT not in out.parents:
        raise ValueError(f"Output must stay inside this project: {out}")
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if out.name != "Figure_3_REBUILD_20261023":
            raise ValueError(f"Resume is restricted to this task's staging directory: {out}")
        stale_stems = ["FIGURE3B_CEBRA_ONLY_PAPER_SETTINGS", "FIGURE3B_LOCAL_NO_AUTOLFADS",
                       "FIGURE3C_PAPER_SETTINGS", "FIGURE3D_PAPER_SETTINGS",
                       "FIGURE3E_PAPER_SETTINGS", "FIGURE3F_PAPER_SETTINGS",
                       "FIGURE3G_PAPER_SETTINGS"]
        stale_stems += [f"FIGURE3{panel}" for panel in "BCDEFGHI"]
        stale_stems += ["FIGURE3H_LOCAL_REPLICATION", "FIGURE3I_LOCAL_REPLICATION"]
        for stem in stale_stems:
            for suffix in (".png", ".pdf"):
                path = out / f"{stem}{suffix}"
                if path.exists():
                    path.unlink()
        stale_files = ["FIGURE_PROVENANCE.json", "PANEL_STATUS.json", "PROVENANCE.json",
                       "FIGURE3H_PROVENANCE.json", "FIGURE3I_RESULT.json",
                       "SOFT_FIGURE3_BDEGI_PROVENANCE.json",
                       "FIGURE3H_SOFT_HELDOUT_SEED_VALUES.csv"]
        stale_files += ["FIGURE3H_SOFT_HELDOUT_DECODING_5SEED.png",
                        "FIGURE3H_SOFT_HELDOUT_DECODING_5SEED.pdf"]
        for name in stale_files:
            path = out / name
            if path.exists():
                path.unlink()
        obsolete_prefixes = (
            "FIGURE3B_SOFT_HELDOUT_SEED_", "FIGURE3D_SOFT_HELDOUT_SEED_",
            "FIGURE3E_SOFT_HELDOUT_SEED_", "FIGURE3F_SOFT_HELDOUT_SEED_",
            "FIGURE3G_SOFT_HELDOUT_SEED_", "FIGURE3I_SOFT_HELDOUT_DECODED_TRAJECTORY_SEED_",
            "FIGURE3_SOFT_HELDOUT_EMBEDDINGS_SEED_",
        )
        for path in out.iterdir():
            if path.is_file() and path.name.startswith(obsolete_prefixes):
                path.unlink()
            elif path.is_file() and (".tmp." in path.name or path.name.endswith(".rendering")):
                path.unlink()
        soft_manifest_path = out / "FIGURE3_SOFT_FULL_SAMPLE_PROVENANCE.json"
        if soft_manifest_path.is_file():
            soft_manifest = json.loads(soft_manifest_path.read_text(encoding="utf-8"))
            recoverable = {
                "FIGURE3B_SOFT_FULL_SAMPLE_SEED_42.png",
                "FIGURE3B_SOFT_FULL_SAMPLE_SEED_42.pdf",
                "FIGURE3C_SOFT_FULL_SAMPLE_SEED_42.png",
            }
            recovered = []
            for name, expected in soft_manifest.get("figure_sha256", {}).items():
                artifact = out / name
                actual = sha256(artifact) if artifact.is_file() else None
                if actual != expected:
                    if name not in recoverable or actual is None:
                        raise RuntimeError(f"Unexpected changed Soft artifact during resume: {name}")
                    soft_manifest["figure_sha256"][name] = actual
                    recovered.append(name)
            if recovered:
                soft_manifest["interrupted_render_recovery"] = recovered
                soft_manifest_path.write_text(json.dumps(soft_manifest, indent=2) + "\n", encoding="utf-8")
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    import plot_figure3_paper as cebra_panels
    import plot_figure3b_comparison as cebra_b
    import plot_soft_figure3_full_sample as soft_full
    import deliver_soft_figure3_panels as soft_test
    import deliver_monkey_figures as soft_summary
    import plot_figure3hi as cebra_hi

    # Render directly into the clean staging folder; source fit outputs stay put.
    cebra_panels.FIGURES = out
    cebra_panels.main()
    cebra_b.FIGURE = out
    cebra_b.main()

    # The CEBRA 3H/3I plots use cached held-out scores/embeddings. No encoder fit is run.
    cebra_hi.ROOT = out
    cebra_hi.panel_i_from_saved_plot()
    cebra_hi.panel_h()

    soft_full.OUT = out
    soft_full.main()
    soft_test.OUT = out
    soft_summary.FIG3 = out
    soft_summary.soft_decoding()

    # Soft held-out support is used only for 3I here; B-G use full available trials.
    soft_i_output_hashes = {}
    soft_i_embedding_hashes = {}
    soft_i_prediction_hashes = {}
    soft_i_cache = {}
    for seed in soft_test.SEEDS:
        for architecture in soft_test.ARCHITECTURES:
            for condition in soft_test.CONDITIONS:
                data = soft_test.load_embedding(architecture, condition, seed)
                soft_i_cache[(architecture, condition, seed)] = data
                label = soft_full.SOFT_OBJECTIVE_LABELS[condition]
                soft_i_embedding_hashes[f"{architecture}/{label}/seed_{seed}"] = sha256(data["_path"])
    for seed in soft_test.SEEDS:
        soft_test.plot_3i(seed, soft_i_cache, soft_i_output_hashes, soft_i_prediction_hashes)
    soft_i_provenance = {
        "renderer": str(Path(__file__).relative_to(PROJECT)),
        "panel": "3I",
        "heldout_support": "active TEST rows only",
        "seeds": list(soft_test.SEEDS),
        "embedding_dimension": 3,
        "decoder_refit": False,
        "saved_test_predictions_reused": True,
        "parent_embedding_sha256": soft_i_embedding_hashes,
        "parent_prediction_sha256": soft_i_prediction_hashes,
        "figure_sha256": soft_i_output_hashes,
        "objective_labels": [soft_full.SOFT_OBJECTIVE_LABELS[name] for name in soft_test.CONDITIONS],
    }
    (out / "FIGURE3I_SOFT_HELDOUT_PROVENANCE.json").write_text(
        json.dumps(soft_i_provenance, indent=2) + "\n", encoding="utf-8")

    source_figure3a = PROJECT / "results" / "monkey_figure3_v1" / "stage00_data" / "FIGURE3A_BEHAVIOR_FROM_OFFICIAL_RAW"

    # Canonical-named CEBRA panels; explanatory source plots remain alongside.
    copy_pair(source_figure3a, out / "FIGURE3A")
    copy_pair(out / "FIGURE3B_LOCAL_NO_AUTOLFADS", out / "FIGURE3B")
    for panel in "CDEFG":
        copy_pair(out / f"FIGURE3{panel}_PAPER_SETTINGS", out / f"FIGURE3{panel}")
    for panel in "HI":
        copy_pair(out / f"FIGURE3{panel}_LOCAL_REPLICATION", out / f"FIGURE3{panel}")

    readme = """# Figure 3 — regenerated panels

Panels B–G are full-sample descriptive views, not held-out plots. For Soft, the saved encoder was trained on the train split, while the plotted embeddings concatenate valid train, validation, and test rows. Only panels H and I are held-out decoding evaluations. No encoder or decoder was retrained by this delivery; CEBRA 3I reuses its saved plot and scores with added labels.

## CEBRA

- `FIGURE3B`: aligned six-method view. CEBRA-Behavior target: direction 1–8; CEBRA-Time: no behavior target. Other method labels are in the panel.
- `FIGURE3C`: CEBRA target position X,Y; points colored by position X or Y.
- `FIGURE3D`: CEBRA has no behavior target; position X,Y only colors the points.
- `FIGURE3E`: separate active and passive fits, target direction 1–8; 4D fit, three latent coordinates plotted.
- `FIGURE3F`: active and passive data, targets direction 1–8 plus condition; 4D fit, three latent coordinates plotted.
- `FIGURE3G`: active and passive data, target position X,Y; 4D fit, three latent coordinates plotted; direction colors are post hoc.
- `FIGURE3H`: held-out position X,Y, direction 1–8, and active/passive condition decoding, five seeds.
- `FIGURE3I`: held-out position X,Y decoding; original saved scatter reused and labeled, with no decoder refit.

## Soft

- `FIGURE3B_SOFT_FULL_SAMPLE_SEED_*`: 4D output, plotted z1–z2; direction-averaged trajectories for CNN and Transformer under time-only, position X,Y-only, and time + position X,Y losses. All valid active rows are shown; direction is post hoc, not a loss target.
- `FIGURE3C_SOFT_FULL_SAMPLE_SEED_*`: 3D output; position X,Y-only loss; points colored by position X or Y.
- `FIGURE3D_SOFT_FULL_SAMPLE_SEED_*`: 3D output; time-only loss; points colored by position X or Y for display.
- `FIGURE3E_SOFT_FULL_SAMPLE_SEED_*`: 3D output; active-only position X,Y fit, closest available Soft analogue to the direction-trajectory panel. Direction is post hoc.
- `FIGURE3F_SOFT_FULL_SAMPLE_SEED_*`: 3D output; active-only time + position X,Y fit. Direction is post hoc; no direction or active/passive loss target.
- `FIGURE3G_SOFT_FULL_SAMPLE_SEED_*`: 3D output; active-only position X,Y fit. This reuses the same available position-only Soft objective as 3E; no separate direction-target Soft fit exists.
- `FIGURE3H_SOFT_HELDOUT_DECODING_5SEED` and CSV: saved held-out test scores across seeds 42–46.
- `FIGURE3I_SOFT_HELDOUT_DECODED_TRAJECTORY_SEED_*`: saved held-out predictions; kNN36/cosine was trained on TRAIN only and was not refit here.

Soft has active trials only. It has no direction or active/passive loss target, so its E–G panels are descriptive analogues, not exact CEBRA objective matches.

## Shared colors and lineage

Direction 1–8 uses `hsv(index/8)` identically for CEBRA and Soft. Position X/Y uses `seismic` with fixed limits −15…+15 cm. `FIGURE3_REBUILD_PROVENANCE.json` records output hashes and confirms no model fit was launched.
"""
    (out / "FIGURE3_PLOT_GUIDE.md").write_text(readme, encoding="utf-8")

    files = sorted(path for path in out.iterdir() if path.is_file())
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "renderer": str(Path(__file__).relative_to(PROJECT)),
        "renderer_sha256": sha256(Path(__file__)),
        "training_performed": False,
        "fit_or_decoder_retrained": False,
        "output_file_count_excluding_this_manifest": len(files),
        "output_file_count_including_this_manifest": len(files) + 1,
        "shared_colors": {
            "direction": "matplotlib hsv(index/8), direction array index 0..7",
            "position": "seismic, fixed vmin=-15 cm and vmax=15 cm",
        },
        "outputs": {path.name: sha256(path) for path in files},
        "notes": [
            "Soft 3E/3F/3G are active-only descriptive analogues and are not exact CEBRA objective replications.",
            "Direction is used for post-hoc grouping/color in Soft trajectories, not as a loss target.",
            "Only panels 3H and 3I use held-out test evaluation; panels 3B-G are full-sample descriptive views.",
        ],
    }
    (out / "FIGURE3_REBUILD_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Rebuilt Figure 3 delivery in {out}")
    print(f"Files: {len(list(out.iterdir()))}; no fit was launched")


if __name__ == "__main__":
    main()
