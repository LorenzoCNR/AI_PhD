# `cebra_replica/` — code and execution map

All project-specific rat replication Python code is kept here; datasets and result artifacts stay at the project root in `data/` and `results/`.

## Main code groups

- Fit/campaign: `rat_benchmark.py`, `rat_campaign.py`, `rat_figure2c_3d.py`, and `rat_conv_pivae_fit.py`.
- Frozen-embedding scoring and tables: `rat_score.py`, `rat_results_table.py`, `rat_paper_comparison.py`, `rat_manifold_scoring.py`.
- Embedding metadata and figures: `rat_embedding_metadata.py`, `rat_colored_embedding_fig1.py`, `rat_figure2c_from_frozen_32d.py`, `rat_pilot_figures.py`, `plot_figure2c_3d_cebra_style.py`, `rat_figure_gap_closure.py`, `rat_figure_gap_closure_analysis.py`, `rat_figure_gap_scoring_completion.py`.
- PiVAE: `rat_pivae_export.py`, `rat_conv_pivae_recover_embeddings.py`.
- Manifold baselines: `rat_manifold_benchmark.py`.

The result-facing, panel-organized view is `../results/PAPER_FIGURE_DELIVERY/`; full, canonical per-run outputs remain under `../results/rat_*`.

## Path and provenance note

The ten scripts formerly at the project root now resolve the project root as their parent directory. `rat_campaign.py` invokes `cebra_replica/rat_benchmark.py`. This is a code-layout change only; no experiment settings or scientific results were changed. The exact scripts as they existed immediately before relocation are preserved in `source_snapshots/pre_layout_2026-10-12/`; the prior one-off organizer is preserved under `archive/`.

Because paths are part of provenance and the fit runner hash is recorded in each result, `rat_campaign.py` accepts only the hash of the current path-adjusted runner or the exact archived pre-layout runner when verifying completed fits. New fits, if deliberately authorized later, use the current runner hash. No campaign was started by this reorganization.
