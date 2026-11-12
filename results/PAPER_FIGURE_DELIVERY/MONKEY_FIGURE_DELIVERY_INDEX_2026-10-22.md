# Current artifact map (updated 2026-11-02)

The dated figure folders under `results/` are the current rendered local outputs. Figure 3 is in `results/figura_3_2026-10-29/`; Extended Data 8 is in `results/figura_ext_8_2026-10-29/`. The 22 paper-reference ED8 panels remain in `PAPER_FIGURE_DELIVERY/Extended_Data/ED8/`.

The old rendered Figure 3 exports formerly listed below were superseded and moved to the Windows Recycle Bin on 2026-11-02. This file is retained as historical provenance, not as a current file inventory. See `results/RESULTS_CLEANUP_LOG_20261102.csv` for the exact cleanup manifest.

# Historical Monkey Figure 3 and Extended Data Figure 8 — 22 October 2026

All plots use white backgrounds. Existing scientific artifacts are reused;
no encoder was trained or refit by this delivery step.

## Figure 3 (paper panels)

`FIGURE3A.png` through `FIGURE3H.png` are the canonical paper-style local
replications. The panel-source map and caveats are in
`results/PAPER_FIGURE_DELIVERY/Figure_3/MONKEY_CURRENT_STATE_2026-10-19.md`.

## Added Soft Figure 3B-D: all active trials

The official 3B-D embedding visualizations are descriptive plots of cached
embeddings; their reference plotting code does not apply a held-out-only
filter. The Soft counterparts are now rendered over all valid active rows by
joining the already-frozen train, validation and test embedding exports:

- `Figure_3/FIGURE3B_SOFT_FULL_SAMPLE_SEED_42` through `_46`: direction-mean
  trajectories for CNN/Transformer × time-only/behavior-only/joint, using the
  existing 4D Soft fits and their fixed first-two-coordinate display.
- `Figure_3/FIGURE3C_SOFT_FULL_SAMPLE_SEED_42` through `_46`: behavior-only
  XY embeddings, colored by x/y position; existing 3D Soft fits.
- `Figure_3/FIGURE3D_SOFT_FULL_SAMPLE_SEED_42` through `_46`: time-only
  embeddings, colored by x/y position; existing 3D Soft fits.

There are 114,063 valid sample rows across 193 active trials, with identical
trial/time support across the six Soft conditions and five seeds. The encoders
remain fitted on their saved training split; plotting all cached splits is
descriptive and is not held-out generalization evidence or a full-sample refit.
No model or decoder was retrained. `Figure_3/FIGURE3_SOFT_FULL_SAMPLE_PROVENANCE.json`
records parent/export and figure hashes, support checks, and the prior draft
figure hashes that were replaced after visual-layout corrections.

## Added Soft held-out outputs

- `FIGURE3_SOFT_HELDOUT_EMBEDDINGS_SEED_42` through `_46`: six panels per
  seed (CNN/Transformer × time-only/behavior-only/joint), active test trials,
  3D output, colored by x-position. One image is provided per seed to keep the
  trajectories readable instead of superimposing seed trajectories.
- `FIGURE3H_SOFT_HELDOUT_DECODING_5SEED`: position R² and direction accuracy,
  all dimensions and five individual seeds. The matching CSV preserves each
  seed value. These test scores are descriptive, not selection evidence.

Soft was fit on active trials with joint XY behavior. It cannot be inserted as
an equivalent method in panels requiring passive or all-trial AP labels (3e-g).
Those panels therefore remain the existing paper-protocol outputs; this is a
data-scope limitation, not a missing fit within the completed Soft campaign.

## Extended Data Figure 8

The existing MSE embedding, held-out AP decoding and direction-timecourse
figures remain in this folder. `EXTENDED8_CEBRA_REFERENCE_LOSS_CURVES` is a
replot of the published reference loss arrays in the official HDF5. The five
`EXTENDED8_SOFT_TRAINING_LOSS_DIM_*` figures summarize the locally completed
Soft training histories, with mean ± SD over seeds 42–46 and separate
shuffled-label controls. Different objectives use different loss scales and
are not compared numerically across panels.

Full lineage and output hashes: `Figure_3/MONKEY_FIGURE_DELIVERY_PROVENANCE_2026-10-22.json`.
