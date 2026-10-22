# Monkey Figure 3 and Extended Data Figure 8 — 22 October 2026

All plots use white backgrounds. Existing scientific artifacts are reused;
no encoder was trained or refit by this delivery step.

## Figure 3 (paper panels)

`FIGURE3A.png` through `FIGURE3H.png` are the canonical paper-style local
replications. The panel-source map and caveats are in
`results/PAPER_FIGURE_DELIVERY/Figure_3/MONKEY_CURRENT_STATE_2026-10-19.md`.

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
