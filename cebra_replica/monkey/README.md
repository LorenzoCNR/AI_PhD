# Monkey Area 2 replication — Figure 3 first

This branch is separate from the rat campaign and from NeuroBridge. The
official CEBRA Area 2 bundles are in `../../data/monkey_reaching_preload_smth_40/`.
`prepare_figure3.py` checks all 12 published MD5 checksums and exact trial
contents, reconstructs split IDs, and renders only the descriptive behavioral
Figure 3a panel. It does not fit a model.

The official Figure 3 and Extended Data Figure 8 plot scripts and HDF5 result
containers are preserved unmodified in `official_reference/`. These are
**reference results from the authors**, not results of our own fits. Their
scripts are the visual authority for later panels; do not silently substitute
new colors, axes, legends, projections, or model dimensions.

Important split detail: `active_valid/test` and `passive_valid/test` are not
identical to filtering `all_valid/test`, because validation is halved after
the session filter. The exact mapping is recorded in
`../../results/monkey_figure3_v1/stage00_data/DATA_AND_SPLIT_PROVENANCE.json`.
Select one split universe per model/evaluation and never combine them.

Stages:

1. `stage00_data`: source verification, trial mapping, Figure 3a behavior.
2. `stage01_fit`: checkpoint and training history per exact model condition.
   The first active-position 3D official-demo-aligned fit, seed 42, is complete
   after 5,000 updates. It is not claimed to be the paper's original fit.
3. `stage02_embedding`: immutable embeddings with trial/time, direction,
   active/passive, position, split, and parent checkpoint hash.
   The first position fit has all/train/valid/test exports.
4. `stage03_figures`: Figure 3b–h rendered from frozen embeddings, following
   the authors' Figure 3 source and declaring any panel not reproduced.
   Figure 3c-style all and held-out views exist for the first position fit.
5. Extended Data 8 in a separate downstream branch after Figure 3.

Sources: https://cebra.ai/docs/cebra-figures/figures/Figure3.html and
https://cebra.ai/docs/cebra-figures/figures/ExtendedDataFigure8.html .

## Current paper-settings branch (2026-10-12)

`results/monkey_figure3_paper_v1/stage01_fit/` contains seven independent
10,000-update full-sample descriptive CEBRA fits for Figure 3b-g. The frozen
raw/unit embeddings and trial/behavior metadata are in `stage02_embedding/`;
the local figures are in `stage03_figures/`. The Figure 3b comparison is complete
in the separate `stage01_fit_baselines/`, `stage02_embedding_baselines/`, and
`stage03_figures/figure3b_comparison/` branches: CEBRA-Time/Behavior, conv-piVAE,
t-SNE, and UMAP are included; autoLFADS is excluded. The piVAE with-label and
without-inference-label trajectories are two outputs from the same
direction-prior-trained checkpoint. UMAP/openTSNE settings follow the existing
rat baseline because the monkey paper does not state their exact settings; this
is a comparability caveat. Figure 3b maps are full-sample descriptive, not
held-out decoding results.

Extended Data Figure 8 has its own output root,
`results/monkey_extended8_paper_v1/`. Three CEBRA MSE fits follow the authors'
primate reaching demo values (offset10, 2D, 5,000 updates, batch 512, learning
rate 5e-5, temperature 0.01, Euclidean distance). The separate stages are
`stage01_fit_mse/` and `stage02_embedding_mse/`; their raw/unit embeddings,
trial metadata and three corresponding official-style figures are complete in
`stage03_figures/mse_panels/`. The all-trial AP-position and AP-direction branches
each have 25/25 fits complete (dimensions 3, 4, 8, 16, 32; seeds 42-46), with
held-out embeddings and test decoding scores. Here AP means active/passive:
AP-position is trained with hand position plus active/passive labels, and
AP-direction with direction plus active/passive labels. The combined decoding
figure and full seed-level table are in
`results/PAPER_FIGURE_DELIVERY/Extended_Data/ED8/`. Its position axis follows the
authors' 50-80% display range, so the 3D/4D values below 50% are outside the plot;
their exact scores remain in the CSV. Direction accuracy over time for CEBRA,
conv-piVAE, and Ridge is also complete. The InfoNCE/InfoMSE loss-curve panels
remain missing because existing histories do not cover all required label
combinations and shuffled-label controls; they were not inferred from unrelated
runs. The comparison source is the unmodified authors'
`official_reference/ExtendedDataFigure8.py`.

`fit_figure3h.py` writes the original Figure 3h held-out branch at
`results/monkey_figure3_paper_v1/stage01_fit_heldout/`. It uses only the
official session-specific train files, with 3 tasks x 5 dimensions x 5 seeds.
The added ED8 AP-position and AP-direction fits use separate roots under
`results/monkey_extended8_paper_v1/stage01_fit_heldout/`.
Each fit has an immutable checkpoint, loss history, and source/config hashes.
Seeds 42-46 are local replication seeds, not asserted to be the authors' exact
random seeds. Figure 3h decoding and Figure 3i held-out trajectory are complete
in `results/monkey_figure3_paper_v1/stage03_figures/figure3hi/`, with PNG/PDF,
provenance and the trajectory result record. Test outcomes are reported
descriptively and are not used for model selection.

## Verified navigation update (2026-10-19)

The current inventory verifies 125 held-out CEBRA checkpoint hashes against
their manifests: 75 Figure 3 fits (position, direction, AP; each 5 dimensions
x 5 seeds) and 50 ED8 fits (AP-position, AP-direction; each 5 dimensions x
5 seeds). These have 375 canonical train/valid/test embedding exports and
125 decoder result records. Existing fits must be reused, not retrained.

The local run tree does not contain fit histories for all 20 ED8 loss
conditions. However, the published curve arrays themselves are present in the
local official-reference container
`cebra_replica/monkey/official_reference/EDFigure8.h5`; they can be replotted
as reference data without fitting. The three completed MSE embedding fits
remain separate from those curves. Do not describe reference curves as local
fits.

Use the dated index
`results/PAPER_FIGURE_DELIVERY/Figure_3/MONKEY_CURRENT_STATE_2026-10-19.md`
for every Figure 3 panel, ED8 artifacts and unresolved own-loss mappings.
Historical `PANEL_STATUS.json`, root STATUS/TODO and older README passages
are not authoritative live campaign counters. This update changes only
documentation: no checkpoint, embedding, split, result or figure was changed.

## Figure delivery and Soft monkey results (2026-10-22)

`results/user_soft_monkey_v1/CAMPAIGN_STATE.json` verifies 250/250 completed
fits: 150 observed and 100 behavior-shuffled controls, CNN and Transformer,
time-only / behavior-only / joint XY, dimensions 3/4/8/16/32, five seeds
42-46. Training histories are stored in each fit's `manifest.json`; held-out
test embeddings and decoding results are already saved. No additional seed or
encoder fit is required to make the current plots.

The organized delivery is indexed by
`results/PAPER_FIGURE_DELIVERY/MONKEY_FIGURE_DELIVERY_INDEX_2026-10-22.md`.
It places Figure 3 A-H together, shows Soft test embeddings separately by
seed, summarizes Soft test decoding over all five seeds, and plots Soft loss
histories by dimension. The ED8 reference-loss plot reuses the published
arrays from the local HDF5; it is explicitly not a new local CEBRA fit.
Soft monkey runs use active trials and joint XY behavior only, so they do not
provide matching Soft fits for passive/all-trial AP panels 3e-g.
