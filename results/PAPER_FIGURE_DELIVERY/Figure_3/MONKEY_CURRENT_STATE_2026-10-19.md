# Monkey replica: verified output index, 19 October 2026

All paths below are relative to `AI_PhD/`. Read-only artifact inspection;
documentation-only update, no fitting, metric recomputation or rendering.
VERIFIED means the named local artifact exists, not bitwise reproduction of
the authors' artwork or experimental settings. Raw/unit remain separate.

## Figure 3

Panel | Status | Exact local artifact | Interpretation
---|---|---|---
3a | VERIFIED | `results/monkey_figure3_v1/stage00_data/FIGURE3A_BEHAVIOR_FROM_OFFICIAL_RAW.png` (+PDF) | Descriptive behavior from official raw bundles
3b | VERIFIED, excluding autoLFADS by request | `results/monkey_figure3_paper_v1/stage03_figures/figure3b_comparison/FIGURE3B_LOCAL_NO_AUTOLFADS.png` (+PDF); `PROVENANCE.json` beside it | Full-sample descriptive CEBRA, conv-piVAE, t-SNE, UMAP; exact monkey UMAP/t-SNE settings not documented in paper, local rat settings reused
3c | VERIFIED | `results/monkey_figure3_paper_v1/stage03_figures/FIGURE3C_PAPER_SETTINGS.png` (+PDF) | Active position, descriptive full-sample
3d | VERIFIED | `results/monkey_figure3_paper_v1/stage03_figures/FIGURE3D_PAPER_SETTINGS.png` (+PDF) | Active time, descriptive full-sample
3e | VERIFIED | `results/monkey_figure3_paper_v1/stage03_figures/FIGURE3E_PAPER_SETTINGS.png` (+PDF) | Passive target, descriptive full-sample
3f | VERIFIED | `results/monkey_figure3_paper_v1/stage03_figures/FIGURE3F_PAPER_SETTINGS.png` (+PDF) | All target/AP, descriptive full-sample
3g | VERIFIED | `results/monkey_figure3_paper_v1/stage03_figures/FIGURE3G_PAPER_SETTINGS.png` (+PDF) | All position/AP, descriptive full-sample
3h | VERIFIED | `results/monkey_figure3_paper_v1/stage03_figures/figure3hi/FIGURE3H_LOCAL_REPLICATION.png` (+PDF); `FIGURE3H_PROVENANCE.json` beside it | Held-out position R2, direction accuracy and AP accuracy across dimensions/seeds
3i | VERIFIED | `results/monkey_figure3_paper_v1/stage03_figures/figure3hi/FIGURE3I_LOCAL_REPLICATION.png` (+PDF); `FIGURE3I_RESULT.json` beside it | Illustrative held-out decoded trajectory, not extra encoder training

This delivery folder currently contains a copy of 3b only. The table links to
the canonical existing other panels instead of duplicating payloads. Historical
`stage03_figures/PANEL_STATUS.json` has a stale 3b-in-progress entry; the actual
3b comparison provenance and files above take precedence. Its scientific
artifact was deliberately not rewritten.

## Fits, embeddings and decoding

- `results/monkey_figure3_paper_v1/stage01_fit_heldout/`:
  75 checkpoints = tasks `position`, `direction`, `ap` x dimensions
  3/4/8/16/32 x seeds 42-46. All 75 SHA256 values matched their manifests.
- `results/monkey_extended8_paper_v1/stage01_fit_heldout/`:
  50 checkpoints = tasks `ap_pos`, `ap_direction` x same dimensions/seeds.
  All 50 SHA256 values matched their manifests.
- Corresponding `stage02_embedding_heldout/` roots contain 225 and 150
  canonical split exports respectively, one per train/valid/test per model.
  Three additional ED8 `combined/` exports are not three additional fits.
- Corresponding `stage03_decoding_heldout/` roots contain 75 and 50
  `DECODING_RESULT.json` records. kNN36/cosine follows the official decoding
  demo; the exact Figure 3h paper K was not verified and is not claimed.
- Figure 3 full-sample descriptive `stage01_fit/` has 7 independent fits,
  and `stage02_embedding/` has 7 exports. Baseline fits/exports are in
  `stage01_fit_baselines/` and `stage02_embedding_baselines/`.
- Conv-piVAE has one trained direction-prior checkpoint. Its with-label and
  without-inference-label emissions are two exports from that one model,
  not two independently trained algorithms. Do not use label-conditioned
  inference to claim label-free held-out decoding.

Active and all-trial validation/test split universes differ. Preserve the
explicit IDs from `results/monkey_figure3_v1/stage00_data/DATA_AND_SPLIT_PROVENANCE.json`;
do not filter one universe and pretend it is the other.

## Extended Data 8

Component | Status | Exact artifacts
---|---|---
MSE target/position/time embeddings and plots | VERIFIED | `results/monkey_extended8_paper_v1/stage01_fit_mse/`, `stage02_embedding_mse/`, `stage03_figures/mse_panels/`; delivery `results/PAPER_FIGURE_DELIVERY/Extended_Data/ED8/EXTENDED8_MSE_TARGET.png`, `EXTENDED8_MSE_POSITION.png`, `EXTENDED8_MSE_TIME.png` (+PDF), `ED8_MSE_PROVENANCE.json`
AP-position/AP-direction held-out decoding | VERIFIED | `results/PAPER_FIGURE_DELIVERY/Extended_Data/ED8/EXTENDED8_HELDOUT_AP_DECODING.png` (+PDF), `ED8_AP_DECODING_SEEDS.csv`, `ED8_AP_DECODING_PROVENANCE.json`
Direction accuracy over time | VERIFIED | `results/PAPER_FIGURE_DELIVERY/Extended_Data/ED8/EXTENDED8_DIRECTION_DECODING_TIMECOURSE.png` (+PDF), `DIRECTION_ACCURACY_BY_TIME.csv`, `ED8_DIRECTION_TIMECOURSE_PROVENANCE.json`
InfoNCE/InfoMSE label and shuffle loss panels | PARTIAL / missing complete curve grid | Explicitly missing in `results/monkey_extended8_paper_v1/STATUS.json`; definition in `cebra_replica/monkey/official_reference/ExtendedDataFigure8.py:124-340`

The last component requires 20 curve conditions, not 20 already verified fits:

- All-trial InfoNCE: `ap`, `pos`, `target`, `pos_target`, `ap_pos`,
  `ap_target`, `ap_pos_target`, `ap_shuffle`, `pos_shuffle`, `target_shuffle`.
- Active InfoNCE: `pos`, `target`, `pos_target`, `pos_shuffle`, `target_shuffle`.
- Active InfoMSE: the same five conditions.

The three completed 2D/5000-update MSE embedding fits do not fill this grid.
No existing runner enumerates all 20 conditions. The plot source displays
10000 steps but does not resolve exact dimensions, training seeds,
combined-label sampler mapping or shuffle unit. Existing unrelated histories
must not be silently substituted. Existing curves may be reused only after
matching full config/source/label provenance.

## Own Soft monkey extension: not implemented or scientifically frozen

The implemented `user_soft` data adapter is rat-only. Before monkey fits,
explicitly record: which of active/passive/all split universes; behavior
geometry for joint x/y and categorical direction/AP; relative component
weights and train-only scales; behavior/time/joint condition roster; stable
versus historical loss roster; dimensions and seed budget; shuffle unit.
These choices are not resolved by the word "all" alone. The local trial
time coordinate is available, but ten input bins mean 10 ms for these monkey
data, versus 250 ms for the rat data. No monkey Soft fit was launched by this
documentation task.

## Source and parent SHA256, verified on 2026-10-19

Source | SHA256
---|---
`cebra_replica/monkey/fit_figure3h.py` | `70d33667a5c8c2667ab5c702eb625ff1302a8e8c0b7fa4de52f7508a0f332402`
`cebra_replica/monkey/official_reference/Figure3.py` | `b72658f63857faabb99c4f073f4c0596c86ef6ac89d20efd330b69460e919276`
`cebra_replica/monkey/official_reference/ExtendedDataFigure8.py` | `ca52706258dafa849dceb4b17f37bfef9e14aea5460fb86bfb4d6354957bb1c5`
`results/monkey_figure3_v1/stage00_data/DATA_AND_SPLIT_PROVENANCE.json` | `c78bfc1659c093b4bc1fc935e32df29471ad7f32b0de51ae93157ea937cfc3b5`

Individual checkpoint, source/config/split and export hashes remain in each
existing fit/export manifest; this index does not replace them. This check
verified checkpoint hashes and artifact cardinalities, not a new full-array
scientific reanalysis. No test result was used for model selection.
