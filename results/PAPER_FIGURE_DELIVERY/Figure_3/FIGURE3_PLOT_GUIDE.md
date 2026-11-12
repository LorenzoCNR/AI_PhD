# Figure 3 — regenerated panels

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
