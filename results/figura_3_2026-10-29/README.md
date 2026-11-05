# Figure 3 — canonical outputs

This is the single delivery folder for Figure 3. Plots are stored in `plot/a` through `plot/i`; process records are under `processo/`.

- Panels 3a–3h use the selected October 29 outputs; panel 3h is the latest rebuild from already-computed scores and does not retrain models or decoders.
- Panel 3i uses the selected October 28 rendering; its source hashes are recorded in `processo/provenienza/i/fonti_sha256.json`.
- Panel 3h keeps the matrix and adds `Figure_3h_model_dots.png` / `.pdf`: individual seed scores are points, embedding dimension is color, the interquartile range is a segment, and the median is a diamond. Its additional provenance is in `processo/provenienza/h/`.
- Per-panel provenance and source hashes are in `processo/provenienza/<lettera>/`.
- `Figura_3_specifiche_modelli.xlsx` contains the model specification table; `RIFERIMENTI.md` indexes the panel sources and provenance.

For 3i, the local CEBRA plot is 16D and the local Soft plots are 3D (seeds 42–46); these are the existing local outputs, not a claim that Soft matches the original paper's 16D model.
