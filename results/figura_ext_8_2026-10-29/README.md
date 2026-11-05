# Extended Data Figure 8 — local rebuild (2026-10-29)

Rendered panels are in `plot/a/` through `plot/f/`; metrics, status, and provenance are in `processo/`. Figure titles, axes, and legends are in English.

This rebuild reuses saved local CEBRA embeddings/scores and the official local `EDFigure8.h5` loss traces. It does not launch training and reads the active Soft campaign without changing its files. Soft figures are written only after the corresponding fit manifests and output hashes are present. Run `python -m cebra_replica.monkey.render_ed8_20261029` again after the Soft campaign finishes to incorporate newly completed artifacts.

Panel e uses 32D for Soft solely as an experimental choice; the plotting code does not specify the original fit dimension. The Soft objective is Stable loss (`T_embed=1`, `T_metadata=0.5`), not the CEBRA InfoNCE/InfoMSE objective.

See `PANEL_STATUS.json` for panel-by-panel readiness, `SOFT_FIT_AUDIT.json` for all 136 planned cells, and `PROVENANCE.json` for SHA-256 input/output hashes.
