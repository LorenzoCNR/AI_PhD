# Figura 1 — indice e riferimenti

Radice: `results/figura_1_2026-11-01/`. Le figure sono in `plot/<lettera>/<ratto>/`; metriche, embedding intermedi e provenienza sono in `processo/`.

## Pannelli

- `plot/c/`: riepilogo descrittivo dei dati dei quattro ratti; codice `cebra_replica/rat_figure_gap_closure_analysis.py`.
- `plot/d/<Rat>/`: output statici Paper e Soft; Achilles conserva anche i plot interattivi. I file dei plot 1D sono stati mantenuti senza rigenerarli né modificarli.
- `processo/embedding/d/<Rat>/tSNE_attempt_*/`: fit t-SNE completi, uno per ratto. Registrazione completa, finestre neurali da 10 bin, output 2D, cosine, perplexity 30, inizializzazione PCA, learning rate auto, early exaggeration 12 per 250 iterazioni, 1.000 iterazioni, seed 42. Le etichette comportamentali servono per colorare il plot, non per il fit.
- `plot/e/`: consistenza cross-rat; specifiche e provenienza in `processo/provenienza/`.

## Codice di Figura 1D

- Fit t-SNE: `cebra_replica/rat_fig1d_tsne_stable_20261029.py`.
- Inserimento dei fit t-SNE nelle figure: `cebra_replica/rat_insert_fig1d_ed3_tsne_20261029.py` (lo script rigenera anche ED3; non eseguirlo per aggiornare soltanto Figura 1).
- Renderer statico: `cebra_replica/rat_render_figura1_saved.py`.
- Specifiche e validazione dei fit: `cebra_replica/rat_rebuild_1d_ed3_20261027.py`.
- Renderer interattivi: `cebra_replica/rat_rebuild_1d_position_scale_20261027.py` e `cebra_replica/rat_figure1_soft_achilles_interactive.py`.

Fit Paper di Figura 1D: `results/rat_paper_replication_2023_v1/ED3/{cebra,pivae_full_recording,manifold}/`. Fit Soft: `results/user_soft_rat_fullrecording_t1_3conditions_seed42_v1/`.
