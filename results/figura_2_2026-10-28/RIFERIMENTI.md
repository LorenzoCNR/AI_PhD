# Figura 2 — riferimenti

Codice: `cebra_replica/rat_figura2_20261028.py`. I plot sono in `plot/b|d/<ratto>/`; metriche, calcoli e provenienza sono in `processo/`. La tavola b usa le stime registrate nei riferimenti per grafico. Due prove di input comportamentale 2D sono esplorative, non configurazioni originali confermate.

In b è disponibile per tutti e quattro i ratti anche il singolo pannello CEBRA Time + Position + Direction, output 5D, coordinate visualizzate 2,3,1. Fit full-recording seed 42, 10000 update; input neurale e label `(position, right, left)`. Provenienza in `processo/provenienza/b/<ratto>/`; codice fit `reproduction/figure2_bc/scripts/fit_time_position_direction_5d.py`, rendering `cebra_replica/rat_figura2b_time_position_direction.py`. Gli altri fit della tavola b restano disponibili solo per Achilles.

Il pannello d ha i plot per ratto in `plot/d/<ratto>/`; i calcoli sono in `processo/metriche/d/`. Gli output attualmente presenti usano 6 seed (42–47) e nei JSON gli input provengono da `rat_paper_fig1e_ed4_6seed_v1`, non dalla successiva `rat_paper_fig1e_ed4_10seed_v2`. Questo riordino non ricalcola i risultati.

Notebook ufficiale: https://cebra.ai/docs/cebra-figures/figures/Figure2.html

## Figura 2c — campagna trial-safe

Risultati, fit e diagnostica: `2c/` e `2c/plot/`.
