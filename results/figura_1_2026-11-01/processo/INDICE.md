# Figura 1 — processo e sorgenti

## Dove sono gli output

- Plot: `../plot/c/`, `../plot/d/<ratto>/`, `../plot/e/`.
- Metriche e tabelle: `metriche/`.
- Manifest, specifiche e note: `provenienza/`.
- Embedding intermedi t-SNE: `embedding/d/<ratto>/tSNE_attempt_*/`.

## Fit riutilizzati

- Figura 1D ed ED3 condividono i fit full-recording: `results/rat_paper_replication_2023_v1/ED3/` e `results/user_soft_rat_fullrecording_t1_3conditions_seed42_v1/`.
- Figura 1E, ED4 e il decoding 2D condividono la campagna CEBRA/Soft in `results/rat_paper_fig1e_ed4_10seed_v2/`.
- Le baseline storiche e i relativi score sono in `results/rat_paper_fig1e_ed4_6seed_v1/`, `results/rat_manifold_baselines_v1/` e `results/rat_rebuild_20261027/`.
- Il tentativo piVAE a 10 seed, interrotto prima di completare qualsiasi fit, è conservato solo come audit in `provenienza/campagne_interrotte/rat_paper_fig1e_ed4_10seed_v1/`; non è una sorgente di risultati.

I fit condivisi restano nella loro unica cartella sorgente; questa pagina li collega senza duplicarli. Gli output dei pannelli e gli hash sono nei manifest di `provenienza/`.
