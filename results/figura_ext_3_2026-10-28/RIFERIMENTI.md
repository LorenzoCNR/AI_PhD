# Extended Data Figure 3 — riferimenti

I plot per ratto sono in `plot/a/Achilles`, `plot/b/Buddy`, `plot/c/Cicero`, `plot/d/Gatsby`; PNG/PDF restano sotto `plot/`. Manifest e hash sono in `processo/provenienza/<lettera>/<ratto>/`.

CEBRA, conv-piVAE, UMAP e Soft riutilizzano i fit full-recording seed 42 descritti per Figura 1D; t-SNE usa invece i fit full-recording 10-bin completati il 29 ottobre. Gli embedding CEBRA e Soft sono nativamente 3D; nella vista ED3 si plottano le coordinate 1 e 2. PiVAE, UMAP e t-SNE sono 2D. Le barre posizione sono a destra: Left e Right. Non sono stati rifittati i modelli già esistenti per creare ED3.

- Renderer: `cebra_replica/rat_render_extended_data_3_4_20261028.py`; lettura e validazione degli embedding: `cebra_replica/rat_rebuild_1d_ed3_20261027.py` e `cebra_replica/rat_render_figura1_saved.py`.
- CEBRA: `results/rat_paper_replication_2023_v1/ED3/cebra/<rat>/{behavior,time}/seed_42/`.
- conv-piVAE, stesso fit con due uscite: `results/rat_paper_replication_2023_v1/ED3/pivae_full_recording/<rat>/seed_42/`.
- Soft Stable: `results/user_soft_rat_fullrecording_t1_3conditions_seed42_v1/<rat>/<condition>/<architecture>/seed_42/`.
- Codice ufficiale ED3: https://cebra.ai/docs/cebra-figures/figures/ExtendedDataFigure3.html
- t-SNE: `results/figura_1_2026-11-01/processo/embedding/d/<rat>/tSNE_attempt_*/`; fit full-recording su finestre da 10 bin, output 2D, cosine, perplexity 30, PCA init, learning rate auto, early exaggeration 12 per 250 iterazioni, 1.000 iterazioni, seed 42; nessuna label comportamentale nel fit. Hash e configurazione nei `PROVENANCE.json` per ratto.
