# Extended Data Figure 4 — risultati rigenerati

## Figure

- `plot/a/<Rat>/ED4_<RAT>_CONSISTENCY_ALL_METHODS.png` / `.pdf`: consistency across subjects e consistency across runs, tutti i metodi insieme, un file per ratto.
- `plot/b/<Rat>/ED4_<RAT>_DECODING_ALL_METHODS.png` / `.pdf`: decoding Position + Direction (R²) ed errore mediano di posizione (cm), tutti i metodi insieme, un file per ratto.
- Punti neri: singoli run/seed; linea arancione: mediana. Ogni run riassume i tre fold.

## Dati e valutazione

- CEBRA e Soft riusano gli stessi fit e seed della nuova Figura 1E: 4 ratti, 3 fold, 10 seed (42–51). Per la consistenza si riusano i punteggi salvati `train_cross_subject_r2` e `train_run_r2` (il campo chiamato `train_run_consistency` nel notebook CEBRA), aggregati prima tra i peer all'interno del fold e poi sui tre fold. Sono metriche sul supporto train, non sul test. Il notebook ED4 specifica che ogni run riassume tre split non sovrapposti; qui sono usati i tre fold della campagna locale.
- Le baseline conv-piVAE, UMAP e t-SNE riusano i punteggi storici salvati (6 seed); PCA è deterministica. I risultati disponibili sono mantenuti con il loro supporto originale.
- Il decoding è distinto dalla consistenza: riusa i punteggi esistenti valutati sul test held-out. kNN cosine seleziona k = 1, 4, 9, 16 o 25 sulla validation; R² congiunto per posizione e direzione, più errore mediano assoluto di posizione.
- Nessun encoder è stato riaddestrato e nessun decoding/kNN è stato ricalcolato. Sono stati aggregati e ripresentati i punteggi già salvati.
- `conv-piVAE q(z|x,u)`: decoding non disponibile nella tabella perché le etichette comportamentali sono usate anche all'inferenza. PCA: consistency across runs non applicabile, perché non ha run indipendenti.

## Tracciabilità

- Renderer: `cebra_replica/rat_render_ed4_all_methods.py`.
- Valori dei grafici: `processo/metriche/calcoli/ed4_plot_values_all_methods.csv`.
- Manifest e hash degli input/output: `processo/provenienza/ED4_ALL_METHODS_PROVENANCE.json`.
- Input CEBRA/Soft: `results/rat_paper_fig1e_ed4_10seed_v2/`.
- Input baseline: `results/rat_rebuild_20261027/Figure_1E_ED4/`.
- Riferimenti ufficiali: [CEBRA Extended Data Figure 4](https://cebra.ai/docs/cebra-figures/figures/ExtendedDataFigure4.html) e [CEBRA Figure 1](https://cebra.ai/docs/cebra-figures/figures/Figure1.html).
