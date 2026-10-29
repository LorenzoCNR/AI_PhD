# Figure 1d / Extended Data Figure 3 — report di replica

Aggiornato: 23 ottobre 2026. Questo report descrive la campagna isolata in `results/rat_paper_replication_2023_v1/ED3` e le figure nuove in `results/rat_paper_fig1d_ed3_v1`. Non certifica come completati i fit UMAP/t-SNE finché non esistono embedding e manifest verificati.

## Decisioni e geometria dei pannelli

| Pannello | Fit | Output del fit | Coordinate disegnate | Supporto usato nel grafico | Scala |
|---|---|---:|---|---|---|
| CEBRA-Behaviour | `offset10-model`, supervisione posizione + Left/Right | 3D nativo | tutte e 3 | full recording, righe CEBRA salvate | output CEBRA normalizzato L2 |
| conv-piVAE con labels | un fit conv-piVAE; posterior `q(z|x,u)` in inferenza | 2D nativo | entrambe | solo le finestre ammesse da `valid_mask`, allineate con `source_id` | coordinate posterior raw |
| conv-piVAE senza labels | lo stesso fit piVAE; posterior `q(z|x)` in inferenza | 2D nativo | entrambe | stessa maschera valida, allineata con `source_id` | coordinate posterior raw |
| CEBRA-Time | `offset10-model`, obiettivo temporale | 3D nativo | tutte e 3 | full recording, righe CEBRA salvate | output CEBRA normalizzato L2 |
| t-SNE | non supervisionato su finestre neurali; cosine, perplexity 29 | 2D nativo | entrambe | centri di tutte le finestre complete (T−9) | coordinate t-SNE raw |
| UMAP | non supervisionato su finestre neurali; cosine, 31 vicini, `min_dist=0.99` | 2D nativo | entrambe | centri di tutte le finestre complete (T−9) | coordinate UMAP raw |

Figure 1d è prodotta per Achilles. ED3 è prodotta separatamente per tutti e quattro i ratti. Per Figure 1d riuso i parametri ED3: è la decisione operativa già fissata, quindi la figura non viene descritta come recupero degli iperparametri eventualmente ottimizzati per Figure 1d.

La tavola Soft è un’estensione separata, non uno dei pannelli del paper: usa i fit Soft esistenti CNN/Transformer per `Position + Direction` e `Time + Behavior`, con tutte e tre le coordinate dei fit nativi 3D. `Position + Direction` è l’etichetta leggibile del caso Soft behavior-only. Le altre due condizioni Soft (time-only e shuffled) restano nei risultati e non sono aggiunte alla tavola principale.

## Parametri di fit fissati

- CEBRA-Behaviour: `output_dimension=3`, batch 512, learning rate `3e-4`, `time_offsets=10`, cosine, temperatura 1, 10.000 update, `conditional=time_delta`.
- CEBRA-Time: `output_dimension=3`, batch 512, learning rate `3e-4`, `time_offsets=10`, cosine, temperatura 2.25, 4.000 update, `conditional=time`.
- conv-piVAE: 2D, 1.000 epoche, learning rate `2.5e-4`. Sono salvati separatamente il posterior raw label-free e quello behavior-informed; quest’ultimo usa le label all’inferenza e non è intercambiabile con il primo.
- t-SNE locale: openTSNE, 2D, cosine, perplexity 29, PCA init, 1.000 iterazioni, learning rate auto, early exaggeration 12 per 250 iterazioni, seed 42, `n_jobs=1`. Il manifest salva parametri effettivi e versione.
- UMAP locale: 2D, cosine, `n_neighbors=31`, `min_dist=0.99`, seed e transform seed 42, `n_jobs=1`. Il manifest salva parametri effettivi e versione.

Gli output Soft non vengono rifittati: vengono letti da `results/user_soft_rat_fullrecording_stable3d_v1`, usando l’array unit-norm `embedding_unit` e la maschera/ID campione salvati con ogni fit.

## Dati, finestre, allineamento

I file ratto contengono spike a bin di 25 ms. I Methods descrivono segmenti di 250 ms con overlap di 225 ms per CEBRA, UMAP, t-SNE e piVAE. Per UMAP/t-SNE il runner crea finestre contigue da 10 bin a stride 1, le appiattisce in ordine channel-major e associa la label al centro tramite `source_id`.

| Ratto | T righe | Neuroni | Righe fit manifold T−9 | Feature manifold 10×N | Righe Soft valide nel plot |
|---|---:|---:|---:|---:|---:|
| Achilles | 10.178 | 120 | 10.169 | 1.200 | 9.800 |
| Buddy | 6.577 | 48 | 6.568 | 480 | 6.316 |
| Cicero | 47.431 | 55 | 47.422 | 550 | 47.017 |
| Gatsby | 17.026 | 66 | 17.017 | 660 | 16.648 |

CEBRA, piVAE e Soft sono fittati sull’intero recording, non su un test split. PiVAE e Soft conservano tutte le righe fit ma la figura applica `valid_mask` per escludere le finestre che attraversano un confine di trial. CEBRA usa le righe prodotte dal loader, che gestisce i bordi con il proprio clamping. UMAP/t-SNE fittano e plottano i T−9 centri con finestra completa. Questi grafici non sono valutazioni held-out.

La fonte pubblica specifica la finestra temporale, ma non documenta la precisa trasformazione in matrice campioni×feature per UMAP/t-SNE. Il repository ufficiale `cebra-figures` dichiara di distribuire plotting su risultati precalcolati, non la pipeline completa dei fit. Pertanto il `window10_flat` locale è una scelta esplicita, coerente con i Methods e riproducibile, ma non va attribuita come codice originale degli autori. Anche il trattamento esatto dei bordi UMAP/t-SNE non è pubblicato; qui si usano solo finestre complete.

## Stato artefatti al 23 ottobre 2026

- CEBRA ED3: 8/8 fit (quattro ratti × Behaviour/Time), array `(T,3)`, finiti e unit-norm; checkpoint e manifest presenti.
- conv-piVAE: 4/4 fit completi, due output `(T,2)` raw per ratto, finiti; maschere valide archiviate.
- Soft: 32/32 fit completi, quattro condizioni × CNN/Transformer × quattro ratti, output `(T,3)` finiti e unit-norm. Le gallerie Soft 3D PNG/PDF/SVG con JSON di provenienza sono già state generate nella sottocartella `Soft_3D_extension`.
- UMAP/t-SNE: i job Buddy, Cicero e Gatsby sono stati avviati con il runner resumable. Al controllo non risultavano ancora embedding/manifest finali; le celle hanno `RUNNING.json`. Per Achilles esiste inoltre un processo preesistente che non è stato attribuito con certezza a metodo/ratto: non è stato terminato né duplicato. Lo stato finale va aggiornato usando i manifest, non i soli PID.
- Di conseguenza, al momento di questo aggiornamento le tavole canoniche Figure 1d/ED3 non sono ancora complete; il generatore le blocca se manca anche un solo metodo richiesto, così un pannello parziale non può sembrare la figura completa.

## Codice e destinazioni

- Fit CEBRA: `cebra_replica/rat_paper_ed3_cebra.py`
- Fit piVAE: `cebra_replica/rat_paper_ed3_pivae_full.py`
- Fit UMAP/t-SNE resumable: `cebra_replica/rat_paper_ed3_manifold_resumable.py`
- Rendering e metadati: `cebra_replica/rat_paper_fig1d_ed3.py`
- Audit config e piano: `cebra_replica/FIG1D_ED3_CONFIGURATION_AUDIT.md`, `cebra_replica/FIG1D_ED3_EXECUTION_PLAN.md`
- Output canonici: `results/rat_paper_fig1d_ed3_v1/`
- Output Soft separati: `results/rat_paper_fig1d_ed3_v1/Soft_3D_extension/`
- Gli output storici in `results/RAT_FIGURES` non sono stati cancellati o sovrascritti.

## Riferimenti

- [Schneider, Lee & Mathis (2023), paper e caption Figure 1d / Extended Data Figure 3](https://www.nature.com/articles/s41586-023-06031-6)
- [Methods dell’articolo su PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10172131/)
- [README ufficiale del repository cebra-figures](https://github.com/AdaptiveMotorControlLab/cebra-figures/blob/main/README.md)
