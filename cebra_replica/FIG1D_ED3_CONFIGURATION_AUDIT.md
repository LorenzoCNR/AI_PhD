# Audit configurazione — Figure 1d ed Extended Data Figure 3

Audit aggiornato: 23 ottobre 2026. Questo documento sostituisce come stato corrente le note preliminari del 22 ottobre; non modifica né promuove i vecchi output in `results/RAT_FIGURES`.

## Decisioni congelate

- Usare le configurazioni pubblicate per ED3 anche per Figure 1d. È la scelta già approvata: Figure 1d non viene presentata come replica esatta dei suoi iperparametri eventualmente ottimizzati, ma come confronto che riusa i parametri ED3.
- La geometria del paper è **CEBRA nativo 3D** e **UMAP, t-SNE e conv-piVAE nativi 2D**. Per Figure 1d l'articolo dice esplicitamente che questa è la dimensione minima equivalente scelta per il confronto; ED3 ripete 3D per CEBRA e 2D per gli altri metodi.
- I fit Soft dell'utente sono un'estensione distinta, non un metodo del paper. I fit Soft disponibili sono già nativi 3D e normalizzati; non vanno ridotti dai 32D né inseriti nella tavola canonica senza etichettarli come estensione.
- autoLFADS è escluso, come richiesto.

## Specifiche per metodo e pannello

| Metodo/pannello | Fit e supervisione | Output del fit | Coordinate visualizzate | Supporto e stato locale |
|---|---|---:|---|---|
| CEBRA-Behaviour | Neural window, `offset10-model`; comportamento continuo posizione + indicatori Left/Right; ED3: temperatura 1, offset temporale 10, batch 512, LR `3e-4`, 10.000 update | **3D nativo** | tutte e 3 le coordinate; nessuna proiezione da 32D | Full recording, 1 fit per ratto; 4/4 ratti completi. Gli array sono finiti e con norma L2 ≈ 1. |
| conv-piVAE senza label in inferenza | Neural window; training conv-piVAE; rappresentazione `q(z|x)` | **2D nativo** | entrambe le coordinate raw | 1 fit per ratto, 1.000 epoche, LR `2.5e-4`; 4/4 completi. Nessuna normalizzazione a sfera. |
| conv-piVAE con label in inferenza | Stesso fit piVAE; posterior behavior-informed `q(z|x,u)` | **2D nativo** | entrambe le coordinate raw | Usa comportamento all'inferenza; non equivale al pannello label-free. Stessi 4 fit piVAE. |
| CEBRA-Time | Neural window; time-only; ED3: temperatura 2.25, offset temporale 10, batch 512, LR `3e-4`, 4.000 update | **3D nativo** | tutte e 3 le coordinate | Full recording, 1 fit per ratto; 4/4 completi. Array finiti e norma L2 ≈ 1. |
| t-SNE (openTSNE) | Unsupervised sui segmenti neurali; cosine, perplexity 29, PCA init, 1.000 iterazioni; seed 42. Early exaggeration 12 e 250 iterazioni iniziali come nei Methods/API; LR `auto`, `n_jobs=1` | **2D nativo** | entrambe le coordinate | Achilles ha directory inizializzata ma nessun embedding/manifest completato; altri ratti da eseguire o verificare. |
| UMAP | Unsupervised sui segmenti neurali; cosine, `n_neighbors=31`, `min_dist=0.99`; seed/transform seed 42, `n_jobs=1` | **2D nativo** | entrambe le coordinate | Achilles ha directory inizializzata ma nessun embedding/manifest completato; altri ratti da eseguire o verificare. |
| Soft CNN/Transformer (estensione) | `soft_v1_stable`, finestra 10, seed 42, fit full-recording; condizioni behavior-only, joint, time-only e shuffle | **3D nativo** | tutte e 3 le coordinate | 32/32 fit completi. Array `(N,3)`, finiti, norma L2 ≈ 1. Non è una configurazione del paper. |

## Dati, finestre e allineamento

Il dataset locale ha attività neurale `X=(T,N_neuroni)` a bin di 25 ms. Conteggi verificati nei file locali:

| Ratto | Righe `T` | Neuroni | Righe UMAP/t-SNE per la finestra locale `T−9` | Feature per finestra appiattita |
|---|---:|---:|---:|---:|
| Achilles | 10.178 | 120 | 10.169 | 1.200 |
| Buddy | 6.577 | 48 | 6.568 | 480 |
| Cicero | 47.431 | 55 | 47.422 | 550 |
| Gatsby | 17.026 | 66 | 17.017 | 660 |

I Methods descrivono segmenti ratto lunghi 250 ms con 225 ms di sovrapposizione per CEBRA, UMAP, t-SNE e piVAE. Dieci bin consecutivi a passo di un bin sono quindi il supporto temporale. Il loader CEBRA locale restituisce `(campioni, neuroni, 10)` e gestisce gli estremi del recording clampando gli indici; non appiattisce l'input per il fit CEBRA.

**Limite importante per UMAP/t-SNE:** il repository ufficiale `cebra-figures` dichiara di contenere plotting e risultati precalcolati, non il codice completo dei fit. I Methods pubblicano la finestra da 250 ms ma non la riga esatta di reshape per questi algoritmi. L'implementazione locale usa quindi un campione per centro, trasforma `(10, N_neuroni)` in `10*N_neuroni` feature in ordine channel-major e associa le label al bin centrale (`source_id`). Questa è una scelta esplicita e riproducibile coerente con la finestra pubblicata, non un dettaglio di codice attribuibile agli autori. La regola locale per gli estremi produce `T−9` finestre; l'esatta gestione autoriale degli estremi non è pubblicata.

I fit CEBRA, piVAE e Soft coprono tutte le righe del recording. PiVAE e Soft archiviano anche `valid_mask`: per i plot va rispettata la maschera delle finestre che non attraversano un confine di trial, e ogni metodo va associato alle label tramite i propri `source_id`. CEBRA archivia invece tutte le `T` righe, inclusi i centri gestiti dal clamp del loader. UMAP/t-SNE archiviano solo i `T−9` centri con finestra neurale completa. In nessun caso questi embedding sono fit held-out; il test split degli indici grafici storici non descrive questa nuova campagna.

## Audit degli output esistenti

| Campagna | Verifica al 23-10-2026 | Riuso |
|---|---|---|
| `results/rat_paper_replication_2023_v1/ED3/cebra` | 8/8 fit (4 ratti × Behaviour/Time); array `(T,3)`, finiti, unit-norm; seed 42; manifest/checkpoint presenti | Sì per ED3 e per Figure 1d con la decisione di condividere la configurazione ED3 |
| `results/rat_paper_replication_2023_v1/ED3/pivae_full_recording` | Campaign state 4/4 complete; array `(T,2)` finiti; `z_mean_raw=q(z|x)` e `post_mean_behavior_informed_raw=q(z|x,u)` distinti | Sì; usare la maschera valida per la visualizzazione e dichiarare il posterior usato |
| `results/user_soft_rat_fullrecording_stable3d_v1` | 32/32 complete; 4 ratti × 4 condizioni × 2 architetture; ogni array `(T,3)`, finito e unit-norm; valid rows: 9.800, 6.316, 47.017, 16.648 per ratto | Sì per una tavola Soft separata; nessun nuovo fit Soft richiesto |
| `results/rat_paper_replication_2023_v1/ED3/manifold` | Nessun embedding/manifest completato al controllo successivo. Sei fit resumable sono in esecuzione: UMAP e t-SNE per Buddy, Cicero e Gatsby, ciascuno con `RUNNING.json`. Per Achilles resta un processo preesistente non attribuibile con certezza; non è stato terminato né duplicato. | Nessun manifold ancora validato; creare le tavole solo dopo manifest completi |

È presente PID 52572 (`rat_manifold_v1\python.exe`, avvio 22-10-2026 12:15, CPU cumulata in aumento) e PID 102924 (`python.exe`, stesso avvio ma CPU pressoché ferma). La sandbox nega la lettura della command line WMI; non assegno quindi con certezza metodo/ratto ai PID. Non sono stati terminati. Il report finale deve aggiornare questo stato e associare i fit tramite manifest, non dedurli dal solo PID.

Gli indici/plot storici in `results/RAT_FIGURES` appartengono a una versione precedente (split held-out e/o fit CEBRA 32D). Non usarli come output di questa replica e non sovrascriverli.

## Fonti

- Schneider, Lee & Mathis (2023), paper e caption Figure 1d / Extended Data Figure 3: https://www.nature.com/articles/s41586-023-06031-6
- Testo dell'articolo indicizzato con i Methods rat e t-SNE: https://pmc.ncbi.nlm.nih.gov/articles/PMC10172131/
- README ufficiale `cebra-figures`, che specifica che il repository distribuisce plotting su risultati precalcolati: https://github.com/AdaptiveMotorControlLab/cebra-figures/blob/main/README.md
- Loader locale CEBRA: `CEBRA/cebra/datasets/hippocampus.py`; espansione delle finestre: `CEBRA/cebra/data/base.py`.
- Manifest locali e script di fit sono i registri primari per l'esecuzione qui presente; i dati numerici sopra sono stati verificati direttamente sugli array salvati il 23-10-2026.
