# Audit preliminare: Figure 1d, 1e, ED3, ED4

Data: 22 ottobre 2026. Stato: audit prima del training. Nessun risultato locale è promosso a replica esatta.

## Fonti e limite delle fonti

- [Notebook ufficiale Figure 1](https://cebra.ai/docs/cebra-figures/figures/Figure1.html): Figure 1d usa 3D CEBRA e 2D per gli altri metodi; Figure 1e legge il campo `train` di `results_v3` e media le matrici dirette su 10 run.
- [Notebook ufficiale ED3](https://cebra.ai/docs/cebra-figures/figures/ExtendedDataFigure3.html): legge `EDFigure3.h5` e `EDFigure3_addition.h5`, dispone 7 metodi × 4 ratti e colora le due direzioni separatamente.
- [Notebook ufficiale ED4](https://cebra.ai/docs/cebra-figures/figures/ExtendedDataFigure4.html): legge `results_v4`; usa `train`, `train_run_consistency`, `test_total_r2`, `test_position_error`; media tre valori per run e mostra 10 run e mediana.
- [README cebra-figures](https://github.com/AdaptiveMotorControlLab/cebra-figures/blob/main/README.md): questo repository distribuisce plotting e risultati precalcolati, non il processo completo di fitting. La provenienza sperimentale dei campi non è ricostruibile dal solo notebook.
- [Codice CEBRA locale](../CEBRA/cebra/datasets/hippocampus.py), revisione `b7e4b9e`, e [runner locale](rat_benchmark.py).

## Classificazione per figura e metodo

| Figura | Metodo | Stato | Differenza verificata / azione |
|---|---|---|---|
| 1d | CEBRA Behaviour, Time | REQUIRES RETRAINING | I fit principali locali sono 32D, 10.000 update, offset10, batch 512, LR 3e-4; Time usa temperatura 1.12. Richiesti fit nativi 3D e verifica dei parametri specifici della figura. |
| 1d | UMAP, t-SNE | UNRESOLVED — PARTI DELLA FONTE VERIFICATE | Methods: input neurale a segmenti 250 ms/225 ms overlap anche per UMAP/t-SNE; fit locale su train fold trial e parametri 24/0.0001 e perplexity 30 non equivalenti. Il codice originale di fitting/reshape non è nel repository pubblico delle figure. |
| 1d | conv-piVAE | REQUIRES RETRAINING | Locale 2D, finestra 10, LR 2.5e-4, 600 epoche su fold trial; selezione e campione originali non verificati. Entrambe le inferenze q(z|x), q(z|x,u) sono disponibili localmente. |
| 1e | tutti | UNRESOLVED — SOURCE VERIFICATION REQUIRED | Il notebook usa `train`; il locale usa test fold, allineamento sulla sola posizione con 100 bin e per CEBRA 32D. Occorrono allineamento posizione × direzione, regessione diretta con intercetta e 10 run comparabili. |
| ED3 | CEBRA Behaviour, Time | REQUIRES RETRAINING | Caption: Behaviour 3D/temp 1/10.000 update; Time 3D/temp 2.25/4.000 update; entrambi offset10 e batch 512. Locali 32D e Time temp 1.12/10.000. |
| ED3 | UMAP, t-SNE | REQUIRES RETRAINING | Methods verificati: segmenti 250 ms/225 ms overlap sul continuo; 2D unsupervised. Caption: UMAP cosine 31/0.99, t-SNE cosine perplexity 29. I fit locali 24/0.0001 e perplexity 30 non sono equivalenti. L'esatta riga di reshape non è pubblicata; decisione e limiti documentati in [audit input UMAP/t-SNE](UMAP_TSNE_INPUT_PROVENANCE.md). |
| ED3 | conv-piVAE | REQUIRES RETRAINING | Caption: 2D, 1.000 epoche, LR 2.5e-4. Locali 600 epoche. |
| ED4 | tutti | UNRESOLVED — SOURCE VERIFICATION REQUIRED | Le metriche locali sono held-out analoghe, non i campi pubblicati. Non è dimostrato che i fold trial locali coincidano con i tre split originali. |

AutoLFADS escluso su richiesta. PCA e metodi Soft locali non appartengono alla replica richiesta.

## Dettagli metodologici locali verificati

| Aspetto | Stato locale | Stato originale / verifica aperta |
|---|---|---|
| Dati | Quattro file `.jl`: Achilles, Buddy, Cicero, Gatsby; bin neurali 25 ms; posizione e due indicatori di direzione. | Confermare hash e versione contro dati sorgente dei risultati pubblicati. |
| Split | `SingleRatTrialSplitDataset`, tre fold trial con train/valid/test disgiunti. | Il notebook ED4 non documenta la generazione dei tre split. |
| CEBRA | `offset10-model`, finestra 10, cosine, dimensione 32, LR 3e-4, batch 512, 10.000 update; temp 1 Behaviour / 1.12 Time; seed 42–51. | Dimensione nativa 3 per le figure qui richieste; architettura nascosta, LR, campionamento e altri parametri da confermare alla versione paper. |
| UMAP | 2D, cosine, 24 vicini, min_dist 0.0001, seed; baseline locale usa finestra 10 appiattita su train fold. | Methods paper supporta segmenti da 10 bin per UMAP/t-SNE; ED3 usa 31/0.99. Il repo pubblico delle figure non espone il fitting originale né la riga di reshape; per questa replica si usa esplicitamente finestra 10 appiattita, mantenendo il controllo single-bin separato. |
| t-SNE | openTSNE 2D, cosine, perplexity 30, PCA init, 1.000 iterazioni, learning rate auto; baseline locale su finestra 10 appiattita. | Methods paper supporta segmenti da 10 bin; ED3 perplexity 29. Verificare early exaggeration e versione software. |
| conv-piVAE | Sorgente modificata in `CEBRA/third_party/pivae`; finestra 10, 2D, 600 epoche, LR 2.5e-4, batch 200, miglior checkpoint validation. | ED3 1.000 epoche; confermare architettura e regola di selezione per 1d/1e/ED4. |
| Consistency | `consistency_score`, 100 bin sulla posizione, embedding test, direzioni non separate. | Posizione × direzione, normalizzazione CEBRA, riempimento bin vuoti, mapping lineare diretto con intercetta; campo `train` pubblicato. |
| Decoding | Posizione R², errore posizione e direction balanced accuracy separati. | `test_total_r2` è un campo distinto; formula e decoder originari da verificare. |
| Visualizzazione | Figure locali spesso fold 0, seed 42, split test; ED3 locale include metodi Soft. | Figure 1d/ED3 usano dati precomputati originali; corrispondenza del campione locale non dimostrata. |

## Artefatti esistenti e riuso

La campagna CEBRA locale ha 840/840 fit completati, ma i fit canonici sono 32D. Le 252 baseline PCA/UMAP/t-SNE sono su fold trial. I 120 fit conv-piVAE sono a 600 epoche. Questi sono riutilizzabili per controlli e confronti, non come checkpoint ED3 equivalenti. Dataset, loader, funzioni di plotting, hash e manifest restano utili. Le metriche 1e/ED4 e le figure possono essere ricalcolate senza fit **solo dopo** che gli embedding configurati correttamente sono disponibili.

## Questioni non risolte

1. Generazione di `results_v3`/`results_v4`, significato di `repeat` e dei tre split, e provenienza dei campi `train`/`train_run_consistency`.
2. Formula esatta di `test_total_r2`, decoder, supporto di evaluation e selezione iperparametri.
3. Parametri Figure 1d e 1e per ogni metodo, dati full recording o sottoinsieme, e possibilità di condividere fit con ED3/ED4.
4. Seme e configurazione esatti degli embedding HDF5 pubblicati; presenza di checkpoint originali riusabili.

La verifica del 22 ottobre nel clone locale CEBRA e negli script `cebra_replica` non ha trovato una routine che generi `results_v3`, `results_v4`, `train_run_consistency` o `test_total_r2`. Le pagine ufficiali consultate mostrano il caricamento e la selezione di questi campi, quindi attestano **cosa viene disegnato**, non come i valori sono stati ottenuti. Il testo «tre split test non sovrapposti» resta in tensione con l'uso di `train` nei pannelli di consistenza; non è stato reinterpretato come split trial locali.

Finché questi punti restano aperti, l'etichetta **EXACT MATCH** non è attribuita a nessuna cella.
