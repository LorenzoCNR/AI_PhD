# Piano di esecuzione: Figure 1d, 1e, ED3, ED4

Data: 22 ottobre 2026. Stato: pre-training; nessun fit avviato.

Output proposto: `results/rat_paper_replication_2023_v1/`, con un sottodirettorio immutabile per `figure/method/rat/split/seed/config_hash`. Un runner deve rifiutare file preesistenti incompatibili, validare gli hash all'eventuale resume e scrivere per ogni fit parametri, seed, split, hash dati/codice, versioni dipendenze, checkpoint e embedding.

## Inventario dei fit

| Blocco | Configurazioni distinte sicuramente mancanti | Conteggio | Nota |
|---|---:|---:|---|
| ED3 CEBRA Behaviour + Time, 4 ratti, un'esemplare visualizzato | 2 × 4 | 8 | Entrambi 3D; Time 4.000 update/temp 2.25. |
| ED3 UMAP + t-SNE, 4 ratti | 2 × 4 | 8 | Fit manifold, non training neurale; parametri ED3 distinti. |
| ED3 conv-piVAE, 4 ratti, un'esemplare visualizzato | 1 × 4 | 4 | 1.000 epoche; un modello produce due tipi di inferenza. |
| 1d | indeterminato | — | Possibile riuso di ED3 solo dopo verifica completa di configurazione e campione. |
| 1e/ED4 | indeterminato | — | 10 run × 3 split potrebbero richiedere 30 fit per ratto e metodo; split e configurazioni non ancora comprovati. |

**Minimo certo: 20 nuovi fit distinti per ED3** (8 CEBRA, 8 manifold, 4 piVAE). Non è un totale dell'intero incarico. A titolo di dimensionamento, il disegno locale 4 ratti × 3 fold × 10 seed richiederebbe 120 fit per metodo: 240 fit per le due condizioni CEBRA e 120 per conv-piVAE, più 120 UMAP e 120 t-SNE se ciascuno richiede lo stesso disegno. Questi sono scenari, non fit approvati né conteggi esatti della replica pubblicata. Le sovrapposizioni tra figure non sono ancora quantificabili.

## Estensione richiesta: loss Soft in 3D

Il confronto della loss Soft richiede **nuovi fit nativi 3D**. Il runner CNN `user_soft/rat_runner.py` fissa oggi `output_dimension=32`; anche il piano Transformer a cinque seed è 32D. Gli embedding esistenti non sono riutilizzabili come fit 3D. Soft è un'estensione sperimentale, da etichettare come tale nelle figure accanto ai metodi pubblicati.

| Disegno Soft 3D | Calcolo | Nuovi fit |
|---|---:|---:|
| Solo visualizzazione ED3, CNN joint stable, un seed per ratto | 4 × 1 | 4 |
| CNN joint stable, 10 seed, 3 fold | 4 × 3 × 10 | 120 |
| CNN joint frozen + stable, 10 seed, 3 fold | 4 × 3 × 10 × 2 | 240 |
| Aggiunta di una seconda condizione Soft alle due versioni | 4 × 3 × 10 × 2 | +240 |
| Aggiunta di un'architettura Transformer con identico disegno | 4 × 3 × 10 × 2 | +240 per condizione |

Per la comparazione primaria propongo **CNN joint stable 3D, 120 fit**, che dà 10 run su tre fold e permette di derivare la visualizzazione ED3 da un fit qualificato. Se l'obiettivo è confrontare anche frozen contro stable, diventano **240**. La scelta della famiglia Soft e delle condizioni va congelata nel protocollo prima del training; nessuna cifra Soft va confusa con i fit richiesti per riprodurre il paper.

## Sequenza

1. Ottenere i file di risultati pubblicati e rintracciare il codice che li genera. Fissare versione software, configurazioni per figura, significato di split/repeat, normalizzazione/allineamento e formula di decoding. Registrare ogni decisione con fonte primaria.
2. Confrontare hash, parametri e supporto campionario di tutti gli artefatti locali; costruire una matrice `reusable / recompute / retrain` per ogni cella. Solo dopo congelare il numero esatto di fit.
3. Implementare runner isolato e riprendibile con `--dry-run`, manifest JSON e verifica degli output; testare su una configurazione piccola. Non modificare i benchmark esistenti.
4. Eseguire ED3, poi Figure 1d, 1e ed ED4 condividendo solo checkpoint dimostrabilmente equivalenti. Salvare metriche per split e run, figure PNG/PDF/SVG e confronto numerico col pubblicato.

Prima del fit verificare ambiente CEBRA/PyTorch/TensorFlow/openTSNE/UMAP, GPU e spazio; registrare tempo e memoria di un singolo fit qualificato per stimare il carico totale. Il tempo di training non può essere stimato onestamente dai soli manifest attuali perché hardware e durata dei fit piVAE/CEBRA paper-specific possono differire.

## Misura reale iniziale

Il primo fit isolato ED3 CEBRA-Behaviour, Achilles, seed 42, 3D, 10.000 update, full recording, ha completato in **73,08 s** sulla NVIDIA GeForce RTX 4070 Laptop GPU con il clone locale CEBRA 0.6.1. Checkpoint ed embedding hanno hash nel manifest sotto `results/rat_paper_replication_2023_v1/ED3/cebra/achilles/behavior/seed_42/`. Questa è una misura di una sola cella, non un tempo garantito per altri ratti, CEBRA-Time, piVAE o Soft. A parità di durata per fit, 120 fit richiederebbero circa **2,44 ore** di sola ottimizzazione; è solo una proiezione, senza export, scoring, overhead e variazioni di dimensione dei dati.
