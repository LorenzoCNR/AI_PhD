# Figura 2b–2c: audit e riproduzione

## Evidenza originale

Il file `Figure2_official.h5` è scaricato da https://github.com/AdaptiveMotorControlLab/cebra-figures/blob/main/data/Figure2.h5. I dati ufficiali sono stati ispezionati direttamente: `data/visualization/{hypothesis,shuffled,discovery,hybrid}` contengono ciascuno `embedding` di forma (10178, 3) e `label` di forma (10178, 3). La caption dichiara un fit 5D per Time + Position, quindi la matrice salvata è una vista troncata e non prova che il fit fosse 3D. I dati del primo pannello comportamentale non figurano tra queste quattro chiavi.

L'hash Git blob locale `e1b2b01cd31c919c3480055200e8a754417a3448` coincide con lo SHA del file remoto ottenuto dal connettore GitHub.

`data/hypothesis_testing/viz` contiene sei array di forma (10178, 8); le sei curve in `data/hypothesis_testing/loss` contengono 5000 campioni. Il notebook ufficiale mostra soltanto le prime tre coordinate: https://cebra.ai/docs/cebra-figures/figures/Figure2.html. Pertanto i fit 3D/10000 indicati nella richiesta sono un esperimento distinto, non identico al file grafico ufficiale. La forma 8D è accertata per gli array salvati, non dimostra da sola ogni parametro di training.

## Implementazione e condizioni

`scripts/run_cebra.py` usa CEBRA installato (0.6.1) e il dataset `rat-hippocampus-single-achilles`. Dataset completo, seed 42, niente split per queste visualizzazioni, nessun preprocessing arbitrario. Il file `dataset_audit.json` registra shape, dtype, range, NaN e checksum. I manifest e gli hash di ogni cella ne descrivono input, parametri effettivi, loss e checkpoint. I controlli smoke sono separati dal training completo. `smoke` contiene una cella 5D fallita prima della correzione della lettura della loss, conservata per tracciabilità; `smoke2` è completato.

Le due ricostruzioni del primo pannello usano rispettivamente input posizione e input posizione + direzione, entrambe con etichetta posizione, encoder `offset10-model-mse`, distanza euclidea e output 2D. La scelta dei canali e della dimensione non è confermata come originale.

Nei fit neurali della richiesta si usa output 3D, tranne Time + Position 5D; batch 512, 10000 update, cosine, learning rate 0.0003. Direction float segue sia i default del demo sia una condizione controllata esplicita. Shuffling permuta intere righe delle etichette e non gli spike. Per Time la temperatura è 1.12, per gli altri 1.

Nel sorgente CEBRA 0.6.1 effettivamente installato nell'ambiente `cebra`, il solver usa `num_behavior_features=3` e `output_mode=overlapping`: la loss posizione usa le prime tre coordinate, la loss temporale tutte e cinque. Questo verifica il funzionamento del solver attuale, non l'identità della versione storica con quella del paper.

Comando smoke: `python reproduction/figure2_bc/scripts/run_cebra.py --mode smoke2`. Comando completo: `python reproduction/figure2_bc/scripts/run_cebra.py --mode full`. Usare l'ambiente `cebra`.

## Campagne specifiche avviate successivamente

Le stime 2c CEBRA native 8D sono isolate in `figure2c_8d` e invocate da `scripts/run_cebra_2c_8d.py`; i fit usano 10000 update come richiesto, mentre la figura ufficiale contiene 5000 valori di loss. Le corrispondenti stime Soft 8D sono isolate in `soft_2c_8d` e invocate da `scripts/run_soft_2c_8d.py`, con condizioni Position, Direction, Position + Direction e rispettivi shuffle, CNN e Transformer, T embedding 1 e T metadata 0.5. Le condizioni Soft position e direction isolano le colonne corrispondenti nella distanza dei metadati, ponendo le altre a zero.

Le prove comportamentali Soft definitive sono isolate in `soft_behavior_2d_v2`: l'input Position + Direction ha due colonne (position, right), non tre. La precedente prova Soft è stata interrotta ed è conservata come parziale in `soft_behavior_2d`. Anche la prova CEBRA precedente su tre colonne è da considerarsi una diversa variante non equivalente alla prova richiesta su due colonne; quella corretta è prevista in `figure2b_missing`.

Gli script Soft derivano da `user_soft/full_recording_runner.py`, aggiornato per verificare e salvare la dimensione configurata invece di assumere sempre 3D. Le campagne riutilizzano i file originali, non riaddestrano quelli 1e/ED4, e lavorano su Achilles seed 42 come da specifica allegata. Le code attendono il processo predecessore e richiedono il numero previsto di manifest completati prima di avviare il successivo; non riavviano automaticamente job falliti.

Tracciabilità del processo Soft 8D: durante l'esecuzione è stato rafforzato solo il codice di audit finale nel file su disco, senza variare il codice di training. Il processo già avviato continua a eseguire la versione originaria, conservata esattamente come `scripts/run_soft_2c_8d_execution_snapshot.py` con SHA-256 `7dd66365b45fa02eac7be881ded3cf46806c725d37e2652993d94a3b33887655`. Alcuni manifest generati successivamente registrano l'hash del file modificato su disco, non quello del codice già caricato in memoria; lo snapshot è il riferimento corretto per il processo in corso.

## Esito della campagna

Le stime richieste per Achilles e seed 42 sono completate. L'audit indipendente `scripts/audit_completion.py` ha verificato 44 fit complessivi, inclusi i fit riutilizzati: forma degli embedding, valori finiti, 10000 aggiornamenti e hash di embedding e checkpoint. L'esito della coda è in `RESOCONTO_RUN.json`. Nessun fit della campagna 1e/ED4 è stato riaddestrato.

| Insieme | Conteggio e condizioni |
|---|---|
| CEBRA 2b | 8 fit: due prove comportamentali 2D, Time e Position 3D, Time + Position 5D, Position shuffled 3D, Position + Direction e relativo shuffle 3D. Tre riutilizzati, cinque nuovi. |
| CEBRA 2c | 8 fit 8D: sei condizioni e varianti controllate Direction / Direction shuffled. |
| Soft input comportamentale 2b | 4 fit 2D: due input per CNN e Transformer. |
| Soft 2c | 12 fit 8D: sei condizioni per CNN e Transformer. |
| Soft 2b aggiuntivi | 6 fit 3D: Position, Position shuffled, Time + Position + Direction shuffled, CNN e Transformer. |
| Soft 2b riutilizzati | 6 fit 3D della campagna full-recording con temperatura embedding 1: Time, Position + Direction, Time + Position + Direction, CNN e Transformer. |

Il controllo verifica integrità e configurazione ma non garantisce la replicazione numerica dei risultati originali. I fit sono sul recording completo e non sono una valutazione held-out. Per 2c sono stati eseguiti 10000 update secondo la richiesta, mentre le curve nel file ufficiale contengono 5000 valori. Le due ricostruzioni comportamentali 2D restano esplorative. Per gli shuffle Soft il tempo rimane invariato nel caso Time + Position + Direction shuffled; negli shuffle delle singole variabili è permutata la variabile indicata. La Figura 2d resta invariata e nessuna nuova figura è stata generata in questa fase.

## Correzione Time + Position + Direction 5D

Eseguiti quattro fit indipendenti, seed 42, 10000 update, uno per ciascun ratto. Input: spike dell'intera registrazione; label ausiliarie: position, right, left; `hybrid=True`, output 5D, `offset10-model`, 32 unità interne, batch 512, LR 0.0003, temperatura 1, cosine e `time_delta`. Checkpoint, embedding, loss, configurazione, hash e durata si trovano in `time_position_direction_5d/<rat>/seed_42/`. Le prime tre coordinate sono visualizzate nell'ordine 2,3,1. Il precedente fit solo Position non è stato cancellato ma non è più usato nella tavola CEBRA 2b di Achilles.
