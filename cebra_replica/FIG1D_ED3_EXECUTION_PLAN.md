# Piano operativo — Figure 1d ed Extended Data Figure 3

Aggiornato: 23 ottobre 2026. L'audit di configurazione e degli output correnti è in `FIG1D_ED3_CONFIGURATION_AUDIT.md`.

## Obiettivo congelato

Riprodurre la tavola canonica di Figure 1d per Achilles e le tavole ED3 per i quattro ratti, con configurazioni ED3 anche per il pannello Figure 1d. Non eseguire autoLFADS, sweep di iperparametri o nuovi fit CEBRA/piVAE/Soft già disponibili. Aggiungere la galleria Soft solo come estensione separata, dichiarando che Soft non appartiene al paper.

La tavola canonica ha sei metodi: CEBRA-Behaviour (3D), conv-piVAE con e senza comportamento nell'inferenza (2D), CEBRA-Time (3D), t-SNE (2D), UMAP (2D). L'output Soft (2 CNN + 2 Transformer, per condizioni principali) non va confuso con quei sei metodi.

## Sequenza

1. **Non interrompere i processi esistenti.** Controllare periodicamente PID 52572/102924 e l'albero `ED3/manifold`. Associare ogni fit solo quando compare un manifest valido con ratto, metodo, seed, configurazione, supporto, versione e hash. Se il processo non termina, non lanciarne un duplicato sullo stesso ratto/metodo.
2. **Riusare i fit con compatibilità verificata:** CEBRA 8/8, conv-piVAE 4/4 e Soft 32/32. Nessuna copia dei checkpoint è necessaria: le figure devono puntare ai file sorgente e conservarne hash e manifest.
3. **Completare i manifold mancanti** con `rat_paper_ed3_manifold_resumable.py`, seed 42, full recording e input `window10_flat`. Le celle già complete vengono validate e riusate. Buddy, Cicero e Gatsby sono stati avviati in celle isolate con un processo per metodo/ratto (`n_jobs=1` dentro ciascun fit); verificare periodicamente stato e manifest. Achilles resta escluso da nuovi lanci finché non è chiarito il processo preesistente. Ogni output è commit atomico e ha un manifest autonomo.
4. **Generare pannelli dalle stime congelate**, non rifittare: sei pannelli canonici per Achilles (Figure 1d), poi gli stessi sei per ogni ratto ED3. Associare posizione e direzione per `source_id`; usare le maschere valide salvate per PiVAE/Soft; non interpolare e non ridurre dimensioni.
5. **Produrre una galleria Soft separata** dai checkpoint 3D già esistenti, senza farla passare per un pannello originale Nature. Per il confronto di base includere Soft behavior-only (Position + Direction) e Soft joint (Time + Behavior), CNN e Transformer; altre condizioni (time-only e shuffle) restano disponibili nei manifest e non vanno aggiunte implicitamente alla figura principale.
6. **Validare e consegnare** PNG, PDF e SVG, indice dei pannelli e metadati JSON per campioni fit/plot, dimensioni, coordinate, normalizzazione, label, iperparametri, versioni, codice sorgente e hash dei parent.

## Celle ancora da completare

Per ciascuno dei quattro ratti esiste una cella UMAP e una t-SNE (8 totali). Alla fotografia iniziale dell'audit nessuna aveva un embedding con manifest. Al successivo controllo risultavano avviate le sei celle Buddy/Cicero/Gatsby, ancora senza embedding/manifest finali; Achilles ha un processo preesistente non associabile con certezza e non viene duplicato. Il conteggio finale sarà ridotto soltanto da manifest completi e compatibili.

| Metodo | Dimensione | Configurazione pubblicata / fissata | Ulteriori impostazioni da registrare |
|---|---:|---|---|
| UMAP | 2 | cosine, 31 vicini, `min_dist=0.99` | `umap-learn 0.5.7`; seed e transform seed 42; `n_jobs=1`; restanti parametri esplicitati dal `get_params()` del modello |
| openTSNE | 2 | cosine, perplexity 29 | `openTSNE 1.0.2`; PCA init, 1.000 iterazioni; seed 42, LR auto, exaggeration iniziale 12 per 250 iterazioni, `n_jobs=1`; restanti parametri esplicitati dal `get_params()` |

Nota: i Methods descrivono early exaggeration 12 per gli altri dataset ED3 con perplexity 30; la caption specifica ED3 riporta perplexity 29. Si usa 29, dato della caption, e 12, valore dei Methods. Con la versione installata `openTSNE 1.0.2`, `early_exaggeration="auto"` si risolve a 12 quando `exaggeration=None`; il runner lo imposta esplicitamente a 12 e registra il parametro effettivo.

## Output e protezioni

- Fit già congelati: `results/rat_paper_replication_2023_v1/ED3/{cebra,pivae_full_recording}` e `results/user_soft_rat_fullrecording_stable3d_v1`.
- Manifold: `results/rat_paper_replication_2023_v1/ED3/manifold/{umap,tsne}/<rat>/window10_flat/seed_42/`.
- Figure nuove: `results/rat_paper_fig1d_ed3_v1/` (separato da `results/RAT_FIGURES` e dagli output di figure precedenti).
- Non cancellare né sostituire checkpoint o figure esistenti. Se una directory contiene artefatti senza manifest/hash coerenti, fermare quella sola cella e conservarla per ispezione.
- Il resume è per ratto/metodo: un output completo e hash-verificato viene riusato; un fit UMAP/t-SNE interrotto prima del commit viene rifatto da zero per quella singola cella, senza perdere le altre.

## Criteri di completamento

- Tutte le 8 celle CEBRA sono native 3D, finite e validabili da manifest.
- Tutte le 4 celle piVAE forniscono entrambi i posteriori 2D con semantica documentata.
- Tutte le 8 celle manifold sono 2D, finite, allineate alle label con `source_id` e salvate atomicamente.
- Tutte le 32 celle Soft riusate sono native 3D; per la galleria principale Soft sono usate solo le condizioni dichiarate al punto 5.
- Figure 1d ed ED3 sono separate dalla galleria Soft, senza cambiare gli output storici.
- Il report distingue repliche di configurazione, decisione ED3-riusata per Fig. 1d, inferenze locali di preprocessing e dettagli non pubblicati.
