# Figura 2 — processo e sorgenti

## Dove sono gli output

- Plot: `../plot/b/<ratto>/`, `../plot/c/<ratto>/`, `../plot/d/<ratto>/`.
- Metriche e calcoli del decoding 2D: `metriche/d/` (tabella per fold e record individuali in `calcoli/`).
- Loss numeriche 2C: `metriche/c/`.
- Manifest e riferimenti per pannello/ratto: `provenienza/`.

## Fit riutilizzati

- I fit dei pannelli 2B e 2C sono tracciati in `reproduction/figure2_bc/`; la campagna diagnostica di 2C è in `results/figure2c_shuffle_diagnostic_2026-11-02/`.
- Nota importante: i risultati 2D attualmente archiviati (`metriche/d/` e i plot in `plot/d/`) hanno 804 record, sei seed 42–47, e i JSON registrano input da `results/rat_paper_fig1e_ed4_6seed_v1/`. Non sono ancora la versione aggiornata basata sulla campagna `rat_paper_fig1e_ed4_10seed_v2/`; la cartella qui è solo riordinata, non ricalcolata.

I fit condivisi non sono copiati. Le note storiche rimaste in `provenienza/legacy/` sono archivio di provenienza, non output correnti.
