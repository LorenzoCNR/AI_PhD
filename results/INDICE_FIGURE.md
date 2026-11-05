# Figure — indice unico

Ogni consegna ha una sola cartella. Dentro ciascuna:

- `plot/<lettera>/<ratto>/` contiene solo figure statiche o interattive.
- `processo/metriche/` contiene CSV, calcoli e risultati numerici.
- `processo/embedding/` contiene embedding intermedi e input derivati.
- `processo/provenienza/` contiene manifest, hash, specifiche e note dei pannelli.
- `processo/codice/` contiene script consegnati insieme a un pannello.

I fit e checkpoint riutilizzati da più figure restano una sola volta nelle cartelle di campagna già esistenti sotto `results/`. Gli indici di ogni figura indicano quali cartelle sono condivise: non sono state create copie dei fit.

| Consegna | Cartella corrente | Pannelli |
|---|---|---|
| Figura 1 | `figura_1_2026-11-01/` | c, d, e |
| Figura 2 | `figura_2_2026-10-28/` | b, c, d |
| Figura 3 | `figura_3_2026-10-29/` | a–i |
| Extended Data 3 | `figura_ext_3_2026-10-28/` | a–d, ratto per pannello |
| Extended Data 4 | `figura_ext_4_2026-10-28/` | a–b, quattro ratti |
| Extended Data 8 | `figura_ext_8_2026-10-29/` | a–f |

Le cartelle di processo e i riferimenti specifici sono descritti in `processo/INDICE.md` dentro ciascuna consegna.

La campagna piVAE interrotta a 10 seed (0 fit completati) è stata tolta dal livello principale e conservata come audit in `figura_1_2026-11-01/processo/provenienza/campagne_interrotte/`; non viene usata nei plot.
