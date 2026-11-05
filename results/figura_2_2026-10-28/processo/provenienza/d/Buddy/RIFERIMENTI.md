# Figura 2d

Le traiettorie `Figura_2d_Buddy_traiettorie_test` mostrano un intervallo consecutivo comune del test, fold 0 e seed 42: posizione reale e previsioni dei decoder kNN e Ridge. Provenienza e indici sono in `RIFERIMENTI_TRAIETTORIE.json`; codice: `cebra_replica/rat_figura2d_trajectory_other_rats.py`.

Metriche per fold: `../../../metriche/d/metriche_fold.csv` (i singoli record con percorsi e hash sono in `../../../metriche/d/calcoli`). Fit encoder della campagna `rat_paper_fig1e_ed4_6seed_v1`, sei seed 42–47, tre fold per trial, train/validation/test separati. kNN cosine, k scelto sulla validation in base alla media degli R² Position e Direction; errore mediano e accuracy sul test. Ridge usa standardizzazione e alpha 0.1, 1, 10 selezionato sulla validation per R² posizione. Punti: media tre fold per seed; barre: mediana. PCA è deterministica. Non si usa q(z|x,u) per decoding. Questi output non sono ancora aggiornati alla campagna `rat_paper_fig1e_ed4_10seed_v2`.
