# Five-seed amendment — 14 October 2026

Classification: user-approved NEW PROTOCOL DECISION concerning replication
budget, not a software bug correction.

The completed joint CNN branch retains seven seeds (42–48). The currently
running time-only CNN also remains at seven seeds, three folds, 168 fits.

Subsequent Transformer joint/time-only and joint-shuffle branches use exactly
five training seeds (42–46), with all three folds (0–2), four rats, and both
frozen/stable losses. Transformer: 120 joint + 120 time-only = 240 cells.
Shuffle: 120 CNN + 120 Transformer = 240 cells. No per-fit numerical,
architecture, sampling, optimization, or evaluation setting changes.

Old seven-seed plans, qualification reports, permutations and artifacts are
retained. Five-seed plans/gates use separate roots:

- `results/user_soft_rat_transformer_5seed_v1/`
- `results/user_soft_rat_shuffle_5seed_v1/`

The four completed initial Transformer fits and matching partial fifth remain
eligible for exact immutable reuse. Updated wrapper qualification must pass
before execution. Comparisons involving all methods must use the common five
seed set; the two additional CNN seeds remain useful supplementary replication,
not extra paired observations for Transformer comparisons.

The old pending seven-seed coordinator was stopped without interrupting the
active CNN worker. The replacement coordinator waits for all 168 CNN time-only
fits and the source-qualified five-seed Transformer GO gate.
