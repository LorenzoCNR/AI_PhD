# UMAP/t-SNE rat input: source audit

Data: 22 October 2026. At the audit snapshot, before the separately configured
paper-window fit, no UMAP/t-SNE fit had been started.

## Finding

For a paper-faithful rat replication, the observation supplied to UMAP/t-SNE is a
250 ms neural segment, not a single 25 ms population vector. The paper's Methods
specify that the continuous 25 ms-binned rat spiking activity was chopped into
250 ms segments with 225 ms overlap specifically for CEBRA, UMAP, t-SNE, and
pi-VAE. With 25 ms bins, that means 10 consecutive bins per sample and a stride
of one bin.

For Achilles, whose local source array has shape `(10178, 120)`, one segment is
`(10, 120)`. UMAP's documented input is a two-dimensional `(samples, features)`
matrix, so this runner represents each 10-bin sample as one flattened vector of
1,200 features. Across rats the rule is `10 * n_neurons`, not an assumed fixed
1,200 features: the local counts are Achilles 120, Buddy 48, Cicero 55, Gatsby
66, yielding 1200/480/550/660 window features respectively. Each segment is
kept as a distinct sample; the overlapping windows are not averaged together.

The paper's precise UMAP preprocessing call (including the reshape line) is not
in the public figure-reproduction code. The official `cebra-figures` README says
that repository contains plotting functions over precomputed results, while
experiment/fitting code is separate. Thus the 10-bin support is directly
documented by the paper; flattening it into a feature vector is the explicit
implementation needed by standard 2D UMAP/t-SNE APIs and is recorded as our
faithful interpretation, not claimed as a recovered author source line. The
single-bin representation is retained only as a separately named diagnostic
alternative, not as the paper protocol.

## Kept-distinct configurations

| Name | Per-sample input | Achilles features | Status |
|---|---|---:|---|
| `window10_flat` | 10 consecutive 25 ms bins × 120 neurons, flattened | 1,200 | Methods-based paper replication |
| `single_bin` | One 25 ms bin × 120 neurons | 120 | Separate alternative; not Methods-faithful |

`rat_paper_ed3_manifold.py` now requires or defaults to an explicit
`--input-representation` choice. Outputs use distinct representation-specific
directories so that 120-feature and 1,200-feature fits cannot be mistaken for
one another or overwrite one another. The Methods-based choice is the default;
`single_bin` must be requested explicitly. Nothing has been fit by this audit.

## Sources

- Schneider, Lee & Mathis (2023), *Nature*, Methods / rat model comparisons:
  [article](https://www.nature.com/articles/s41586-023-06031-6). The Methods
  describe 250 ms segments, 225 ms overlap, and state that this setup matched
  CEBRA, UMAP, t-SNE, and pi-VAE.
- Authors' figure repository:
  [CEBRA-figures README](https://github.com/AdaptiveMotorControlLab/cebra-figures/blob/main/README.md)
  says it contains plotting code applied to precomputed results, not the full
  experiment/fitting pipeline.
- Authors' rat dataset loader:
  [CEBRA `hippocampus.py`](https://github.com/AdaptiveMotorControlLab/CEBRA/blob/main/cebra/datasets/hippocampus.py)
  records 25 ms spike-count bins and returns rat samples as
  `[samples, neurons, 10]` for the offset-10 dataset.
- UMAP's documented API:
  [UMAP API guide](https://umap-learn.readthedocs.io/en/latest/api.html)
  specifies `X` as `(n_samples, n_features)` unless using a precomputed
  distance matrix; the runner uses cosine distance on feature vectors.

## Local verification

- Achilles source data: `data/rat_hippocampus/achilles.jl`, `spikes.shape ==
  (10178, 120)`.
- Window support in the runner: center rows `5` through `N-5`, inclusive;
  offsets `[-5, +4]`; one center per bin; 10 bins per segment. This excludes
  the first five and last four centers to retain complete windows. The paper
  does not expose its exact edge-handling code, so this is a stated local policy,
  not a claim that the authors used identical endpoint handling.
- There were no active Python fit processes and no embedding/manifest files in
  the ED3 manifold output tree at the time of inspection.
