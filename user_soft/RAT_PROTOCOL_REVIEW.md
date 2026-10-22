# Rat protocol and downstream adapter review

Reviewed 2026-10-14. Source review only; no old fits or results modified.

## Existing reference campaign

`cebra_replica/rat_campaign.py` implements four rats (Achilles, Buddy, Cicero,
Gatsby), three deterministic trial folds (0, 1, 2), seeds 42 through 51, and
seven conditions: time, position, direction, joint position_direction, and the
three corresponding behavior-shuffled controls. This is 840 reference fits,
not ten splits by three seeds. A live manifest count found 120 `result.json`
records and 120 `scores.json` records for each condition, 840 of each in total.
These counts verify presence, not replacement for full parent hash checks.

The Figure 2d method-comparison label `CEBRA-Behavior` maps specifically to
`position_direction`, not position-only: `rat_manifold_scoring.py:42` and
`rat_figure_gap_closure_analysis.py:75-77`. The joint input is the existing
three-column behavior vector (position, right indicator, left indicator).
The two direction coordinates are redundant and remain unchanged in the
reference. Their dtype is float; sklearn CEBRA therefore treats this joint
vector as continuous metadata. The comparison table is generated as
`figure2d/FIGURE2D_METHOD_METRICS_SEED_LEVEL.csv` and the figure stem is
`FIGURE2D_METHOD_COMPARISON_TEST` under the existing figure-delivery output.

Frozen reference numerical settings in `rat_benchmark.py`: official
`offset10-model`, hidden width 32, embedding dimension 32, batch size 512,
10,000 optimizer updates, learning rate 3e-4, cosine distance, time offset 10;
temperature 1.12 for time and 1.0 for behavior. Adam uses the installed official
API defaults: betas (0.9, 0.999), eps 1e-8, weight decay 0, no early stopping.
Loss replacement necessarily changes the objective; this must remain explicit
even when architecture, data and optimizer settings match.

The initial new comparison scope is joint position_direction: two soft-loss
versions x four rats x three folds x ten seeds = 240 fits. Applying both soft
versions to all seven reference conditions would instead require 1680 fits;
that wider campaign is not implied by implementing this adapter.

## Official encoder and trial metadata

`CEBRA/cebra/models/model.py:263-303` registers `offset10-model` through
`OffsetNModel`. Construct the official registry model with num_neurons equal
to the recording channels, num_units=32, num_output=32. Input shape is
batch x channels x 10; offset is (5,5), corresponding to t-5 through t+4.
The native output already includes the official row-normalization layer.
Thus native/raw and unit-normalized outputs can coincide; preserve both
representation names rather than inventing a pre-normalization encoder output.

`CEBRA/cebra/datasets/hippocampus.py:206-261` defines round-trip trials using
every second direction transition, then deterministic nested threefold splits
with shuffle=False. Preserve its exact slice convention. Recover global trial
IDs by matching selected source slices to full-source trial slices; local time
is arange(trial_length), with 25-ms sampling. Session identity is the rat.
Do not infer local time from the concatenated selected-split row number.
Similar local times across different trials intentionally remain eligible in
the new loss; this is distinct from convolution input-window boundary handling.

## Reported boundary limitation and authorized compatibility policy

The existing reference `rat_benchmark.py` concatenates selected trials, then
passes the array to sklearn `model.fit`. The sklearn dataset calls global
`expand_index`, which clamps whole-array boundaries only. Consequently a
10-bin input adjacent to an artificial join can combine nonadjacent trials.
This is a valid legacy reference policy with a scientific limitation, not a
train/test leakage finding: arrays are split before fitting and transforming.

The authorized controlled campaign keeps this `legacy_concat` input policy
to match frozen references. A strictly trial-safe option is useful for focused
qualification, but must not silently replace campaign inputs. Both methods
are evaluated on common explicit interior support: for a source trial of
length L, a valid center satisfies t>=5 and t+4<L. This excludes contaminated
join rows from scoring but does not change how the old encoder was trained.
A later trial-safe training comparison would need a paired reference under
the same input policy; it is not performed by the current decoder module.

## Downstream module

`user_soft/downstream.py` consumes frozen arrays with explicit source_id,
trial_id, time_id, Boolean valid_mask, behavior, and native/raw embedding.
Missing trial metadata in old NPZ files must be reconstructed by a separate
trusted source adapter; the scorer refuses to guess trial boundaries.
All source rows and whole-trial IDs must be disjoint across train/valid/test.
Raw and unit embeddings are reported separately on identical supplied support.

Reference position decoding is fixed cosine kNN36, reporting median absolute
position error in cm and R2; reference direction decoding is cosine kNN36,
reporting balanced accuracy and the additionally requested ordinary accuracy.
Ridge position decoding is newly implemented downstream: StandardScaler is
fitted on training embeddings only, followed by Ridge fitted on train.
The exact recovered alpha grid is (0.1,1.0,10.0), from
`Neuro_Bridge/tools/final_thesis_core_metrics.py:70`. Choose maximal validation
position R2, with exact ties retaining the first entry in that fixed order.
Complete both raw/unit selections before reading any test embedding or behavior.
No refit on train+validation is performed. Test source/trial IDs may be read
earlier only for split-disjointness checks. This is downstream probe selection,
not encoder/HPO selection. Test outcomes do not influence alpha selection.

The existing rat code already implemented balanced accuracy; rat Ridge was
still marked pending at `TODO.md:70`. Monkey RidgeClassifier is a different
analysis and is not imported as rat regression.

The CLI accepts `--run` and a distinct new `--output`, validates declared
parent embedding hashes when present, records all actual parent content hashes,
parent manifest hash, scorer source hash, fixed grids and support rules, and
refuses an existing destination or an output nested inside the frozen run.
It never changes a checkpoint or encoder. New outputs are
`RAT_DECODING_RESULTS.csv`, `RAT_DECODING_RESULTS.json`, `PROVENANCE.json`.
No new figure styling is introduced by this module.

## S12 — concurrent JSON publication risk (reported before wrapper correction)

Where: `baseline_reference.py`, `_write_json`, `path.open("x")` followed by
`json.dump`. Current behavior creates the final manifest/metrics pathname before
the complete JSON bytes have been written. A concurrent comparison reader could
therefore observe an incomplete JSON. This is a prospective software/I/O race,
not an observed corruption or a scientific/numerical failure. Severity MINOR;
affected outputs would be transient reporting/aggregation, with no checkpoint,
embedding values, decoder fit, alpha selection or scientific metric changes.
No old result requires recomputation. Required action: I/O correction only in a
new independent wrapper, preserving sealed `baseline_reference.py` source.

Proposed smallest fix: `baseline_campaign.py` overrides only `_write_json` with
unique temporary-file publication, complete write/flush/fsync, and Windows
rename to a non-existing target. Existing immutable targets are rejected; one
exclusive reference-worker lock prevents duplicate workers. The wrapper records
its source hash and the explicit I/O override in its own provenance. The worker
does not publish the root comparison CSV; the existing root aggregator remains
its single writer. CPU/BLAS worker threads are limited to one.

Before/after: final JSON pathname visible during write -> complete temporary
write plus atomic rename -> only complete JSON visible to comparison readers.
Verification: targeted publication checks must confirm full parseable content,
existing destination rejection and temporary-file cleanup before launching the
reference worker. No methodological or protocol setting is changed.

S12 resolved evidence: `python -m user_soft.baseline_campaign --smoke-only`
passed complete JSON parse, immutable-target rejection with unchanged hash,
temporary cleanup, and failed/nonfinite serialization remaining unpublished.
The sealed baseline verifier/exporter source was not edited. The new worker
checks the saved qualification decision GO, zero failed tests and current plan
sources before scientific downstream execution.
