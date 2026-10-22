# Independent soft-loss qualification

Review sealed: 2026-10-14 13:34:34 UTC. Scope: software and evaluation-boundary qualification, not statistical evidence of rat decoding performance.

**GO for the controlled CNN rat comparison.** Independent execution of `python -m unittest discover -s user_soft/tests -v` passed all **28 tests** in **3.589 seconds** using the existing `cebra` Python environment. The only warning was an upstream Torch JIT deprecation. No real model fit or scientific test metric was computed by this reviewer.

## Requested invariant coverage

Requirement | Independent check | Outcome
---|---|---
Q/P normalization | Valid rows sum to one; uniform targets checked against 1/(B-1) | PASS
Diagonal exclusion | Exact zero diagonal in both distributions; reference formula masks self candidates | PASS
Finite gradients to encoder | Frozen/stable backward, double-precision gradcheck, official CNN plus recovered CNN/Transformer | PASS
Uniform/concentrated cases | Uniform prediction gives log(B-1); metadata temperature 1e-8 and embedding temperature 1e-5 remain finite | PASS
Coherent batch permutations | Loss invariant and Q/P/W equivariant under joint metadata/embedding permutation | PASS
CNN/Transformer compatibility | Actual official offset10 CNN uses unchanged 32 hidden units; recovered encoders tested independently | PASS
Intentional cross-trial pairs | Equal trial-relative time/behavior across distinct trials receives positive, stronger weight | PASS
Train/test separation | Nontrain scale fitting rejected; fixed scales unchanged by evaluation outliers; sampler rejects test; row/trial overlap rejected; test outcomes cannot select Ridge alpha | PASS
Fixed-seed reproducibility | Identical CPU updates and toy parameter hashes; interrupted fit exactly matches uninterrupted fit | PASS
Mathematical formula | Independent float64 frozen and stable reference distributions and cross-entropy | PASS

## Staged runner checks

Five CPU toy tests cover completed-checkpoint reuse without calling the fit objective, refusal to overwrite when relevant configuration changes, embedding export cardinality and metadata correspondence, export reuse without loading an encoder, interrupted optimizer/RNG resume, and exact context equivalence with official CEBRA `TensorDataset.expand_index`.

The successful interruption test stops before attempted update 3, retains the committed checkpoint at update 2, resumes to update 4, and matches the final parameter hash of uninterrupted four-update training. Periodic checkpoints may replay work since the last committed checkpoint; replay does not represent additional committed optimizer updates.

Independent read-only inspection also compared cached TRAIN windows from **all 12 rat/fold combinations** against official CEBRA context expansion. For each cache, ten selected centers include recording boundaries and the first three concatenated trial joins. Every neural context matched exactly. This check accessed TRAIN data only.

## Issues found and resolved

ID | Direct evidence before correction | Resolution verified
---|---|---
S08 | Wrong internal trial/session/time metadata at row 8 could leave center 10 eligible under endpoint-only validation | Full ten-row context validation; adversarial discontinuity test passes
S09 | NumPy int64 TRAIN IDs raised JSON serialization TypeError | Native integer conversion; NumPy and Torch ID provenance hashes match
S10 | Uninterrupted four-update hash `69f3f936...` differed from interrupted/resumed hash `f78f0a0f...` because the failed batch advanced RNG | Initial and committed periodic checkpoints preserved; exact-resume test now passes

These defects were reported before corrections. No prior rat scientific result was invalidated: they were found during new software qualification before rat execution. The permanent issue log is `AUDIT_ISSUES_AND_FIXES.md`.

## CPU training record

`results/user_soft_rat_v1/qualification/cpu_seed42_20261014_1515/` contains `CPU_SMOKE_RESULTS.json` and `PROVENANCE.json`. Both versions completed 30 CPU optimizer updates, with finite gradients and fixed-support loss **6.41559267 -> 3.32660365**. Parameter reproducibility is checked independently. Equality between versions is expected on this toy, where training extrema and minibatch extrema coincide; it does not establish equivalence on rat data or scientific efficacy.

## Dependency review and interpretation

`rat_runner.sources()` hashes training dependencies individually, including `frozen_config`, `make_encoder`, `tensors`, `encode_batch`, `objective`, and `fit_cell`. It excludes downstream scorer/decoder/exporter functions and does not hash the entire runner as a training dependency. Downstream-only changes therefore do not invalidate a fit signature; embedding and metric stages keep their own dependencies. Whole `data.py` and `loss.py` remain upstream dependencies, conservatively.

The new soft objective deliberately uses dense batch-pair distributions; official CEBRA uses reference/positive/negative sampling. A fixed encoder and matching optimization settings do not make the two candidate distributions identical. Minibatch diagonal exclusion removes identical batch positions, while independently resampled duplicate source rows remain eligible off-diagonal candidates. Official native encoder output is unit normalized, so exported `raw` means native output, not hidden pre-normalization activations.

The sealed dry run prepares 12 caches and enumerates 240 soft fits: four rats x three folds x ten seeds x two soft versions. It performs zero optimizer updates. Statistical power, confidence intervals, and rat scientific effect estimates are outside this qualification; these checks establish implementation readiness.

Machine-readable independent verification: `results/user_soft_rat_v1/qualification/INDEPENDENT_TEST_SUMMARY_20261014T133434Z.json`.
