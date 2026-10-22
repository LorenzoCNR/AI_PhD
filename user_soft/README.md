# Controlled user soft comparison on rats

This package is separate from the unchanged CEBRA replication. Historical source copies carry their original variable names and exact contents.

Versions: `soft_v1_frozen` reconstructs the generic NeuroBridge batch-distance API and original cross entropy. `soft_v1_stable` replaces batch-wise distance scaling with training-only fixed bounds and evaluates targets and probabilities in log space. These are different target distributions; stability is not evidence of better scientific performance.

Joint rat metadata use exactly `(position,right,left)` as one continuous Euclidean vector plus local trial time. Right/left are redundant one-hot encodings of one direction. No synthetic eight-condition circle is used. Weights are time=0.5, joint behavior=0.5; metadata temperature=0.5 and embedding temperature=0.1 recover historical soft settings, while optimizer/input/architecture/budget follow the rat baseline. A categorical direction decomposition or separate position/direction weighting is a future protocol, not silently introduced here.

The stable continuous scale is the Euclidean norm of the training feature-wise range, an upper bound on pair distances. Time scale is the training time range. Constant components use scale1 with an explicit flag. Validation/test cannot fit scales. Values outside the training extent are neither rescaled nor clipped. Trial times use 25ms bins from the start of each exact official round-trip trial. Cross-trial pairs remain candidates at compatible local times. Training batches contain one rat/session and one split.

Official `offset10-model` is instantiated directly from the installed CEBRA registry, with 32 hidden units and 32 outputs; no architecture rewrite. Native output is already normalized. `embedding_raw` means native frozen output, and `embedding_unit` is its explicit unit-normalized counterpart; for this model they should agree numerically. Recovered original Transformer is tested but not included in the first 240 fits.

Campaign: 4 rats x3 official folds x10 seeds42..51 x2 versions x1 joint position/direction condition=240 new fits. Existing baseline joint fits are parents for comparison, not rerun. Adam lr3e-4, betas(.9,.999), eps1e-8, wd0, batch512, 10000 updates; no early stopping or HPO. Soft temperatures are part of its specified objective, not selected from test outcomes. Reference draws are uniform with replacement; off-diagonal duplicates remain eligible as in the historical dense loss. They are logged, not silently filtered.

Input contexts follow the baseline's concatenated-split construction (`legacy_concat`): centers are clamped to [5,N-5] and then expanded by [-5,+4]. This retains the known trial-join confound in both baseline and new input paths. Full metadata and strict trial-interior validity are exported. All downstream comparisons use the same interior support. Trial-safe windows are available for tests/future separately declared controls; switching the campaign to them requires a matched baseline. See AUDIT_ISSUES_AND_FIXES.md.

Stages and commands (from AI_PhD):

```powershell
python -m unittest discover -s user_soft/tests -v
python -m user_soft.smoke --output results/user_soft_rat_v1/qualification/cpu_seed42 --seed 42
python -m user_soft.rat_runner --dry-run
python -m user_soft.rat_runner --execute
```

`stage00_data` caches inputs/splits/metadata/scales, `stage01_fit` stores resumable checkpoints/history, `stage02_embedding` stores frozen exports and hashes, and `stage03_metrics` contains separate downstream probes. Every stage verifies parent and source hashes before reuse. No existing experiment output is writable through this package. Campaign failures remain in the manifest; no overwrite or silent invalidation. New protocol requires a new run root/version.

W is reconstructed from metadata and saved scales; diagnostic reference batches save IDs and small W/Q/P matrices. Entire BxB matrices are not written at every update. Diagnostics report entropy, exp(entropy) effective neighbors, sum(p²) concentration, top1 probability, candidate counts, cross-trial mass and minibatch duplicate fraction. Sensitivity probes vary temperatures, weights and batch composition on qualification observations, without selecting hyperparameters on test.

Rat panel references: Figures1c-e (context/manifold/consistency),2b-d (hypotheses/loss/manifold/decoding), ED4 (decoding/dimension comparisons); Figure3 and ED8 concern monkey and are outside this campaign. New rows append to the comparison tables; existing figure styles/renderers remain the authority. No new arbitrary figure style is introduced.
