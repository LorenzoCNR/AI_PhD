# Matched CEBRA-style TRAIN joint-label shuffle

USER-APPROVED NEW PROTOCOL DECISION (budget amendment): successive shuffle/Transformer structures use FIVE seeds42-46 with3 folds. Shuffle now has240 cells(120CNN+120Transformer), sharing60 TRAIN permutations, in new root results/user_soft_rat_shuffle_5seed_v1. Existing seven-seed CNN time-only168 is unchanged. Default --seed-count7 preserves old336 enumeration; old sealed dry-run roots remain untouched. Source-qualified new gates are required, and no completed model is invalidated. This is an approved budget decision, not a bug fix.

```powershell
python -m user_soft.shuffle_runner --dry-run --seed-count 5 --output results/user_soft_rat_shuffle_5seed_v1 --transformer-output results/user_soft_rat_transformer_5seed_v1 --transformer-fit-count 240
python -m user_soft.shuffle_runner --execute --seed-count 5 --output results/user_soft_rat_shuffle_5seed_v1 --transformer-output results/user_soft_rat_transformer_5seed_v1 --transformer-fit-count 240
```

The five-seed shuffle executes only after the original CNN time-only168 and the explicitly configured new Transformer240 producer are complete. Predecessor path/count are sealed in the plan; rerouting a sealed plan is rejected.

S11 reported before correction: the prerequisite check originally required status=complete, but the existing CNN time-only runner can finish all168 cells and set status=requested_limit because its default new-fit cap is168. This would wrongly prevent the next valid stage. The shuffle-only correction accepts complete/requested_limit only when the declared completed count and exact complete-record count both equal the required168/336 and the producer RUNNING.lock is absent. No producer, checkpoint or scientific setting changes. Active/incomplete campaigns remain blocked.

Reference inspected: cebra_replica/rat_benchmark.py lines102-113. rng=np.random.default_rng(training_seed+10000); the complete joint behavior Nx3 matrix is indexed by one rng.permutation(N). Components move together. This is a training-control row permutation, not a complete-trial fixed-embedding permutation test and not1000 inferential replicates.

At each rat/fold/seed, one exact permutation is shared across CNN/Transformer and frozen/stable versions. Neural rows, relative time, trial/session/source IDs are unchanged. Permute TRAIN behavior only; validation behavior remains original. Fit target combines shuffled joint(position,right,left) distance with unchanged local-time distance. Original labels are used in exported embeddings and all downstream decoder evaluations. Explicitly: training and validation losses correspond to different behavior-target worlds; they are diagnostic, not a model/HPO/early-stop selection criterion.

Stable TRAIN distance scales are copied from the original parent because a row permutation preserves the complete feature ranges and temporal bounds exactly. No scale re-estimation uses validation/test. Stored TRAIN permutation NPZ includes recipient/source/trial identities, donor behavior/source/trial identities and seeds; manifest hashes the parent cache, TRAIN file, permutation and source function. No shuffled neural-data copy is stored.

Design:4 rats x3 folds x7 seeds42-48 x2 soft versions x2 architectures=336 new fits(168CNN,168Transformer). Same matched official10-bin inputs,32D,batch512,Adam3e-4,10000updates and unchanged temperatures/architecture. Transformer dropout/LayerNorm remain unchanged. No time-only behavior-shuffle fits: pure time loss is algebraically unaffected by behavior permutation. A temporal-order null would be a separately defined experiment.

TRAIN fit inputs receive shuffled metadata while the encoder sees identical neural rows. Encoder-independent soft loss remains unchanged. Exports attach original behavior. TRAIN sensitivity diagnostics explicitly reconstruct shuffled target metadata using the saved permutation and also save donor IDs; reading original-label exports without this reconstruction would describe the wrong training target, so the wrapper supplies a dedicated diagnostic adapter. Existing joint diagnostic/loss/runner source is untouched.

Output root results/user_soft_rat_shuffle_v1 has cnn/ and transformer/ stage branches, shared stage00_shuffle permutations and aggregate tables/state. Existing models/caches/split records are retained. Dry-run prepares immutable permutation records and performs zero optimizer updates. Dedicated qualification checks permutation sharing, original exports, stable-scale identity and unchanged neural/time/source observations before GO.

Execution follows completion of CNN time-only168 and expanded Transformer joint/time336. The runner checks those prerequisite campaign state counts/status before starting a real fit. No empirical decoder result controls this sequence. No GPU execution is launched by creating the runner.

```powershell
python -m user_soft.shuffle_runner --dry-run
python -m user_soft.shuffle_runner --execute
```

Interpretation: this control removes neural alignment with the joint behavioral TRAIN labels, preserving the time prior and all neural observations. It need not yield chance decoding for every task because temporal structure and neural information remain; above-shuffle comparison supports a narrower claim than full unsupervised evidence. Validation retains original targets by fidelity to the existing TRAIN-only shuffle control.
