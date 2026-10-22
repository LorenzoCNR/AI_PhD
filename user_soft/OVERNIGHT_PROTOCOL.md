# Overnight staged replication amendment

S12 reported before correction during qualification: the first draft verified original fit signatures/checkpoint hashes but indexed downstream manifests without independently checking each frozen embedding's content hash or metric-parent linkage. This was an incomplete new-index provenance check, not evidence of any corrupted historical result. Read-only verification now checks embedding/checkpoint/config linkage, every embedding NPZ hash, scorer/embedding parent signature, and all registered diagnostic artifact hashes before a record is marked complete. No historical file is overwritten and no metric is recomputed by these checks.

User-approved NEW PROTOCOL DECISION: preserve the existing seven-seed CNN joint/time campaigns and currently running five-seed Transformer. After the latter completes, first extend both Transformer conditions to seeds47/48, giving seven complete seeds for all four core groups. Then execute matched joint-label shuffle for both architectures through seven seeds. Only subsequently add seed49 to all six groups, then50, then51 if permitted by the execution ceiling and available time. No model hyperparameter, architecture, split, loss, metric or input setting changes.

Order/counts beyond the576 existing/ongoing core references: Transformer47/48=96 fits; seven-seed shuffle=336; each extra balanced seed across four core plus two shuffle groups=144. Potential models through ten seeds:1440. Core seven seeds account for672 and seven-seed shuffles for336, total1008. The full potential ten-seed plan is sealed before any fit; max-seeds/deadline are external execution directives and do not mutate model configs or training signatures.

Within each phase/seed, families are interleaved for each matched rat/fold/loss cell. Completed-prefix counts differ by at most one among the phase's families, reducing partial-seed imbalance at a deadline without changing any fit setting or selecting by outcomes.

Default execution ceiling is seven seeds. The explicit overnight directive permits ten and a hard start cutoff2026-10-15T07:00:00+00:00 (09:00Europe/Rome). Deadline is checked before every new fit, including resuming an incomplete checkpoint, and rechecked immediately before its optimizer transaction. A fit already in progress finishes, checkpoints and downstream outputs flush, then no next fit starts. Completed fit reuse remains permitted beyond the deadline. Full completion by the deadline is not promised; per-family/per-seed progress distinguishes completed core, incomplete shuffle and any later seed.

Existing five-seed Transformer worker is neither edited nor relaunched. The extension waits for both CNN168 states and Transformer240 state to contain exact complete records, with inactive locks. Its deadline does not alter an independent producer process; an external supervisor must enforce a deadline on that worker separately if it fails to finish beforehand. The extension owns its own exclusive lock and does not delete producer locks.

New root results/user_soft_rat_overnight_v1 contains POTENTIAL_PLAN.json, immutable execution_directives records, referenced/shared input provenance, families stage branches and an artifact index in CAMPAIGN_STATE.json. Existing fit/checkpoint/embedding/metric paths are referenced rather than copied. Every reused fit checks its exact original configuration, parent ID, training-source mapping and checkpoint hash. Wrapper provenance is separate, allowing historical training signatures to remain identical. Both soft versions, native32D encoders, dropout/LayerNorm, batch512 and terminal10000-update training stay unchanged.

Original rat_runner/time_only_runner/transformer_campaign/shuffle_runner factories, scale projections and training-source identities are reused exactly. Parent TRAIN and split hashes, shuffle permutations and donors are sealed in the potential plan. Shuffle remains behavior-only TRAIN row permutation shared across architectures/versions with original exports/decoder labels. No time-only duplicate shuffle is introduced.

Dedicated qualification checks the1440-job potential plan,432 jobs to seven seeds,864 possible extension jobs, ordering, original config/source identity, completed/partial artifact routing, deadline boundary behavior, external ceilings and zero-fit dry-run. Execution requires a matching-source and matching-plan overnight GO gate plus all existing upstream gates. New deadline/ceiling directives do not affect any fit's parent/source/config identity.

```powershell
python -m user_soft.overnight_runner --dry-run
python -m user_soft.overnight_runner --execute --wait --max-seeds 10 --deadline-utc 2026-10-15T07:00:00+00:00
```

SIX_GROUP_PROGRESS.csv and the state expose counts/completed seed lists independently for cnn_joint,cnn_time,transformer_joint,transformer_time,shuffle_cnn,shuffle_transformer. Waiting/deadline/ceiling/failure statuses are explicit. No GPU fit is started by implementation or qualification.
