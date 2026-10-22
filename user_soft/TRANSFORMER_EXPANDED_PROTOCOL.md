# Seven-seed recovered Transformer extension

USER-APPROVED NEW PROTOCOL DECISION (budget amendment): successive Transformer and shuffle structures use FIVE seeds42-46, retaining all3 folds. Existing CNN time-only168 fits/seven seeds remain unchanged. The five-seed Transformer design has240 cells(120joint+120time), new root results/user_soft_rat_transformer_5seed_v1. Use --seed-count5 explicitly. Default7 remains for reproducible enumeration of the old336-cell plan, whose sealed dry-run root is retained and never rewritten. Joint configs/source signatures stay identical at shared seeds; four complete initial models and the update7500 partial attempt remain valid reusable parents. No HPO/settings/metrics change. This is an approved budget decision, not a bug fix.

```powershell
python -m user_soft.transformer_campaign --dry-run --seed-count 5 --output results/user_soft_rat_transformer_5seed_v1
python -m user_soft.transformer_campaign --execute --seed-count 5 --output results/user_soft_rat_transformer_5seed_v1
```

Authorized expanded design:4 rats x3 official folds x7 seeds42-48 x2 versions x2 conditions(joint,time_only)=336 cells. This new wrapper does not rewrite the initial8-cell sealed plan or initial runner. Execution is queued after the separate CNN time-only branch by the orchestrator. Settings stay matched to each corresponding CNN condition: exact10 bins,32D,Adam3e-4,batch512,10000 updates,unchanged soft versions/temperatures,dropout0.1 and LayerNorm preserved.

Joint target remains the original temporal plus joint(position,right,left) distances; time-only uses local trial time weight1 and intentional cross-trial candidates. TRAIN time scales are projected without re-estimation, with parent hashes. Metadata targets/eligibility remain encoder independent. No HPO or test-outcome selection occurs.

New root: results/user_soft_rat_transformer_campaign_v1. Separate joint/ and time_only/ stage branches prevent artifact collisions. The wrapper's aggregate state/tables point to actual artifact paths and hashes. Initial seed42fold0 joint checkpoints already present in results/user_soft_rat_transformer_v1 keep their exact source/config/data signatures and original files. Complete fits are reused; the incomplete attempt resumes its stored optimizer/RNG transaction. Nothing is relabeled as a newly completed fit. Original embeddings/metrics are reused under existing hash checks; new cells use the expanded root. This avoids checkpoint/embedding copies and preserves immutable provenance.

Joint training signatures remain transformer_runner.sources() exactly, permitting original fit reuse. Time-only signatures additionally identify metadata-spec/projection/binding and expanded configuration source. The wrapper plan/provenance is separate from original model dependencies. GO gate checks expanded plan/control invariants and the original Transformer CPU dropout/resume qualification. Completed/partial counts, actual optimizer updates and any replayed steps remain recorded by the existing fit transaction mechanism.

Two exclusive locks guard the expanded wrapper and original initial-runner root, preventing simultaneous writes to a reused checkpoint. No file is unlocked or deleted by guessing whether another process is alive. Existing root locks must be resolved by the orchestrator only after verifying process state.

Native-normalized raw and explicit unit exports retain trial/time/source/session IDs, behavior and validity masks. Decoders, TRAIN sensitivity diagnostics and per-condition compute tables use the existing qualified components. GPU concurrency caveats remain explicit; observed speed is not an isolated efficiency comparison.

The CEBRA-style TRAIN joint-label shuffle is planned as a separate branch, not silently activated here. Its existing reference permutes complete joint behavior rows with rng default_rng(seed+10000), preserves neural rows/local time and retains original behavior for evaluation. Behavior shuffle is algebraically ineffective for the pure time-only loss and must not trigger redundant time-only fits.

```powershell
python -m user_soft.transformer_campaign --dry-run
python -m user_soft.transformer_campaign --execute
```

Dry-run performs zero optimizer updates. Execution requires the dedicated matching-source expanded GO gate and the original Transformer GO gate. No GPU execution is launched by constructing this wrapper.
