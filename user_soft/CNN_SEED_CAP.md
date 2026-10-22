# CNN campaign execution amendment — 2026-10-14

User instruction: stop the CNN comparison at seven training seeds per rat,
then proceed with the preliminary Transformer comparison.

Authorized CNN subset: seeds 42–48, all four rats, folds 0–2, both frozen
and stable soft objectives: 168 completed fits. Seeds 49–51 are not to be
started. The historical sealed 240-fit plan is retained unchanged as provenance;
its planned count is not the amended execution target.

At the stop request, 163 fits were complete. The active fit had no completed
manifest. Its last committed checkpoint is retained for resumption. The worker
was stopped; after verifying its PID was absent, its stale exclusive lock was
removed. Resume uses the existing runner with `--execute --max-new-fits 5`.
Previously completed fits are reused, not trained again. This limit counts new
completed fits, and the remaining five belong to seed 48.

Transformer: separate initial eight-fit branch (four rats, fold 0, seed 42,
two objectives), launched only after its own CPU qualification gate passes.
No change to data, loss, CNN architecture, optimizer, per-fit update budget,
splits, or completed scientific results is authorized by this amendment.
