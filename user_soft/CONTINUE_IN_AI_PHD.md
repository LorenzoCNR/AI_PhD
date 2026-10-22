# Continue this work from a new AI_PhD Codex chat

Project root: C:/Users/loren/Desktop/USB_Content/New project/AI_PhD.
Old Neuro_Bridge is read-only source history. Do not work in it.

The user has authorized240joint-rat neural fits:4rats x3officialfolds x10seeds42-51
x2lossversions (soft_v1_frozen/soft_v1_stable), actual official offset10 CNN32D,
batch512,Adam3e-4,10000updates. No Transformer campaign or time-only soft fits yet.
Review user_soft/README.md and frozen results/user_soft_rat_v1/CAMPAIGN_PLAN.json.
29tests pass; CPU smoke and exact interrupted resume pass;240Behavior/Time
baseline checkpoint/source/embedding proofs pass. Three specialist reports and
root integrated report exist. Source histories copied standalone with hashes.

FIRST check results/user_soft_rat_v1/CAMPAIGN_STATE.json and RUNNING.lock.
A worker was launched by the previous chat with `python -m user_soft.rat_runner
--execute`. Do NOT launch a duplicate if its PID remains alive. Use read-only
status/progress inspection. The lock deliberately refuses a second process.
If stopped, inspect PID and last checkpoint; safely remove only a proven stale
lock in this exact study folder, then run the same command in the cebra env.
The runner refuses changed dependencies and reuses valid complete checkpoints
and embeddings. Changing a decoder/report does not invalidate fit dependencies.

Initial baseline2decoder cells were recomputed on common trial-interior support.
The remaining238reference cells need downstream evaluation via
`python -m user_soft.baseline_reference --evaluate`, without any encoder fit.
CEBRA-Behavior in this comparison means joint(position,right,left), Time is label-free.
Baseline840historical fits/scored manifests already exist; they must not be retrained.

Current result tables: RAT_SOFT_COMPARISON.csv, COMPUTE_PERFORMANCE.csv; sources
user_soft/rat_runner.py stages00data,01fit,02embedding,03metrics,04diagnostics.
W/Q/P reference matrices and TRAIN sensitivity diagnostics are in stage04.
Do not tune from test values. Keep raw/unit explicit (official CNN native output
already normalized, so its native raw and unit copies should numerically agree).
Fitting retains legacy concatenated input contexts for control; valid masks mark
trial-interior common support for scoring. Trial-safe training is separate future
paired-control design, not a silent change to this campaign.

S11 first reporting collision was fixed after first frozen fit completed.29test
gate and unchanged training signature prove no retraining. Retain audit log.
Figures remain to render from frozen outputs with EXISTING paper-style renderers
and references (ratsFig1c-e,2b-d,ED4). Do not invent new figure styling.
No Git add/commit/push by agent; user is performing Git commands personally.
