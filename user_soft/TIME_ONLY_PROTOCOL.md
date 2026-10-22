# Pure time soft CNN control

Approved new condition: the metadata target uses only official trial-relative time_id in25ms bins, with temporal absolute distance and weight1. Behavior position/right/left remains exported for downstream decoding but does not enter W,Q or model inputs. Cross-trial relationships at equal/similar local trial time remain intentional. No same-trial restriction is imposed. This is a time-metadata soft cross-entropy experiment, not a reconstruction of original CEBRA-Time's sampled offset-positive InfoNCE objective.

Frozen version scales local-time distances by current batch maximum, exactly as the generic original API. Stable version projects the existing complete TRAIN time scale from the parent joint scale record, retaining TRAIN ID hash, constant-component flag and exact numeric bound. The projected record adds an explicit parent-scale hash and one-spec list; no train/validation/test scale is re-estimated. Q/P exclude diagonal; logits, temperatures and eligibility are unchanged.

Same official offset10-model architecture,32 units/32 outputs, exact10-bin contexts[-5,+4] and legacy_concat input construction; Adam3e-4,betas0.9/0.999,eps1e-8,weight_decay0,batch512,10000 updates,no early stopping/HPO,tau_metadata0.5,tau_embedding0.1. Design:4 rats x3 official folds x7 seeds42-48 x2 loss versions=168 new fits. These are separate models because the training target changed. No parent model is overwritten or invalidated.

Parent caches at results/user_soft_rat_v1/stage00_data are checked using the original unbound preparation function before metadata-spec injection. Parent source, official split and cache content hashes remain unchanged. New stage00 manifest links the original files and projected scale provenance. The separate process changes only rat_runner.SPECS within a finally-restored context; existing CNN/Transformer processes and source files are untouched. Dedicated source-signature GO gate and CPU qualification are required before actual training.

S10 discovered before implementation: the existing joint diagnostic writer requires two metadata components and dual-weight probes, so feeding one-spec metadata would fail downstream after a valid fit. This is an API scope limitation, not a training/metric bug. The time-only runner implements its own one-component diagnostic writer without editing the running parent sources. It records the same W/Q/P, entropy/concentration/effective-neighbor diagnostics, prespecified temperature/composition probes, input IDs and parent hashes. A one-component normalized weight is always1; no meaningless dual-weight probe is added.

Output root: results/user_soft_rat_time_only_v1, staged data pointers, checkpoints/history, frozen raw/unit embeddings, decoder metrics, TRAIN reference diagnostics, comparison/compute tables and manifest. Raw is native normalized official encoder output; unit is explicit normalization. Joint metadata and all original IDs remain in exports for scoring with the same interior-valid support, while loss construction consumes time only. New comparison rows explicitly identify condition=time_only and architecture=offset10-model.

```powershell
python -m user_soft.time_only_runner --dry-run
python -m user_soft.time_only_runner --execute --max-new-fits 168
```

Dry-run produces zero optimizer updates. Execution is not launched by code construction or qualification. Completed fits/resumable checkpoints are reused under the exact source/config/data signature; mismatches require a new branch rather than overwrite. Seed variability, windows/support and decoders remain matched to the joint experiment. Test results never choose this control's settings.
