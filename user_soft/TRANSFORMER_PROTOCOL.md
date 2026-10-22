# Recovered Transformer: controlled first replicas

The first authorized block is8 fits:4 rats x fold0 x seed42 x2 soft versions. No full240-fit Transformer campaign is launched by this runner. Both versions use the CNN's exact prepared TRAIN/validation/test caches, split/source IDs and TRAIN scale metadata. Only the architecture changes.

Architecture verified in original staged_shared_latent.py:104-107 and final Real trial_record.json for ft-final-w201-TOTAL65-transformer-soft-c3-s1201-9bdf50edb2dd: transformer_dim64,heads4,layers2,dropout0.1. The recovered TemporalTransformerEncoder uses sinusoidal positional encoding, TransformerEncoderLayer with GELU/feedforward256, temporal mean pooling, projection and L2 normalization. Output is32 for parity with the rat CNN, an explicit architecture/output mapping rather than the historical NeuroBridge3D output. The wrapper only transposes[B,N,10] to[B,10,N]; it does not modify time support or weights.

Input is the same exact10 rat bins used by offset10-model, indices[-5,+4] around baseline-clamped centers, at25ms/bin. No new longer observation window is introduced. The known legacy concatenated-split boundary context remains matched to the CNN and baseline; strict trial-interior masks are exported and reused in downstream scoring. TemporalTransformer sees the same observations; its receptive operation differs by architecture.

Control settings: Adam lr3e-4,betas0.9/0.999,eps1e-8,weight_decay0,batch512,10000 optimizer updates,terminal checkpoint,no early stopping/HPO; both soft versions unchanged. Metadata time and joint(position,right,left),weights0.5/0.5,tau_metadata0.5,tau_embedding0.1,cross-trial candidates unchanged. Training seed42 first; seeds42..51/folds0..2 describe the potential larger design but are not executed yet. Validation remains diagnostics only. No test outcome influences settings or model selection.

Runner reuses the root fit transaction/optimizer/RNG resume mechanism through explicit process-local encoder-factory dependency injection. The binding is restored in finally. The CNN process and files remain untouched. Signature records actual wrapper/config/function hashes and recovered source hash. A dedicated CPU qualification gate must verify dropout/RNG interrupted resume before GPU fits. Neither initial dry-run nor gate preparation launches a real fit.

New output root: results/user_soft_rat_transformer_v1. stage01_fit records checkpoints/history/compute; stage02_embedding exports native normalized raw and explicit unit representations plus trial/time/behavior/source IDs and validity masks; stage03_metrics reuses existing decoders; stage04_diagnostics reuses TRAIN probes. Tables identify TemporalTransformerEncoder explicitly. Native raw is already L2 normalized by recovered architecture, so raw/unit need not differ.

GPU fitting may run concurrently with the separate CNN campaign. Observed training/inference time and peak allocated/reserved memory remain recorded; they cannot establish an isolated architecture speed ranking in this context. Scientific embedding/decoder semantics remain matched. This caveat is included in config, embedding provenance and compute tables.

Commands from AI_PhD:

```powershell
python -m user_soft.transformer_runner --dry-run
python -m user_soft.transformer_runner --execute --max-new-fits 8
```

Execution requires both the dedicated matching-source Transformer GO gate and parent CNN GO gate. Existing complete fits/checkpoint hashes are verified and reused; mismatches stop instead of overwriting. The immutable plan includes only8 configurations. Additional Transformer replication requires a separately authorized campaign expansion.
