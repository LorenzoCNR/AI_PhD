"""Encoder-independent reconstruction and controlled stable soft objective."""
from dataclasses import dataclass
import hashlib
import json
import math
import torch
import torch.nn.functional as F
from . import legacy_infonce, legacy_batch_similarity


def _positive(value, name):
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _validate(batch, specs):
    if not specs:
        raise ValueError("specs cannot be empty")
    count = None
    for spec in specs:
        value = batch[spec['key']]
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError("metadata must be finite tensors")
        if value.ndim not in (1, 2):
            raise ValueError("metadata must have shape N or NxD")
        if count is not None and len(value) != count:
            raise ValueError("metadata cardinalities differ")
        count = len(value)
        weight = float(spec.get('weight', 1.))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("weights must be finite and nonnegative")
    if sum(float(s.get('weight', 1.)) for s in specs) <= 0:
        raise ValueError("at least one weight must be positive")
    return count


def fit_distance_scales(batch, specs, *, split='train', source_ids=None):
    """Fit deterministic distance bounds on TRAIN or an explicit full recording."""
    if split not in ('train', 'full_recording'):
        raise ValueError("distance scales require train or explicit full_recording scope")
    count = _validate(batch, specs)
    if not count:
        raise ValueError("train metadata cannot be empty")
    scales = []
    for spec in specs:
        values = batch[spec['key']].detach().cpu().double()
        geometry = spec.get('geometry', 'continuous').lower()
        if geometry == 'temporal':
            scale = float(values.max() - values.min())
        elif geometry in ('continuous', 'euclidean'):
            if values.ndim == 1:
                values = values[:, None]
            p = 2. if geometry == 'euclidean' else float(spec.get('p', 2.))
            _positive(p, 'p')
            scale = float(torch.linalg.vector_norm(values.max(0).values-values.min(0).values, ord=p))
        elif geometry == 'categorical':
            scale = 1.
        elif geometry == 'circular':
            scale = int(spec.get('num_labels', 8))/2.
        else:
            raise ValueError(f"unsupported geometry {geometry}")
        constant = scale == 0
        scales.append(dict(key=spec['key'], geometry=geometry, scale=1. if constant else scale, constant=constant))
    if source_ids is None:
        raise ValueError("exact training source_ids are required for provenance")
    ids = source_ids.detach().cpu().tolist() if isinstance(source_ids, torch.Tensor) else list(source_ids)
    native_ids = []
    for value in ids:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric != int(value):
            raise ValueError("training source_ids must be finite integers")
        native_ids.append(int(value))
    ids = native_ids
    if len(ids) != count:
        raise ValueError("training source_ids cardinality mismatch")
    id_hash = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
    return dict(split=split,
                policy=('train_range_or_feature_bounding_box_no_clipping' if split == 'train'
                        else 'full_recording_range_or_feature_bounding_box_no_clipping'),
                scales=scales,
                train_id_sha256=id_hash if split == 'train' else None,
                full_recording_id_sha256=id_hash if split == 'full_recording' else None,
                train_count=count if split == 'train' else None,
                full_recording_count=count if split == 'full_recording' else None,
                specs=[dict(s) for s in specs])


@dataclass
class SoftResult:
    loss: torch.Tensor
    Q: torch.Tensor
    P: torch.Tensor
    W: torch.Tensor
    log_W: torch.Tensor
    valid_rows: torch.Tensor
    candidate_mask: torch.Tensor
    trial_id: torch.Tensor | None = None

    def diagnostics(self):
        def summarize(distribution, prefix):
            p = distribution.detach()[self.valid_rows]
            entropy = -(p * p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(1)
            concentration = p.square().sum(1)
            return {prefix+'_entropy_mean': float(entropy.mean()),
                    prefix+'_effective_neighbors_mean': float(entropy.exp().mean()),
                    prefix+'_concentration_mean': float(concentration.mean()),
                    prefix+'_inverse_participation_mean': float(concentration.reciprocal().mean()),
                    prefix+'_top1_mean': float(p.max(1).values.mean())}
        out = dict(valid_rows=int(self.valid_rows.sum()), skipped_rows=int((~self.valid_rows).sum()),
                   candidate_count_min=int(self.candidate_mask.sum(1).min()),
                   candidate_count_max=int(self.candidate_mask.sum(1).max()),
                   W_min=float(self.W.min()), W_max=float(self.W.max()),
                   W_row_mass_mean=float(self.W.sum(1).mean()))
        if self.valid_rows.any():
            out.update(summarize(self.Q, 'Q')); out.update(summarize(self.P, 'P'))
        if self.trial_id is not None:
            cross = self.trial_id[:, None] != self.trial_id[None, :]
            out['cross_trial_Q_mass_mean'] = float((self.Q.detach()*cross).sum(1)[self.valid_rows].mean()) if self.valid_rows.any() else 0.
        return out


def evaluate_soft(z, batch, specs, version='soft_v1_stable', scales=None,
                  temperature=.1, metadata_temperature=.5, candidate_mask=None, empty_rows='error',
                  allow_full_recording_scales=False):
    """Return loss and diagnostic distributions without changing the encoder."""
    if version not in ('soft_v1_frozen', 'soft_v1_stable'):
        raise ValueError("unknown soft version")
    if z.ndim != 2 or not len(z) or not torch.isfinite(z).all():
        raise ValueError("z must be finite nonempty NxD")
    if _validate(batch, specs) != len(z):
        raise ValueError("metadata and embedding cardinality mismatch")
    _positive(temperature, 'temperature'); _positive(metadata_temperature, 'metadata_temperature')
    if empty_rows not in ('error', 'skip'):
        raise ValueError("empty_rows must be error or skip")
    n = len(z)
    mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    if candidate_mask is not None:
        if version == 'soft_v1_frozen':
            raise ValueError("frozen version supports original diagonal exclusion only")
        if candidate_mask.shape != (n,n) or candidate_mask.dtype != torch.bool:
            raise ValueError("candidate_mask must be boolean NxN")
        mask &= candidate_mask.to(z.device)
    device_batch = {k: v.detach().to(z.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
    if version == 'soft_v1_frozen':
        similarity = legacy_batch_similarity.batch_structured_similarity_from_specs(
            device_batch, specs, tau=metadata_temperature, normalize=True).detach().to(z.dtype)
        loss = legacy_infonce.soft_contrastive_loss(z, similarity, temperature=temperature)
        W = similarity.masked_fill(~mask, 0)
        mass = W.sum(1, keepdim=True)
        valid = mass[:,0] > 1e-8
        Q = W/mass.clamp_min(1e-8)
        log_W = similarity.log().masked_fill(~mask, -torch.inf)
    else:
        valid_scope = (scales is not None and scales.get('split') == 'train')
        full_scope = (allow_full_recording_scales and scales is not None
                      and scales.get('split') == 'full_recording')
        if not (valid_scope or full_scope) or scales.get('specs') != [dict(s) for s in specs]:
            raise ValueError("stable version requires matching TRAIN scales or explicitly allowed full-recording scales")
        if len(scales['scales']) != len(specs):
            raise ValueError("scale cardinality mismatch")
        distance = torch.zeros((n,n), device=z.device, dtype=z.dtype)
        total = sum(float(s.get('weight',1.)) for s in specs)
        for spec, scale in zip(specs, scales['scales']):
            _positive(scale['scale'], 'distance scale')
            component = legacy_batch_similarity.batch_distance_from_spec(device_batch, spec).to(z.dtype)
            if not torch.isfinite(component).all():
                raise ValueError("nonfinite distance")
            distance += (float(spec.get('weight',1.))/total)*component/scale['scale']
        log_W = (-distance/metadata_temperature).masked_fill(~mask, -torch.inf).detach()
        valid = mask.any(1)
        if not valid.all() and empty_rows == 'error':
            raise ValueError("rows without valid candidates")
        if not valid.any():
            raise ValueError("no valid candidate rows")
        safe_log_W = torch.where(valid[:,None], log_W, torch.zeros_like(log_W))
        Q = F.softmax(safe_log_W, 1).masked_fill(~mask, 0).detach()
        W = log_W.exp()
    logits = legacy_infonce.pairwise_cosine_logits(z, temperature).masked_fill(~mask,-torch.inf)
    safe_logits = torch.where(mask.any(1)[:,None], logits, torch.zeros_like(logits))
    log_P = F.log_softmax(safe_logits, 1)
    P = log_P.exp().masked_fill(~mask, 0)
    if version == 'soft_v1_stable':
        safe_log_P = log_P.masked_fill(~mask, 0)
        loss = -(Q*safe_log_P).sum(1)[valid].mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite stable soft loss")
    return SoftResult(loss,Q,P,W,log_W,valid,mask,device_batch.get('trial_id'))


def soft_v1_frozen(z, batch, specs, **kwargs):
    return evaluate_soft(z,batch,specs,version='soft_v1_frozen',**kwargs)


def soft_v1_stable(z, batch, specs, **kwargs):
    return evaluate_soft(z,batch,specs,version='soft_v1_stable',**kwargs)
