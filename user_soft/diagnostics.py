"""Prespecified TRAIN-only objective diagnostics; never select hyperparameters."""
import copy
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from .loss import evaluate_soft


def write_diagnostics(output, z, batch, specs, scales, version, seed):
    """Record reference matrices and sensitivity on caller-supplied TRAIN rows.

    Scales retain the complete TRAIN fit bounds/ID hash. Sensitivity changes
    metadata/embedding temperatures and weights as descriptive probes only.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"immutable diagnostics output exists: {output}")
    if scales.get('split') != 'train':
        raise ValueError('diagnostics require TRAIN fitted scale provenance')
    if len(specs) != 2:
        raise ValueError('prespecified rat sensitivity requires two metadata components')
    if len(z) < 2:
        raise ValueError('diagnostics need at least two TRAIN rows')
    supplied_split = batch.get('split')
    if supplied_split is not None:
        entries = [supplied_split] if isinstance(supplied_split,str) else list(supplied_split)
        if any(value != 'train' for value in entries):
            raise ValueError('diagnostic inputs must contain TRAIN rows only')
    n = min(64, len(z))
    ids = torch.arange(n, device=z.device)
    def subset(count):
        result = {}
        for key,value in batch.items():
            if isinstance(value,torch.Tensor) and value.ndim and len(value)==len(z):
                result[key] = value[:count]
        return result
    base_batch = subset(n)
    rows = []
    def probe(kind, metadata_temperature=.5, temperature=.1, weights=None, count=n):
        current_specs = copy.deepcopy(specs)
        current_scales = copy.deepcopy(scales)
        if weights is not None:
            for spec,weight in zip(current_specs,weights):
                spec['weight'] = weight
            current_scales['specs'] = copy.deepcopy(current_specs)
        with torch.no_grad():
            result = evaluate_soft(z[:count],subset(count),current_specs,
                version=version,scales=current_scales,temperature=temperature,
                metadata_temperature=metadata_temperature)
        row = dict(probe=kind,version=version,seed=int(seed),split='train',rows=count,
                   metadata_temperature=metadata_temperature,embedding_temperature=temperature,
                   weight0=current_specs[0].get('weight',1.),weight1=current_specs[1].get('weight',1.),
                   loss=float(result.loss),pair01_log_W=float(result.log_W[0,1]),
                   pair01_Q=float(result.Q[0,1]))
        row.update(result.diagnostics())
        rows.append(row)
        return result
    reference = probe('reference')
    for value in (.25,.5,1.):
        probe('metadata_temperature',metadata_temperature=value)
    for value in (.05,.1,.2):
        probe('embedding_temperature',temperature=value)
    for weights in ((0.,1.),(.5,.5),(1.,0.)):
        probe('metadata_weights',weights=weights)
    if n > 32:
        probe('batch_composition_first32',count=32)
        probe('batch_composition_first64_or_available',count=n)
    payload = dict(Q=reference.Q.detach().cpu().numpy(),P=reference.P.detach().cpu().numpy(),
                   W=reference.W.detach().cpu().numpy(),log_W=reference.log_W.detach().cpu().numpy(),
                   reference_input_indices=ids.cpu().numpy())
    for key,value in base_batch.items():
        payload[key] = value.detach().cpu().numpy()
    def tensor_hash(value):
        values = value.detach().cpu().contiguous().numpy()
        return hashlib.sha256(str(values.dtype).encode()+str(values.shape).encode()+values.tobytes()).hexdigest()
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(),version=version,seed=int(seed),
        split='train',scope='descriptive prespecified probes; no hyperparameter selection',
        parent_embedding_sha256=tensor_hash(z),parent_metadata_hashes={key:tensor_hash(value) for key,value in batch.items() if isinstance(value,torch.Tensor)},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        loss_source_sha256=hashlib.sha256(Path(__file__).with_name('loss.py').read_bytes()).hexdigest(),
        train_scales=scales,reference_input_indices=list(range(n)),
        config=dict(metadata_temperatures=[.25,.5,1.],embedding_temperatures=[.05,.1,.2],
                    weights=[[0.,1.],[.5,.5],[1.,0.]],batch_composition='first32 versus first64_or_available TRAIN rows',
                    scale_reestimation=False),test_selection=False)
    output.mkdir(parents=True,exist_ok=False)
    with (output/'TRAIN_SENSITIVITY.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(output/'TRAIN_REFERENCE_W_Q_P.npz',**payload)
    (output/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    return dict(output=str(output),probes=len(rows),reference_rows=n)
