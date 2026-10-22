"""Downstream-only recovery: keep the original sealed fit-source signature."""
from __future__ import annotations
import csv
import io
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
from . import behavior_only_shuffle as campaign, behavior_only_runner as b, rat_runner as rr
from .data import sha256, config_hash
from .loss import evaluate_soft


def diagnostic(branch, embedding_folder, original, shuffled, permutation, parent, config, record):
    from .downstream import _LazyArchive
    path = branch/'stage04_diagnostics'/config['rat']/f"fold_{config['fold']}"/config['version']/f"seed_{config['seed']}"
    marker = path/'STAGE_SIGNATURE.json'
    signature = config_hash(dict(source=sha256(__file__), config=config, shuffle=record['signature'],
        embedding=sha256(embedding_folder/'train_embedding.npz'), scales=parent['train_scales']))
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved['signature'] != signature or any(sha256(path/n)!=h for n,h in saved['artifact_hashes'].items()):
            raise RuntimeError('Recovery diagnostic hash mismatch')
        return
    if path.exists():
        raise RuntimeError('Partial diagnostic output retained; inspect before recovery')
    if parent['train_scales'].get('split') != 'train' or original.split != 'train' or shuffled.split != 'train':
        raise ValueError('TRAIN diagnostic only')
    data = _LazyArchive(embedding_folder/'train_embedding.npz')
    if not np.array_equal(data['source_id'],original.source_id):
        raise RuntimeError('Diagnostic embedding source mismatch')
    if not np.array_equal(shuffled.behavior,original.behavior[permutation]):
        raise RuntimeError('Diagnostic shuffled target mismatch')
    ids = torch.randperm(len(data['embedding']),generator=torch.Generator().manual_seed(config['seed']+30000))[:64].numpy()
    z = torch.as_tensor(data['embedding'][ids])
    batch = {k:torch.as_tensor(getattr(shuffled,k)[ids]) for k in ('behavior','time_id','trial_id','source_id','session_id')}
    rows = []
    def probe(kind, meta=config['metadata_temperature'], embedding=config['embedding_temperature'], count=len(ids)):
        with torch.no_grad():
            result = evaluate_soft(z[:count],{k:v[:count] for k,v in batch.items()},b.SPECS,
                version=config['version'],scales=parent['train_scales'],temperature=embedding,metadata_temperature=meta)
        row = dict(probe=kind,split='train',seed=config['seed'],condition='behavior_only_shuffled',rows=count,
            metadata_temperature=meta,embedding_temperature=embedding,loss=float(result.loss),**result.diagnostics())
        rows.append(row)
        return result
    reference = probe('reference')
    for value in (.25,.5,1.):
        probe('metadata_temperature',meta=value)
    for value in (.05,.1,.2):
        probe('embedding_temperature',embedding=value)
    if len(ids)>32:
        probe('batch_composition_first32',count=32)
        probe('batch_composition_first64_or_available')
    path.mkdir(parents=True,exist_ok=False)
    with (path/'TRAIN_SENSITIVITY.csv').open('w',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(path/'TRAIN_REFERENCE_W_Q_P.npz',Q=reference.Q.numpy(),P=reference.P.numpy(),
        W=reference.W.numpy(),log_W=reference.log_W.numpy(),reference_train_row_indices=ids,
        behavior_donor_source_id=original.source_id[permutation[ids]],**{k:v.numpy() for k,v in batch.items()})
    rr.atomic_json(path/'PROVENANCE.json',dict(created_utc=rr.now(),config=config,source_sha256=sha256(__file__),
        loss_source_sha256=sha256(Path(__file__).with_name('loss.py')),shuffle_record=record,
        parent_embedding_sha256=sha256(embedding_folder/'train_embedding.npz'),train_scales=parent['train_scales'],
        metadata_specs=b.SPECS,no_time_contribution=True,no_scale_refit=True,no_test_access=True,
        weight_sensitivity='NOT_APPLICABLE: exactly one component; no added artificial component',
        probe_policy='existing temperature/composition grid, descriptive only; no model selection'))
    rr.atomic_json(marker,dict(signature=signature,artifact_hashes={p.name:sha256(p) for p in path.iterdir() if p.is_file()}))


def qualify():
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName('user_soft.tests.test_resume_behavior_shuffle'))
    evidence=dict(tests_run=result.testsRun,passed=result.wasSuccessful(),output=stream.getvalue(),
        created_utc=rr.now(),recovery_source_sha256=sha256(__file__),
        test_source_sha256=sha256(Path(__file__).parent/'tests/test_resume_behavior_shuffle.py'),
        original_fit_sources=campaign.source_record())
    if not result.wasSuccessful():
        rr.atomic_json(campaign.OUTPUT/f'RECOVERY_VALIDATION_FAILED_{time.time_ns()}.json',evidence)
        raise RuntimeError(stream.getvalue())
    state=json.loads((campaign.OUTPUT/'CAMPAIGN_STATE.json').read_text())
    fit_sources=campaign.source_record()
    reused=[]
    for key,entry in state['records'].items():
        folder=Path(entry['fit_manifest_path']).parent
        fit=json.loads((folder/'manifest.json').read_text())
        before=sha256(folder/'checkpoint.pt')
        with patch.object(rr,'make_encoder',side_effect=AssertionError('Recovery retraining forbidden')):
            checked,created=rr.fit_cell(folder,None,None,fit['config'],None,fit['parent_id'],fit_sources,torch.device('cpu'))
        if created or sha256(folder/'checkpoint.pt')!=before or checked['checkpoint_sha256']!=before:
            raise RuntimeError('Real checkpoint reuse verification failed')
        reused.append(dict(id=key,checkpoint_sha256=before,optimizer_updates_during_recovery=0))
    evidence.update(real_checkpoint_reuse=reused,decision='GO',
        before_after='two-component diagnostic rejects behavior-only -> isolated single-component adapter -> CPU end-to-end + exact checkpoint reuse pass')
    rr.atomic_json(campaign.OUTPUT/f'RECOVERY_VALIDATION_{time.time_ns()}.json',evidence)
    print(stream.getvalue(),flush=True)


def main():
    qualify()
    if '--qualify-only' in sys.argv:
        return
    original=campaign.diagnostic
    argv=sys.argv
    campaign.diagnostic=diagnostic
    try:
        sys.argv=[__file__,'--wait-and-execute']
        campaign.main()
    except BaseException as exc:
        rr.atomic_json(campaign.OUTPUT/'SEQUENCE_STATE.json',dict(status='failed',error=repr(exc),updated_utc=rr.now()))
        raise
    finally:
        campaign.diagnostic=original
        sys.argv=argv


if __name__=='__main__':
    main()
