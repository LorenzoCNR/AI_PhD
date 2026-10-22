"""Approved monkey roster only; no fit is permitted without a behavioral target freeze."""
from __future__ import annotations
import itertools
import json
from . import rat_runner as rr
from .data import config_hash,sha256

DIMENSIONS=(3,4,8,16,32)
SEEDS=(42,43,44,45,46)
CONDITIONS=('time_only','joint','behavior_only')
ARCHITECTURES=('cnn','transformer')


def roster():
    entries=[]
    for seed,dimension,condition,architecture in itertools.product(SEEDS,DIMENSIONS,CONDITIONS,ARCHITECTURES):
        for shuffle in (False,True):
            if shuffle and condition=='time_only':
                continue
            role='behavior_shuffle' if shuffle else 'observed'
            key=f'{role}/{architecture}/{condition}/dim_{dimension}/seed_{seed}'
            entries.append(dict(trial_id=key,role=role,architecture=architecture,condition=condition,
                embedding_dimension=dimension,training_seed=seed,
                shuffle_seed=seed+10000 if shuffle else None))
    return entries


def require_target_freeze(specification):
    required=('behavior_target','session_universe','behavior_geometry','behavior_coordinate_encoding')
    if specification.get('approved') is not True or any(not specification.get(k) for k in required):
        raise RuntimeError('Monkey behavioral target not frozen; no automatic scientific defaults or fits')


def main():
    trials=roster()
    split=rr.ROOT/'results/monkey_figure3_v1/stage00_data/DATA_AND_SPLIT_PROVENANCE.json'
    record=dict(created_utc=rr.now(),status='DRAFT_TARGET_FREEZE_PENDING',execution_allowed=False,
        approved_counts=dict(observed=150,behavior_shuffle=100,total=250),
        dimensions=list(DIMENSIONS),seeds=list(SEEDS),conditions=list(CONDITIONS),
        architectures=list(ARCHITECTURES),trials=trials,roster_hash=config_hash(trials),
        source_sha256=sha256(__file__),frozen_split_metadata_sha256=sha256(split),
        behavior_target=None,session_universe=None,
        no_model_fit=True,optimizer_updates=0,no_test_scientific_outcomes_read=True,
        scope='one official split universe and one behavioral target; no multiplication by5CEBRA target tasks',
        unresolved=['behavior target and its coordinate encoding/geometry; active vsall universe',
            'controlled common support for comparison: old CEBRA exports mark allrowsvalid, rat-derived support excludes trial edges'],
        input_note='10bins at1ms are10ms onmonkey, not rat250ms; preserve official monkey offset10')
    output=rr.ROOT/'results/user_soft_monkey_v1'
    rr.atomic_json(output/'DRAFT_CAMPAIGN_PLAN.json',record)
    print(json.dumps(dict(status=record['status'],counts=record['approved_counts'],output=str(output),optimizer_updates=0)),flush=True)


if __name__=='__main__':
    main()
