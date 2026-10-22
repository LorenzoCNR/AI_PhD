"""Shared frozen-artifact readers for the explicitly requested rat figure revision."""
from __future__ import annotations
import csv
from functools import lru_cache
import hashlib
import inspect
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DELIVERY=ROOT/'results/RAT_FIGURES'
STAGE=DELIVERY/'.revision_20261021'
RATS=('achilles','buddy','cicero','gatsby')
SEEDS=tuple(range(42,48))
SOFT_KEYS=tuple((a,c) for a in ('CNN','Transformer') for c in ('joint','time_only','behavior_only'))
DISPLAY={(a,c):f'Soft {a}\n'+{'joint':'Time + Behavior','time_only':'Time','behavior_only':'Behavior'}[c] for a,c in SOFT_KEYS}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def csv_rows(path):
    if not Path(path).is_file():return []
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def write_csv(path,rows):
    if not rows:raise ValueError(f'No output rows:{path}')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


@lru_cache(maxsize=1)
def overnight_records():
    return json.loads((ROOT/'results/user_soft_rat_overnight_v1/CAMPAIGN_STATE.json').read_text())['records']


def soft_root(architecture,condition,rat,seed=42,fold=0,shuffle=False):
    architecture={'cnn':'CNN','CNN1D':'CNN','transformer':'Transformer'}.get(architecture,architecture)
    a='cnn' if architecture=='CNN' else 'transformer'
    cell=f'{rat}/fold_{fold}/soft_v1_stable/seed_{seed}'
    if condition=='behavior_only':
        name='user_soft_rat_behavior_only_shuffle_v1' if shuffle else 'user_soft_rat_behavior_only_v1'
        root=ROOT/'results'/name/a
        if (root/'stage01_fit'/cell/'manifest.json').is_file():return root
    family=f'shuffle_{a}' if shuffle else f'{a}_'+('joint' if condition=='joint' else 'time')
    key=f'{family}/{cell}'
    record=overnight_records().get(key)
    if record and record.get('status')=='complete':
        return Path(record['artifact_root'])
    candidates=[]
    if not shuffle:
        if a=='cnn':candidates=[ROOT/'results'/('user_soft_rat_v1' if condition=='joint' else 'user_soft_rat_time_only_v1')]
        else:candidates=[ROOT/'results/user_soft_rat_transformer_5seed_v1'/condition,
                         ROOT/'results/user_soft_rat_transformer_v1']
    for root in candidates:
        if (root/'stage01_fit'/cell/'manifest.json').is_file():return root
    raise FileNotFoundError(f'Frozen Soft cell missing:{architecture}/{condition}/{cell}/shuffle={shuffle}')


def fit_path(architecture,condition,rat,seed=42,fold=0,shuffle=False):
    return soft_root(architecture,condition,rat,seed,fold,shuffle)/'stage01_fit'/rat/f'fold_{fold}'/'soft_v1_stable'/f'seed_{seed}'/'manifest.json'


def fit_manifest(architecture,condition,rat,seed=42,fold=0,shuffle=False):
    path=fit_path(architecture,condition,rat,seed,fold,shuffle)
    return json.loads(path.read_text()),dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path))


def embedding(architecture,condition,rat,seed=42,fold=0,split='test',shuffle=False):
    root=soft_root(architecture,condition,rat,seed,fold,shuffle)
    folder=root/'stage02_embedding'/rat/f'fold_{fold}'/'soft_v1_stable'/f'seed_{seed}'
    path=folder/f'{split}_embedding.npz';marker=folder/'manifest.json'
    record=json.loads(marker.read_text());expected=record['embeddings'][split]['embedding_sha256']
    actual=sha(path)
    if actual!=expected:raise RuntimeError(f'Frozen embedding hash mismatch:{path}')
    with np.load(path,allow_pickle=False) as saved:
        arrays={k:saved[k].copy() for k in ('embedding_raw','embedding_unit','behavior','source_id','trial_id','time_id','valid_mask','session_id')}
    n=len(arrays['source_id'])
    if any(len(v)!=n for v in arrays.values()) or arrays['embedding_raw'].ndim!=2 or not np.isfinite(arrays['embedding_raw']).all():
        raise RuntimeError(f'Embedding cardinality/finiteness mismatch:{path}')
    if arrays['valid_mask'].dtype!=np.bool_ or len(np.unique(arrays['source_id']))!=n:raise RuntimeError('Invalid support metadata')
    return arrays,dict(path=path.relative_to(ROOT).as_posix(),sha256=actual,
        manifest=marker.relative_to(ROOT).as_posix(),manifest_sha256=sha(marker),split=split)


def save(fig,base,panel,parents=None,metadata=None):
    import matplotlib.pyplot as plt
    base=Path(base)
    if STAGE.resolve() not in base.resolve().parents:raise ValueError('Revision writes must stay in staging')
    base.parent.mkdir(parents=True,exist_ok=True)
    fig.patch.set_facecolor('white')
    for ext in ('png','pdf'):
        fig.savefig(base.with_suffix('.'+ext),dpi=220 if ext=='png' else None,bbox_inches='tight',facecolor='white',transparent=False)
    caller=Path(inspect.stack()[1].filename)
    record=dict(created_utc=datetime.now(timezone.utc).isoformat(),panel=panel,parents=parents or [],
        renderer_sha256=sha(caller),common_reader_sha256=sha(__file__),encoder_retrained=False,
        output_hashes={ext:sha(base.with_suffix('.'+ext)) for ext in ('png','pdf')})
    record.update(metadata or {})
    base.with_name(base.name+'_PROVENANCE.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    plt.close(fig)
