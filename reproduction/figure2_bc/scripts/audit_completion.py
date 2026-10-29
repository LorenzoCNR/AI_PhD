"""Read-only validation of completed Figure 2b/2c experiment artifacts."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'reproduction/figure2_bc'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def check(path,dimension,soft):
    record=json.loads(path.read_text(encoding='utf-8'))
    cp=path.parent/'checkpoint.pt'
    embedding=path.parent/('embedding_full.npz' if soft else 'embedding.npz')
    expected_cp=record.get('checkpoint_sha256') if soft else record['sha256']['checkpoint.pt']
    expected_emb=record.get('embedding_sha256') if soft else record['sha256']['embedding.npz']
    updates=record.get('actual_updates') if soft else record['steps']
    with np.load(embedding,allow_pickle=False) as a:
        z=a['embedding']; shape=list(z.shape); finite=bool(np.isfinite(z).all())
    valid=sha(cp)==expected_cp and sha(embedding)==expected_emb and shape==[10178,dimension] and finite and updates==10000
    return {'manifest':str(path.relative_to(ROOT)),'embedding_shape':shape,'finite':finite,'updates':updates,'hash_ok':valid}


def main():
    records=[]
    for condition in ('behavior_position_input','time','position'):
        dim=2 if condition=='behavior_position_input' else 3
        records.append(check(BASE/'full'/condition/'seed_42/manifest.json',dim,False))
    for condition in ('behavior_position_direction_input','time_position_5d','position_shuffled','position_direction','position_direction_shuffled'):
        dim=2 if condition=='behavior_position_direction_input' else (5 if condition=='time_position_5d' else 3)
        records.append(check(BASE/'figure2b_missing/full'/condition/'seed_42/manifest.json',dim,False))
    for condition in ('position','direction_paper_demo','position_direction','position_shuffled','direction_shuffled_paper_demo','position_direction_shuffled','direction_controlled','direction_shuffled_controlled'):
        records.append(check(BASE/'figure2c_8d/full'/condition/'seed_42/manifest.json',8,False))
    for condition in ('position_input','position_direction_input'):
        for arch in ('cnn','transformer'):
            records.append(check(BASE/'soft_behavior_2d_v2/full/achilles'/condition/arch/'seed_42/manifest.json',2,True))
    for condition in ('position','direction','position_direction','position_shuffled','direction_shuffled','position_direction_shuffled'):
        for arch in ('cnn','transformer'):
            records.append(check(BASE/'soft_2c_8d/full/achilles'/condition/arch/'seed_42/manifest.json',8,True))
    for condition in ('position','position_shuffled','position_direction_shuffled'):
        for arch in ('cnn','transformer'):
            records.append(check(BASE/'soft_2b_shuffle_3d/full/achilles'/condition/arch/'seed_42/manifest.json',3,True))
    for condition in ('time_only','direction_position','time_direction_position'):
        for arch in ('cnn','transformer'):
            records.append(check(ROOT/'results/user_soft_rat_fullrecording_t1_3conditions_seed42_v1/achilles'/condition/arch/'seed_42/manifest.json',3,True))
    result={'complete':all(r['hash_ok'] for r in records),'count':len(records),'records':records}
    print(json.dumps(result,indent=2))
    if not result['complete']: raise SystemExit(1)


if __name__=='__main__':main()
