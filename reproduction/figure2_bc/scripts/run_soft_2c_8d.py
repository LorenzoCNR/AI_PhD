"""Full-recording Soft 8D hypothesis experiments for Achilles, seed 42."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from user_soft import full_recording_runner as runner
from user_soft import rat_runner as rr
from user_soft.loss import fit_distance_scales

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reproduction/figure2_bc/soft_2c_8d'
CONDITIONS=('position','direction','position_direction','position_shuffled','direction_shuffled','position_direction_shuffled')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true'); args=parser.parse_args()
    output=OUT/('smoke' if args.smoke else 'full')
    rows, source=runner.load_full_recording('achilles')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2)
    specs=[{'key':'behavior','geometry':'euclidean','weight':1.0}]
    for condition in CONDITIONS:
        observed=rows.behavior.copy()
        if condition.startswith('position_direction'): pass
        elif condition.startswith('position'): observed[:,1:]=0
        elif condition.startswith('direction'): observed[:,0]=0; observed[:,2]=0
        permutation=np.random.default_rng(10042).permutation(len(observed)) if condition.endswith('_shuffled') else np.arange(len(observed))
        training=runner.rows_with_behavior(rows,observed[permutation])
        values=rr.tensors(training,torch.device('cpu'))
        scales=fit_distance_scales({'behavior':values['behavior'],'time_id':values['time_id']},specs,split='full_recording',source_ids=rows.source_id)
        for architecture in ('cnn','transformer'):
            cfg=runner.condition_config(architecture,'achilles','direction_position',embedding_temperature=1.)
            cfg.update(condition=condition,output_dimension=8,metadata_specs=specs,max_updates=10 if args.smoke else 10000,
                       checkpoint_interval=10 if args.smoke else 500,diagnostic_interval=10 if args.smoke else 500,
                       metadata_supervision=condition,shuffle_seed=10042 if condition.endswith('_shuffled') else None,
                       condition_interpretation='Only indicated behavior columns used; shuffled rows together when applicable')
            sources=runner.source_record(architecture)
            sources['experiment_script_sha256']=runner.sha256(Path(__file__))
            print(f'START {condition}/{architecture}',flush=True)
            runner.fit_cell(output,rows,training,specs,scales,'achilles',condition,architecture,cfg,sources,device)
    print('COMPLETE Soft 2c 8D',flush=True)
    if not args.smoke:
        shuffle_out=ROOT/'reproduction/figure2_bc/soft_2b_shuffle_3d/full'
        shuffled_rows, shuffle_specs, shuffle_scales, _ = runner.prepare_condition(
            shuffle_out,rows,'achilles','position_direction_shuffled')
        for architecture in ('cnn','transformer'):
            cfg=runner.condition_config(architecture,'achilles','position_direction_shuffled',embedding_temperature=1.)
            sources=runner.source_record(architecture)
            sources['experiment_script_sha256']=runner.sha256(Path(__file__))
            print(f'START 2b time + position + direction shuffled/{architecture}',flush=True)
            runner.fit_cell(shuffle_out,rows,shuffled_rows,shuffle_specs,shuffle_scales,
                            'achilles','position_direction_shuffled',architecture,cfg,sources,device)
        for condition in ('position','position_shuffled'):
            observed=rows.behavior.copy(); observed[:,1:]=0
            permutation=np.random.default_rng(10042).permutation(len(observed)) if condition.endswith('_shuffled') else np.arange(len(observed))
            training=runner.rows_with_behavior(rows,observed[permutation])
            values=rr.tensors(training,torch.device('cpu'))
            scales=fit_distance_scales({'behavior':values['behavior'],'time_id':values['time_id']},specs,split='full_recording',source_ids=rows.source_id)
            for architecture in ('cnn','transformer'):
                cfg=runner.condition_config(architecture,'achilles','direction_position',embedding_temperature=1.)
                cfg.update(condition=condition,metadata_specs=specs,metadata_supervision=condition,
                           shuffle_seed=10042 if condition.endswith('_shuffled') else None,
                           condition_interpretation='Position only; position permuted for shuffled condition')
                sources=runner.source_record(architecture); sources['experiment_script_sha256']=runner.sha256(Path(__file__))
                print(f'START 2b {condition}/{architecture}',flush=True)
                runner.fit_cell(shuffle_out,rows,training,specs,scales,'achilles',condition,architecture,cfg,sources,device)
        roots={'CEBRA 2c 8D':ROOT/'reproduction/figure2_bc/figure2c_8d/full',
               'CEBRA 2b aggiuntivi':ROOT/'reproduction/figure2_bc/figure2b_missing/full',
               'Soft 2b input comportamentale 2D':ROOT/'reproduction/figure2_bc/soft_behavior_2d_v2/full',
               'Soft 2c 8D':output,'Soft 2b 3D aggiuntivi':shuffle_out}
        expected={'CEBRA 2c 8D':8,'CEBRA 2b aggiuntivi':5,
                  'Soft 2b input comportamentale 2D':4,'Soft 2c 8D':12,'Soft 2b 3D aggiuntivi':6}
        audit={}
        for label,root in roots.items():
            manifests=sorted(p for p in root.rglob('manifest.json') if (p.parent/'checkpoint.pt').exists()) if root.exists() else []
            records=[]
            for path in manifests:
                record=json.loads(path.read_text(encoding='utf-8'))
                checkpoint=path.parent/'checkpoint.pt'
                expected_hash=record.get('checkpoint_sha256') or record.get('sha256',{}).get('checkpoint.pt')
                valid=checkpoint.exists() and expected_hash is not None and runner.sha256(checkpoint)==expected_hash
                embedding_path=path.parent/('embedding_full.npz' if 'Soft' in label else 'embedding.npz')
                embedding_hash=record.get('embedding_sha256') or record.get('sha256',{}).get('embedding.npz')
                embedding_ok=embedding_path.exists() and embedding_hash is not None and runner.sha256(embedding_path)==embedding_hash
                expected_dim=(8 if '2c' in label else (2 if 'comportamentale' in label else 3))
                if label=='CEBRA 2b aggiuntivi':
                    expected_dim=record.get('parameters',{}).get('output_dimension')
                shape=None; finite=False
                if embedding_ok:
                    with np.load(embedding_path,allow_pickle=False) as archive:
                        z=archive['embedding']; shape=list(z.shape); finite=bool(np.isfinite(z).all())
                updates=record.get('actual_updates',record.get('steps'))
                valid=bool(valid and embedding_ok and shape==[len(rows.neural),expected_dim] and finite and updates==10000)
                records.append({'manifest':str(path.relative_to(ROOT)),'checkpoint_hash_ok':valid,
                                'embedding_sha256_ok':bool(embedding_ok),'shape':shape,'updates':updates})
            audit[label]={'expected':expected[label],'found':len(records),'records':records,
                          'complete':len(records)==expected[label] and all(r['checkpoint_hash_ok'] for r in records)}
        existing=[]
        for name in ('behavior_position_input','time','position'):
            path=ROOT/'reproduction/figure2_bc/full'/name/'seed_42'/'manifest.json'
            if path.exists():
                record=json.loads(path.read_text(encoding='utf-8'))
                checkpoint=path.parent/'checkpoint.pt'
                valid=checkpoint.exists() and runner.sha256(checkpoint)==record.get('sha256',{}).get('checkpoint.pt')
                existing.append({'manifest':str(path.relative_to(ROOT)),'checkpoint_hash_ok':valid})
        audit['CEBRA 2b fit riutilizzati']={'expected':3,'found':len(existing),'records':existing,
                                            'complete':len(existing)==3 and all(r['checkpoint_hash_ok'] for r in existing)}
        existing=[]
        for condition in ('time_only','direction_position','time_direction_position'):
            for architecture in ('cnn','transformer'):
                path=ROOT/'results/user_soft_rat_fullrecording_t1_3conditions_seed42_v1'/'achilles'/condition/architecture/'seed_42'/'manifest.json'
                if path.exists():
                    record=json.loads(path.read_text(encoding='utf-8')); checkpoint=path.parent/'checkpoint.pt'
                    valid=checkpoint.exists() and runner.sha256(checkpoint)==record.get('checkpoint_sha256')
                    existing.append({'manifest':str(path.relative_to(ROOT)),'checkpoint_hash_ok':valid})
        audit['Soft 2b fit riutilizzati']={'expected':6,'found':len(existing),'records':existing,
                                           'complete':len(existing)==6 and all(r['checkpoint_hash_ok'] for r in existing)}
        audit['complete']=all(v['complete'] for v in audit.values())
        report=ROOT/'reproduction/figure2_bc/RESOCONTO_RUN.json'
        report.write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
        print('AUDIT COMPLETO',audit['complete'],flush=True)


if __name__=='__main__': main()
