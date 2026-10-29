"""Full-recording 8D Soft 2c losses for Buddy, Cicero and Gatsby."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from user_soft import full_recording_runner as runner
from user_soft import rat_runner as rr
from user_soft.loss import fit_distance_scales

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reproduction/figure2_bc/soft_2c_8d_other_rats'
FIG=ROOT/'results/figura_2_2026-10-28/c'
RATS=('buddy','cicero','gatsby')
CONDITIONS=('position','direction','position_direction','position_shuffled','direction_shuffled','position_direction_shuffled')
LABELS=('Position','Direction','Position + Direction','Position shuffled','Direction shuffled','Position + Direction shuffled')


def plot(rat,architecture):
    fig,ax=plt.subplots(figsize=(7,4.5));refs=[]
    for i,(condition,label) in enumerate(zip(CONDITIONS,LABELS)):
        folder=OUT/rat/condition/architecture/'seed_42';path=folder/'training_history.csv'
        m=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if m['actual_updates']!=10000 or runner.sha256(folder/'checkpoint.pt')!=m['checkpoint_sha256']:raise ValueError(f'Invalid fit {folder}')
        a=np.loadtxt(path,delimiter=',',skiprows=1)
        if a.shape!=(10000,2) or not np.isfinite(a).all():raise ValueError(f'Invalid loss {path}')
        ax.plot(a[:,0],a[:,1],color='gray' if i>=3 else 'deepskyblue',alpha=(.35,.65,1)[i%3],lw=.8,label=label)
        refs.append({'condition':condition,'loss':str(path.relative_to(ROOT)),'sha256':runner.sha256(path),
                     'manifest':str((folder/'manifest.json').relative_to(ROOT)),'checkpoint_sha256':m['checkpoint_sha256']})
    ax.set_xlabel('Update');ax.set_ylabel('Soft Stable training loss');ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();dest=FIG/rat.capitalize();dest.mkdir(parents=True,exist_ok=True)
    stem=dest/f'Figura_2c_{rat.capitalize()}_SOFT_{architecture.upper()}_loss'
    for ext in ('.png','.pdf','.svg'):
        path=stem.with_suffix(ext)
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=220,bbox_inches='tight')
    plt.close(fig)
    (dest/f'RIFERIMENTI_SOFT_{architecture.upper()}.json').write_text(json.dumps({'rat':rat,'seed':42,'source':refs,'fit_scope':'full recording','fit_dimension':8,'updates':10000},indent=2),encoding='utf-8')


def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(2)
    specs=[{'key':'behavior','geometry':'euclidean','weight':1.0}]
    for rat in RATS:
        rows,_=runner.load_full_recording(rat)
        for condition in CONDITIONS:
            observed=rows.behavior.copy()
            if condition.startswith('position_direction'):pass
            elif condition.startswith('position'):observed[:,1:]=0
            elif condition.startswith('direction'):observed[:,0]=0;observed[:,2]=0
            perm=np.random.default_rng(10042).permutation(len(observed)) if condition.endswith('_shuffled') else np.arange(len(observed))
            training=runner.rows_with_behavior(rows,observed[perm])
            values=rr.tensors(training,torch.device('cpu'))
            scales=fit_distance_scales({'behavior':values['behavior'],'time_id':values['time_id']},specs,split='full_recording',source_ids=rows.source_id)
            for arch in ('cnn','transformer'):
                cfg=runner.condition_config(arch,rat,'direction_position',embedding_temperature=1.)
                cfg.update(condition=condition,output_dimension=8,metadata_specs=specs,metadata_supervision=condition,
                           shuffle_seed=10042 if condition.endswith('_shuffled') else None,
                           condition_interpretation='Only indicated behavior columns used; shuffled rows together when applicable')
                sources=runner.source_record(arch);sources['experiment_script_sha256']=runner.sha256(Path(__file__))
                print('START',rat,condition,arch,flush=True)
                runner.fit_cell(OUT,rows,training,specs,scales,rat,condition,arch,cfg,sources,device)
        for arch in ('cnn','transformer'):plot(rat,arch)
        print('FINISHED RAT',rat,flush=True)


if __name__=='__main__':main()
