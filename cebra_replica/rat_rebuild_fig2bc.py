"""Render rat 2b/2c from completed, hash-verified fits; no fitting."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/rat_rebuild_20261027/Figure_2'
CEBRA = ROOT/'results/rat_paper_fig2bc_32d_v1'
SOFT = ROOT/'results/rat_paper_fig1e_ed4_6seed_v1/soft/stage01_fit'
HYBRID = ROOT/'results/rat_figure_gap_closure_v1/figure2b_fits'
RATS = ('achilles','buddy','cicero','gatsby')
CONDS = ('position','direction','position_direction','position_shuffled','direction_shuffled','position_direction_shuffled','time')
TITLES = {'position':'Position','direction':'Direction','position_direction':'Position + direction',
          'position_shuffled':'Position shuffled','direction_shuffled':'Direction shuffled',
          'position_direction_shuffled':'Position + direction shuffled','time':'Time'}
SC = {'time_only':'Time','direction_position':'Direction + position',
      'time_direction_position':'Time + direction + position'}

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def verified(path, expected):
    actual=digest(path)
    if actual != expected: raise RuntimeError(f'Hash mismatch: {path}')
    return {'path':str(path.relative_to(ROOT)), 'sha256':actual}

def save(fig, stem, provenance):
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths=[stem.with_suffix('.png'),stem.with_suffix('.pdf'),stem.with_suffix('.json')]
    if any(p.exists() for p in paths): raise FileExistsError(f'Refusing to overwrite {stem}')
    fig.savefig(paths[0],dpi=180,bbox_inches='tight'); fig.savefig(paths[1],bbox_inches='tight'); plt.close(fig)
    provenance['outputs']={p.name:digest(p) for p in paths[:2]}
    paths[2].write_text(json.dumps(provenance,indent=2),encoding='utf-8')

def render_2b(rat):
    panels=[]; parents=[]; reference=None
    for condition in CONDS:
        folder=CEBRA/rat/condition/'seed_42'; marker=folder/'manifest.json'
        rec=json.loads(marker.read_text()); path=folder/'all_embedding.npz'
        parents.extend([{'path':str(marker.relative_to(ROOT)),'sha256':digest(marker)},verified(path,rec['artifact_sha256'][path.name])])
        if rec['seed']!=42 or rec['condition']!=condition or rec['config']['parameters']['output_dimension']!=32: raise ValueError(marker)
        with np.load(path,allow_pickle=False) as a: z=a['embedding'].copy(); y=a['behavior'].copy(); ids=a['source_id'].copy()
        if z.shape!=(len(ids),32) or y.shape!=(len(ids),3) or len(np.unique(ids))!=len(ids) or not np.isfinite(z).all(): raise ValueError(path)
        if reference is None: reference=(ids,y)
        elif not (np.array_equal(ids,reference[0]) and np.array_equal(y,reference[1])): raise ValueError('2b source mismatch')
        panels.append((f'CEBRA: {TITLES[condition]}',z[:,:3],y))
    folder=HYBRID/rat/'hybrid_position_time_5d/seed_42'; marker=folder/'result.json'; rec=json.loads(marker.read_text()); path=folder/'all_embedding.npz'
    parents.extend([{'path':str(marker.relative_to(ROOT)),'sha256':digest(marker)},verified(path,rec['artifacts'][path.name])])
    with np.load(path,allow_pickle=False) as a: z=a['embedding'].copy(); y=a['behavior'].copy(); ids=a['source_id'].copy()
    if z.shape!=(len(ids),5) or not np.array_equal(ids,reference[0]) or not np.array_equal(y,reference[1]): raise ValueError('Hybrid support mismatch')
    panels.append(('CEBRA: Time + position (Hybrid)',z[:,[1,2,0]],y))
    fig=plt.figure(figsize=(16,8)); norm=Normalize(vmin=float(reference[1][:,0].min()),vmax=float(reference[1][:,0].max()))
    for i,(title,z,y) in enumerate(panels):
        ax=fig.add_subplot(2,4,i+1,projection='3d')
        for col,cmap in ((2,'cool'),(1,'viridis')):
            m=y[:,col]==1; ax.scatter(*z[m].T,c=y[m,0],cmap=cmap,norm=norm,s=1.0,alpha=.8,linewidths=0,rasterized=True)
        ax.set_title(title,fontsize=9); ax.set_axis_off()
    fig.suptitle(rat.title(),fontsize=13)
    for x,cmap,label in ((.92,'cool','Left'),(.94,'viridis','Right')):
        cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([x,.34,.012,.32]),orientation='vertical');cb.set_ticks([])
        fig.text(x+.006,.32,label,ha='center',va='top',fontsize=8)
    fig.subplots_adjust(top=.91,bottom=.04,right=.89,wspace=.02,hspace=.10)
    save(fig,OUT/'2B'/f'{rat}_cebra_vertical_close',{'rat':rat,'seed':42,'fit':'contiguous first 80%; not trial split','display':'all transformed rows; includes training and test',
         'conditions':list(CONDS)+['time_position_hybrid'],'dimensions':'32D except Hybrid 5D; projections, not separately fitted 3D','parents':parents})

def render_2c(rat):
    fig,axes=plt.subplots(1,3,figsize=(17,5)); parents=[]
    colors={'position':'#4477aa','direction':'#cc6677','position_direction':'#228833'}
    for cond in CONDS[:-1]:
        base=cond.replace('_shuffled',''); curves=[]
        for seed in range(42,48):
            folder=CEBRA/rat/cond/f'seed_{seed}'; marker=folder/'manifest.json'; rec=json.loads(marker.read_text()); p=folder/'loss_history.csv'
            parents.append(verified(p,rec['artifact_sha256'][p.name])); parents.append({'path':str(marker.relative_to(ROOT)),'sha256':digest(marker)})
            a=np.loadtxt(p,delimiter=',',skiprows=1)
            if a.shape!=(5000,2) or not np.array_equal(a[:,0],np.arange(1,5001)) or not np.isfinite(a).all(): raise ValueError(p)
            curves.append(a[:,1])
        vals=np.asarray(curves); mean=vals.mean(axis=0); sd=vals.std(axis=0)
        axes[0].plot(np.arange(1,5001),mean,color=colors[base],ls='--' if cond.endswith('shuffled') else '-',label=TITLES[cond])
        axes[0].fill_between(np.arange(1,5001),mean-sd,mean+sd,color=colors[base],alpha=.07)
    for j,arch in enumerate(('cnn','transformer'),1):
        for ci,(condition,title) in enumerate(SC.items()):
            folder=SOFT/condition/arch/rat/'fold_0/soft_v1_stable/seed_42'; marker=folder/'manifest.json'; cell=folder/'CELL_MANIFEST.json'
            record=json.loads(marker.read_text()); cellrec=json.loads(cell.read_text()); parents.append(verified(marker,cellrec['fit_manifest_sha256']))
            h=record['history']; updates=np.asarray([r['update'] for r in h]); values=np.asarray([r['train_loss'] for r in h])
            if len(h)!=10000 or not np.array_equal(updates,np.arange(1,10001)) or not np.isfinite(values).all(): raise ValueError(marker)
            axes[j].plot(updates,values,label=title,color=('#555555','#228833','#4477aa')[ci],lw=.9)
        axes[j].set_title(f'Soft {arch.upper()} · fold 0 · seed 42')
    axes[0].set_title('CEBRA · seeds 42–47 (mean ± SD)')
    for ax in axes: ax.set_xlabel('Optimizer update');ax.set_ylabel('Training objective');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(f'{rat.title()} | Figure 2c: training loss',fontsize=13);fig.tight_layout()
    save(fig,OUT/'2C'/f'{rat}_training_loss',{'rat':rat,'cebra_fit':'32D contiguous 80% training; 5000 updates; six seeds',
         'soft_fit':'3D trial split train; fold 0 seed 42; 10000 updates; no shuffled Soft fits claimed',
         'warning':'CEBRA and Soft use different losses, split policies and budgets; scales not directly comparable', 'parents':parents})

if __name__=='__main__':
    for rat in RATS:
        render_2b(rat);print('2b vertical',rat,flush=True)
