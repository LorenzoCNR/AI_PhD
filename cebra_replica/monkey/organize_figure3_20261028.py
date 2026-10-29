"""Organize Figure 3 plots from saved local fits; never retrain or delete."""
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plot_figure3b_comparison as comparison
import plot_figure3_paper as paper
from figure3_style import DIRECTION_COLORS

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/figura_3_2026-10-28'
RENDER=ROOT/'results/monkey_figure3_paper_v1/figure3_render_20261028'
CORRECTED=ROOT/'results/monkey_figure3_paper_v1/figure3g_position_only_correction_v1/stage02_embedding/figure3g_all_position_only_4d_corrected/seed_42'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def copy_pair(src,dst):
    for ext in ('.png','.pdf'):
        a=src.with_suffix(ext);b=dst.with_suffix(ext)
        if not a.is_file():raise FileNotFoundError(a)
        if b.exists() and sha(a)!=sha(b):raise FileExistsError(b)
        if not b.exists():shutil.copy2(a,b)

def paper_3b():
    output=OUT/'b';output.mkdir(parents=True,exist_ok=True)
    prior=output/'RIFERIMENTI_CEBRA_3b.json'
    if prior.is_file() and all((output/f'Figura_3b_CEBRA_convpiVAE_tSNE_UMAP.{ext}').is_file() for ext in ('png','pdf')):
        return
    base=comparison.ROOT/'stage02_embedding'; baselines=comparison.ROOT/'stage02_embedding_baselines'
    specs=[
      ('CEBRA\nDirection',base/'figure3b_e_active_target_4d/seed_42/all_trials.npz','embedding_raw',base/'figure3b_e_active_target_4d/seed_42/EMBEDDING_PROVENANCE.json',(2,0),False),
      ('conv-piVAE\nDirection: con label',baselines/'conv_pivae/seed_42/all_trials.npz','with_label_raw',baselines/'conv_pivae/seed_42/EMBEDDING.json',(2,3),True),
      ('CEBRA\nTime',base/'figure3b_active_time_4d/seed_42/all_trials.npz','embedding_raw',base/'figure3b_active_time_4d/seed_42/EMBEDDING_PROVENANCE.json',(0,1),False),
      ('conv-piVAE\nDirection: senza label',baselines/'conv_pivae/seed_42/all_trials.npz','without_label_raw',baselines/'conv_pivae/seed_42/EMBEDDING.json',(2,3),True),
      ('t-SNE',baselines/'opentsne/seed_42/all_trials.npz','embedding_raw',baselines/'opentsne/seed_42/EMBEDDING.json',(0,1),True),
      ('UMAP',baselines/'umap/seed_42/all_trials.npz','embedding_raw',baselines/'umap/seed_42/EMBEDDING.json',(0,1),True)]
    prov=json.loads(comparison.TRIAL_PROVENANCE.read_text(encoding='utf8'))
    ids=np.asarray(prov['session_trial_indices_by_split']['active']['all'],dtype=np.int64); length=int(prov['trial_length_bins'])
    keys=np.sort(np.repeat(ids,length)*length+np.tile(np.arange(length,dtype=np.int64),len(ids)))
    loaded=[comparison.read(path,key,manifest,ordinal,ids,length,keys) for _,path,key,manifest,_,ordinal in specs]
    if not all(np.array_equal(t[2],loaded[0][2]) and np.array_equal(t[1],loaded[0][1]) for t in loaded):raise ValueError('Support mismatch')
    fig,axes=plt.subplots(1,6,figsize=(24,4.8))
    for ax,(title,_,_,_,coords,_),(embedding,direction,_,_) in zip(axes,specs,loaded):
        trials=embedding.reshape(-1,length,embedding.shape[1]);labels=direction.reshape(-1,length)[:,0]
        for label in range(8):
            mean=trials[labels==label].mean(axis=0);ax.plot(mean[:,coords[0]],mean[:,coords[1]],color=DIRECTION_COLORS[label],lw=2)
        ax.set_title(title,fontsize=12,pad=12);ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
    fig.subplots_adjust(left=.02,right=.98,top=.83,bottom=.06,wspace=.32)
    stem=output/'Figura_3b_CEBRA_convpiVAE_tSNE_UMAP'
    for ext in ('png','pdf'):
        path=stem.with_suffix('.'+ext)
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=220,bbox_inches='tight')
    plt.close(fig)
    ref={'models':[{'title':t,'embedding':str(p.relative_to(ROOT)),'sha256':r[3],'coordinates_zero_indexed':coords} for (t,p,k,m,coords,o),r in zip(specs,loaded)],
         'note':'CEBRA fit full active; conv-piVAE fit train only, full active transformed; UMAP/tSNE fit full active. Direction post-hoc average. UMAP/tSNE parameters are rat-derived; autoLFADS unavailable.',
         'script':str(Path(__file__).relative_to(ROOT)),'reference':str(comparison.TRIAL_PROVENANCE.relative_to(ROOT))}
    (output/'RIFERIMENTI_CEBRA_3b.json').write_text(json.dumps(ref,indent=2),encoding='utf8')

def main():
    refs={}
    for panel in 'abcdefg':(OUT/panel).mkdir(parents=True,exist_ok=True)
    for p in 'acdef':
        src=RENDER/('FIGURE3'+p.upper());dst=OUT/p/('Figura_3'+p+'_CEBRA' if p!='a' else 'Figura_3a_movimenti_attivi_passivi')
        copy_pair(src,dst);refs[str(dst.relative_to(ROOT))]={'source':str(src.relative_to(ROOT)),'png_sha256':sha(src.with_suffix('.png'))}
    paper_3b()
    for p in 'bcdefg':
        for seed in (42,43,44,45,46):
            src=RENDER/f'FIGURE3{p.upper()}_SOFT_FULL_SAMPLE_SEED_{seed}'
            desc={'b':'Time_Position_Time_e_Position','c':'Position_XY','d':'Time','e':'Position_XY_solo_attivo','f':'Time_e_Position_XY_solo_attivo','g':'Position_XY_solo_attivo'}[p]
            dst=OUT/p/f'Figura_3{p}_Soft_CNN_Transformer_{desc}_seed_{seed}'
            copy_pair(src,dst);refs[str(dst.relative_to(ROOT))]={'source':str(src.relative_to(ROOT)),'png_sha256':sha(src.with_suffix('.png'))}
    paper.FIGURES=OUT/'g'
    prior=paper.load
    def corrected(name):
        if name=='corrected_position_only':
            path=CORRECTED/'all_trials.npz'
            rec=json.loads((CORRECTED/'EMBEDDING_PROVENANCE.json').read_text(encoding='utf8'))
            if sha(path)!=rec['embedding_sha256']:raise ValueError('Corrected 3g embedding hash mismatch')
            with np.load(path) as v:return {k:v[k] for k in v.files}
        return prior(name)
    paper.load=corrected
    paper.plot_3efg('corrected_position_only',None,'CEBRA — Position X,Y',(0,1,2),'row','Figura_3g_CEBRA_Position_XY')
    refs['g/Figura_3g_CEBRA_Position_XY']={'source':str((CORRECTED/'all_trials.npz').relative_to(ROOT)),'sha256':sha(CORRECTED/'all_trials.npz'),'fit':'position only; active + passive, 4D'}
    for panel in 'acdefg':
        subset={k:v for k,v in refs.items() if ('/'+panel+'/' in k.replace('\\','/') or k.startswith(panel+'/'))}
        (OUT/panel/'RIFERIMENTI.json').write_text(json.dumps(subset,indent=2),encoding='utf8')
    print('OK organized',len(refs),'figure stems',flush=True)

if __name__=='__main__':main()
