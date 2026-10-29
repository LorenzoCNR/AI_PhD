"""Figure 2b Soft from the shared 1e/ED4 embeddings; no inference or fit."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from rat_rebuild_fig2d import ROOT,OUT,RATS,SC,cache,load_soft,exclusive_json

def main():
    for rat in RATS:
        data=cache(rat,0); mask=data['test']['valid_mask']; y=data['test']['behavior'][mask]
        norm=Normalize(float(y[:,0].min()),float(y[:,0].max()))
        for arch in ('cnn','transformer'):
            panels=[];parents=[]
            for cond,title in SC.items():
                result=load_soft(rat,0,42,cond,arch,data)
                if result is None: raise FileNotFoundError(f'Missing shared embeddings {rat}/{arch}/{cond}')
                arr,paths=result; z=arr['test'][mask]
                if z.shape!=(len(y),3) or not np.isfinite(z).all():raise ValueError('Soft embedding shape or finiteness')
                panels.append((title,z));parents+=paths
            fig=plt.figure(figsize=(13,4.8))
            for i,(title,z) in enumerate(panels):
                ax=fig.add_subplot(1,3,i+1,projection='3d')
                for col,cmap in ((2,'cool'),(1,'viridis')):
                    selected=y[:,col]==1
                    ax.scatter(*z[selected].T,c=y[selected,0],cmap=cmap,norm=norm,s=2,alpha=.8,linewidths=0,rasterized=True)
                ax.set_title(f'Soft {arch.upper()}: {title}',fontsize=10);ax.set_axis_off()
            fig.suptitle(rat.title(),fontsize=12)
            fig.subplots_adjust(bottom=.05,top=.86,right=.87,wspace=.02)
            for x,cmap,label in ((.90,'cool','Left'),(.92,'viridis','Right')):
                cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([x,.29,.014,.42]),orientation='vertical');cb.set_ticks([])
                fig.text(x+.007,.27,label,ha='center',va='top',fontsize=8,rotation=90)
            dest=OUT.parent/'2B';dest.mkdir(parents=True,exist_ok=True);stem=dest/f'{rat}_soft_{arch}_vertical_close_v2'
            for ext in ('.png','.pdf'):
                p=stem.with_suffix(ext)
                if p.exists():raise FileExistsError(p)
                fig.savefig(p,dpi=180,bbox_inches='tight')
            plt.close(fig)
            exclusive_json(stem.with_suffix('.json'),{'rat':rat,'architecture':arch,'seed':42,'fold':0,
                'embedding_dimension':3,'plot_coordinates':[0,1,2],'fit_split':'training trials',
                'display_split':'held-out test trials; within-trial valid context only',
                'comparison_warning':'CEBRA 2b gallery displays all transformed rows and has a different fit/split protocol.',
                'parents':parents})
            print(rat,arch,flush=True)

if __name__=='__main__':main()
