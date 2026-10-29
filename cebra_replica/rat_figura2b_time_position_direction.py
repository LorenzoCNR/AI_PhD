"""Update the CEBRA 2b gallery and render one corrected 5D fit per rat."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

import rat_figura2_20261028 as fig2

ROOT=fig2.ROOT
FIT=ROOT/'reproduction/figure2_bc/time_position_direction_5d'
PANEL=ROOT/'results/figura_2_2026-10-28/b'
STAGE=FIT/'rendering'
RATS=('achilles','buddy','cicero','gatsby')


def main():
    models={rat:fig2.source_cebra(FIT/rat/'seed_42') for rat in RATS}
    if any(z.shape[1]!=5 for z,y,ref in models.values()):raise ValueError('Expected 5D fit')
    ce=fig2.REPRO/'full';missing=fig2.REPRO/'figure2b_missing/full'
    specs=[('CEBRA — Position input',ce/'behavior_position_input/seed_42',[0,1]),
           ('CEBRA — Position + Direction input',missing/'behavior_position_direction_input/seed_42',[0,1]),
           ('CEBRA — Time',ce/'time/seed_42',[0,1,2]),('CEBRA — Position',ce/'position/seed_42',[0,1,2]),
           ('CEBRA — Position + Direction',missing/'position_direction/seed_42',[0,1,2]),
           ('CEBRA — Time + Position + Direction',FIT/'achilles/seed_42',[1,2,0]),
           ('CEBRA — Position shuffled',missing/'position_shuffled/seed_42',[0,1,2]),
           ('CEBRA — Position + Direction shuffled',missing/'position_direction_shuffled/seed_42',[0,1,2])]
    entries=[]
    for title,path,dims in specs:
        z,y,ref=fig2.source_cebra(path);entries.append((title,z,y,dims,ref))
    fig2.DEST=STAGE
    fig2.render_b('CEBRA',entries)
    for rat,(z,y,ref) in models.items():
        fig=plt.figure(figsize=(7,5));ax=fig.add_subplot(111,projection='3d')
        fig2.colored(ax,z,y,[1,2,0]);ax.set_title('CEBRA — Time + Position + Direction')
        fig.subplots_adjust(left=.03,right=.80,bottom=.08,top=.9)
        for x,cmap,label in ((.83,'cool','Left'),(.91,'viridis','Right')):
            bar=fig.colorbar(ScalarMappable(norm=fig2.NORM,cmap=cmap),cax=fig.add_axes([x,.24,.025,.49]),ticks=[0,.5,1,1.5])
            bar.ax.tick_params(labelsize=7);fig.text(x+.012,.19,label,ha='center',fontsize=8)
        fig2.save(fig,STAGE/'single'/rat.capitalize()/f'CEBRA_Time_Position_Direction_{rat.capitalize()}')
        folder=PANEL/rat.capitalize();folder.mkdir(parents=True,exist_ok=True)
        (folder/'RIFERIMENTI_TIME_POSITION_DIRECTION.json').write_text(json.dumps({'rat':rat,'seed':42,'fit':ref,'input':'spikes; label: position, right, left','output_dimension':5,'coordinates_one_based':[2,3,1],'fit_and_plot':'full recording','script':str(Path(__file__).relative_to(ROOT))},indent=2),encoding='utf-8')
        for extension in ('.png','.pdf','.svg'):
            source=STAGE/'single'/rat.capitalize()/f'CEBRA_Time_Position_Direction_{rat.capitalize()}{extension}'
            source.replace(folder/source.name)
        if rat!='achilles':
            (folder/'RIFERIMENTI.md').write_text('Disponibile il grafico CEBRA Time + Position + Direction 5D per questo ratto. Gli altri fit 2b della campagna corrente sono disponibili solo per Achilles.\n',encoding='utf-8')
    gallery=STAGE/'b/Achilles';target=PANEL/'Achilles'
    for name in ('Figura_2b_Achilles_CEBRA.png','Figura_2b_Achilles_CEBRA.pdf','Figura_2b_Achilles_CEBRA.svg','RIFERIMENTI_CEBRA.json'):
        (gallery/name).replace(target/name)
    print('COMPLETE: updated Achilles gallery and four rat-specific panels',flush=True)


if __name__=='__main__':main()
