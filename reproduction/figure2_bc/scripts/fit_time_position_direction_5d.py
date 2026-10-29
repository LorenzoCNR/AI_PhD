"""Full-recording CEBRA Time + Position + Direction 5D, four rats, seed 42."""
import csv
import hashlib
import json
import os
import random
import time
import traceback
from pathlib import Path

import cebra
import cebra.datasets
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reproduction/figure2_bc/time_position_direction_5d'
RATS=('achilles','buddy','cicero','gatsby')
PARAMS=dict(model_architecture='offset10-model',num_hidden_units=32,output_dimension=5,
            batch_size=512,learning_rate=3e-4,optimizer='adam',criterion='infonce',
            distance='cosine',temperature_mode='constant',temperature=1.,max_iterations=10000,
            conditional='time_delta',time_offsets=10,hybrid=True,device='cuda_if_available',verbose=False)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''):h.update(part)
    return h.hexdigest()


def main():
    torch.set_num_threads(2)
    for rat in RATS:
        ds=cebra.datasets.init(f'rat-hippocampus-single-{rat}')
        x=ds.neural.numpy();y=ds.continuous_index.numpy()
        if x.ndim!=2 or y.shape!=(len(x),3) or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.allclose(y[:,1]+y[:,2],1):raise ValueError(f'Invalid data {rat}')
        folder=OUT/rat/'seed_42';folder.mkdir(parents=True,exist_ok=True)
        manifest=folder/'manifest.json';source=ROOT/'data/rat_hippocampus'/f'{rat}.jl'
        if manifest.exists():
            m=json.loads(manifest.read_text(encoding='utf-8'))
            if m['parameters']!=PARAMS or m['source_sha256']!=sha(source) or m['label_columns']!=['position','right','left'] or m['cebra_version']!=cebra.__version__:raise ValueError(f'Incompatible fit {manifest}')
            for name,digest in m['sha256'].items():
                if sha(folder/name)!=digest:raise ValueError(f'Changed output {folder/name}')
            print('REUSE',rat,flush=True);continue
        if any(folder.iterdir()):raise RuntimeError(f'Partial run; inspect before resuming: {folder}')
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
        running=folder/'RUNNING.json'
        running.write_text(json.dumps({'pid':os.getpid(),'rat':rat,'parameters':PARAMS},indent=2),encoding='utf-8')
        start=time.time();print('START',rat,flush=True)
        try:
            model=cebra.CEBRA(**PARAMS);model.fit(x,y)
            z=model.transform(x)
            raw=model.state_dict_['loss']
            loss=raw.detach().cpu().numpy() if isinstance(raw,torch.Tensor) else np.asarray(raw)
            if z.shape!=(len(x),5) or len(loss)!=10000 or not np.isfinite(z).all() or not np.isfinite(loss).all():raise ValueError('invalid output')
            model.save(str(folder/'checkpoint.pt'))
            np.savez_compressed(folder/'embedding.npz',embedding=z,behavior=y,source_id=np.arange(len(x)))
            with (folder/'loss.csv').open('x',newline='',encoding='utf-8') as f:
                writer=csv.writer(f);writer.writerow(('update','loss'));writer.writerows(enumerate(loss,1))
            m={'status':'complete','rat':rat,'seed':42,'parameters':PARAMS,'input_shape':list(x.shape),
               'label_shape':list(y.shape),'label_columns':['position','right','left'],'fit_scope':'full_recording',
               'elapsed_seconds':time.time()-start,'source_sha256':sha(source),'script_sha256':sha(Path(__file__)),
               'cebra_version':cebra.__version__,'torch_version':torch.__version__,'loss_final':float(loss[-1]),
               'sha256':{name:sha(folder/name) for name in ('checkpoint.pt','embedding.npz','loss.csv')}}
            manifest.write_text(json.dumps(m,indent=2),encoding='utf-8');running.unlink()
            print('COMPLETE',rat,round(m['elapsed_seconds'],1),flush=True)
        except Exception:
            (folder/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
