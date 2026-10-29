"""Experimental Soft encoders on behavior input, guided by position."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from user_soft import full_recording_runner as runner
from user_soft import rat_runner as rr
from user_soft.loss import fit_distance_scales

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reproduction/figure2_bc/soft_behavior_2d_v2'


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true'); args=parser.parse_args()
    output=OUT/('smoke' if args.smoke else 'full')
    original,_=runner.load_full_recording('achilles')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.set_num_threads(2)
    specs=[{'key':'behavior','geometry':'euclidean','weight':1.0}]
    for condition in ('position_input','position_direction_input'):
        input_data=original.behavior[:,:1].copy() if condition=='position_input' else original.behavior[:,:2].copy()
        rows=SimpleNamespace(neural=input_data,behavior=original.behavior,source_id=original.source_id,
                             trial_id=original.trial_id,time_id=original.time_id,session_id=original.session_id)
        label=original.behavior.copy(); label[:,1:]=0
        training=runner.rows_with_behavior(rows,label)
        values=rr.tensors(training,torch.device('cpu'))
        scales=fit_distance_scales({'behavior':values['behavior'],'time_id':values['time_id']},specs,split='full_recording',source_ids=rows.source_id)
        for architecture in ('cnn','transformer'):
            cfg=runner.condition_config(architecture,'achilles','direction_position',embedding_temperature=1.)
            cfg.update(condition=condition,output_dimension=2,metadata_specs=specs,max_updates=10 if args.smoke else 10000,
                       checkpoint_interval=10 if args.smoke else 500,diagnostic_interval=10 if args.smoke else 500,
                       metadata_supervision='position_only',input_features=condition,
                       condition_interpretation='Exploratory behavior-input fit; exact original configuration unverified')
            sources=runner.source_record(architecture); sources['experiment_script_sha256']=runner.sha256(Path(__file__))
            print(f'START {condition}/{architecture}',flush=True)
            runner.fit_cell(output,rows,training,specs,scales,'achilles',condition,architecture,cfg,sources,device)
    print('COMPLETE Soft behavioral inputs 2D',flush=True)


if __name__=='__main__': main()
