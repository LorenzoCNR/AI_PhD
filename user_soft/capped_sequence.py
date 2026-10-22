"""Execute the authorized CNN-seven-seed -> eight-Transformer-fit sequence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .rat_runner import ROOT, atomic_json, now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout-seconds', type=int, default=3600)
    args = parser.parse_args()
    cnn = ROOT / 'results/user_soft_rat_v1'
    target = ROOT / 'results/user_soft_rat_transformer_v1'
    target.mkdir(parents=True, exist_ok=True)
    state_path = target / 'SEQUENCE_STATE.json'
    deadline = time.monotonic() + args.timeout_seconds
    status = dict(created_utc=now(),cnn_target=168,cnn_seeds=list(range(42,49)),
                  transformer_target=8,command=[sys.executable,'-m','user_soft.transformer_runner','--execute'])
    while time.monotonic() < deadline:
        state = json.loads((cnn/'CAMPAIGN_STATE.json').read_text())
        status.update(updated_utc=now(),status='waiting_CNN_and_qualification',cnn_completed=state.get('completed',0))
        atomic_json(state_path,status)
        if state.get('status') in ('failed','interrupted'):
            raise RuntimeError('CNN worker stopped unexpectedly; no Transformer launched')
        if state.get('status') == 'requested_limit' and not (cnn/'RUNNING.lock').exists():
            complete = {k for k,v in state['records'].items() if v['status']=='complete'}
            expected = {f'{rat}/fold_{fold}/{version}/seed_{seed}'
                        for rat in ('achilles','buddy','cicero','gatsby') for fold in range(3)
                        for version in ('soft_v1_frozen','soft_v1_stable') for seed in range(42,49)}
            if complete != expected:
                raise RuntimeError('CNN completed subset does not exactly equal authorized 168 fits')
            gate = target/'qualification/GATE.json'
            if gate.exists():
                qualification = json.loads(gate.read_text())
                if qualification.get('decision')!='GO':
                    raise RuntimeError('Transformer qualification is not GO')
                status.update(status='launching_Transformer',updated_utc=now())
                atomic_json(state_path,status)
                # Runner independently checks the current source signature before any fit.
                result = subprocess.run(status['command'],cwd=ROOT,check=False)
                status.update(status='Transformer_worker_finished' if result.returncode==0 else 'failed',
                              returncode=result.returncode,updated_utc=now())
                atomic_json(state_path,status)
                if result.returncode:
                    raise RuntimeError(f'Transformer worker exited {result.returncode}')
                return
        time.sleep(15)
    status.update(status='timeout_no_Transformer_launch',updated_utc=now())
    atomic_json(state_path,status)
    raise RuntimeError('Timed out awaiting capped CNN completion and Transformer gate')


if __name__=='__main__':
    main()
