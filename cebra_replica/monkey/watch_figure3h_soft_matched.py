"""Wait for the three specific fit processes; score only a complete grid."""
import json
import time
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[2]
TASKS={'position':('monkey_fig3h_soft_position_v1',6096,'monkey_fig3h_position'),
       'direction':('monkey_fig3h_soft_direction_v1',46096,'monkey_fig3h_direction'),
       'ap':('monkey_fig3h_soft_ap_v1',44432,'monkey_fig3h_ap')}
OUTPUT=ROOT/'results/monkey_fig3h_soft_matched_decoding_v1'
def process_matches(pid,expected):
    try:
        p=psutil.Process(pid)
        return p.is_running() and expected in ' '.join(p.cmdline())
    except (psutil.NoSuchProcess,psutil.AccessDenied):return False
def main():
    OUTPUT.mkdir(parents=True,exist_ok=True); state=OUTPUT/'WATCH_STATE.json'
    while True:
        tasks={}
        for task,(folder,pid,module) in TASKS.items():
            base=ROOT/'results'/folder/'stage02_embedding'
            completed=sum((base/arch/f'dim_{d}'/f'seed_{s}'/'manifest.json').is_file()
                for arch in ('cnn','transformer') for d in (3,4,8,16,32) for s in (42,43,44,45,46))
            tasks[task]={'embeddings_complete':completed,'total':50,'pid':pid,'process_active':process_matches(pid,module)}
        summary={'updated_utc':datetime.now(timezone.utc).isoformat(),'status':'waiting_fits','tasks':tasks}
        if any(v['embeddings_complete']<50 and not v['process_active'] for v in tasks.values()):
            summary['status']='fit_process_stopped_before_completion';state.write_text(json.dumps(summary,indent=2),encoding='utf8')
            print(json.dumps(summary),flush=True);return 1
        if all(v['embeddings_complete']==50 for v in tasks.values()):
            summary['status']='scoring';state.write_text(json.dumps(summary,indent=2),encoding='utf8')
            result=subprocess.run([sys.executable,str(ROOT/'cebra_replica/monkey/score_plot_figure3h_soft_matched.py')],cwd=ROOT)
            summary['status']='complete' if result.returncode==0 else 'scoring_failed'
            summary['scoring_exit_code']=result.returncode;state.write_text(json.dumps(summary,indent=2),encoding='utf8')
            return result.returncode
        state.write_text(json.dumps(summary,indent=2),encoding='utf8');time.sleep(20)
if __name__=='__main__':sys.exit(main())
