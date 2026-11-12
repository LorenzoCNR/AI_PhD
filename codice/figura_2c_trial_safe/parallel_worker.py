"""Run a disjoint tail of the campaign while the main runner starts at its head."""
import hashlib
import json
from pathlib import Path

import run_campaign as c


def main():
    required = [c.ROOT / 'data' / 'PREPARED.json', c.ROOT / 'data' / 'PREFLIGHT.json', Path(c.__file__),
                Path(c.__file__).with_name('cebra_trial_safe.py'), Path(c.__file__).with_name('soft_trial_safe.py')]
    signature = hashlib.sha256(''.join(c.sha256(p) for p in required).encode()).hexdigest()
    state_path = c.ROOT / 'PARALLEL_STATE.json'
    tasks = [('full', 'gatsby', model, condition)
             for model in ('soft_stable_transformer', 'soft_stable_cnn')
             for condition in reversed(c.CONDITIONS)]
    state = dict(total=len(tasks), done=0, current=None, signature=signature)
    for task in tasks:
        primary = json.loads((c.ROOT / 'CAMPAIGN_STATE.json').read_text(encoding='utf-8'))
        current = primary.get('current') or {}
        if primary.get('error') or (current.get('scope') == 'full' and current.get('rat') == 'gatsby'):
            state['stopped'] = 'primary reached Gatsby full or reported an error'
            break
        state['current'] = dict(zip(('scope', 'rat', 'model', 'condition'), task))
        c.write_json(state_path, state)
        print('START', *task, flush=True)
        try:
            result = c.run_fit(*task, signature)
        except Exception:
            import traceback
            state['error'] = traceback.format_exc()
            c.write_json(state_path, state)
            raise
        state['done'] += 1
        print(result.upper(), *task, flush=True)
    state['current'] = None
    state['finished'] = True
    c.write_json(state_path, state)


if __name__ == '__main__': main()
