"""Figure 2c 8D trainable experiment, isolated from 3D reproduction."""
import sys
from pathlib import Path
import run_cebra as base

original_setup = base.setup
def setup_8d(name, steps):
    params = original_setup(name, steps)
    params['output_dimension'] = 8
    return params

base.setup = setup_8d
base.OUT = Path(__file__).resolve().parents[1] / 'figure2c_8d'
base.CONDITIONS = ('position', 'direction_paper_demo', 'position_direction',
                   'position_shuffled', 'direction_shuffled_paper_demo',
                   'position_direction_shuffled', 'direction_controlled',
                   'direction_shuffled_controlled')

if __name__ == '__main__':
    base.main()
