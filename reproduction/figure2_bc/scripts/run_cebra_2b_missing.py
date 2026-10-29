"""Isolated remaining CEBRA 2b conditions, full recording."""
from pathlib import Path
import run_cebra as base

base.OUT = Path(__file__).resolve().parents[1] / 'figure2b_missing'
base.CONDITIONS = ('behavior_position_direction_input', 'time_position_5d', 'position_shuffled', 'position_direction', 'position_direction_shuffled')

if __name__ == '__main__':
    base.main()
