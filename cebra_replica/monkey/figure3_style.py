"""Shared Figure 3 color mappings for CEBRA and Soft renderers."""

from __future__ import annotations

import matplotlib.pyplot as plt


# Match the official Figure 3 source: direction values 0..7 map to hsv(i / 8).
DIRECTION_COLORS = tuple(plt.cm.hsv(index / 8) for index in range(8))

# Match the official continuous position colors and common physical scale.
POSITION_CMAP = "seismic"
POSITION_LIMITS = (-15.0, 15.0)


SOFT_OBJECTIVE_LABELS = {
    "time_only": "Time",
    "behavior_only": "Position X,Y",
    "joint": "Time + Position X,Y",
}
