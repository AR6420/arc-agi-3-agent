"""Shared constants for the discovery agent."""

from __future__ import annotations

GRID = 64
N_COLORS = 16          # cells in [0, 15]
BG = 0                 # background color

# Action ids (ARC-AGI-3 SDK GameAction values)
RESET = 0
ACTION1 = 1            # directional (up)
ACTION2 = 2            # directional (down)
ACTION3 = 3            # directional (left)
ACTION4 = 4            # directional (right)
ACTION5 = 5            # often select/cycle or release
ACTION6 = 6            # click (x, y)
ACTION7 = 7            # often undo

DIRECTIONAL = (ACTION1, ACTION2, ACTION3, ACTION4)

# World-model learning thresholds
MIN_TRIALS = 2                 # observations before an ActionEffect is "known"
NOOP_HI = 0.95                 # noop_rate >= this and trials>=MIN_TRIALS => dead action
CONTROLLABLE_CONFIRM = 2       # consistent single-translation obs to set controllable_sig
DETERMINISTIC_FRAC = 0.8       # dominant translation vector fraction to call deterministic
MATCH_MAX_DIST = 8             # initial nearest-centroid radius for object matching
STUCK_K = 8                    # identical state-keys -> stuck escape
