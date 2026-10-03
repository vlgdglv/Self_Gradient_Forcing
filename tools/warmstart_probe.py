from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is a soft dependency for iteration only
    def tqdm(iterable, **kwargs):
        return iterable

# The 5 saved ODE states correspond to flow-matching timesteps at indices
# [0, 12, 24, 36, -1] of the 48-step schedule used in wan/generate_ode_trajectories.py.
TIMESTEPS = [1000.0, 937.5, 833.3, 625.0, 0.0]
N_STATES = len(TIMESTEPS)
EPS = 1e-8


# ---------------------------------------------------------------------------
# Loading / shape handling
# ---------------------------------------------------------------------------

def load_states(path: Path, layout: str) -> torch.Tensor:
    """Load one sample and return states as float32 `[S, T, C, H, W]`.
    States.shape: torch.Size([1, 5, 21, 16, 60, 104]) ([B, S, T, C, H, W])
    Handles an optional leading batch dim (asserted to be 1) and either
    `TCHW` (default, matching wan/generate_ode_trajectories.py) or `CTHW`
    post-ODE-state axis order.
    """
    data = torch.load(path, map_location="cpu")
    states = data["states"]
    if not torch.is_tensor(states):
        raise ValueError(f"{path}: 'states' is not a tensor (got {type(states)})")

    if states.ndim == 6:
        assert states.shape[0] == 1, (
            f"{path}: expected batch size 1, got shape {tuple(states.shape)}"
        )
        states = states[0]
    elif states.ndim != 5:
        raise ValueError(
            f"{path}: expected 5 or 6 dims (with/without batch), got shape {tuple(states.shape)}"
        )

    assert states.shape[0] == N_STATES, (
        f"{path}: expected {N_STATES} ODE states in dim 0, got shape {tuple(states.shape)}"
    )

    states = states.float()

    if layout == "CTHW":
        # [S, C, T, H, W] -> [S, T, C, H, W]
        states = states.transpose(1, 2).contiguous()
    elif layout != "TCHW":
        raise ValueError(f"Unknown layout {layout!r}, expected 'TCHW' or 'CTHW'")

    return states  # [S, T, C, H, W]


