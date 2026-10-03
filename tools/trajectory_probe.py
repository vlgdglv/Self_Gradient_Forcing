"""
ODE Trajectory Probe — offline, observational analysis of saved teacher ODE states.
=====================================================================================

Pure post-hoc analysis of the ~16k bidirectional Wan2.1 teacher ODE trajectories
produced by `wan/generate_ode_trajectories.py`. No model inference, no VAE decode,
no trained probes, no random-projection / MI / CKA estimators. Everything here is a
closed-form measurement computed directly on the saved latents.

Each `.pt` file holds:

    {
        "states": Tensor,   # [B, S, T, C, H, W], usually B=1, S=5
        "prompt": str,
        "seed": int,
        "sample_id": int,
    }

`S=5` are ODE states at flow-matching timesteps [1000, 937.5, 833.3, 625, 0]
(indices [0, 12, 24, 36, -1] of the 48-step schedule). `T` is the latent temporal
axis (21 for the standard 81-frame / 16fps / 4x-VAE-compression setting), `C` is
the latent channel count (16), `H, W` the latent spatial size (60, 104).

Per `wan/generate_ode_trajectories.py`, latents are built as
`torch.randn([1, T, C, H, W], ...)`, i.e. the axis order after the batch dim is
`[T, C, H, W]` (frames before channels) — NOT `[C, T, H, W]`. This script assumes
that layout by default (`--layout TCHW`); pass `--layout CTHW` if your dump uses
the other convention. Shapes are asserted, not guessed silently.

For every temporal position `i` we treat the latent chunk `z_i^t \\in R^{C*H*W}`
(flattened over channel + spatial dims) as a single vector, and measure how pairs
of these per-position vectors relate to each other across temporal lag `d` and
ODE timestep `t`.

Probes implemented (see README-style docstrings on each `run_*` function):

  1. Temporal structure emergence:      R_t(d) = ||z_i - z_{i+d}||^2 / (0.5*(||z_i||^2+||z_{i+d}||^2))
  2. ODE update alignment:              cos(tilde-Delta_i^k, tilde-Delta_{i+d}^k), mean-update removed
  3. Artificial AR chunk-boundary break: (1) and (2) at lag 1, split within-chunk vs cross-boundary
  4. Noise->endpoint residual:          z_i^t projected onto span{eps_i, x_i}; residual energy + cosine

Usage
-----
    python tools/trajectory_probe.py \\
        --data-dir /path/to/data \\
        --output-dir ./results \\
        --chunk-size 3 \\
        --max-lag 8 \\
        --max-samples -1

Smoke-test on synthetic data (no real dataset needed):

    python tools/trajectory_probe.py --self-test --output-dir ./results_selftest
"""

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
# Streaming aggregation
# ---------------------------------------------------------------------------

class Accumulator:
    """Streaming mean accumulator keyed by arbitrary hashable keys.

    Values are added incrementally (per-file) so the full dataset never needs
    to be held in memory at once.
    """

    def __init__(self):
        self._sum = defaultdict(float)
        self._count = defaultdict(int)

    def add(self, key, values):
        if torch.is_tensor(values):
            values = values.detach()
            n = values.numel()
            if n == 0:
                return
            self._sum[key] += values.double().sum().item()
            self._count[key] += n
        else:
            if values != values:  # NaN guard
                return
            self._sum[key] += float(values)
            self._count[key] += 1

    def mean(self, key):
        c = self._count.get(key, 0)
        return self._sum[key] / c if c > 0 else float("nan")

    def count(self, key):
        return self._count.get(key, 0)

    def keys(self):
        return list(self._sum.keys())

    def as_dict(self):
        return {str(k): {"mean": self.mean(k), "n": self.count(k)} for k in self.keys()}


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


def flatten_states(states: torch.Tensor):
    """`[S, T, C, H, W] -> [S, T, D]` with `D = C*H*W`."""
    S, T, C, H, W = states.shape
    flat = states.reshape(S, T, C * H * W)
    return flat, T


# ---------------------------------------------------------------------------
# Core pairwise measurements (operate on a single `[T, D]` slice)
# ---------------------------------------------------------------------------

def pairwise_temporal_distance(vec: torch.Tensor, d: int):
    """R(d)_i = ||v_i - v_{i+d}||^2 / (0.5*(||v_i||^2 + ||v_{i+d}||^2)), for all valid i."""
    T = vec.shape[0]
    if d < 1 or d >= T:
        return None
    a, b = vec[: T - d], vec[d:]
    diff_sq = (a - b).pow(2).sum(-1)
    denom = 0.5 * (a.pow(2).sum(-1) + b.pow(2).sum(-1))
    return diff_sq / (denom + EPS)


def pairwise_cosine(vec: torch.Tensor, d: int):
    """cos(v_i, v_{i+d}) for all valid i."""
    T = vec.shape[0]
    if d < 1 or d >= T:
        return None
    a, b = vec[: T - d], vec[d:]
    num = (a * b).sum(-1)
    den = a.norm(dim=-1) * b.norm(dim=-1)
    return num / (den + EPS)


def chunk_boundary_mask(T: int, K: int) -> torch.Tensor:
    """Boolean mask over adjacent pairs (i, i+1), i in [0, T-2]: True = within the
    same artificial AR chunk of size K, False = crosses a chunk boundary."""
    idx = torch.arange(T - 1)
    return (idx // K) == ((idx + 1) // K)


# ---------------------------------------------------------------------------
# Probe 1 — temporal structure emergence
# ---------------------------------------------------------------------------

def run_temporal_structure(flat: torch.Tensor, max_lag: int, acc: Accumulator):
    """R_t(d) aggregated over temporal position and (across files) over samples,
    keyed by (state_idx, lag)."""
    for t_idx in range(flat.shape[0]):
        vec = flat[t_idx]  # [T, D]
        for d in range(1, max_lag + 1):
            r = pairwise_temporal_distance(vec, d)
            if r is not None:
                acc.add((t_idx, d), r)


# ---------------------------------------------------------------------------
# Probe 2 — ODE update alignment
# ---------------------------------------------------------------------------

def run_update_alignment(flat: torch.Tensor, max_lag: int, acc: Accumulator):
    """cos(tilde-Delta_i^k, tilde-Delta_{i+d}^k), keyed by (transition_idx, lag).

    tilde-Delta_i^k = Delta_i^k - mean_j Delta_j^k, i.e. the per-video common
    (temporally-global) update component is subtracted before comparing.
    """
    for k in range(flat.shape[0] - 1):
        delta = flat[k + 1] - flat[k]  # [T, D]
        tilde = delta - delta.mean(dim=0, keepdim=True)
        for d in range(1, max_lag + 1):
            c = pairwise_cosine(tilde, d)
            if c is not None:
                acc.add((k, d), c)


# ---------------------------------------------------------------------------
# Probe 3 — artificial AR chunk-boundary analysis (uses lag-1 outputs of 1 & 2)
# ---------------------------------------------------------------------------

def run_boundary_analysis(
    flat: torch.Tensor, chunk_size: int, acc_latent: Accumulator, acc_update: Accumulator
):
    """Split lag-1 latent distance / update-cosine into within-chunk vs
    cross-boundary groups, for every ODE state / transition respectively."""
    T = flat.shape[1]
    if T < 2:
        return
    mask = chunk_boundary_mask(T, chunk_size)  # [T-1] bool, True = within-chunk

    for t_idx in range(flat.shape[0]):
        r = pairwise_temporal_distance(flat[t_idx], 1)
        if r is None:
            continue
        acc_latent.add((t_idx, "within"), r[mask])
        acc_latent.add((t_idx, "cross"), r[~mask])

    for k in range(flat.shape[0] - 1):
        delta = flat[k + 1] - flat[k]
        tilde = delta - delta.mean(dim=0, keepdim=True)
        c = pairwise_cosine(tilde, 1)
        if c is None:
            continue
        acc_update.add((k, "within"), c[mask])
        acc_update.add((k, "cross"), c[~mask])


# ---------------------------------------------------------------------------
# Probe 4 — noise -> endpoint trajectory residual
# ---------------------------------------------------------------------------

def project_onto_span(eps: torch.Tensor, x: torch.Tensor, z: torch.Tensor):
    """Per-temporal-position analytic least-squares projection of `z` onto
    `span{eps, x}` (both `[T, D]`, shared per-position basis).

    Solves the 2x2 normal-equations system
        [[eps.eps, eps.x], [eps.x, x.x]] [a, b]^T = [eps.z, x.z]^T
    independently for every row i, fully vectorized over T.

    Returns `(proj, residual)`, each `[T, D]`.
    """
    G11 = (eps * eps).sum(-1)
    G12 = (eps * x).sum(-1)
    G22 = (x * x).sum(-1)
    r1 = (eps * z).sum(-1)
    r2 = (x * z).sum(-1)

    det = G11 * G22 - G12 * G12
    det_safe = torch.where(det.abs() < EPS, torch.full_like(det, EPS), det)

    a = (G22 * r1 - G12 * r2) / det_safe
    b = (G11 * r2 - G12 * r1) / det_safe

    proj = a.unsqueeze(-1) * eps + b.unsqueeze(-1) * x
    residual = z - proj
    return proj, residual


def run_noise_endpoint_residual(
    flat: torch.Tensor,
    max_lag: int,
    chunk_size: int,
    acc_energy: Accumulator,
    acc_cos_lag: Accumulator,
    acc_cos_boundary: Accumulator,
):
    """For each intermediate ODE state (excludes the endpoints, which are the
    basis vectors themselves and trivially have zero residual), compute residual
    energy and residual-cosine structure across temporal lag / chunk boundary."""
    eps = flat[0]   # z^1000, [T, D]
    x = flat[-1]    # z^0,    [T, D]
    T = flat.shape[1]
    mask = chunk_boundary_mask(T, chunk_size) if T >= 2 else None

    for t_idx in range(1, flat.shape[0] - 1):
        z = flat[t_idx]
        _, residual = project_onto_span(eps, x, z)

        energy = residual.pow(2).sum(-1) / (z.pow(2).sum(-1) + EPS)
        acc_energy.add(t_idx, energy)

        for d in range(1, max_lag + 1):
            c = pairwise_cosine(residual, d)
            if c is None:
                continue
            acc_cos_lag.add((t_idx, d), c)
            if d == 1 and mask is not None:
                acc_cos_boundary.add((t_idx, "within"), c[mask])
                acc_cos_boundary.add((t_idx, "cross"), c[~mask])


# ---------------------------------------------------------------------------
# Per-file driver
# ---------------------------------------------------------------------------

class Accumulators:
    def __init__(self):
        self.temporal_distance = Accumulator()          # (state_idx, lag) -> R_t(d)
        self.update_alignment = Accumulator()            # (k, lag) -> cos
        self.boundary_latent = Accumulator()              # (state_idx, within/cross)
        self.boundary_update = Accumulator()              # (k, within/cross)
        self.residual_energy = Accumulator()              # state_idx -> energy
        self.residual_cos_lag = Accumulator()             # (state_idx, lag) -> cos
        self.residual_cos_boundary = Accumulator()        # (state_idx, within/cross)
        self.n_files_ok = 0
        self.n_files_failed = 0
        self.T_values = []


def process_file(path: Path, layout: str, max_lag: int, chunk_size: int, acc: Accumulators):
    states = load_states(path, layout)
    flat, T = flatten_states(states)
    acc.T_values.append(T)

    run_temporal_structure(flat, max_lag, acc.temporal_distance)
    run_update_alignment(flat, max_lag, acc.update_alignment)
    run_boundary_analysis(flat, chunk_size, acc.boundary_latent, acc.boundary_update)
    run_noise_endpoint_residual(
        flat, max_lag, chunk_size,
        acc.residual_energy, acc.residual_cos_lag, acc.residual_cos_boundary,
    )
    acc.n_files_ok += 1


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(acc: Accumulators, max_lag: int, chunk_size: int, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    lags = list(range(1, max_lag + 1))

    # --- Plot 1: temporal structure emergence heatmap (lag x ODE timestep) ---
    grid = np.full((len(lags), N_STATES), np.nan)
    for i, d in enumerate(lags):
        for t_idx in range(N_STATES):
            grid[i, t_idx] = acc.temporal_distance.mean((t_idx, d))

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(N_STATES))
    ax.set_xticklabels([f"{t:g}" for t in TIMESTEPS])
    ax.set_yticks(range(len(lags)))
    ax.set_yticklabels(lags)
    ax.set_xlabel("ODE timestep")
    ax.set_ylabel("temporal lag d")
    ax.set_title("Probe 1: normalized temporal distance R_t(d)")
    fig.colorbar(im, ax=ax, label="R_t(d)")
    fig.tight_layout()
    fig.savefig(output_dir / "01_temporal_structure_heatmap.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: ODE update alignment vs lag, one line per transition ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for k in range(N_STATES - 1):
        ys = [acc.update_alignment.mean((k, d)) for d in lags]
        ax.plot(lags, ys, marker="o", label=f"{TIMESTEPS[k]:g} -> {TIMESTEPS[k + 1]:g}")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("temporal lag d")
    ax.set_ylabel("cos(tilde-Delta_i, tilde-Delta_{i+d})")
    ax.set_title("Probe 2: mean-removed ODE update alignment")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "02_update_alignment.png", dpi=150)
    plt.close(fig)

    # --- Plot 3a: AR boundary — latent distance, within vs cross, per ODE state ---
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(N_STATES)
    width = 0.35
    within = [acc.boundary_latent.mean((t, "within")) for t in range(N_STATES)]
    cross = [acc.boundary_latent.mean((t, "cross")) for t in range(N_STATES)]
    ax.bar(x - width / 2, within, width, label="within chunk")
    ax.bar(x + width / 2, cross, width, label="cross boundary")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:g}" for t in TIMESTEPS])
    ax.set_xlabel("ODE timestep")
    ax.set_ylabel("R_t(1)")
    ax.set_title(f"Probe 3a: lag-1 latent distance, chunk size K={chunk_size}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "03a_boundary_latent_distance.png", dpi=150)
    plt.close(fig)

    # --- Plot 3b: AR boundary — update cosine, within vs cross, per transition ---
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(N_STATES - 1)
    within = [acc.boundary_update.mean((k, "within")) for k in range(N_STATES - 1)]
    cross = [acc.boundary_update.mean((k, "cross")) for k in range(N_STATES - 1)]
    ax.bar(x - width / 2, within, width, label="within chunk")
    ax.bar(x + width / 2, cross, width, label="cross boundary")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{TIMESTEPS[k]:g}->{TIMESTEPS[k + 1]:g}" for k in range(N_STATES - 1)], fontsize=8)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("ODE transition")
    ax.set_ylabel("cos(tilde-Delta_i, tilde-Delta_{i+1})")
    ax.set_title(f"Probe 3b: lag-1 update alignment, chunk size K={chunk_size}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "03b_boundary_update_alignment.png", dpi=150)
    plt.close(fig)

    # --- Plot 4a: residual energy per intermediate ODE state ---
    mid_states = list(range(1, N_STATES - 1))
    fig, ax = plt.subplots(figsize=(6, 5))
    ys = [acc.residual_energy.mean(t) for t in mid_states]
    ax.bar([f"{TIMESTEPS[t]:g}" for t in mid_states], ys, color="tab:orange")
    ax.set_xlabel("ODE timestep")
    ax.set_ylabel("||r||^2 / ||z||^2")
    ax.set_title("Probe 4a: residual energy outside span{noise, endpoint}")
    fig.tight_layout()
    fig.savefig(output_dir / "04a_residual_energy.png", dpi=150)
    plt.close(fig)

    # --- Plot 4b: residual cosine vs lag, one line per intermediate state ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for t_idx in mid_states:
        ys = [acc.residual_cos_lag.mean((t_idx, d)) for d in lags]
        ax.plot(lags, ys, marker="o", label=f"t={TIMESTEPS[t_idx]:g}")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("temporal lag d")
    ax.set_ylabel("cos(r_i, r_{i+d})")
    ax.set_title("Probe 4b: residual cosine similarity vs lag")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "04b_residual_cosine_vs_lag.png", dpi=150)
    plt.close(fig)

    # --- Plot 4c: residual boundary, within vs cross, per intermediate state ---
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(mid_states))
    within = [acc.residual_cos_boundary.mean((t, "within")) for t in mid_states]
    cross = [acc.residual_cos_boundary.mean((t, "cross")) for t in mid_states]
    ax.bar(x - width / 2, within, width, label="within chunk")
    ax.bar(x + width / 2, cross, width, label="cross boundary")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{TIMESTEPS[t]:g}" for t in mid_states])
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("ODE timestep")
    ax.set_ylabel("cos(r_i, r_{i+1})")
    ax.set_title(f"Probe 4c: lag-1 residual cosine, chunk size K={chunk_size}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "04c_residual_boundary.png", dpi=150)
    plt.close(fig)


def save_stats(acc: Accumulators, args, output_dir: Path):
    stats = {
        "n_files_ok": acc.n_files_ok,
        "n_files_failed": acc.n_files_failed,
        "T_values_seen": sorted(set(acc.T_values)),
        "args": vars(args),
        "temporal_distance": acc.temporal_distance.as_dict(),
        "update_alignment": acc.update_alignment.as_dict(),
        "boundary_latent": acc.boundary_latent.as_dict(),
        "boundary_update": acc.boundary_update.as_dict(),
        "residual_energy": acc.residual_energy.as_dict(),
        "residual_cos_lag": acc.residual_cos_lag.as_dict(),
        "residual_cos_boundary": acc.residual_cos_boundary.as_dict(),
    }
    with open(output_dir / "aggregated_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Self-test (synthetic data, no real dataset required)
# ---------------------------------------------------------------------------

def make_synthetic_file(path: Path, T=9, C=4, H=3, W=3, seed=0, layout="TCHW"):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(T, C, H, W, generator=g)       # clean endpoint
    eps = torch.randn(T, C, H, W, generator=g)       # noise
    fracs = [1.0, 0.9375, 0.8333, 0.625, 0.0]        # matches TIMESTEPS/1000
    states = torch.stack([f * eps + (1 - f) * x0 for f in fracs], dim=0)  # [S, T, C, H, W]
    if layout == "CTHW":
        states = states.transpose(1, 2).contiguous()  # [S, C, T, H, W]
    states = states.unsqueeze(0).to(torch.float16)  # [B=1, S, ...]
    torch.save(
        {"states": states, "prompt": f"synthetic prompt {seed}", "seed": seed, "sample_id": seed},
        path,
    )


def self_test(output_dir: Path, layout: str = "TCHW"):
    print("Running self-test on synthetic data...")
    tmp_dir = Path(tempfile.mkdtemp(prefix="trajectory_probe_selftest_"))
    try:
        for i in range(5):
            make_synthetic_file(tmp_dir / f"{i:06d}.pt", seed=i, layout=layout)

        args = argparse.Namespace(
            data_dir=str(tmp_dir),
            output_dir=str(output_dir),
            chunk_size=3,
            max_lag=4,
            max_samples=-1,
            layout=layout,
        )
        run(args)
        assert (output_dir / "aggregated_stats.json").exists()
        assert (output_dir / "01_temporal_structure_heatmap.png").exists()
        print("Self-test PASSED.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.pt"))
    if not files:
        files = sorted(data_dir.glob("*.pth"))
    if not files:
        raise FileNotFoundError(f"No .pt/.pth files found under {data_dir}")

    if args.max_samples is not None and args.max_samples >= 0:
        files = files[: args.max_samples]

    print(f"Found {len(files)} files. chunk_size={args.chunk_size}, max_lag={args.max_lag}, layout={args.layout}")

    acc = Accumulators()
    for path in tqdm(files, desc="processing trajectories"):
        try:
            process_file(path, args.layout, args.max_lag, args.chunk_size, acc)
        except Exception as e:  # noqa: BLE001 - keep going on bad files, report at the end
            acc.n_files_failed += 1
            print(f"[WARN] failed on {path}: {e}")
            traceback.print_exc()

    print(f"Processed OK: {acc.n_files_ok}, failed: {acc.n_files_failed}")
    if acc.n_files_ok == 0:
        raise RuntimeError("No files processed successfully; aborting before plotting.")

    save_stats(acc, args, output_dir)
    make_plots(acc, args.max_lag, args.chunk_size, output_dir)
    print(f"Wrote figures + aggregated_stats.json to {output_dir}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=str, default=None, help="Directory of saved .pt ODE trajectory files")
    p.add_argument("--output-dir", type=str, required=True, help="Directory to write figures + stats")
    p.add_argument("--chunk-size", type=int, default=3, help="Artificial AR chunk size K for boundary analysis")
    p.add_argument("--max-lag", type=int, default=8, help="Maximum temporal lag d to probe")
    p.add_argument("--max-samples", type=int, default=-1, help="Limit number of files processed (-1 = all)")
    p.add_argument(
        "--layout", type=str, default="TCHW", choices=["TCHW", "CTHW"],
        help="Axis order after the ODE-state dim: TCHW (default, matches "
             "wan/generate_ode_trajectories.py) or CTHW",
    )
    p.add_argument("--self-test", action="store_true", help="Run on synthetic data and exit (ignores --data-dir)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test(Path(args.output_dir), layout=args.layout)
        return
    if args.data_dir is None:
        raise SystemExit("--data-dir is required unless --self-test is set")
    run(args)


if __name__ == "__main__":
    main()
