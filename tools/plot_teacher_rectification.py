"""
Teacher Rectification Probe — Visualization
=============================================
Reads probe_records.json and produces three figures:

  1. teacher_rectification_curve.png
       Raw R_t and median-normalized R_t vs. rollout time.
  2. teacher_rectification_map.png
       Heatmap: probe time × frame position → per-frame rectification energy.
  3. teacher_rectification_concentration.png
       Lorenz-style cumulative R_t curve; prints top-20% concentration stat.

Usage
-----
python tools/plot_teacher_rectification.py \\
    --records outputs/teacher_rectification/<run>/probe_records.json \\
    --output_dir outputs/teacher_rectification/<run>

Optional: --run_name "my experiment"  (label on plots)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True,
                   help="Path to probe_records.json")
    p.add_argument("--output_dir", default=None,
                   help="Output dir for PNGs (default: same dir as records)")
    p.add_argument("--run_name", default="",
                   help="Optional experiment label for plot titles")
    return p.parse_args()


def load_scored_records(path: str) -> list[dict]:
    with open(path) as f:
        all_records = json.load(f)
    scored = [r for r in all_records if r.get("status") == "scored"
              and isinstance(r.get("rectification_energy"), float)]
    return scored


# ---------------------------------------------------------------------------
# Plot 1 — rectification energy curve
# ---------------------------------------------------------------------------

def plot_curve(records: list[dict], output_dir: str, run_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = [r["rollout_time_sec"] for r in records]
    R_vals = [r["rectification_energy"] for r in records]

    # Median-normalized series (normalize by median of first K ≤ 3 points)
    K = min(3, len(R_vals))
    baseline = sorted(R_vals[:K])[K // 2] if K > 0 else 1.0
    if baseline == 0.0:
        baseline = 1.0
    R_norm = [v / baseline for v in R_vals]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    title_suffix = f" — {run_name}" if run_name else ""

    ax0 = axes[0]
    ax0.plot(times, R_vals, marker="o", linewidth=1.5, markersize=5, color="#2563EB")
    ax0.set_ylabel("Frontier Teacher Rectification Energy  $R_t$")
    ax0.set_title(f"Teacher Rectification Energy (Last AR Block){title_suffix}")
    ax0.grid(True, alpha=0.3)
    ax0.set_ylim(bottom=0)

    ax1 = axes[1]
    ax1.plot(times, R_norm, marker="s", linewidth=1.5, markersize=5, color="#DC2626",
             label=f"normalized (÷ median of first {K} points = {baseline:.4f})")
    ax1.set_ylabel("$R_t$ (frontier) / early-median")
    ax1.set_xlabel("Rollout time (seconds)")
    ax1.set_title(f"Median-normalized Frontier Rectification Energy{title_suffix}")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    plt.tight_layout()
    out = os.path.join(output_dir, "teacher_rectification_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2 — rectification heatmap (map)
# ---------------------------------------------------------------------------

def plot_map(records: list[dict], output_dir: str, run_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    per_frame_lists = [r.get("rectification_per_frame") for r in records]
    per_frame_lists = [pf for pf in per_frame_lists if pf is not None]
    if not per_frame_lists:
        print("No per-frame data available — skipping heatmap.")
        return

    # Find common frame count (trim to min if ragged)
    min_frames = min(len(pf) for pf in per_frame_lists)
    matrix = np.array([pf[:min_frames] for pf in per_frame_lists])  # [T_probes, F_win]

    times = [r["rollout_time_sec"] for r in records
             if r.get("rectification_per_frame") is not None][:len(per_frame_lists)]
    frame_indices = list(range(min_frames))

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        matrix.T,
        aspect="auto",
        origin="lower",
        cmap="plasma",
        interpolation="nearest",
        extent=[
            min(times) - (times[1] - times[0]) / 2 if len(times) > 1 else min(times) - 1,
            max(times) + (times[1] - times[0]) / 2 if len(times) > 1 else max(times) + 1,
            -0.5,
            min_frames - 0.5,
        ],
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Per-frame Rectification Energy")

    ax.set_xlabel("Rollout probe time (seconds)")
    ax.set_ylabel("Frame position in 5s window")
    title_suffix = f" — {run_name}" if run_name else ""
    ax.set_title(f"Teacher Rectification Map{title_suffix}")

    plt.tight_layout()
    out = os.path.join(output_dir, "teacher_rectification_map.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3 — concentration curve
# ---------------------------------------------------------------------------

def plot_concentration(records: list[dict], output_dir: str, run_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    R_vals = [r["rectification_energy"] for r in records]
    if len(R_vals) < 2:
        print("Too few records for concentration plot — skipping.")
        return

    # Sort descending
    sorted_R = sorted(R_vals, reverse=True)
    total = sum(sorted_R)
    if total == 0:
        print("All R_t are zero — skipping concentration plot.")
        return

    cumulative = np.cumsum(sorted_R) / total
    frac_points = np.arange(1, len(sorted_R) + 1) / len(sorted_R)

    # Top-20% statistic
    cutoff = int(math.ceil(0.2 * len(sorted_R)))
    top20_fraction = cumulative[cutoff - 1] if cutoff <= len(cumulative) else 1.0
    print(f"Top 20% probe points explain {100 * top20_fraction:.1f}% of total rectification energy.")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(frac_points, cumulative, linewidth=2, color="#7C3AED")
    ax.axvline(0.2, color="#6B7280", linestyle="--", linewidth=1, label="top 20%")
    ax.axhline(top20_fraction, color="#6B7280", linestyle=":", linewidth=1)
    ax.annotate(
        f"top 20% → {100*top20_fraction:.1f}%",
        xy=(0.2, top20_fraction),
        xytext=(0.3, top20_fraction - 0.08),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )
    ax.set_xlabel("Fraction of probe points (sorted by $R_t$ descending)")
    ax.set_ylabel("Cumulative fraction of total $R_t$")
    title_suffix = f" — {run_name}" if run_name else ""
    ax.set_title(f"Rectification Energy Concentration{title_suffix}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    out = os.path.join(output_dir, "teacher_rectification_concentration.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

import math  # noqa: E402  (imported late so non-matplotlib deps don't block help text)


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.dirname(args.records)
    os.makedirs(output_dir, exist_ok=True)

    records = load_scored_records(args.records)
    if not records:
        print(f"No scored records found in {args.records}. Run the scorer first.")
        sys.exit(1)

    print(f"Loaded {len(records)} scored records.")
    # Sort by rollout time so plots are chronological
    records.sort(key=lambda r: r.get("rollout_time_sec", 0))

    plot_curve(records, output_dir, args.run_name)
    plot_map(records, output_dir, args.run_name)
    plot_concentration(records, output_dir, args.run_name)
    print("Done.")


if __name__ == "__main__":
    main()
