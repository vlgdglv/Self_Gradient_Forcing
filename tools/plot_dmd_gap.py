"""
DMD Score-Gap Map — Visualization
====================================
Reads dmd_score_gap_records.json and produces three figures:

  1. dmd_score_gap_curve.png
       Primary DMD gap vs rollout time (+ median-normalized panel).
  2. dmd_score_gap_diagnostics.png
       raw_score_gap and angular_disagreement vs rollout time.
  3. dmd_score_gap_concentration.png
       Lorenz-style cumulative gap curve; prints top-10/20/30% stats.

Usage
-----
python tools/plot_dmd_gap.py \\
    --records outputs/dmd_score_gap/sample0000/dmd_score_gap_records.json \\
    --output_dir outputs/dmd_score_gap/sample0000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True,
                   help="dmd_score_gap_records.json")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--run_name", default="",
                   help="Optional label for plot titles")
    return p.parse_args()


def load_records(path: str) -> list[dict]:
    with open(path) as f:
        all_recs = json.load(f)
    good = [r for r in all_recs if r.get("status") == "scored"]
    return sorted(good, key=lambda r: r.get("rollout_time_sec", 0))


def _primary_key(records: list[dict]) -> tuple[str, str]:
    """Return (metric_key, display_title) for whichever mode was used."""
    mode = records[0].get("score_mode", "fake_score") if records else "fake_score"
    if mode == "fake_score":
        return "dmd_gap_rms", "DMD Score Gap  $G_t$  (Last AR Block)"
    else:
        return "generator_proxy_gap_rms", "Generator-Proxy Score Gap  (Last AR Block)"


# ---------------------------------------------------------------------------
# Plot 1 — main gap curve
# ---------------------------------------------------------------------------

def plot_curve(records: list[dict], output_dir: str, run_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pk, ptitle = _primary_key(records)
    times = [r["rollout_time_sec"] for r in records if isinstance(r.get(pk), float)]
    vals = [r[pk] for r in records if isinstance(r.get(pk), float)]

    K = min(3, len(vals))
    baseline = sorted(vals[:K])[K // 2] if K > 0 else 1.0
    baseline = baseline or 1.0
    vals_norm = [v / baseline for v in vals]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    sfx = f" — {run_name}" if run_name else ""

    ax0 = axes[0]
    ax0.plot(times, vals, marker="o", linewidth=1.5, markersize=5, color="#2563EB")
    ax0.set_ylabel(pk)
    ax0.set_title(f"{ptitle}{sfx}")
    ax0.grid(True, alpha=0.3)
    ax0.set_ylim(bottom=0)

    ax1 = axes[1]
    ax1.plot(times, vals_norm, marker="s", linewidth=1.5, markersize=5, color="#DC2626",
             label=f"normalized (÷ median of first {K} = {baseline:.4f})")
    ax1.set_ylabel(f"{pk} / early-median")
    ax1.set_xlabel("Rollout time (seconds)")
    ax1.set_title(f"Median-normalized {ptitle.split('(')[0].strip()}{sfx}")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    plt.tight_layout()
    out = os.path.join(output_dir, "dmd_score_gap_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2 — diagnostics: raw gap + angular disagreement
# ---------------------------------------------------------------------------

def plot_diagnostics(records: list[dict], output_dir: str, run_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sfx = f" — {run_name}" if run_name else ""
    times = [r["rollout_time_sec"] for r in records]

    raw = [r.get("raw_score_gap", float("nan")) for r in records]
    ang = [r.get("angular_disagreement", float("nan")) for r in records]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(times, raw, marker="o", linewidth=1.5, markersize=5, color="#059669")
    axes[0].set_ylabel("raw_score_gap\n||fake−real|| / ||real||")
    axes[0].set_title(f"Raw Score Gap{sfx}")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)

    axes[1].plot(times, ang, marker="s", linewidth=1.5, markersize=5, color="#D97706")
    axes[1].set_ylabel("angular_disagreement\n1 − cos(fake, real)")
    axes[1].set_xlabel("Rollout time (seconds)")
    axes[1].set_title(f"Angular Disagreement (1 − cosine){sfx}")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(bottom=0)

    plt.tight_layout()
    out = os.path.join(output_dir, "dmd_score_gap_diagnostics.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3 — concentration
# ---------------------------------------------------------------------------

def plot_concentration(records: list[dict], output_dir: str, run_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pk, _ = _primary_key(records)
    vals = [r[pk] for r in records if isinstance(r.get(pk), float)]
    if len(vals) < 2:
        print("Too few records for concentration plot — skipping.")
        return

    sorted_v = sorted(vals, reverse=True)
    total = sum(sorted_v)
    if total == 0:
        print("All gaps are zero — skipping concentration plot.")
        return

    cum = np.cumsum(sorted_v) / total
    frac = np.arange(1, len(sorted_v) + 1) / len(sorted_v)

    def top_k(k_frac):
        idx = max(0, int(math.ceil(k_frac * len(sorted_v))) - 1)
        return float(cum[idx])

    top10, top20, top30 = top_k(0.10), top_k(0.20), top_k(0.30)
    print(f"Top 10% states explain {100*top10:.1f}% of total DMD score-gap magnitude.")
    print(f"Top 20% states explain {100*top20:.1f}% of total DMD score-gap magnitude.")
    print(f"Top 30% states explain {100*top30:.1f}% of total DMD score-gap magnitude.")

    sfx = f" — {run_name}" if run_name else ""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(frac, cum, linewidth=2, color="#7C3AED")
    for kf, kv, ls in [(0.10, top10, ":"), (0.20, top20, "--"), (0.30, top30, "-.")]:
        ax.axvline(kf, color="#6B7280", linestyle=ls, linewidth=1)
        ax.axhline(kv, color="#6B7280", linestyle=ls, linewidth=1,
                   label=f"top {int(kf*100)}% → {100*kv:.1f}%")
    ax.set_xlabel("Fraction of probe states (sorted by gap ↓)")
    ax.set_ylabel("Cumulative fraction of total gap")
    ax.set_title(f"DMD Score-Gap Concentration{sfx}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    out = os.path.join(output_dir, "dmd_score_gap_concentration.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.dirname(args.records)
    os.makedirs(output_dir, exist_ok=True)

    records = load_records(args.records)
    if not records:
        print(f"No scored records in {args.records}.")
        sys.exit(1)

    mode = records[0].get("score_mode", "unknown")
    print(f"Loaded {len(records)} scored records.  score_mode={mode}")

    plot_curve(records, output_dir, args.run_name)
    plot_diagnostics(records, output_dir, args.run_name)
    plot_concentration(records, output_dir, args.run_name)
    print("Done.")


if __name__ == "__main__":
    main()
