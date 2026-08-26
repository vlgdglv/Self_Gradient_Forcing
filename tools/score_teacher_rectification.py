"""
Offline Teacher Rectification Scorer
=====================================
Stage 2 of the Teacher Rectification Probe.

Usage
-----
python tools/score_teacher_rectification.py \\
    --manifest outputs/teacher_rectification/<run>/probe_manifest.json \\
    --output_dir outputs/teacher_rectification/<run>

This script:
1. Reads the probe manifest written by TeacherRectificationProbe.finalize().
2. Loads the frozen bidirectional teacher (Wan2.1-T2V-14B real_score).
3. For each pending window:
   a. Generates deterministic probe noise (isolated RNG, same seed as probe).
   b. Applies the repository's FlowMatchScheduler.add_noise() at tau.
   c. Runs the teacher (WanDiffusionWrapper, bidirectional / non-causal).
   d. Computes Teacher Rectification Energy R_t and per-frame energy.
4. Writes probe_records.json + probe_records.csv.

All teacher calls are wrapped in torch.inference_mode() + teacher.eval().
No gradient is ever created.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys

# make repo root importable
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from utils.teacher_rectification_probe import TeacherRectificationProbe, _rectification_energy
from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import FlowMatchScheduler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# teacher loading
# ---------------------------------------------------------------------------

def load_teacher(model_name: str, device: torch.device, dtype: torch.dtype) -> WanDiffusionWrapper:
    """
    Load the frozen bidirectional teacher (is_causal=False).

    Mirrors BaseModel._initialize_models():
        self.real_score = WanDiffusionWrapper(model_name=self.real_model_name, is_causal=False)
        self.real_score.model.requires_grad_(False)
    """
    log.info("Loading teacher %s on %s …", model_name, device)
    teacher = WanDiffusionWrapper(model_name=model_name, is_causal=False)
    teacher.model.requires_grad_(False)
    teacher.eval()
    teacher = teacher.to(device=device, dtype=dtype)
    log.info("Teacher loaded.")
    return teacher


# ---------------------------------------------------------------------------
# single-window scoring
# ---------------------------------------------------------------------------

@torch.inference_mode()
def score_window(
    window: torch.Tensor,          # [B, F, C, H, W], CPU, possibly bf16/fp16
    teacher: WanDiffusionWrapper,
    scheduler: FlowMatchScheduler,
    teacher_timestep: int,
    probe_seed: int,
    prompt_embeds: torch.Tensor,    # [B, L, D], already on device
    device: torch.device,
    dtype: torch.dtype,
    num_frame_per_block: int = 1,
    log_frontier: bool = False,
) -> dict:
    """
    Compute Teacher Rectification Energy for one window.

    Primary metric (rectification_energy) is computed on the last AR block only
    (the causal frontier).  The teacher still receives the full ~5s window.
    A secondary whole_window_rectification_energy field is written for debugging.

    Returns a dict with keys:
        rectification_energy           — frontier (last AR block) metric  ← primary
        delta_norm                     — ||teacher_frontier - student_frontier||
        window_norm                    — ||student_frontier||
        whole_window_rectification_energy  — secondary, whole-window metric
        rectification_per_frame        — per latent-frame R across full window
    """
    assert not torch.is_grad_enabled(), "Must run inside inference_mode"

    B, F, C, H, W = window.shape
    window_dev = window.to(device=device, dtype=dtype)

    # 1. Deterministic probe noise — isolated RNG, never touches global state.
    eps = TeacherRectificationProbe.make_probe_noise(
        shape=(B * F, C, H, W),
        seed=probe_seed,
        device=device,
        dtype=dtype,
    )
    eps = eps.unflatten(0, (B, F))   # [B, F, C, H, W]

    # 2. Add noise at tau using the repository's FlowMatchScheduler.add_noise().
    #    signature: add_noise(original_samples, noise, timestep)
    #    where all three are [B*T, C, H, W] and timestep is [B*T].
    t_tensor = torch.full(
        (B * F,), teacher_timestep, device=device, dtype=torch.long
    )
    noisy_window = scheduler.add_noise(
        window_dev.flatten(0, 1),   # [B*F, C, H, W]
        eps.flatten(0, 1),          # [B*F, C, H, W]
        t_tensor,
    ).unflatten(0, (B, F))          # [B, F, C, H, W]

    # 3. Build conditional_dict that WanDiffusionWrapper.forward() expects.
    conditional_dict = {"prompt_embeds": prompt_embeds}

    # 4. Build per-frame timestep tensor [B, F].
    #    WanDiffusionWrapper.forward() with uniform_timestep=True uses timestep[:, 0].
    #    Since the teacher is non-causal (uniform_timestep=True), [B, F] is fine.
    timestep = torch.full(
        (B, F), teacher_timestep, device=device, dtype=torch.long
    )

    # 5. Run teacher — returns (flow_pred, pred_x0), both [B, F, C, H, W].
    #    We use the pred_x0 that WanDiffusionWrapper already computes inside forward().
    #    Teacher input is the FULL window (unchanged from before).
    _, teacher_pred_x0 = teacher(
        noisy_image_or_video=noisy_window,
        conditional_dict=conditional_dict,
        timestep=timestep,
    )

    # 6. Compute metrics on CPU in float32.
    teacher_pred_x0_cpu = teacher_pred_x0.detach().to("cpu", dtype=torch.float32)
    window_cpu = window.float()

    # Log frontier details on first scored window.
    if log_frontier:
        log.info(
            "[TeacherRectProbe] teacher window latent frames: %d\n"
            "                   AR block latent frames: %d\n"
            "                   scoring frontier frames: [-%d:]\n"
            "                   teacher input: full %d-frame window\n"
            "                   metric input: newest AR block only",
            F, num_frame_per_block, num_frame_per_block, F,
        )

    R, delta_norm, window_norm, R_whole, per_frame = _rectification_energy(
        teacher_pred_x0_cpu, window_cpu, num_frame_per_block=num_frame_per_block
    )

    if not math.isfinite(R):
        log.warning("R_t is not finite (R=%.6g) — skipping this record.", R)
        return None

    return {
        "rectification_energy": round(R, 8),
        "delta_norm": round(delta_norm, 6),
        "window_norm": round(window_norm, 6),
        "whole_window_rectification_energy": round(R_whole, 8),
        "rectification_per_frame": [round(x, 8) for x in per_frame],
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Offline teacher rectification scorer")
    p.add_argument("--manifest", required=True,
                   help="Path to probe_manifest.json written by the probe")
    p.add_argument("--output_dir", default=None,
                   help="Output dir for probe_records.json/.csv (default: same as manifest dir)")
    p.add_argument("--teacher_model_name", default="Wan2.1-T2V-14B",
                   help="Teacher model directory under wan_models/")
    p.add_argument("--device", default="cuda",
                   help="Device for teacher scoring (default: cuda)")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"],
                   help="dtype for teacher (default: bfloat16)")
    p.add_argument("--skip_scored", action="store_true",
                   help="Skip windows that already have rectification_energy in the manifest")
    return p.parse_args()


def main():
    args = parse_args()

    manifest_path = args.manifest
    output_dir = args.output_dir or os.path.dirname(manifest_path)
    os.makedirs(output_dir, exist_ok=True)

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    # Load manifest
    with open(manifest_path) as f:
        records = json.load(f)
    log.info("Loaded manifest: %d records from %s", len(records), manifest_path)

    # Load teacher once
    teacher = load_teacher(args.teacher_model_name, device=device, dtype=dtype)
    teacher_scheduler = teacher.scheduler
    teacher_scheduler.timesteps = teacher_scheduler.timesteps.to(device)
    teacher_scheduler.sigmas = teacher_scheduler.sigmas.to(device)

    # Cache prompt embeddings per path (avoid reloading)
    embed_cache: dict[str, torch.Tensor] = {}

    scored_records = []
    first_score = True
    for i, rec in enumerate(records):
        if args.skip_scored and rec.get("status") == "scored":
            scored_records.append(rec)
            continue

        window_path = rec.get("window_path")
        if not window_path or not os.path.exists(window_path):
            log.warning("Window file not found: %s — skipping.", window_path)
            rec["status"] = "missing_window"
            scored_records.append(rec)
            continue

        # Load window
        saved = torch.load(window_path, map_location="cpu", weights_only=False)
        window: torch.Tensor = saved["window"]    # [B, F, C, H, W]
        embed_path = saved.get("embed_path")

        # Derive num_frame_per_block from the stored window_shape.
        # window_shape is [B, F_win, C, H, W]; num_frame_per_block is stored in meta.
        num_frame_per_block = int(rec.get("num_frame_per_block", saved.get("meta", {}).get("num_frame_per_block", 1)))

        # Load prompt embeddings
        if embed_path and embed_path not in embed_cache:
            embed_cache[embed_path] = torch.load(embed_path, map_location="cpu", weights_only=False)
        if embed_path and embed_path in embed_cache:
            prompt_embeds = embed_cache[embed_path].to(device=device, dtype=dtype)
        else:
            # Fallback: zero embeddings (scores will be meaningless but won't crash)
            log.warning("No prompt embeddings for record %d — using zeros.", i)
            B = window.shape[0]
            prompt_embeds = torch.zeros(B, 512, 4096, device=device, dtype=dtype)

        log.info(
            "[%d/%d] Scoring block %d, window %s, tau=%d, F_block=%d …",
            i + 1, len(records),
            rec["global_block"], list(window.shape), rec["teacher_timestep"],
            num_frame_per_block,
        )

        metrics = score_window(
            window=window,
            teacher=teacher,
            scheduler=teacher_scheduler,
            teacher_timestep=rec["teacher_timestep"],
            probe_seed=rec["probe_seed"],
            prompt_embeds=prompt_embeds,
            device=device,
            dtype=dtype,
            num_frame_per_block=num_frame_per_block,
            log_frontier=first_score,
        )
        first_score = False

        if metrics is None:
            rec["status"] = "non_finite_skipped"
        else:
            rec.update(metrics)
            rec["status"] = "scored"
            log.info(
                "  R_t(frontier)=%.6f  R_t(whole)=%.6f  delta_norm=%.2f  window_norm=%.2f",
                rec["rectification_energy"], rec.get("whole_window_rectification_energy", float("nan")),
                rec["delta_norm"], rec["window_norm"],
            )

        scored_records.append(rec)

    # Write probe_records.json
    records_path = os.path.join(output_dir, "probe_records.json")
    with open(records_path, "w") as f:
        json.dump(scored_records, f, indent=2)
    log.info("Wrote %s", records_path)

    # Write probe_records.csv
    csv_path = os.path.join(output_dir, "probe_records.csv")
    scalar_keys = [
        "sample_id", "global_block", "rollout_time_sec",
        "window_start_sec", "window_end_sec", "actual_window_sec",
        "teacher_timestep", "probe_seed",
        "rectification_energy", "delta_norm", "window_norm",
        "whole_window_rectification_energy", "status",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        for rec in scored_records:
            writer.writerow({k: rec.get(k, "") for k in scalar_keys})
    log.info("Wrote %s", csv_path)

    n_scored = sum(1 for r in scored_records if r.get("status") == "scored")
    log.info("Done. %d / %d records scored.", n_scored, len(scored_records))


if __name__ == "__main__":
    main()
