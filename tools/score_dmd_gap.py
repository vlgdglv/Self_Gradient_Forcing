"""
DMD Score-Gap Map — Offline Scorer
====================================
Reads windows captured by TeacherRectificationProbe and computes the
DMD Score-Gap: the disagreement between real_score and fake_score on each
saved latent window.

Exact SGF/DMD semantics reproduced from model/dmd.py:
  - real_score:  WanDiffusionWrapper(model_name=real_name, is_causal=False)
                 Always uses CFG:
                   pred_real = pred_real_cond + (pred_real_cond - pred_real_uncond) * real_guidance_scale
  - fake_score:  WanDiffusionWrapper(model_name=fake_name, is_causal=False)
                 Default fake_guidance_scale=0.0 (conditional only).
  - Normalizer (DMD paper eq. 8):
                 |estimated_clean - pred_real|.mean(dim=[1,2,3,4], keepdim=True)
  - grad:        pred_fake - pred_real
  - Primary metric: dmd_gap_rms = sqrt(mean((grad_normalized[:, -F_block:])^2))

Two modes:
  --score_mode fake_score       (preferred; requires fake_score checkpoint)
  --score_mode generator_proxy  (fallback; uses causal generator as proxy)

Usage
-----
# Exact fake-score mode:
python tools/score_dmd_gap.py \\
    --manifest outputs/teacher_rectification/sample0000/probe_manifest.json \\
    --output_dir outputs/dmd_score_gap/sample0000 \\
    --real_model_name Wan2.1-T2V-14B \\
    --fake_model_name Wan2.1-T2V-1.3B \\
    --fake_ckpt checkpoints/critic/critic_step050000.pt \\
    --score_mode fake_score

# Generator-proxy mode (no fake_score checkpoint needed):
python tools/score_dmd_gap.py \\
    --manifest outputs/teacher_rectification/sample0000/probe_manifest.json \\
    --output_dir outputs/dmd_score_gap/sample0000 \\
    --real_model_name Wan2.1-T2V-14B \\
    --generator_ckpt checkpoints/model.pt \\
    --negative_prompt "" \\
    --score_mode generator_proxy

Notes
-----
- fake_score and real_score are both WanModel (bidirectional, is_causal=False).
- Generator proxy (CausalWanModel) is fed the full 5s window as a single
  independent batch; no KV-cache is constructed. The result is a proxy only.
- Metrics scored on frontier slice only: window[:, -frames_per_ar_block:].
- No backward pass. Runs fully under torch.inference_mode().
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F

from utils.teacher_rectification_probe import TeacherRectificationProbe
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
from utils.scheduler import FlowMatchScheduler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_real_score(
    model_name: str, device: torch.device, dtype: torch.dtype
) -> WanDiffusionWrapper:
    """Frozen bidirectional real score (Wan2.1-T2V-14B by default)."""
    log.info("Loading real_score  %s …", model_name)
    m = WanDiffusionWrapper(model_name=model_name, is_causal=False)
    m.model.requires_grad_(False)
    m.eval()
    m = m.to(device=device, dtype=dtype)
    log.info("real_score loaded.")
    return m


def load_fake_score(
    model_name: str, ckpt_path: str | None,
    device: torch.device, dtype: torch.dtype
) -> WanDiffusionWrapper:
    """
    Bidirectional fake-score critic (is_causal=False).
    If ckpt_path is provided, load trained critic weights on top of the
    pretrained base (matching distillation.py trainer behaviour).
    """
    log.info("Loading fake_score base  %s …", model_name)
    m = WanDiffusionWrapper(model_name=model_name, is_causal=False)
    if ckpt_path:
        log.info("Loading fake_score critic weights from  %s …", ckpt_path)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Trainer saves either raw state_dict or wrapped in a key
        if "fake_score" in sd:
            sd = sd["fake_score"]
        elif "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        m.model.load_state_dict(sd, strict=False)
    m.model.requires_grad_(False)
    m.eval()
    m = m.to(device=device, dtype=dtype)
    log.info("fake_score loaded (checkpoint: %s).", ckpt_path or "none — pretrained only")
    return m


def load_generator_proxy(
    model_name: str, ckpt_path: str | None,
    device: torch.device, dtype: torch.dtype
) -> WanDiffusionWrapper:
    """
    Causal generator used as fake-side proxy.
    WARNING: The causal generator expects block-autoregressive KV context that
    is unavailable offline.  This is fed the full 5s window without KV-cache
    and is a rough diagnostic proxy, NOT a rigorous fake score.
    """
    log.info("Loading generator proxy  %s …", model_name)
    m = WanDiffusionWrapper(model_name=model_name, is_causal=True)
    if ckpt_path:
        log.info("Loading generator weights from  %s …", ckpt_path)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        for key in ("generator_ema", "generator", "model"):
            if key in sd:
                sd = sd[key]
                break
        fixed = {}
        for k, v in sd.items():
            fixed[k.replace("model._fsdp_wrapped_module.", "model.", 1)] = v
        m.model.load_state_dict(fixed, strict=False)
    m.model.requires_grad_(False)
    m.eval()
    m = m.to(device=device, dtype=dtype)
    log.info("Generator proxy loaded.")
    return m


def load_text_encoder(device: torch.device) -> WanTextEncoder:
    log.info("Loading text encoder …")
    enc = WanTextEncoder()
    enc.eval()
    enc = enc.to(device=device)
    log.info("Text encoder loaded.")
    return enc


# ---------------------------------------------------------------------------
# Conditioning helpers
# ---------------------------------------------------------------------------

def encode_text(
    text_encoder: WanTextEncoder,
    prompts: list[str],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    with torch.inference_mode():
        out = text_encoder(text_prompts=prompts)
    return out["prompt_embeds"].to(device=device, dtype=dtype)


def build_unconditional_embeds(
    text_encoder: WanTextEncoder,
    negative_prompt: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    cache: dict,
) -> torch.Tensor:
    if negative_prompt not in cache:
        cache[negative_prompt] = encode_text(
            text_encoder, [negative_prompt], device, dtype
        )
    base = cache[negative_prompt]   # [1, L, D]
    return base.expand(batch_size, -1, -1)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _compute_dmd_gap_metrics(
    pred_fake_x0: torch.Tensor,      # [B, F, C, H, W] float32 CPU
    pred_real_x0: torch.Tensor,      # [B, F, C, H, W] float32 CPU
    estimated_clean: torch.Tensor,   # [B, F, C, H, W] float32 CPU  (= W_t)
    num_frame_per_block: int,
    eps: float = 1e-8,
) -> dict:
    """
    Reproduce exact DMD normalisation from model/dmd.py _compute_kl_grad():

        grad          = pred_fake - pred_real
        normalizer    = |estimated_clean - pred_real|.mean(dim=[1,2,3,4])
        grad_norm     = grad / normalizer

    Primary metrics are computed on the causal frontier slice only:
        frontier = slice [:, -num_frame_per_block:]

    Returns a flat dict of scalars.
    """
    fb = num_frame_per_block

    # Full-window tensors (float32)
    fake = pred_fake_x0.float()
    real = pred_real_x0.float()
    clean = estimated_clean.float()

    # --- Frontier slices ---
    fake_fr = fake[:, -fb:]    # [B, fb, C, H, W]
    real_fr = real[:, -fb:]
    clean_fr = clean[:, -fb:]

    # --- DMD gradient (full window, same as training) ---
    grad_full = fake - real   # [B, F, C, H, W]

    # --- Normaliser (eq. 8, full window, keepdim) ---
    p_real_full = clean - real                                      # [B, F, C, H, W]
    normalizer_val = torch.abs(p_real_full).mean(dim=[1, 2, 3, 4])  # [B]
    normalizer_scalar = normalizer_val.mean().item()

    # --- Frontier grad + normalised grad ---
    grad_fr = grad_full[:, -fb:]   # [B, fb, C, H, W]
    p_real_fr = clean_fr - real_fr
    normalizer_fr = torch.abs(p_real_fr).mean(dim=[1, 2, 3, 4], keepdim=True)  # [B,1,1,1,1]
    grad_fr_normed = grad_fr / (normalizer_fr + eps)                             # [B, fb, C, H, W]

    # Metric A — raw gap (frontier)
    raw_gap = (grad_fr.norm() / (real_fr.norm() + eps)).item()

    # Metric B — native DMD-normalised gap (frontier)
    dmd_gap_rms = grad_fr_normed.pow(2).mean().sqrt().item()
    dmd_gap_l1 = grad_fr_normed.abs().mean().item()

    # Metric C — cosine disagreement (frontier, flatten all but batch)
    real_flat = real_fr.flatten(1)   # [B, N]
    fake_flat = fake_fr.flatten(1)
    cosine = F.cosine_similarity(real_flat, fake_flat, dim=1).mean().item()
    angular_disagreement = 1.0 - cosine

    # Metric D — normaliser value (for diagnosing whether spikes come from tiny denominator)
    real_residual_norm = normalizer_scalar

    return {
        "raw_score_gap": round(raw_gap, 8),
        "dmd_gap_rms": round(dmd_gap_rms, 8),
        "dmd_gap_l1": round(dmd_gap_l1, 8),
        "real_fake_cosine": round(cosine, 8),
        "angular_disagreement": round(angular_disagreement, 8),
        "normalizer": round(real_residual_norm, 8),
    }


# ---------------------------------------------------------------------------
# Per-window scoring
# ---------------------------------------------------------------------------

@torch.inference_mode()
def score_window(
    window: torch.Tensor,           # [B, F, C, H, W] CPU
    real_score: WanDiffusionWrapper,
    fake_model: WanDiffusionWrapper,
    scheduler: FlowMatchScheduler,
    teacher_timestep: int,
    probe_seed: int,
    prompt_embeds: torch.Tensor,    # [B, L, D] on device
    uncond_embeds: torch.Tensor,    # [B, L, D] on device
    device: torch.device,
    dtype: torch.dtype,
    num_frame_per_block: int,
    real_guidance_scale: float,
    score_mode: str,
    sanity_verbose: bool = False,
) -> dict | None:
    assert not torch.is_grad_enabled()

    B, F, C, H, W = window.shape
    window_dev = window.to(device=device, dtype=dtype)

    # ---- Deterministic probe noise (identical to Teacher Rectification) ----
    eps = TeacherRectificationProbe.make_probe_noise(
        shape=(B * F, C, H, W), seed=probe_seed, device=device, dtype=dtype
    ).unflatten(0, (B, F))  # [B, F, C, H, W]

    # ---- Add noise with native FlowMatchScheduler.add_noise ----
    t_tensor = torch.full((B * F,), teacher_timestep, device=device, dtype=torch.long)
    noisy_window = scheduler.add_noise(
        window_dev.flatten(0, 1),
        eps.flatten(0, 1),
        t_tensor,
    ).unflatten(0, (B, F))   # [B, F, C, H, W]

    # ---- Timestep tensor [B, F] for WanDiffusionWrapper ----
    #      uniform_timestep=True → only timestep[:, 0] is used internally.
    timestep = torch.full((B, F), teacher_timestep, device=device, dtype=torch.long)

    cond_dict = {"prompt_embeds": prompt_embeds}
    uncond_dict = {"prompt_embeds": uncond_embeds}

    # ---- Real score with CFG (exact replica of model/dmd.py lines 98–112) ----
    _, pred_real_cond = real_score(
        noisy_image_or_video=noisy_window,
        conditional_dict=cond_dict,
        timestep=timestep,
    )
    _, pred_real_uncond = real_score(
        noisy_image_or_video=noisy_window,
        conditional_dict=uncond_dict,
        timestep=timestep,
    )
    pred_real = (
        pred_real_cond + (pred_real_cond - pred_real_uncond) * real_guidance_scale
    )

    # ---- Fake score (conditional only, fake_guidance_scale=0.0 by default) ----
    _, pred_fake = fake_model(
        noisy_image_or_video=noisy_window,
        conditional_dict=cond_dict,
        timestep=timestep,
    )

    # ---- Move to CPU float32 for metric computation ----
    pred_real_cpu = pred_real.detach().to("cpu", dtype=torch.float32)
    pred_fake_cpu = pred_fake.detach().to("cpu", dtype=torch.float32)
    window_cpu = window.float()

    if sanity_verbose:
        f_block = num_frame_per_block
        log.info(
            "[DMD-Gap sanity]\n"
            "  score_mode          = %s\n"
            "  window shape        = %s\n"
            "  noisy shape         = %s\n"
            "  pred_real shape     = %s\n"
            "  pred_fake shape     = %s\n"
            "  frontier slice      = [:, -%d:]\n"
            "  timestep            = %d",
            score_mode, list(window.shape), list(noisy_window.shape),
            list(pred_real_cpu.shape), list(pred_fake_cpu.shape),
            f_block, teacher_timestep,
        )
        assert torch.isfinite(pred_real_cpu).all(), "pred_real has non-finite values"
        assert torch.isfinite(pred_fake_cpu).all(), "pred_fake has non-finite values"

    metrics = _compute_dmd_gap_metrics(
        pred_fake_x0=pred_fake_cpu,
        pred_real_x0=pred_real_cpu,
        estimated_clean=window_cpu,
        num_frame_per_block=num_frame_per_block,
    )

    if sanity_verbose:
        log.info(
            "  normalizer          = %.6f\n"
            "  raw_score_gap       = %.6f\n"
            "  dmd_gap_rms         = %.6f\n"
            "  real_fake_cosine    = %.6f",
            metrics["normalizer"], metrics["raw_score_gap"],
            metrics["dmd_gap_rms"], metrics["real_fake_cosine"],
        )

    if not all(math.isfinite(v) for v in metrics.values()):
        log.warning("Non-finite metric detected — skipping record.")
        return None

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DMD Score-Gap offline scorer")
    p.add_argument("--manifest", required=True,
                   help="probe_manifest.json from TeacherRectificationProbe.finalize()")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--score_mode", default="fake_score",
                   choices=["fake_score", "generator_proxy"],
                   help="fake_score (preferred) or generator_proxy (fallback)")

    p.add_argument("--real_model_name", default="Wan2.1-T2V-14B")
    p.add_argument("--fake_model_name", default="Wan2.1-T2V-1.3B",
                   help="Base model for fake_score (or generator proxy)")
    p.add_argument("--fake_ckpt", default=None,
                   help="Trained critic checkpoint for fake_score mode")
    p.add_argument("--generator_ckpt", default=None,
                   help="Generator checkpoint for generator_proxy mode")

    p.add_argument("--real_guidance_scale", type=float, default=3.0,
                   help="CFG scale for real_score (matches training default)")
    p.add_argument("--negative_prompt", type=str,
                   default=(
                       "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
                       "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
                       "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
                       "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
                   ),
                   help="Negative prompt for real_score CFG (matches SGF training default)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--skip_scored", action="store_true")
    p.add_argument("--sanity_windows", type=int, default=3,
                   help="Print verbose diagnostics for the first N windows")
    return p.parse_args()


def main():
    args = parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    # ---- Announce mode ----
    if args.score_mode == "fake_score":
        print(
            "[DMD Score-Gap] Mode: EXACT FAKE SCORE\n"
            f"  real_score : {args.real_model_name}\n"
            f"  fake_score : {args.fake_model_name}  ckpt={args.fake_ckpt or 'pretrained-only'}\n"
            f"  real CFG   : guidance_scale={args.real_guidance_scale}\n"
            f"  metric     : dmd_gap_rms  (frontier, last AR block)"
        )
    else:
        print(
            "[DMD Score-Gap] Mode: GENERATOR PROXY (fallback)\n"
            f"  real_score : {args.real_model_name}\n"
            f"  generator  : {args.fake_model_name}  ckpt={args.generator_ckpt or 'pretrained-only'}\n"
            f"  real CFG   : guidance_scale={args.real_guidance_scale}\n"
            "  NOTE: Generator is causal; served the full 5s window without KV-cache.\n"
            "        Results labelled 'generator_proxy_gap'. NOT exact DMD score gap."
        )

    # ---- Load manifest ----
    with open(args.manifest) as f:
        records = json.load(f)
    log.info("Manifest: %d records from %s", len(records), args.manifest)

    output_dir = args.output_dir or os.path.dirname(args.manifest)
    os.makedirs(output_dir, exist_ok=True)

    # ---- Scheduler (same shift=8.0 as WanDiffusionWrapper default) ----
    scheduler = FlowMatchScheduler(shift=8.0)

    # ---- Load models ----
    real_score = load_real_score(args.real_model_name, device, dtype)

    if args.score_mode == "fake_score":
        fake_model = load_fake_score(args.fake_model_name, args.fake_ckpt, device, dtype)
    else:
        fake_model = load_generator_proxy(args.fake_model_name, args.generator_ckpt, device, dtype)

    # ---- Load text encoder for negative-prompt conditioning ----
    text_encoder = load_text_encoder(device)

    # ---- Embedding caches ----
    embed_cache: dict[str, torch.Tensor] = {}    # prompt path → tensor
    uncond_cache: dict[str, torch.Tensor] = {}   # negative_prompt str → tensor

    # ---- Score loop ----
    scored_records = []
    n_sanity = args.sanity_windows

    for i, rec in enumerate(records):
        if args.skip_scored and rec.get("status") == "scored":
            scored_records.append(rec)
            continue

        window_path = rec.get("window_path")
        if not window_path or not os.path.exists(window_path):
            log.warning("Window missing: %s — skip.", window_path)
            rec["status"] = "missing_window"
            scored_records.append(rec)
            continue

        saved = torch.load(window_path, map_location="cpu", weights_only=False)
        window: torch.Tensor = saved["window"]   # [B, F, C, H, W]
        embed_path: str | None = saved.get("embed_path")

        num_frame_per_block = int(
            rec.get("num_frame_per_block",
                    saved.get("meta", {}).get("num_frame_per_block", 1))
        )

        # Prompt embeddings
        if embed_path and embed_path not in embed_cache:
            embed_cache[embed_path] = torch.load(
                embed_path, map_location="cpu", weights_only=False
            )
        if embed_path and embed_path in embed_cache:
            prompt_embeds = embed_cache[embed_path].to(device=device, dtype=dtype)
        else:
            log.warning("No prompt embeds for record %d — using zeros (metrics meaningless).", i)
            B_win = window.shape[0]
            prompt_embeds = torch.zeros(B_win, 512, 4096, device=device, dtype=dtype)

        B_win = window.shape[0]
        uncond_embeds = build_unconditional_embeds(
            text_encoder, args.negative_prompt, B_win, device, dtype, uncond_cache
        )

        teacher_timestep = int(rec.get("teacher_timestep", 500))
        probe_seed = int(rec.get("probe_seed", 1234))

        log.info(
            "[%d/%d] block=%d  t=%.2fs  τ=%d  F_block=%d  mode=%s",
            i + 1, len(records),
            rec["global_block"], rec["rollout_time_sec"],
            teacher_timestep, num_frame_per_block, args.score_mode,
        )

        metrics = score_window(
            window=window,
            real_score=real_score,
            fake_model=fake_model,
            scheduler=scheduler,
            teacher_timestep=teacher_timestep,
            probe_seed=probe_seed,
            prompt_embeds=prompt_embeds,
            uncond_embeds=uncond_embeds,
            device=device,
            dtype=dtype,
            num_frame_per_block=num_frame_per_block,
            real_guidance_scale=args.real_guidance_scale,
            score_mode=args.score_mode,
            sanity_verbose=(n_sanity > 0),
        )
        n_sanity -= 1

        out_rec = dict(rec)
        if metrics is None:
            out_rec["status"] = "non_finite_skipped"
        else:
            # Rename primary metric when using generator proxy so it's unambiguous.
            if args.score_mode == "generator_proxy":
                metrics["generator_proxy_gap_rms"] = metrics.pop("dmd_gap_rms")
                metrics["generator_proxy_gap_l1"] = metrics.pop("dmd_gap_l1")
            out_rec.update(metrics)
            out_rec["score_mode"] = args.score_mode
            out_rec["frames_per_ar_block"] = num_frame_per_block
            out_rec["real_guidance_scale"] = args.real_guidance_scale
            out_rec["status"] = "scored"
            primary = metrics.get("dmd_gap_rms") or metrics.get("generator_proxy_gap_rms", float("nan"))
            log.info(
                "  primary_gap=%.6f  raw_gap=%.6f  cosine=%.4f  normalizer=%.6f",
                primary, metrics["raw_score_gap"],
                metrics["real_fake_cosine"], metrics["normalizer"],
            )

        scored_records.append(out_rec)

    # ---- Write JSON ----
    out_json = os.path.join(output_dir, "dmd_score_gap_records.json")
    with open(out_json, "w") as f:
        json.dump(scored_records, f, indent=2)
    log.info("Wrote %s", out_json)

    # ---- Write CSV ----
    primary_col = "dmd_gap_rms" if args.score_mode == "fake_score" else "generator_proxy_gap_rms"
    csv_keys = [
        "sample_id", "global_block", "rollout_time_sec",
        "window_start_sec", "window_end_sec",
        "teacher_timestep", "probe_seed",
        "score_mode", "frames_per_ar_block", "real_guidance_scale",
        "raw_score_gap", primary_col, "dmd_gap_l1",
        "generator_proxy_gap_l1",
        "real_fake_cosine", "angular_disagreement", "normalizer",
        "status",
    ]
    out_csv = os.path.join(output_dir, "dmd_score_gap_records.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_keys, extrasaction="ignore")
        writer.writeheader()
        for rec in scored_records:
            writer.writerow({k: rec.get(k, "") for k in csv_keys})
    log.info("Wrote %s", out_csv)

    # ---- Numerical summary ----
    good = [r for r in scored_records if r.get("status") == "scored"]
    N = len(good)
    log.info("Done. %d / %d records scored.", N, len(scored_records))

    if N == 0:
        return

    mode_tag = good[0].get("score_mode", args.score_mode)
    primary_key = "dmd_gap_rms" if mode_tag == "fake_score" else "generator_proxy_gap_rms"
    vals = [r[primary_key] for r in good if isinstance(r.get(primary_key), float)]
    times = [r["rollout_time_sec"] for r in good if isinstance(r.get(primary_key), float)]

    if len(vals) < 2:
        return

    import numpy as np
    vals_arr = np.array(vals)
    times_arr = np.array(times)

    # Early / late 20%
    n20 = max(1, int(math.ceil(0.2 * len(vals_arr))))
    early_med = float(np.median(vals_arr[:n20]))
    late_med = float(np.median(vals_arr[-n20:]))
    ratio = late_med / (early_med + 1e-12)

    # Pearson
    if len(vals_arr) >= 2:
        t_c = times_arr - times_arr.mean()
        v_c = vals_arr - vals_arr.mean()
        denom = (np.sqrt((t_c**2).sum()) * np.sqrt((v_c**2).sum()))
        pearson = float((t_c * v_c).sum() / (denom + 1e-12))
    else:
        pearson = float("nan")

    # Spearman (optional)
    try:
        from scipy.stats import spearmanr
        spearman = spearmanr(times_arr, vals_arr).correlation
    except ImportError:
        spearman = None

    # Concentration
    sorted_v = np.sort(vals_arr)[::-1]
    total_v = sorted_v.sum()
    cum = np.cumsum(sorted_v) / (total_v + 1e-12)
    def _top_k_contrib(frac):
        idx = max(0, int(math.ceil(frac * len(sorted_v))) - 1)
        return float(cum[idx])
    top10 = _top_k_contrib(0.10)
    top20 = _top_k_contrib(0.20)
    top30 = _top_k_contrib(0.30)

    print(
        f"\n[DMD Score-Gap Summary]\n"
        f"  N points       : {N}\n"
        f"  mode           : {mode_tag}\n"
        f"  primary metric : {primary_key}\n"
        f"\n"
        f"  early median   : {early_med:.6f}  (first {n20} points)\n"
        f"  late  median   : {late_med:.6f}  (last  {n20} points)\n"
        f"  late / early   : {ratio:.4f}\n"
        f"\n"
        f"  Pearson(time, gap)   : {pearson:.4f}\n"
        f"  Spearman(time, gap)  : {spearman if spearman is not None else 'n/a (scipy missing)'}\n"
        f"\n"
        f"  top 10% contribution : {100*top10:.1f}%\n"
        f"  top 20% contribution : {100*top20:.1f}%\n"
        f"  top 30% contribution : {100*top30:.1f}%"
    )
    print(f"\nTop 20% states explain {100*top20:.1f}% of total DMD score-gap magnitude.")


if __name__ == "__main__":
    main()

