"""
Teacher Rectification Probe — diagnostic-only, zero-gradient, offline two-stage.

Stage 1 (during inference):  CausalInferencePipeline calls maybe_capture() after
    each final denoised AR block.  This saves lightweight detached CPU windows and
    metadata.  The real generation trajectory is never touched.

Stage 2 (offline):  tools/score_teacher_rectification.py loads the saved windows,
    runs the frozen bidirectional teacher (real_score, Wan2.1-T2V-14B), and writes
    JSON records + CSV summary.  Plots live in tools/plot_teacher_rectification.py.

Trajectory-invariance guarantee
--------------------------------
* The probe uses only .detach().clone() on final generated latents.
* All probe tensors are immediately moved to CPU.
* An isolated torch.Generator (never the global RNG) produces probe noise.
* The probe never writes to kv_cache, crossattn_cache, or any student tensor.
* When enabled=False the code path is a single boolean check with negligible cost.

Latent geometry (Wan2.1, standard settings)
--------------------------------------------
* VAE temporal compression: 4x  →  latent FPS = pixel_fps / 4 = 4.0 Hz
  (81 pixel frames → 21 latent frames ≈ 5.0 s at 16 fps)
* Each framewise AR block: nfpb=1 latent / 0.25 s
* Each chunkwise AR block: nfpb=3 latents / 0.75 s
* ~5 s window  →  20 latent frames  →  ~20 framewise or ~7 chunkwise blocks
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import deque
from typing import Any, Dict, List, Optional

import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rectification_energy(
    teacher_pred_x0: torch.Tensor,
    window: torch.Tensor,
    num_frame_per_block: int = 1,
    eps: float = 1e-8,
) -> tuple[float, float, float, float, list[float]]:
    """
    Compute frontier (last AR block) and whole-window rectification energy.

    Both tensors are expected in shape [B, F, C, H, W] (float32, CPU).
    The primary metric scores only the last `num_frame_per_block` latent frames
    (the causal frontier).  The teacher still received the full window — only the
    metric slice changes.

    Returns:
        R_frontier      — primary metric (last AR block only)
        delta_norm      — ||teacher_frontier - student_frontier||
        window_norm     — ||student_frontier||
        R_whole_window  — secondary/debug whole-window metric (preserved for comparison)
        per_frame       — per-frame R across the full window (dim 1)
    """
    t = teacher_pred_x0.float()
    w = window.float()

    # --- frontier slice: last `num_frame_per_block` latent frames ---
    F_block = num_frame_per_block
    t_frontier = t[:, -F_block:]   # [B, F_block, C, H, W]
    w_frontier = w[:, -F_block:]   # [B, F_block, C, H, W]
    frontier_delta = t_frontier - w_frontier

    delta_norm  = frontier_delta.norm().item()
    window_norm = w_frontier.norm().item()
    R_frontier  = delta_norm / (window_norm + eps)

    # --- whole-window (secondary, for debugging) ---
    delta_whole  = t - w
    R_whole      = delta_whole.norm().item() / (w.norm().item() + eps)

    # --- per-frame across full window (dim 1) ---
    num_frames = w.shape[1]
    per_frame = []
    for f in range(num_frames):
        d_f = (t[:, f] - w[:, f])
        r_f = d_f.norm().item() / (w[:, f].norm().item() + eps)
        per_frame.append(r_f)

    return R_frontier, delta_norm, window_norm, R_whole, per_frame


# ---------------------------------------------------------------------------
# main probe class
# ---------------------------------------------------------------------------

class TeacherRectificationProbe:
    """
    Lightweight offline-mode probe for Teacher Rectification Energy.

    Parameters
    ----------
    enabled : bool
    probe_every_blocks : int
        Capture a window once every this many AR blocks.
    window_seconds : float
        Target window duration in seconds (approximate; rounded to whole blocks).
    teacher_timestep : int
        The flow-matching "t" (0–1000 scale) at which to query the teacher.
    probe_seed : int
        Seed for the isolated RNG that generates probe noise.
    output_dir : str
        Directory where windows/ and probe_records.json are written.
    latent_fps : float
        Latent frames per second.  Default 4.0 (Wan2.1 with 16 fps pixel video).
    num_frame_per_block : int
        AR block size in latent frames.
    sample_id : str
        Identifier for the current generation run (e.g., prompt slug).
    """

    def __init__(
        self,
        enabled: bool = False,
        probe_every_blocks: int = 4,
        window_seconds: float = 5.0,
        teacher_timestep: int = 500,
        probe_seed: int = 1234,
        output_dir: str = "outputs/teacher_rectification",
        latent_fps: float = 4.0,
        num_frame_per_block: int = 1,
        sample_id: str = "sample",
    ):
        self.enabled = enabled
        self.probe_every_blocks = probe_every_blocks
        self.window_seconds = window_seconds
        self.teacher_timestep = teacher_timestep
        self.probe_seed = probe_seed
        self.output_dir = output_dir
        self.latent_fps = latent_fps
        self.num_frame_per_block = num_frame_per_block
        self.sample_id = sample_id

        if not enabled:
            return

        # Derived constants — computed once.
        # Window size in latent frames, rounded up to a whole number of blocks.
        target_latent_frames = window_seconds * latent_fps
        blocks_needed = max(1, math.ceil(target_latent_frames / num_frame_per_block))
        self._window_blocks = blocks_needed
        self._window_latent_frames = blocks_needed * num_frame_per_block
        self._actual_window_sec = self._window_latent_frames / latent_fps

        # Rolling deque of (block_index, cpu_tensor [B, nfpb, C, H, W])
        self._block_buffer: deque = deque()
        self._buffer_total_frames: int = 0

        # Saved records (written by finalize())
        self._records: List[Dict[str, Any]] = []
        self._logged_shape: bool = False

        # Prepare output directories
        self._windows_dir = os.path.join(output_dir, "windows")
        os.makedirs(self._windows_dir, exist_ok=True)

        log.info(
            "[TeacherRectProbe] Probe enabled. "
            "Window target=%.1fs → %d blocks → %d latent frames (actual=%.3fs). "
            "probe_every_blocks=%d, tau=%d, seed=%d, out=%s",
            window_seconds, self._window_blocks, self._window_latent_frames,
            self._actual_window_sec, probe_every_blocks, teacher_timestep,
            probe_seed, output_dir,
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def maybe_capture(
        self,
        final_denoised_block: torch.Tensor,
        global_block: int,
        rollout_time_sec: float,
        prompt_embeds: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Called immediately after each final denoised AR block is available.

        Parameters
        ----------
        final_denoised_block : torch.Tensor
            Shape [B, nfpb, C, H, W] — the final clean prediction for this block.
            Must be the exact tensor committed to the generated video.
        global_block : int
            0-indexed block counter.
        rollout_time_sec : float
            Wall-clock rollout position in seconds (block_end_latent_frame / latent_fps).
        prompt_embeds : torch.Tensor or None
            The prompt_embeds tensor from conditional_dict["prompt_embeds"].
            Saved once (for offline scoring); not used here.
        """
        if not self.enabled:
            return

        # --- store block in rolling buffer (CPU, detached) ---
        cpu_block = final_denoised_block.detach().to("cpu", non_blocking=False)
        self._block_buffer.append((global_block, cpu_block))
        self._buffer_total_frames += cpu_block.shape[1]

        # Trim oldest blocks if buffer exceeds window size
        while self._buffer_total_frames - self._block_buffer[0][1].shape[1] >= self._window_latent_frames:
            _, removed = self._block_buffer.popleft()
            self._buffer_total_frames -= removed.shape[1]

        # --- log shape once ---
        if not self._logged_shape:
            log.info(
                "[TeacherRectProbe] window shape = [B=%d, F=%d, C=%d, H=%d, W=%d], "
                "actual duration = %.3f sec, tau = %d",
                *cpu_block.shape[:1], self._window_latent_frames,
                *cpu_block.shape[2:], self._actual_window_sec, self.teacher_timestep,
            )
            self._logged_shape = True

        # --- decide whether to probe ---
        if (global_block + 1) % self.probe_every_blocks != 0:
            return
        if self._buffer_total_frames < self._window_latent_frames:
            log.debug(
                "[TeacherRectProbe] block %d: not enough history yet "
                "(%d / %d latent frames). Skipping.",
                global_block, self._buffer_total_frames, self._window_latent_frames,
            )
            return

        # --- build window tensor ---
        window_frames = torch.cat(
            [blk for _, blk in self._block_buffer], dim=1
        )[:, -self._window_latent_frames:]  # [B, F_win, C, H, W]

        window_start_frame = (
            (global_block + 1) * self.num_frame_per_block - self._window_latent_frames
        )
        window_start_sec = window_start_frame / self.latent_fps
        window_end_sec = (global_block + 1) * self.num_frame_per_block / self.latent_fps

        # --- build metadata dict ---
        meta = {
            "sample_id": self.sample_id,
            "global_block": global_block,
            "rollout_time_sec": round(rollout_time_sec, 4),
            "window_start_sec": round(window_start_sec, 4),
            "window_end_sec": round(window_end_sec, 4),
            "actual_window_sec": round(self._actual_window_sec, 4),
            "teacher_timestep": self.teacher_timestep,
            "probe_seed": self.probe_seed,
            "window_shape": list(window_frames.shape),
            "num_frame_per_block": self.num_frame_per_block,
        }

        # --- save window + metadata for offline scoring ---
        tag = f"block{global_block:05d}"
        window_path = os.path.join(self._windows_dir, f"{self.sample_id}_{tag}.pt")
        embed_path = None
        if prompt_embeds is not None:
            embed_path = os.path.join(self._windows_dir, f"{self.sample_id}_prompt_embeds.pt")
            if not os.path.exists(embed_path):
                torch.save(prompt_embeds.detach().to("cpu"), embed_path)
        torch.save({
            "window": window_frames,
            "meta": meta,
            "embed_path": embed_path,
        }, window_path)

        meta["window_path"] = window_path
        meta["embed_path"] = embed_path
        meta["status"] = "pending_scoring"
        self._records.append(meta)

        log.info(
            "[TeacherRectProbe] Captured block %d → %.2f s window [%.2f–%.2f s], "
            "shape %s, saved to %s",
            global_block, self._actual_window_sec,
            window_start_sec, window_end_sec,
            list(window_frames.shape), window_path,
        )

    def finalize(self) -> str:
        """
        Write the probe manifest (all pending records) to JSON.

        Returns the path to the JSON file.
        """
        if not self.enabled:
            return ""

        manifest_path = os.path.join(self.output_dir, "probe_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(self._records, f, indent=2)

        log.info(
            "[TeacherRectProbe] finalize(): wrote %d records to %s",
            len(self._records), manifest_path,
        )
        return manifest_path

    # ------------------------------------------------------------------
    # deterministic probe noise (public for offline scorer to reuse)
    # ------------------------------------------------------------------

    @staticmethod
    def make_probe_noise(
        shape: tuple,
        seed: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Generate deterministic probe noise using an isolated Generator.

        Same seed + shape → identical tensor every time.
        Never touches the global RNG state.
        """
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        eps = torch.randn(shape, generator=gen, device=device, dtype=dtype)
        return eps
