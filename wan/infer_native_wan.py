import os
import math

import imageio
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.wan_wrapper import (
    WanDiffusionWrapper,
    WanTextEncoder,
    WanVAEWrapper,
)
from utils.scheduler import FlowMatchScheduler
from utils.distributed import launch_distributed_job


# ============================================================
# Config
# ============================================================


SEED = 42

# Wan2.1 T2V-1.3B native-quality setting
TIMESTEP_SHIFT = 5.0
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 6.0
NUM_SAMPLES = 1
CAPTION_PATH = "prompts/vbench_all_dimension.txt"
OUTPUT_FOLDER = f"outputs/wan21_native_vbench_step{TIMESTEP_SHIFT}_timeshift{NUM_INFERENCE_STEPS}_cfg{GUIDANCE_SCALE}"


# 81 RGB frames -> 21 latent frames
LATENT_SHAPE = [1, 21, 16, 60, 104]

FPS = 16

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，"
    "画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"
    "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
    "三条腿，背景人很多，倒着走"
)


# ============================================================
# Dataset
# ============================================================

class SimpleTextDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            self.texts = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(self.texts)} prompts from {data_path}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


# ============================================================
# Model
# ============================================================

def init_models(device):

    # --------------------------------------------------------
    # Native bidirectional Wan
    # --------------------------------------------------------
    model = WanDiffusionWrapper(
        model_name="Wan2.1-T2V-1.3B",
        is_causal=False,
        timestep_shift=TIMESTEP_SHIFT,
    ).to(device).to(torch.float32)

    model.eval()

    # --------------------------------------------------------
    # Text encoder
    # --------------------------------------------------------
    encoder = WanTextEncoder().to(device).to(torch.float32)
    encoder.eval()

    # --------------------------------------------------------
    # VAE
    # --------------------------------------------------------
    vae = WanVAEWrapper().to(device)
    vae.eval()

    # --------------------------------------------------------
    # Native FlowMatch sampling schedule
    # --------------------------------------------------------
    scheduler = FlowMatchScheduler(
        shift=TIMESTEP_SHIFT,
        sigma_min=0.0,
        extra_one_step=True,
    )

    scheduler.set_timesteps(
        num_inference_steps=NUM_INFERENCE_STEPS,
        denoising_strength=1.0,
    )

    scheduler.sigmas = scheduler.sigmas.to(device)

    # --------------------------------------------------------
    # Negative prompt embedding
    # --------------------------------------------------------
    unconditional_dict = encoder(
        text_prompts=[NEGATIVE_PROMPT]
    )

    return model, encoder, vae, scheduler, unconditional_dict


# ============================================================
# Video saving
# ============================================================

def save_video(video, path, fps=16):
    """
    video:
        [1, F, C, H, W]
        range [-1, 1]
    """

    video = video[0]

    video = (
        (video * 0.5 + 0.5)
        .clamp(0, 1)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )

    video = (video * 255).round().astype(np.uint8)

    imageio.mimsave(
        path,
        list(video),
        fps=fps,
        codec="libx264",
        quality=8,
    )


# ============================================================
# Main
# ============================================================

@torch.no_grad()
def main():

    launch_distributed_job()

    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.cuda.current_device()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if global_rank == 0:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    dist.barrier()

    model, encoder, vae, scheduler, unconditional_dict = init_models(
        device=device
    )

    dataset = SimpleTextDataset(CAPTION_PATH)

    num_local_samples = math.ceil(len(dataset) / world_size)

    iterator = tqdm(
        range(num_local_samples),
        disable=(global_rank != 0),
    )
    for local_idx in iterator:

        prompt_idx = local_idx * world_size + global_rank

        if prompt_idx >= len(dataset):
            continue

        prompt = dataset[prompt_idx]

        for sample_idx in range(NUM_SAMPLES):
        
            output_path = os.path.join(
                OUTPUT_FOLDER,
                f"{prompt}-{sample_idx}.mp4",
            )

            if os.path.exists(output_path):
                continue

            # ----------------------------------------------------
            # Deterministic per-prompt seed
            # ----------------------------------------------------
            sample_seed = SEED + prompt_idx

            generator = torch.Generator(device=device)
            generator.manual_seed(sample_seed)

            # ----------------------------------------------------
            # Text condition
            # ----------------------------------------------------
            conditional_dict = encoder(
                text_prompts=[prompt]
            )

            # ----------------------------------------------------
            # Initial Gaussian noise
            #
            # [B, latent_frames, C, H, W]
            #
            # 21 latent frames -> 81 RGB frames
            # ----------------------------------------------------
            latents = torch.randn(
                LATENT_SHAPE,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )

            # ----------------------------------------------------
            # Full-sequence bidirectional diffusion
            # ----------------------------------------------------
            for t in scheduler.timesteps:

                # every latent frame has the same diffusion timestep
                timestep = t * torch.ones(
                    [1, LATENT_SHAPE[1]],
                    device=device,
                    dtype=torch.float32,
                )

                # ------------------------------------------------
                # Batched CFG
                #
                # batch 0 = negative
                # batch 1 = positive
                # ------------------------------------------------
                batched_latents = torch.cat(
                    [latents, latents],
                    dim=0,
                )

                batched_timestep = torch.cat(
                    [timestep, timestep],
                    dim=0,
                )

                batched_condition = {
                    "prompt_embeds": torch.cat(
                        [
                            unconditional_dict["prompt_embeds"],
                            conditional_dict["prompt_embeds"],
                        ],
                        dim=0,
                    )
                }

                # ------------------------------------------------
                # WanDiffusionWrapper
                #
                # Current Self/Causal-Forcing wrapper returns
                # (flow_pred, pred_x0).
                # ------------------------------------------------
                model_output = model(
                    batched_latents,
                    batched_condition,
                    batched_timestep,
                )

                # Compatible with wrapper returning either:
                #   flow_pred
                # or
                #   (flow_pred, pred_x0)
                if isinstance(model_output, tuple):
                    flow_pred_batched = model_output[0]
                else:
                    flow_pred_batched = model_output

                flow_pred_uncond, flow_pred_cond = (
                    flow_pred_batched.chunk(2, dim=0)
                )

                # CFG
                flow_pred = (
                    flow_pred_uncond
                    + GUIDANCE_SCALE
                    * (flow_pred_cond - flow_pred_uncond)
                )

                # ------------------------------------------------
                # Flow matching Euler update
                # ------------------------------------------------
                latents = scheduler.step(
                    flow_pred.flatten(0, 1),
                    timestep.flatten(0, 1),
                    latents.flatten(0, 1),
                ).unflatten(
                    0,
                    flow_pred.shape[:2],
                )

            # ----------------------------------------------------
            # Decode:
            #
            # [1, 21, 16, 60, 104]
            #       ->
            # [1, 81, 3, 480, 832]
            # ----------------------------------------------------
            video = vae.decode_to_pixel(latents)

            save_video(
                video,
                output_path,
                fps=FPS,
            )

    dist.barrier()


if __name__ == "__main__":
    main()