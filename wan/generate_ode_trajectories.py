from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.scheduler import FlowMatchScheduler
from utils.distributed import launch_distributed_job
from torch.utils.data import Dataset
import torch.distributed as dist
from tqdm import tqdm
import argparse
import torch
import math
import os
import random


def sample_subset():
    src = "prompts/vidprom_filtered_extended.txt"
    dst = "prompts/vidprom_sample_16k.txt"

    random.seed(42)

    with open(src, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sampled = random.sample(lines, 16_000)

    with open(dst, "w", encoding="utf-8") as f:
        f.writelines(sampled)

    print(f"Sampled {len(sampled)} prompts -> {dst}")


class SimpleTextDataset(Dataset):
    def __init__(self, data_path, start_index=0, end_index=-1):
        self.texts = []
        with open(data_path, "r") as f:
            for line in f:
                self.texts.append(line.strip())
        if end_index != -1:
            self.texts = self.texts[:end_index]
        if start_index != 0:
            self.texts = self.texts[start_index:]
        print("Text data loaded from {}, starting from {} to {}, total {} entries.".format(data_path, start_index, end_index, len(self.texts)))
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def init_model(device, timestep_shift=5.0, num_inference_steps=48):
    assert timestep_shift == 5.0
    assert num_inference_steps == 48
    model = WanDiffusionWrapper(is_causal=False, timestep_shift=timestep_shift).to(device).to(torch.float32)
    encoder = WanTextEncoder().to(device).to(torch.float32)
    scheduler = FlowMatchScheduler(
        shift=timestep_shift, 
        sigma_min=0.0, 
        extra_one_step=True
    )
    scheduler.set_timesteps(
        num_inference_steps=num_inference_steps, 
        denoising_strength=1.0
    )
    scheduler.sigmas = scheduler.sigmas.to(device)

    # strange Chinese characters
    sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

    unconditional_dict = encoder(
        text_prompts=[sample_neg_prompt]
    )

    return model, encoder, scheduler, unconditional_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    parser.add_argument("--output_folder", type=str, default="dataset/vidprom_sample_16k")
    parser.add_argument("--caption_path", type=str, default="prompts/vidprom_sample_16k.txt")
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    launch_distributed_job()
    global_rank = dist.get_rank()
    
    device = torch.cuda.current_device()
    batched_cfg = True
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed + global_rank)
    torch.cuda.manual_seed(args.seed + global_rank)

    model, encoder, scheduler, unconditional_dict = init_model(device=device)

    dataset = SimpleTextDataset(args.caption_path, args.start_index, args.end_index)

    if global_rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)
    dist.barrier()

    for index in tqdm(range(int(math.ceil(len(dataset) / dist.get_world_size()))), disable=dist.get_rank() != 0):
        prompt_index = index * dist.get_world_size() + dist.get_rank()
        if prompt_index >= len(dataset):
            continue
        prompt = dataset[prompt_index]

        global_prompt_index = args.start_index + prompt_index
        sample_seed = args.seed + global_prompt_index
        
        output_path = os.path.join(args.output_folder, f"{global_prompt_index:06d}.pt")
        if os.path.exists(output_path):
            continue

        generator = torch.Generator(device=device)
        generator.manual_seed(sample_seed)

        conditional_dict = encoder(
            text_prompts=[prompt]
        )

        latents = torch.randn(
            [1, 21, 16, 60, 104], dtype=torch.float32, device=device,
            generator=generator
        )

        noisy_input = []

        for progress_id, t in enumerate(scheduler.timesteps):
            timestep = t * \
                torch.ones([1, 21], device=device, dtype=torch.float32)

            noisy_input.append(latents)

            if batched_cfg:
                # CFG batching: [uncond, cond]

                batched_latents = torch.cat(
                    [latents, latents],
                    dim=0
                )

                batched_timestep = torch.cat(
                    [timestep, timestep],
                    dim=0
                )

                batched_conditional_dict = {
                    "prompt_embeds": torch.cat(
                        [
                            unconditional_dict["prompt_embeds"],
                            conditional_dict["prompt_embeds"],
                        ],
                        dim=0,
                    )
                }

                flow_pred_batched = model(
                    batched_latents,
                    batched_conditional_dict,
                    batched_timestep,
                    return_x0=False,
                )

                flow_pred_uncond, flow_pred_cond = flow_pred_batched.chunk(2, dim=0)
               
            else:
                flow_pred_cond, _ = model(
                    latents, conditional_dict, timestep
                )

                flow_pred_uncond, _ = model(
                    latents, unconditional_dict, timestep
                )

            flow_pred = flow_pred_uncond + args.guidance_scale * (
                flow_pred_cond - flow_pred_uncond
            )


            latents = scheduler.step(
                flow_pred.flatten(0, 1),
                timestep.flatten(0, 1),
                latents.flatten(0, 1),
            ).unflatten(0, flow_pred.shape[:2])

        noisy_input.append(latents)

        noisy_inputs = torch.stack(noisy_input, dim=1)

        noisy_inputs = noisy_inputs[:, [0, 12, 24, 36, -1]]

        stored_data = noisy_inputs.to(torch.float16)

        torch.save(
            {
                "states": stored_data.cpu(),
                "prompt": prompt,
                "seed": sample_seed,
                "sample_id": global_prompt_index,
            },
            output_path
        )

    dist.barrier()


if __name__ == "__main__":
    # sample_subset()
    main()