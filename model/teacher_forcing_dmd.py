import torch.nn.functional as F
from typing import Tuple
import random
import torch
import torch.distributed as dist
from torch import nn

from model.base import BaseModel
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class TeacherForcingDMD(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        # Initialize the device and arguments.
        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        
        # Initialize models.
        self.iscausal = getattr(args, "causal", True)
        self.timestep_shift = getattr(args, "timestep_shift", 5.0)
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=self.iscausal)
        self.generator.model.requires_grad_(True)

        self.fake_model_name = getattr(args, "fake_name", "Wan2.1-T2V-1.3B")
        self.fake_score = WanDiffusionWrapper(model_name=self.fake_model_name, timestep_shift=self.timestep_shift, is_causal=False)
        self.fake_score.model.requires_grad_(True)

        self.real_model_name = getattr(args, "real_name", "Wan2.1-T2V-1.3B")
        self.real_score = WanDiffusionWrapper(model_name=self.real_model_name, timestep_shift=self.timestep_shift, is_causal=False)
        self.real_score.model.requires_grad_(False)
        self.real_score.to(device=self.device, dtype=self.dtype)
        
        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)
        
        if getattr(args, "generator_ckpt", False):
            print(f"Loading pretrained generator from {args.generator_ckpt}")
            state_dict = torch.load(args.generator_ckpt, map_location="cpu")['generator']
            self.generator.load_state_dict(state_dict, strict=True)
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

        if hasattr(args, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long, device=self.device)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).to(self.device)
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        # Optional: separate denoising schedule for the first chunk (block 0).
        # If the config does not provide `denoising_step_list_first_chunk`, all
        # blocks share `denoising_step_list` (backwards compatible).
        # This technique is proposed by [ASD](https://github.com/BigAandSmallq/SAD) for 1/2-step DMD. 
        # We thank ASD for its contribution.
        if hasattr(args, "denoising_step_list_first_chunk") and args.denoising_step_list_first_chunk is not None:
            self.denoising_step_list_first_chunk = torch.tensor(
                args.denoising_step_list_first_chunk, dtype=torch.long, device=self.device)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).to(self.device)
                self.denoising_step_list_first_chunk = timesteps[1000 - self.denoising_step_list_first_chunk]
        else:
            self.denoising_step_list_first_chunk = None

        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 21)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        self.real_guidance_scale = getattr(args, "guidance_scale", 5.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None
    
    def log(self, msg):
        print(f"[TFDMD] {msg}")

    def critic_loss(
        self,
        ode_latent: torch.Tensor,
        conditional_dict: dict,
        global_step
    ):
        # [B, T, C, H, W]
        clean_latent = ode_latent[:, -1]
        batch_size = clean_latent.shape[0]

        # --------------------------------------------------
        # 1. Sample one AR chunk
        # --------------------------------------------------
        total_chunks = (clean_latent.shape[1] // self.num_frame_per_block)

        # if dist.get_rank() == 0:
        #     chunk_idx = torch.randint(0, total_chunks, (1,), device=self.device)
        # else:
        #     chunk_idx = torch.zeros(1, dtype=torch.long, device=self.device)

        # dist.broadcast(chunk_idx, src=0)
        # chunk_idx = int(chunk_idx.item())
        
        chunk_idx = (global_step % total_chunks)
        
        ctx_end = chunk_idx * self.num_frame_per_block
        x_ctx = clean_latent[:, :ctx_end]
        current_gt = clean_latent[:, ctx_end:ctx_end + self.num_frame_per_block]

        # --------------------------------------------------
        # 2. Generate fake current chunk
        #
        # IMPORTANT:
        # generator receives NO gradient during critic update
        # --------------------------------------------------
        with torch.no_grad():
            eps_gen = torch.randn_like(current_gt)
            x_fake = self.four_step_generate_with_grad(
                initial_noise=eps_gen,
                conditional_dict=conditional_dict,
                clean_context=x_ctx,
            ).detach()

        prefix_fake = torch.cat([x_ctx, x_fake], dim=1)
        prefix_frames = prefix_fake.shape[1]

        # --------------------------------------------------
        # 3. Sample critic timestep
        # --------------------------------------------------
        tau = torch.randint(0, 1000, (batch_size, 1), device=self.device).float()
        u = tau / 1000.0
        tau = self.timestep_shift * u / (1.0 + (self.timestep_shift - 1.0) * u) * 1000.0
        tau = tau.clamp(20.0, 980.0)
        critic_timestep = tau.repeat(1, prefix_frames)

        # --------------------------------------------------
        # 4. Add noise to the whole prefix
        # --------------------------------------------------
        critic_noise = torch.randn_like(prefix_fake, dtype=self.dtype, device=self.device)

        prefix_tau = self.scheduler.add_noise(
            prefix_fake.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1),
        ).unflatten(0, prefix_fake.shape[:2])

        # --------------------------------------------------
        # 5. Bidirectional fake critic
        # --------------------------------------------------
        _, pred_fake_prefix = self.fake_score(
            noisy_image_or_video=prefix_tau,
            conditional_dict=conditional_dict,
            timestep=critic_timestep,
        )

        # --------------------------------------------------
        # 6. x0 prediction -> flow prediction
        # --------------------------------------------------
        flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_prefix.flatten(0, 1),
                xt=prefix_tau.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1),
            ).unflatten(0, prefix_fake.shape[:2])
        

        # Wan flow-matching target: v* = eps - x0
        flow_target = critic_noise - prefix_fake

        # --------------------------------------------------
        # 7. ONLY optimize generated leaf
        # --------------------------------------------------
        leaf = slice(-self.num_frame_per_block, None)
        loss = F.mse_loss(flow_pred[:, leaf].float(), flow_target[:, leaf].float())

        log_dict = {
            "critic_loss": loss.detach().item(),
            "critic_timestep": critic_timestep.detach()
        }
        return loss, log_dict

    def generator_loss(
        self,
        ode_latent: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        global_step
    ):
        # Teacher synthetic PF-ODE endpoint.
        clean_latent = ode_latent[:, -1] # [B, 21, C, H, W]
        batch_size = clean_latent.shape[0]
        
        # 1. sample one AR leaf
        total_frames = clean_latent.shape[1] // self.num_frame_per_block
        
        # if dist.get_rank() == 0:
        #     chunk_idx = torch.randint(low=0, high=total_frames, size=(1,), device=self.device)
        # else:
        #     chunk_idx = torch.zeros(1, dtype=torch.long, device=self.device)
        # dist.broadcast(chunk_idx, src=0)
        # chunk_idx = int(chunk_idx.item())
        total_chunks = (clean_latent.shape[1] // self.num_frame_per_block)
        chunk_idx = (global_step % total_chunks)
        
        ctx_end = chunk_idx * 3
        x_ctx = clean_latent[:, :ctx_end]

        # 2. teacher-forced causal generation with deployment context tau=0
        eps_gen = torch.randn_like(clean_latent[:, ctx_end:ctx_end+3])

        x_fake = self.four_step_generate_with_grad(
            initial_noise=eps_gen,
            conditional_dict=conditional_dict,
            clean_context=x_ctx,
        )
        prefix_clean = torch.cat([x_ctx, x_fake.detach()], dim=1)
        
        # 3. independent DMD timestep
        tau = torch.randint(low=0, high=1000, size=(batch_size, 1), device=self.device,).float()
        u = tau / 1000.0
        tau = (self.timestep_shift * u / (1.0 + (self.timestep_shift - 1.0) * u)* 1000.0)
        tau = tau.clamp(20.0, 980.0)
        
        noise = torch.randn_like(prefix_clean, dtype=self.dtype, device=self.device)

        prefix_tau = self.scheduler.add_noise(
            prefix_clean.flatten(0, 1),
            noise.flatten(0, 1),
            tau.repeat(1, prefix_clean.shape[1]).flatten(0, 1),
        ).unflatten(0, prefix_clean.shape[:2])
        
        score_timestep = tau.repeat(1, prefix_tau.shape[1])

        with torch.no_grad():
            # 4. bidirectional Real score
            _, pred_real_cond_prefix = self.real_score(
                noisy_image_or_video=prefix_tau,
                conditional_dict=conditional_dict,
                timestep=score_timestep,
            )
            _, pred_real_uncond_prefix = self.real_score(
                noisy_image_or_video=prefix_tau,
                conditional_dict=unconditional_dict,
                timestep=score_timestep,
            )

            pred_real_prefix = pred_real_cond_prefix + self.real_guidance_scale * (pred_real_cond_prefix - pred_real_uncond_prefix)
            pred_real = pred_real_prefix[:, -self.num_frame_per_block:]
            
            # --------------------------------------------------
            # 5. causal SELF fake-score
            # --------------------------------------------------
            _, pred_fake_prefix = self.fake_score(
                noisy_image_or_video=prefix_tau,
                conditional_dict=conditional_dict,
                timestep=score_timestep,
            )

            pred_fake = pred_fake_prefix[:, -self.num_frame_per_block:]

        # 6. DMD gradient
        grad = pred_fake - pred_real

        normalizer = (x_fake.detach() - pred_real).abs().mean(
            dim=(1, 2, 3, 4),
            keepdim=True,
        ).clamp_min(1e-6)
        grad = torch.nan_to_num(grad / normalizer)

        # 7. make loss
        target = (x_fake - grad).detach()
        
        loss = 0.5 * F.mse_loss(x_fake.float(), target.float())

        with torch.no_grad():
            grad_mse_per_sample = F.mse_loss(
                x_fake.detach(),
                target.detach(),
                reduction="none",
            ).mean(dim=[1, 2, 3, 4])

        log_dict = {
            "generetor_loss_per_sample": grad_mse_per_sample,
            "generetor_timestep": tau,
        }

        return loss, log_dict

    def four_step_generate_with_grad(
        self,
        initial_noise,
        conditional_dict,
        clean_context,
    ):
        batch_size, frame_len = initial_noise.shape[0], initial_noise.shape[1]
        num_denoising_steps = len(self.denoising_step_list)
        
        # --------------------------------------------------
        # 1. GT context -> read-only deployment KV cache
        # --------------------------------------------------
        kv_cache, crossattn_cache, frame_seq_len = self._prefill_context_cache(
            model=self.generator,
            context=clean_context,
            conditional_dict=conditional_dict,
            context_timestep=torch.tensor(0, device=self.device),
        )

        # --------------------------------------------------
        # 2. Only the current chunk participates in autograd
        # --------------------------------------------------
        current_start = (clean_context.shape[1] * frame_seq_len)
        noisy_input = initial_noise
        
        # self.log(f"clean_context: {clean_context.shape}, initial_noise: {initial_noise.shape}")

        for index, current_timestep in enumerate(self.denoising_step_list):
            timestep = torch.ones(
                [batch_size, frame_len], device=self.device, dtype=torch.int64
            ) * current_timestep

            _, x0 = self.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=timestep,

                # deployment causal context
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                
                # IMPORTANT:
                # current noisy chunk must not contaminate
                # the clean context cache.
                update_kv_cache=False,
            )

            if index < num_denoising_steps - 1:
                next_timestep = (self.denoising_step_list[index + 1])

                noisy_input = self.scheduler.add_noise(
                    x0.flatten(0, 1),
                    torch.randn_like(x0.flatten(0, 1)),
                    torch.full(
                        [batch_size * frame_len],
                        next_timestep,
                        device=self.device,
                        dtype=torch.long,
                    ),
                ).unflatten(0, x0.shape[:2],)

        return x0

    def _init_kv_cache(
        self,
        batch_size,
        height,
        width,
        dtype,
        max_frames=None,
    ):
        model = self.generator.model

        if max_frames is None:
            max_frames = self.num_training_frames

        patch_h = model.patch_size[1]
        patch_w = model.patch_size[2]

        frame_seq_len = (height // patch_h) * (width // patch_w)

        num_layers = len(model.blocks)
        num_heads = model.num_heads
        head_dim = model.dim // model.num_heads

        kv_cache_size = max_frames * frame_seq_len

        kv_cache = []

        for _ in range(num_layers):
            kv_cache.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=self.device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=self.device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
            })

        return kv_cache, frame_seq_len

    def _initialize_crossattn_cache(self, batch_size):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        dtype, device = self.dtype, self.device
        model = self.generator.model
        num_layers = len(model.blocks)
        num_heads = model.num_heads
        head_dim = model.dim // model.num_heads

        crossattn_cache = []

        for _ in range(num_layers):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "is_init": False
            })
        return crossattn_cache

    @torch.no_grad()
    def _prefill_context_cache(
        self,
        model,
        context,
        conditional_dict,
        context_timestep
    ):
        B, F, C, H, W = context.shape

        kv_cache, frame_seq_len = self._init_kv_cache(
            batch_size=B,
            height=H,
            width=W,
            dtype=context.dtype,
        )

        crossattn_cache = self._initialize_crossattn_cache(
            batch_size=context.shape[0]    
        )

        block = self.num_frame_per_block

        assert F % block == 0, (
            f"context length {F} must be divisible by "
            f"num_frame_per_block={block}"
        )

        for start in range(0, F, block):
            ctx_chunk = context[:, start:start + block]

            # timestep = torch.zeros(
            #     [B, block],
            #     device=self.device,
            #     dtype=torch.long,
            # )
            ctx_t = self._expand_timestep(
                timestep=context_timestep,
                batch_size=B,
                frame_len=ctx_chunk.shape[1],
            )

            model(
                noisy_image_or_video=ctx_chunk,
                conditional_dict=conditional_dict,
                timestep=ctx_t,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=start * frame_seq_len,
                update_kv_cache=True,
            )

        return kv_cache, crossattn_cache, frame_seq_len


    def _expand_timestep(
        self,
        timestep,
        batch_size,
        frame_len,
    ):
        """
        timestep:
            scalar tensor, e.g. shape [1]
            or batch tensor shape [B]
        return:
            [B, frame_len]
        """
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=self.device, dtype=torch.long,)

        timestep = timestep.to(device=self.device, dtype=torch.long)

        if timestep.numel() == 1:
            return timestep.reshape(1, 1).expand(batch_size, frame_len)

        return timestep.reshape(batch_size, 1).expand(batch_size, frame_len)