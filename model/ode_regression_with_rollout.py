import torch.nn.functional as F
from typing import Tuple
import torch
import torch.distributed as dist

from model.ode_regression import ODERegression
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class ODERegressionOriginalCausVid(ODERegression):
    def __init__(self, args, device):
        """
            Copied from CausVid: https://github.com/tianweiy/CausVid/blob/master/causvid/ode_regression.py
        """
        super().__init__(args, device)
        print("[ODERegressionOriginalCausVid] Using original CausVid ODE regression!!!")


    def _process_timestep(self, timestep):
        """
        Pre-process the randomly generated timestep based on the generator's task type.
        Input:
            - timestep: [batch_size, num_frame] tensor containing the randomly generated timestep.

        Output Behavior:
            - image: check that the second dimension (num_frame) is 1.
            - bidirectional_video: broadcast the timestep to be the same for all frames.
            - causal_video: broadcast the timestep to be the same for all frames **in a block**.
        """
        if self.args.generator_task == "image":
            assert timestep.shape[1] == 1
            return timestep
        elif self.args.generator_task == "bidirectional_video":
            for index in range(timestep.shape[0]):
                timestep[index] = timestep[index, 0]
            return timestep
        elif self.args.generator_task == "causal_video":
            # make the noise level the same within every motion block
            timestep = timestep.reshape(
                timestep.shape[0], -1, self.num_frame_per_block)
            timestep[:, :, 1:] = timestep[:, :, 0:1]
            timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def _prepare_generator_input(self, ode_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a tensor containing the whole ODE sampling trajectories,
        randomly choose an intermediate timestep and return the latent as well as the corresponding timestep.
        Input:
            - ode_latent: a tensor containing the whole ODE sampling trajectories [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
        Output:
            - noisy_input: a tensor containing the selected latent [batch_size, num_frames, num_channels, height, width].
            - timestep: a tensor containing the corresponding timestep [batch_size].
        """
        batch_size, num_denoising_steps, num_frames, num_channels, height, width = ode_latent.shape

        # Step 1: Randomly choose a timestep for each frame
        index = torch.randint(0, len(self.denoising_step_list), [
            batch_size, num_frames], device=self.device, dtype=torch.long)

        index = self._process_timestep(index)

        noisy_input = torch.gather(
            ode_latent, dim=1,
            index=index.reshape(batch_size, 1, num_frames, 1, 1, 1).expand(
                -1, -1, -1, num_channels, height, width)
        ).squeeze(1)

        timestep = self.denoising_step_list[index]
        return noisy_input, timestep

    def generator_loss(self, ode_latent: torch.Tensor, conditional_dict: dict) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noisy latents and compute the ODE regression loss.
        Input:
            - ode_latent: a tensor containing the ODE latents [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
            They are ordered from most noisy to clean latents.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - loss: a scalar tensor representing the generator loss.
            - log_dict: a dictionary containing additional information for loss timestep breakdown.
        """
        # Step 1: Run generator on noisy latents
        target_latent = ode_latent[:, -1]

        noisy_input, timestep = self._prepare_generator_input(
            ode_latent=ode_latent)

        _, pred_image_or_video = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        # Step 2: Compute the regression loss
        mask = timestep != 0

        loss = F.mse_loss(
            pred_image_or_video[mask], target_latent[mask], reduction="mean")

        log_dict = {
            "unnormalized_loss": F.mse_loss(pred_image_or_video, target_latent, reduction='none').mean(dim=[1, 2, 3, 4]).detach(),
            "timestep": timestep.float().mean(dim=1).detach()
        }

        return loss, log_dict


class ODERegressionWithRollout(ODERegression):
    def __init__(self, args, device):
        super().__init__(args, device)
        
        self.num_transformer_blocks = len(self.generator.model.blocks)
        self.frame_seq_length = 1560

        local_attn_size = int(self.generator.model.local_attn_size)

        if local_attn_size != -1:
            self.kv_cache_size = (local_attn_size* self.frame_seq_length)
        else:
            self.kv_cache_size = (21 * self.frame_seq_length)

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            if rank == 0:
                print(f"ODERegressionWithRollout initialized with num_transformer_blocks: {self.num_transformer_blocks}")
                print(f"ODERegressionWithRollout initialized with frame_seq_length: {self.frame_seq_length}")
                print(f"ODERegressionWithRollout initialized with local_attn_size: {local_attn_size}")
                print(f"ODERegressionWithRollout initialized with kv_cache_size: {self.kv_cache_size}")
        
    def generator_loss(
        self,
        ode_latent: torch.Tensor,
        conditional_dict: dict,
    ):

        # Same initial Gaussian noise as original ODE dataset.
        initial_noise = ode_latent[:, 0]

        # Same teacher PF-ODE endpoint.
        target_latent = ode_latent[:, -1]

        # ------------------------------------------------
        # Pass 1: exact self rollout, NO GRAD
        # ------------------------------------------------
        noisy_at_t, rollout_clean, train_timestep = (
            self._collect_student_rollout(
                noise=initial_noise,
                conditional_dict=conditional_dict,
            )
        )

        # ------------------------------------------------
        # Pass 2: parallel rollout-conditioned ODE regression
        # ------------------------------------------------
        _, pred = self.generator(
            noisy_image_or_video=noisy_at_t.detach(),
            conditional_dict=conditional_dict,
            timestep=train_timestep,
            clean_x=rollout_clean.detach(),
            aug_t=torch.zeros_like(train_timestep),
        )

        loss = F.mse_loss(
            pred,
            target_latent,
            reduction="mean",
        )

        log_dict = {
            "unnormalized_loss": F.mse_loss(
                pred,
                target_latent,
                reduction="none",
            ).mean(dim=[1, 2, 3, 4]).detach(),

            "timestep": train_timestep.float().mean(dim=1).detach(),

            # log the actual training state, not z1000
            "input": noisy_at_t.detach(),

            "output": pred.detach(),
        }

        return loss, log_dict

    @torch.no_grad()
    def _collect_student_rollout(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
    ):

        B, F, C, H, W = noise.shape
        block_size = self.num_frame_per_block

        assert F % block_size == 0
        num_blocks = F // block_size

        device = noise.device

        self._initialize_kv_cache(
            batch_size=B,
            dtype=noise.dtype,
            device=device,
        )
        self._initialize_crossattn_cache(
            batch_size=B,
            dtype=noise.dtype,
            device=device,
        )

        num_steps = len(self.denoising_step_list)

        # same timestep for every AR block:
        # matches original ODE regression's uniform_timestep=True
        exit_idx = torch.randint(
            low=0,
            high=num_steps - 1,
            size=(B, num_blocks),
            device=device,
        )

        noisy_at_t = torch.zeros_like(noise)
        final_clean = torch.zeros_like(noise)

        current_start = 0
        train_timestep = torch.zeros((B, F), device=device, dtype=torch.long,)
        
        for block_idx in range(num_blocks):
            start = block_idx * block_size
            end = start + block_size

            noisy_input = noise[:, start:end]

            for step_idx, current_timestep in enumerate(
                self.denoising_step_list
            ):

                timestep = torch.full(
                    (B, block_size),
                    current_timestep,
                    device=device,
                    dtype=torch.long,
                )
                select = (exit_idx[:, block_idx] == step_idx) 
                
                # Record the *student's own* trajectory state.
                if select.any():
                    noisy_at_t[select, start:end].copy_(noisy_input[select])
                    train_timestep[select, start:end] = current_timestep
                    
                _, x0 = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start=start * self.frame_seq_length,
                )

                if step_idx < num_steps - 1:
                    next_t = self.denoising_step_list[step_idx + 1]

                    noisy_input = self.scheduler.add_noise(
                        x0.flatten(0, 1),
                        torch.randn_like(x0.flatten(0, 1)),
                        torch.full(
                            (B * block_size,),
                            next_t,
                            device=device,
                            dtype=torch.long,
                        ),
                    ).unflatten(0, x0.shape[:2])

            # IMPORTANT:
            # x0 here = output after ALL denoising steps.
            final_clean[:, start:end].copy_(x0)

            # Exact inference-style context refresh.
            context_timestep = torch.zeros(
                (B, block_size),
                device=device,
                dtype=torch.long,
            )
            # x_ctx = self.scheduler.add_noise(
            #     x0.flatten(0, 1),
            #     torch.randn_like(x0.flatten(0, 1)),
            #     context_timestep.flatten(0, 1).long(),
            # ).unflatten(0, x0.shape[:2])
            self.generator(
                noisy_image_or_video=x0,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache,
                crossattn_cache=self.crossattn_cache,
                current_start=start * self.frame_seq_length,
            )

            current_start = end

        return (
            noisy_at_t,
            final_clean,
            train_timestep,
        )
        
    
    def _initialize_kv_cache(self, batch_size, dtype, device):
        kv_cache = []
        for _ in range(self.num_transformer_blocks):
            kv_cache.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })
        self.kv_cache = kv_cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False,
            })
        self.crossattn_cache = crossattn_cache
