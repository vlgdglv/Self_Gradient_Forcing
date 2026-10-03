import torch.nn.functional as F
from typing import Tuple
import torch
import torch.distributed as dist

from model.ode_regression import ODERegression
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper



class ODERegressionWithForcing(ODERegression):
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
                print(f"ODERegressionWithForcing initialized with num_transformer_blocks: {self.num_transformer_blocks}")
                print(f"ODERegressionWithForcing initialized with frame_seq_length: {self.frame_seq_length}")
                print(f"ODERegressionWithForcing initialized with local_attn_size: {local_attn_size}")
                print(f"ODERegressionWithForcing initialized with kv_cache_size: {self.kv_cache_size}")
    
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
            timestep = timestep.reshape(timestep.shape[0], -1, self.num_frame_per_block)
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

        # Step 1: Randomly choose a timestep for each frame except 0
        index = torch.randint(0, len(self.denoising_step_list)-1, [batch_size, num_frames], device=self.device, dtype=torch.long)

        index = self._process_timestep(index)

        noisy_input = torch.gather(
            ode_latent, dim=1,
            index=index.reshape(batch_size, 1, num_frames, 1, 1, 1).expand(-1, -1, -1, num_channels, height, width)
        ).squeeze(1)

        timestep = self.denoising_step_list[index]
        return noisy_input, timestep

    def generator_loss(
        self,
        ode_latent: torch.Tensor,
        conditional_dict: dict,
    ):

        # Same initial Gaussian noise as original CausVid ODE dataset.
        # initial_noise = ode_latent[:, 0]

        # Same teacher PF-ODE endpoint.
        target_latent = ode_latent[:, -1]

        noisy_input, timestep = self._prepare_generator_input(ode_latent=ode_latent)

        _, pred = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clean_x=target_latent.detach(),
            aug_t=torch.zeros_like(timestep),
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

            "timestep": timestep.float().mean(dim=1).detach(),

            "input": noisy_input.detach(),

            "output": pred.detach(),
        }

        return loss, log_dict



class ODERegressionWithHistoryInit(ODERegression):
    
    def __init__(self, *args, device):
        super().__init__(*args, device)

        self.num_transformer_blocks = len(self.generator.model.blocks)
        self.frame_seq_length = 1560
        local_attn_size = int(self.generator.model.local_attn_size)
        if local_attn_size != -1:
            self.kv_cache_size = (local_attn_size * self.frame_seq_length)
        else:
            self.kv_cache_size = (21 * self.frame_seq_length)

        # ---- warm-start hyperparams ----
        self.warm_start_gamma = float(getattr(self.args, "warm_start_gamma", 0.9375))
        self.warm_start_p1000 = float(getattr(self.args, "warm_start_p1000", 0.5))
        self.warm_history_mode = str(getattr(self.args, "warm_history_mode", "prev_block_copy"))

        assert 0.0 <= self.warm_start_gamma <= 1.0
        assert 0.0 <= self.warm_start_p1000 <= 1.0

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            if rank == 0:
                print(f"ODERegressionWithForcing initialized with num_transformer_blocks: {self.num_transformer_blocks}")
                print(f"ODERegressionWithForcing initialized with frame_seq_length: {self.frame_seq_length}")
                print(f"ODERegressionWithForcing initialized with local_attn_size: {local_attn_size}")
                print(f"ODERegressionWithForcing initialized with kv_cache_size: {self.kv_cache_size}")
                print(f"A0 warm-start gamma: {self.warm_start_gamma}")
                print(f"A0 warm-start P(t=1000): {self.warm_start_p1000}")
                print(f"A0 warm history mode: {self.warm_history_mode}")
    

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
            timestep = timestep.reshape(timestep.shape[0], -1, self.num_frame_per_block)
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

        
        index = self._sample_timestep_index(batch_size, num_frames)
        index = self._process_timestep(index)

        noisy_input = torch.gather(
            ode_latent, dim=1,
            index=index.reshape(batch_size, 1, num_frames, 1, 1, 1).expand(-1, -1, -1, num_channels, height, width)
        ).squeeze(1)

        timestep = self.denoising_step_list[index]
        return noisy_input, timestep, index
    
    def _sample_timestep_index(self, batch_size: int, num_frames: int) -> torch.Tensor:
        num_candidate_steps = len(self.denoising_step_list) - 1
        
        if num_candidate_steps == 1:
            index = torch.zeros(
                batch_size, num_frames,
                device=self.device, dtype=torch.long
            )
            return index
        
        probs = torch.full(
            (num_candidate_steps,),
            (1.0 - self.warm_start_p1000) / (num_candidate_steps - 1),
            device=self.device,
            dtype=torch.float32,              
        )
        probs[0] = self.warm_start_p1000
        
        index = torch.multinomial(
            probs,
            num_samples=batch_size*num_frames,
            replacement=True,
        ).view(batch_size, num_frames)
        
        return index

    def _build_history_prior(self, target_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build causal teacher-history prior H from the clean target_latent.

        Input:
            target_latent: [B, F, C, H, W]

        Output:
            history_prior: [B, F, C, H, W]
            history_valid: [B, F] bool, whether each frame position has a valid history prior

        Modes:
            - prev_block_copy: current block copies the previous block, aligned by intra-block offset
            - last_frame_hold: current block is filled with the last frame of the previous block
        """
        batch_size, num_frames, num_channels, height, width = target_latent.shape
        K = self.num_frame_per_block
        
        history_prior = torch.zeros_like(target_latent)
        history_valid = torch.zeros(batch_size, num_frames, device=target_latent.device, dtype=torch.bool)
        
        num_blocks = (num_frames + K - 1) // K
        mode = self.warm_history_mode
        
        for blk in range(1, num_blocks):
            cur_start = blk * K
            cur_end = min((blk+1)*K, num_frames)
            cur_len = cur_end - cur_start
            
            prev_start = (blk - 1) * K
            prev_end = min(blk * K, num_frames)
            prev_len = prev_end - prev_start

            if prev_len <= 0 or cur_len <= 0:
                continue
        
            if mode == "prev_block_copy":
                if prev_len >= cur_len:
                    history_prior[:, cur_start:cur_end] = target_latent[:, prev_start:prev_start+cur_len]
                else:
                    # unlikely, but be robust: copy what exists, pad with previous block's last frame
                    history_prior[:, cur_start:cur_start + prev_len] = target_latent[:, prev_start:prev_end]
                    pad_len = cur_len - prev_len
                    last_prev = target_latent[:, prev_end - 1:prev_end]  # [B,1,C,H,W]
                    history_prior[:, cur_start + prev_len:cur_end] = last_prev.expand(-1, pad_len, -1, -1, -1)
            elif mode == "last_frame_hold":
                last_prev = target_latent[:, prev_end-1:prev_end]
                history_prior[:, cur_start:cur_end] = last_prev.expand(-1, cur_len, -1, -1, -1)
            else:
                raise NotImplementedError(f"Unknown warm_history_mode: {mode}")
            
            history_valid[:, cur_start:cur_end] = True

        return history_prior, history_valid
    
    def _apply_warm_start_source(
        self,
        noisy_input: torch.Tensor,
        index: torch.Tensor,
        initial_noise: torch.Tensor,
        target_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Replace source only for blocks whose sampled timestep index == 0 (i.e. t=1000).

        Input:
            noisy_input:    [B, F, C, H, W]
            index:          [B, F]
            initial_noise:  [B, F, C, H, W]
            target_latent:  [B, F, C, H, W]

        Output:
            noisy_input: updated [B, F, C, H, W]
            warm_mask:   [B, F] bool
        """
        history_prior, history_valid = self._build_history_prior(target_latent)
        
        warm_mask = (index == 0) & history_valid
        
        gamma = self.warm_start_gamma
        warm_source = gamma * initial_noise + (1.0 - gamma) * history_prior
        
        noisy_input = torch.where(
            warm_mask[..., None, None, None],
            warm_source,
            noisy_input,
        )
        
        return noisy_input, warm_mask
        
    def generator_loss(
        self,
        ode_latent: torch.Tensor,
        conditional_dict: dict,
    ):
        # Same teacher PF-ODE endpoint.
        target_latent = ode_latent[:, -1]

        noisy_input, timestep, index = self._prepare_generator_input(ode_latent=ode_latent)

        initial_noise = ode_latent[:, 0] # [B, F, C, H, W]
        # A0: replace source only when sampled timestep is 1000
        noisy_input, warm_mask = self._apply_warm_start_source(
            noisy_input=noisy_input,
            index=index,
            initial_noise=initial_noise,
            target_latent=target_latent,
        )
        
        _, pred = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clean_x=target_latent.detach(),
            aug_t=torch.zeros_like(timestep),
        )

        loss = F.mse_loss(
            pred,
            target_latent,
            reduction="mean",
        )

        # logging / debugging
        per_sample_loss = F.mse_loss(
            pred,
            target_latent,
            reduction="none",
        ).mean(dim=[1, 2, 3, 4]).detach()

        log_dict = {
            "unnormalized_loss": per_sample_loss,
            "timestep": timestep.float().mean(dim=1).detach(),
            "input": noisy_input.detach(),
            "output": pred.detach(),
            "warm_ratio": warm_mask.float().mean(dim=1).detach(),  # fraction of frames warm-started
            "num_warm_frames": warm_mask.sum(dim=1).detach(),
            "num_t1000_frames": (index == 0).sum(dim=1).detach(),
        }

        return loss, log_dict
