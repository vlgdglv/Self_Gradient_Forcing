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
