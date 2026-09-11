import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.attention.flex_attention import create_block_mask

from model.ode_regression_with_rollout import ODERegressionWithRollout


def prepare_blockwise_sink_window_mask(
    device,
    num_frames: int,
    frame_seqlen: int,
    num_frame_per_block: int,
    local_attn_size: int,
    sink_size: int,
):
    """
    Blockwise causal mask with sliding-window + persistent sink, matching the
    inference-time KV-cache topology exactly:
      - First sink_size frames are always visible (pinned, StreamingLLM-style)
      - Sliding window of the last local_attn_size frames within causal bounds
    This is the training mask for warmup Phase 0 of ODERegressionWithWarmup.
    """
    assert local_attn_size > 0, "local_attn_size must be > 0 for sink+window mask"
    assert sink_size % num_frame_per_block == 0, (
        f"sink_size={sink_size} must be a multiple of num_frame_per_block={num_frame_per_block}"
    )

    total_length = num_frames * frame_seqlen
    padded_length = math.ceil(total_length / 128) * 128 - total_length

    ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
    frame_indices = torch.arange(
        start=0,
        end=total_length,
        step=frame_seqlen * num_frame_per_block,
        device=device,
    )
    for tmp in frame_indices:
        ends[tmp : tmp + frame_seqlen * num_frame_per_block] = (
            tmp + frame_seqlen * num_frame_per_block
        )

    sink_tokens = sink_size * frame_seqlen
    window_tokens = local_attn_size * frame_seqlen

    def attention_mask(b, h, q_idx, kv_idx):
        in_window = (kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - window_tokens))
        is_sink = kv_idx < sink_tokens
        return in_window | is_sink | (q_idx == kv_idx)

    block_mask = create_block_mask(
        attention_mask,
        B=None, H=None,
        Q_LEN=total_length + padded_length,
        KV_LEN=total_length + padded_length,
        _compile=False,
        device=device,
    )

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(
            f"[warmup mask] blockwise causal with sink={sink_size} frames, "
            f"window={local_attn_size} frames, block={num_frame_per_block} frames"
        )
        print(block_mask)

    return block_mask


class ODERegressionWithWarmup(ODERegressionWithRollout):
    """
    Two-phase ODE regression with a CausVid-style warmup before the rollout phase.

    Phase 0  (global_step < warmup_rollout_steps):
        Single-pass ODE regression with no rollout and no clean_x — like CausVid —
        but using a sliding-window + sink attention mask instead of CausVid's strict
        causal mask.  This exactly matches the inference KV-cache topology, closing
        the train-inference gap before any student rollout is attempted.

    Phase 1  (global_step >= warmup_rollout_steps):
        Full rollout-conditioned ODE regression from ODERegressionWithRollout,
        unchanged.

    Implementation note: CausalWanModel._forward_train caches self.block_mask.
    We hold a direct reference to the CausalWanModel instance (captured before FSDP
    wrapping in __init__) and set block_mask on it to inject our warmup mask without
    modifying any existing code.  At the phase boundary we reset block_mask=None so
    _forward_train recomputes the TF mask (2F sequence) for Phase 1.
    """

    def __init__(self, args, device):
        super().__init__(args, device)
        self.warmup_rollout_steps = getattr(args, "warmup_rollout_steps", 0)
        self._warmup_block_mask = None
        # Capture a direct reference to CausalWanModel BEFORE the Trainer applies
        # FSDP wrapping to self.generator (WanDiffusionWrapper). FSDP does not
        # replace inner module instances, so this reference stays valid.
        self._causal_model = self.generator.model
        print("[ODERegressionWithRollout] Warm up steps: ", self.warmup_rollout_steps)

    def _get_or_create_warmup_mask(self, device, num_frames: int):
        if self._warmup_block_mask is None:
            self._warmup_block_mask = prepare_blockwise_sink_window_mask(
                device=device,
                num_frames=num_frames,
                frame_seqlen=self.frame_seq_length,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=int(self._causal_model.local_attn_size),
                sink_size=int(getattr(self._causal_model, "sink_size", 0)),
            )
            print("[ODERegressionWithRollout] Created block wise sink window mask.")
        return self._warmup_block_mask
    
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
        timestep = timestep.reshape(timestep.shape[0], -1, self.num_frame_per_block)
        timestep[:, :, 1:] = timestep[:, :, 0:1]
        timestep = timestep.reshape(timestep.shape[0], -1)
        return timestep
       
    def _sample_warmup_input(self, ode_latent: torch.Tensor):
        """
        Sample per-AR-block timesteps from the ODE trajectory, matching CausVid's
        _prepare_generator_input.  index is [B, F], sampled uniformly from
        [0, num_steps); _process_timestep enforces the same index within each
        num_frame_per_block chunk so each AR block gets one timestep.
        The loss mask (timestep != 0) skips any frames that land on t=0.
        """
        B, num_steps, F, C, H, W = ode_latent.shape

        index = torch.randint(
            0, num_steps, [B, F], device=ode_latent.device, dtype=torch.long
        )
        index = self._process_timestep(index)  # align to AR block granularity

        noisy_input = torch.gather(
            ode_latent, dim=1,
            index=index.reshape(B, 1, F, 1, 1, 1).expand(-1, -1, -1, C, H, W),
        ).squeeze(1)

        timestep = self.denoising_step_list[index]
        return noisy_input, timestep

    def _warmup_loss(self, ode_latent: torch.Tensor, conditional_dict: dict):
        """CausVid-style single-pass ODE regression with inference-topology mask."""
        target_latent = ode_latent[:, -1]  # teacher PF-ODE endpoint (confirmed CausVid format)
        noisy_input, timestep = self._sample_warmup_input(ode_latent)

        # Inject the sink+window mask before the forward pass.
        # _forward_train checks `if self.block_mask is None` — by setting it here
        # we override the cache without modifying causal_model.py.
        self._causal_model.block_mask = self._get_or_create_warmup_mask(
            device=noisy_input.device,
            num_frames=noisy_input.shape[1],
        )

        _, pred = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep,
            # no clean_x: pure CausVid-style single pass
        )

        mask = timestep != 0
        loss = F.mse_loss(pred[mask], target_latent[mask], reduction="mean")

        log_dict = {
            "unnormalized_loss": F.mse_loss(pred, target_latent, reduction="none")
                .mean(dim=[1, 2, 3, 4])
                .detach(),
            "timestep": timestep.float().mean(dim=1).detach(),
            "input": noisy_input.detach(),
            "output": pred.detach(),
        }
        return loss, log_dict

    def generator_loss(
        self,
        ode_latent: torch.Tensor,
        conditional_dict: dict,
        global_step: int = 0,
    ):
        if global_step < self.warmup_rollout_steps:
            return self._warmup_loss(ode_latent, conditional_dict)

        if global_step == self.warmup_rollout_steps:
            # Phase boundary: clear cached warmup mask so _forward_train recomputes
            # the TF mask (2F sequence length) on the first Phase 1 forward call.
            self._causal_model.block_mask = None

        return super().generator_loss(ode_latent, conditional_dict)
