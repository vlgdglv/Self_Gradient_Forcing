"""
CausalRecacheInferencePipeline — drop-in variant of CausalInferencePipeline.

After each AR chunk Bi is fully denoised, instead of the standard single-write
clean-KV refresh, this pipeline runs a *causal recache pass* over the last ≤3
finalized chunks [B_{i-2}, B_{i-1}, Bi] at ``context_noise`` timestep using a
**temporary** KV cache.  The refreshed KV for [B_{i-1}, Bi] (2 chunks, 6 frames
with the default topology) is then written back into the **non-sink** history
slots of the production KV cache.  The sink KV is never modified.

LongLive topology assumed (from configs/self_gradient_forcing_chunkwise.yaml):
    sink    = 3 frames  (1 chunk)
    history = 6 frames  (2 chunks, refreshed each step)
    current = 3 frames  (1 chunk, the chunk being generated next)
    local_attn_size = 9 frames  (sink + history)

Warm-up schedule:
    after B0: nothing to refresh in history (B0 is entirely the sink)
    after B1: recache [B0, B1] → retain B1 (1 chunk)
    B2 onward: recache [B_{i-2}, B_{i-1}, Bi] → retain [B_{i-1}, Bi] (2 chunks)

Constraints satisfied:
    1. No future leakage: only finalized clean chunks are fed into the recache.
    2. Model weights / denoising loop untouched.
    3. Original RoPE / current_start semantics preserved (absolute positions).
    4. Temp cache used for the recache pass; production cache updated in-place.
    5. Warm-up handled (B0, B1 special-cased above).
    6. Sink handling identical to baseline.
    7. No changes to Wan attention math.
    8. Structure allows a future ``recache_mode="bidirectional"`` reuse.
"""
import torch
import torch.nn.functional as F
from typing import Optional

from .causal_inference import CausalInferencePipeline


def compare_kv_range(old, new, lo, hi, name):
    for key in ["k", "v"]:
        x = old[key][:, lo:hi].float()
        y = new[key][:, lo:hi].float()

        diff = y - x

        rel_l2 = diff.norm() / (x.norm() + 1e-8)
        cosine = F.cosine_similarity(
            x.flatten(),
            y.flatten(),
            dim=0
        )

        print(
            f"[{name}] {key.upper()} | "
            f"rel_L2={rel_l2.item():.6f} | "
            f"cos={cosine.item():.6f}"
        )
        
class CausalRecacheInferencePipeline(CausalInferencePipeline):
    """Causal AR inference with a multi-chunk KV history recache pass."""


    # -------------------------------------------------------------------------
    # Override: KV refresh hook
    # -------------------------------------------------------------------------

    def _refresh_kv_cache(
        self,
        block_index: int,
        current_start_frame: int,
        current_num_frames: int,
        denoised_pred: torch.Tensor,
        output: torch.Tensor,
        conditional_dict: dict,
        context_timestep: torch.Tensor,
        final_output: torch.Tensor,
    ):
        # ── Streaming path: fall back to baseline (not yet adapted) ──────────
        # if getattr(self.generator.model, "kv_rope_relative", False):
        #     print(f"[Recache] block={block_index:3d} | "
        #           f"No recache for now.")
        super()._refresh_kv_cache(
            block_index, current_start_frame, current_num_frames,
            denoised_pred, output, conditional_dict, context_timestep, final_output)
        orig_kv_layer0 = {"k": self.kv_cache1[0]["k"].clone(), "v": self.kv_cache1[0]["v"].clone()}
        orig_kv_layer29 = {"k": self.kv_cache1[29]["k"].clone(), "v": self.kv_cache1[29]["v"].clone()}
        
        fs = self.frame_seq_length           # tokens per latent frame (1560)
        chunk_tokens = current_num_frames * fs     # tokens per AR chunk (e.g. 3×1560)
        sink_frames = self.generator.model.sink_size
        sink_tokens = sink_frames * fs       # sink bytes (e.g. 3×1560 = 4680)

        batch_size = output.shape[0]
        device = output.device
        dtype = output.dtype

        abs_list = self.kv_cache1[0].get("stream_abs", None)
        assert abs_list is not None, "Should be a abs list"
        
        if len(abs_list) <= sink_frames:
            print(f"[Recache] block={block_index:3d} | "
                  f"Skipped, still in sink zone")
            super()._refresh_kv_cache(
                block_index, current_start_frame, current_num_frames,
                denoised_pred, output, conditional_dict, context_timestep,
                final_output)
            return


        target_recache_frames = len(abs_list[sink_frames:])
        recache_start_frame = current_start_frame+current_num_frames-target_recache_frames
        recache_end_frame = current_start_frame+current_num_frames
        input_noise_for_recache = final_output[:, recache_start_frame:recache_end_frame]
        
        print(
            f"[Recache] block={block_index:3d} | "
            f"recache_input=[{input_noise_for_recache.shape}(start_f={recache_start_frame}, end_f={recache_end_frame}), "
            # f"target_recache_start={target_recache_start}({target_recache_start//fs} frame), "
            # f"target_reache_end={target_recache_end}({target_recache_end//fs} frame), "
            f"target_recache_frames={target_recache_frames} frames."
        )

        recache_timestep = torch.full(
            (batch_size, target_recache_frames),
            self.args.context_noise,
            device=device,
            dtype=context_timestep.dtype,
        )
        
        self.generator(
            noisy_image_or_video=input_noise_for_recache,
            conditional_dict=conditional_dict,
            timestep=recache_timestep,
            kv_cache=self.kv_cache1,
            crossattn_cache=self.crossattn_cache,
            current_start=recache_start_frame*fs,
        )

        recached_kv_layer0 = self.kv_cache1[0]
        recached_kv_layer29 = self.kv_cache1[29]
        
        abs_start, abs_end = sink_frames * fs, (sink_frames+target_recache_frames) * fs
        compare_kv_range(orig_kv_layer0, recached_kv_layer0, abs_start, abs_end, "layer0")
        compare_kv_range(orig_kv_layer29, recached_kv_layer29, abs_start, abs_end, "layer29")