import gc
import logging
from utils.dataset import ODERegressionLMDBDataset, cycle, ODERegressionPTDataset
from model import TeacherForcingDMD

from collections import defaultdict
from utils.misc import (
    set_seed
)
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os

from utils.distributed import EMA_FSDP, barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        # assert config.trainer == "teacher_forcing_dmd"
        # self.model = ODERegression(config, device=self.device)
        model_cls = TeacherForcingDMD
            
        if self.is_main_process:
            print("[TFDMD Trainer] cls: ", model_cls)
        self.model = model_cls(config, device=self.device)
        
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )
        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=False
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            cpu_offload=False
        )
        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        data_store_ext = getattr(config, "data_store_ext", "lmdb")
        if data_store_ext == "lmdb":
            data_cls = ODERegressionLMDBDataset
        elif data_store_ext == "pt":
            data_cls = ODERegressionPTDataset
        else:
            data_cls = ODERegressionLMDBDataset
        dataset = data_cls(config.data_path, max_pair=getattr(config, "max_pair", int(1e8)))
        
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler, num_workers=8)
        self.dataloader = cycle(dataloader)
        if self.is_main_process:
            print(f"[Trainer] Dataset loaded. {len(dataset)} entries in total.")
            print(f"[Trainer] full config: ")
            print(config)
        self.step = 0

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        # if getattr(config, "generator_ckpt", False):
        #     print(f"Loading pretrained generator from {config.generator_ckpt}")
        #     state_dict = torch.load(config.generator_ckpt, map_location="cpu")
        #     if "generator" in state_dict:
        #         state_dict = state_dict["generator"]
        #         fixed = {}
        #         for k, v in state_dict.items():
        #             if k.startswith("model._fsdp_wrapped_module."):
        #                 k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
        #             fixed[k] = v
        #         state_dict = fixed
        #     elif "model" in state_dict:
        #         state_dict = state_dict["model"]
        #     elif "generator_ema" in state_dict:
        #         gen_sd = state_dict["generator_ema"]
        #         fixed = {}
        #         for k, v in gen_sd.items():
        #             if k.startswith("model._fsdp_wrapped_module."):
        #                 k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
        #             fixed[k] = v
        #         state_dict = fixed
        #     self.model.generator.load_state_dict(
        #         state_dict, strict=True
        #     )

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        # No ema for now
        # rename_param = (
        #     lambda name: name.replace("_fsdp_wrapped_module.", "")
        #     .replace("_checkpoint_wrapped_module.", "")
        #     .replace("_orig_mod.", "")
        # )
        # self.name_to_trainable_params = {}
        # for n, p in self.model.generator.named_parameters():
        #     if not p.requires_grad:
        #         continue

        #     renamed_n = rename_param(n)
        #     self.name_to_trainable_params[renamed_n] = p
        # ema_weight = config.ema_weight
        self.generator_ema = None
        # if (ema_weight is not None) and (ema_weight > 0.0):
        #     print(f"Setting up EMA with weight {ema_weight}")
        #     self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        self.max_grad_norm = 10.0
        self.previous_time = None

    def _debug(self, msg):
        print(
            f"[rank {dist.get_rank()}] "
            f"[step {self.step}] {msg}",
            flush=True,
        )
        
    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(self.model.generator)
        state_dict = {
            "generator": generator_state_dict
        }

        critic_state_dict = {"critic": fsdp_state_dict(self.model.fake_score)}

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            torch.save(critic_state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "critic_model.pt"))
            
            print("Model saved to", os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}", 
            "model.pt and critic_model.pt"))

    def train_one_step_generator(self, ode_latent, conditional_dict, loss_scale=1.0):
        self.model.eval()  # prevent any randomness (e.g. dropout)
        self.generator_optimizer.zero_grad(set_to_none=True)

        # Train the generator
        generator_loss, log_dict = self.model.generator_loss(
            ode_latent=ode_latent,
            conditional_dict=conditional_dict,
            unconditional_dict=self.unconditional_dict,
            global_step=self.step
        )

        (generator_loss * loss_scale).backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm)
        self.generator_optimizer.step()

        # Logging
        if self.is_main_process:
            if not self.disable_wandb:
                wandb_loss_dict = {
                    "generator_loss": generator_loss.item(),
                    "generator_grad_norm": generator_grad_norm.item(),
                }
                wandb.log(wandb_loss_dict, step=self.step)
                if self.step % 25 == 0:
                    print("Step: ", self.step, ", generator_loss: ", generator_loss.item(), "generator_grad_norm: ", generator_grad_norm.item())
            elif self.step % self.config.console_print_interval == 0:
                print("Step: ", self.step, ", generator_loss: ", generator_loss.item(), "generator_grad_norm: ", generator_grad_norm.item())
            else:
                pass

    def train_one_step_critic(self, ode_latent, conditional_dict, loss_scale=1.0):
        self.critic_optimizer.zero_grad(set_to_none=True)

        critic_loss, critic_log_dict = self.model.critic_loss(
            ode_latent, 
            conditional_dict,
            global_step=self.step
        )

        (critic_loss * loss_scale).backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm)
        self.critic_optimizer.step()

        critic_log_dict.update({
            "critic_loss": critic_loss,
            "critic_grad_norm": critic_grad_norm
        })

        # Logging
        if self.is_main_process:
            if not self.disable_wandb:
                wandb_loss_dict = {
                    "critic_loss": critic_loss.item(),
                    "critic_grad_norm": critic_grad_norm.item(),
                }
                wandb.log(wandb_loss_dict, step=self.step)
                if self.step % 25 == 0:
                    print("Step: ", self.step, ", critic_loss: ", critic_loss.item(), "critic_grad_norm: ", critic_grad_norm.item())
            elif self.step % self.config.console_print_interval == 0:
                print("Step: ", self.step, ", critic_loss: ", critic_loss.item(), "critic_grad_norm: ", critic_grad_norm.item())
            else:
                pass

    def train(self):
        
        max_steps = int(getattr(self.config, "max_steps", -1))
        batch_size = int(getattr(self.config, "batch_size", 1))
        if self.is_main_process:
            print("Training begin.")

        with torch.no_grad():
            self.unconditional_dict = self.model.text_encoder(
                text_prompts=[self.config.negative_prompt] * batch_size
            )
            
        while max_steps < 0 or self.step < max_steps:
            do_train_generator = self.step > 0 and self.step % self.config.dfake_gen_update_ratio == 0
            do_train_critic = True
            
            # Get the next batch of text prompts
            batch = next(self.dataloader)
            text_prompts = batch["prompts"]
            ode_latent = batch["ode_latent"].to(device=self.device, dtype=self.dtype)
                
            # Extract the conditional infos
            with torch.no_grad():
                conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
            
            # Train the generator
            if do_train_generator:
                self.train_one_step_generator(ode_latent, conditional_dict)

                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

            if do_train_critic:
                # Train the critic
                self.train_one_step_critic(ode_latent, conditional_dict)

            if (not self.config.no_save) and self.step % self.config.log_iters == 0 and self.step > 0:
                self.save()
                torch.cuda.empty_cache()
            
            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                # barrier()
                
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            self.step += 1
