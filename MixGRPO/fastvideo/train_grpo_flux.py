# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# SPDX-License-Identifier: [Apache License 2.0] 
#
# This file has been modified by [ByteDance Ltd. and/or its affiliates.] in 2025.
#
# Original file was released under [Apache License 2.0], with the full license text
# available at [https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE].
#
# This modified file is released under the same license.

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from argparse import Namespace

import cv2
import numpy as np
import torch
import torch.distributed as dist
import wandb
import wandb.util
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, FluxTransformer2DModel, FluxPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from PIL import Image
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from fastvideo.dataset.latent_flux_rl_datasets import LatentDataset, latent_collate_function
from fastvideo.dataset.text_dataset import TextPromptDataset, JsonPromptDataset
from fastvideo.reward.clip_score import CLIPScoreRewardModel
from fastvideo.reward.hps_score import HPSClipRewardModel
from fastvideo.reward.image_reward import ImageRewardModel
from fastvideo.reward.ocr_score import OcrRewardModel
from fastvideo.reward.pick_score import PickScoreRewardModel
from fastvideo.reward.unified_reward import UnifiedRewardModel
from fastvideo.reward.utils import balance_pos_neg, compute_reward
from fastvideo.utils.checkpoint import save_checkpoint, save_lora_checkpoint
from fastvideo.utils.communications_flux import sp_parallel_dataloader_wrapper
from fastvideo.utils.dataset_utils import LengthGroupedSampler
from fastvideo.utils.fsdp_util import apply_fsdp_checkpointing, get_dit_fsdp_kwargs
from fastvideo.utils.grpo_states import GRPOTrainingStates
from fastvideo.utils.load import load_transformer
from fastvideo.utils.logging_ import main_print
from fastvideo.utils.parallel_states import (
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    initialize_sequence_parallel_state,
    nccl_info,
)
from fastvideo.utils.sampling_utils import (
    dance_grpo_denoising_step,
    dpm_denoising_step,
    flow_grpo_denoising_step,
    sample_diffusion_trajectory,
    timesteps_shift,
)
# from fastvideo.utils.validation import log_validation

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")


# Some constants

EPSILON = 1e-8
SPATIAL_DOWNSAMPLE = 8
IN_CHANNELS = 16
    

def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"


class DistributedKRepeatSampler(DistributedSampler):
    def __init__(self, dataset : Dataset, num_replicas=None, rank=None, shuffle=True, seed=0, k=1):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)
        """
        DistributedKRepeatSampler is a distributed sampler that repeats each index k times.
        """
        
        # Number of times to repeat each index
        self.k = k
        # Original dataset len
        self.original_len = len(self.dataset)

    def __iter__(self):
        # Get the indices for the current replica
        indices_per_replica = list(super().__iter__())
        
        # Repeat each index k times
        repeated_indices = [idx for idx in indices_per_replica for _ in range(self.k)]
        
        return iter(repeated_indices)
    
    def __len__(self):
        return super().__len__() * self.k


def compute_text_embeddings(pipeline : FluxPipeline, prompt : str, device : torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(prompt)
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        text_ids = text_ids.to(device)

    return prompt_embeds, pooled_prompt_embeds, text_ids

def prepare_latent_image_ids(height : int, width : int, device : torch.device, dtype : torch.dtype) -> torch.Tensor:
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def pack_latents(latents : torch.Tensor, batch_size : int, num_channels_latents : int, height : int, width : int) -> torch.Tensor:
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

def unpack_latents(latents : torch.Tensor, height : int, width : int, vae_scale_factor : int) -> torch.Tensor:
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents

def compute_log_probs(
    args : Namespace,
    latents : torch.Tensor,
    pre_latents : torch.Tensor,
    encoder_hidden_states : torch.Tensor,
    pooled_prompt_embeds : torch.Tensor,
    text_ids : torch.Tensor,
    image_ids : torch.Tensor,
    transformer : FluxTransformer2DModel,
    timesteps : List[int],
    i : int,
    sigma_schedule : List[float],
) -> torch.Tensor:
    B = encoder_hidden_states.shape[0]
    transformer.train()
    with torch.autocast("cuda", torch.bfloat16):
        pred= transformer(
            hidden_states=latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep= timesteps / 1000,
            guidance=torch.tensor(
                [3.5],
                device=latents.device,
                dtype=torch.bfloat16
            ),
            txt_ids=text_ids.repeat(encoder_hidden_states.shape[1], 1), # No batch dimension is required here, same as img_ids
            pooled_projections=pooled_prompt_embeds,
            img_ids=image_ids, # No batch dimension is required here, same as txt_ids
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    
    if args.dpm_algorithm_type == "null" or ("dpmsolver" in args.dpm_algorithm_type and args.dpm_apply_strategy == "post"):
        if args.flow_grpo_sampling:
            z, pred_original, log_prob, prev_latents_mean, std_dev_t = flow_grpo_denoising_step(
                model_output=pred,
                latents=latents.to(torch.float32),
                eta=args.eta,
                sigmas=sigma_schedule,
                index=i,
                prev_sample=pre_latents.to(torch.float32),
                determistic=False,
            )
        else:
            z, pred_original, log_prob = dance_grpo_denoising_step(pred, latents.to(torch.float32), args.eta, sigma_schedule, i, prev_sample=pre_latents.to(torch.float32), grpo=True, sde_solver=True)
    elif "dpmsolver" in args.dpm_algorithm_type:
        z, pred_original, log_prob = dpm_denoising_step(
            args,
            model_output=pred,
            sample=latents.to(torch.float32),
            step_index=i,
            timesteps=sigma_schedule[:-1],
            dpm_state=None,
            generator=torch.Generator(device=latents.device),
            sde_solver=True,
            sigmas=sigma_schedule,
        )
    return log_prob

def sample_reference_model(
    args : Namespace,
    device : torch.device,
    transformer : FluxTransformer2DModel,
    vae : AutoencoderKL,
    encoder_hidden_states : torch.Tensor, 
    pooled_prompt_embeds : torch.Tensor, 
    text_ids : torch.Tensor,
    reward_model : RewardModel,
    captions : List[str],
    timesteps_train : List[int],
    global_step : int
):
    w, h, t = args.w, args.h, args.t
    sample_steps = args.sampling_steps
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1).to(device)
    
    sigma_schedule = timesteps_shift(args.shift, sigma_schedule) # [1, 0], length=17

    assert_eq(
        len(sigma_schedule),
        sample_steps + 1,
        "sigma_schedule must have length sample_steps + 1",
    )
    w, h, t = args.w, args.h, args.t
    B = encoder_hidden_states.shape[0]
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE
    image_ids = prepare_latent_image_ids(latent_h // 2, latent_w // 2, device, torch.bfloat16)

    if args.init_same_noise:
        input_latents = torch.randn(
                (1, IN_CHANNELS, latent_h, latent_w),
                device=device,
                dtype=torch.bfloat16,
            )
        # Repeat input_latents on the dim=0
        input_latents = input_latents.repeat(B, 1, 1, 1)
    else:
        input_latents = torch.randn(
                    (B, IN_CHANNELS, latent_h, latent_w),
                    device=device,
                    dtype=torch.bfloat16,
                )

    input_latents = pack_latents(input_latents, B, IN_CHANNELS, latent_h, latent_w)

    grpo_sample = True

    if args.training_strategy == "part":
        determistic = [True] * sample_steps
        for i in timesteps_train:
            determistic[i] = False
    elif args.training_strategy == "all":
        determistic = [False] * sample_steps

    progress_bar = tqdm(range(0, sample_steps), desc="Sampling Progress", disable=not dist.is_initialized() or dist.get_rank() != 0)

    z, latents, all_latents, all_log_probs = sample_diffusion_trajectory(
            args,
            input_latents,
            progress_bar,
            sigma_schedule,
            transformer,
            encoder_hidden_states,
            pooled_prompt_embeds,
            text_ids, # 2D tensor
            image_ids, # 2D tensor
            grpo_sample,
            determistic=determistic,
        )

    vae.enable_tiling()
    
    image_processor = VaeImageProcessor(do_resize=True)
    
    with torch.inference_mode():
        with torch.autocast("cuda", torch.bfloat16):
            latents_unpacked = unpack_latents(latents, h, w, 8)
            latents_unpacked = (latents_unpacked / 0.3611) + 0.1159
            # Decode all images in one go
            images_tensor = vae.decode(latents_unpacked, return_dict=False)[0]
            # decoded_images = [image_processor.postprocess(img)[0] for img in images_tensor]
            decoded_images = image_processor.postprocess(images_tensor, output_type='pil')

    rewards, successes = compute_reward(
            decoded_images, 
            captions,
            reward_model
        )

    return rewards, all_latents, all_log_probs, sigma_schedule


def gather_tensor(tensor : torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)

def gather_objects(local_results : Any, rank, world_size) -> List[Any]:
    gathered_results = [_ for _ in range(world_size)]
    dist.gather_object(local_results, gathered_results, dst=rank)
    
    # Merge all object
    all_results = []
    for results in gathered_results:
        all_results.append(results)

    return all_results

def train_one_step(
    args : Namespace,
    device : torch.device,
    transformer : FluxTransformer2DModel,
    vae : AutoencoderKL,
    reward_model : RewardModel,
    optimizer : torch.optim.Optimizer,
    lr_scheduler : torch.optim.lr_scheduler.LambdaLR,
    train_batch : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]],
    max_grad_norm : float,
    timesteps_train : List[int],
    global_step : int
) -> Dict[str, Union[float, torch.Tensor]]:
    total_loss = 0.0
    kl_total_loss = 0.0
    policy_total_loss = 0.0
    total_clip_frac = 0.0

    optimizer.zero_grad()

    (
        encoder_hidden_states,
        pooled_prompt_embeds,
        text_ids,
        caption,
    ) = train_batch

    B = encoder_hidden_states.shape[0] # = args.num_generations * args.train_batch_size
    w, h = args.w, args.h
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE
    # Latest flux only requires 2-D dimensional tensor of image_ids, no batch dimention.
    image_ids = prepare_latent_image_ids(latent_h // 2, latent_w // 2, device, torch.bfloat16)

    # Note: No batch dimension required for both text_ids and img_ids, and all text_ids are the same.
    # refer to https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux.py#L631
    # line 700-711
    # So reshape it to (1, 3) for canonicalization.
    text_ids = text_ids[0].unsqueeze(0)

    rewards, all_latents, all_log_probs, sigma_schedule = sample_reference_model(
            args,
            device,
            transformer,
            vae,
            encoder_hidden_states,
            pooled_prompt_embeds,
            text_ids,
            reward_model,
            caption,
            timesteps_train,
            global_step
        )

    timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][:args.sampling_steps]
    timestep_values = [timestep_value[:] for _ in range(B)]
    device = all_latents.device
    timesteps =  torch.tensor(timestep_values, device=all_latents.device, dtype=torch.long)

    samples = {
        "timesteps": timesteps.detach().clone()[:, :-1],
        "latents": all_latents[:, :-2],  # each entry is the latent before timestep t
        "next_latents": all_latents[:, 1:-1],  # each entry is the latent after timestep t
        "log_probs": all_log_probs[:, :-1],
        "encoder_hidden_states": encoder_hidden_states,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "rewards": rewards
    }

    gathered_reward = gather_tensor(rewards)

    if dist.get_rank()==0:
        print(f"gathered_{args.reward_model}:", gathered_reward)

    # Compute advantages
    if args.use_group:
        # Compute advantages for each prompt
        n = len(samples["rewards"]) // (args.num_generations)
        advantages = torch.zeros_like(samples["rewards"])
        
        for i in range(n):
            start_idx = i * args.num_generations
            end_idx = (i + 1) * args.num_generations
            group_rewards = samples["rewards"][start_idx:end_idx]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
        
        samples["advantages"] = advantages
    else:
        # Compute advantages globally
        advantages = (samples["rewards"] - gathered_reward.mean())/(gathered_reward.std()+1e-8)
        samples["advantages"] = advantages

    time_perms = torch.stack(
        [
            torch.randperm(len(samples["timesteps"][0]))
            for _ in range(B)
        ]
    ).to(device)
    for key in ["timesteps", "latents", "next_latents", "log_probs"]:
        samples[key] = samples[key][
            torch.arange(B).to(device)[:, None],
            time_perms,
        ]

    samples_batched = {
        k: v.unsqueeze(1)
        for k, v in samples.items()
        if k not in ['rewards']
    }
    # All values of samples_batched should have the same length
    # assert len(set(len(v) for v in samples_batched.values())) == 1

    # dict of lists -> list of dicts for easier iteration
    samples_batched_list = [
        dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
    ]
    
    if dist.get_rank() == 0:
        optimize_sampling_time = 0

    for i, sample in enumerate(samples_batched_list):
        for t in timesteps_train:
            if dist.get_rank() == 0:
                meta_optimize_sampling_time = time.time()
            clip_range = args.clip_range
            adv_clip_max = args.adv_clip_max
            new_log_probs = compute_log_probs(
                args,
                sample["latents"][:,t],
                sample["next_latents"][:,t],
                sample["encoder_hidden_states"],
                sample["pooled_prompt_embeds"],
                text_ids,
                image_ids,
                transformer,
                sample["timesteps"][:,t],
                time_perms[i][t],
                sigma_schedule,
            )

            if dist.get_rank() == 0:
                meta_optimize_sampling_time = time.time() - meta_optimize_sampling_time
                optimize_sampling_time += meta_optimize_sampling_time

            advantages = torch.clamp(
                sample["advantages"],
                -adv_clip_max,
                adv_clip_max,
            )

            ratio = torch.exp(new_log_probs - sample["log_probs"][:,t])

            unclipped_loss = -advantages * ratio
            clipped_loss = -advantages * torch.clamp(
                ratio,
                1.0 - clip_range,
                1.0 + clip_range,
            )
            clip_frac = torch.mean((torch.abs(ratio - 1.0) > clip_range).float())
            policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (args.gradient_accumulation_steps * len(timesteps_train))
            kl_loss = 0.5 * torch.mean((new_log_probs - sample["log_probs"][:, t]) ** 2) / (args.gradient_accumulation_steps * len(timesteps_train))
            loss = policy_loss + args.kl_coeff * kl_loss

            loss.backward()

            # Use one single tensor to stack all loss tensors
            loss_info = torch.stack([l.detach().clone() for l in [loss, policy_loss, kl_loss, clip_frac]])
            # Use one single tensor for efficient communication
            dist.all_reduce(loss_info, op=dist.ReduceOp.AVG)
            for l, total in zip(loss_info, [total_loss, policy_total_loss, kl_total_loss, total_clip_frac]):
                total += l.item()

        if (i + 1) % args.gradient_accumulation_steps == 0:
            grad_norm = transformer.clip_grad_norm_(max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        if dist.get_rank() == 0:
            main_print(f"##### Optimize sampling time per step: {optimize_sampling_time / (i+1)} seconds")

        if dist.is_initialized():
            dist.barrier()

    return {
        'loss': total_loss,
        'grad_norm': grad_norm.item(),
        'policy_loss': policy_total_loss,
        'kl_loss': kl_total_loss,
        'clip_frac': total_clip_frac,
        'reward': gathered_reward
    }


def evaluate(
    args,
    device,
    transformer,
    vae,
    reward_model,
    global_step,
    test_dataloader,
    num_eval_samples=6
):
    """
    Evaluate the model on the given prompts and log on wandb
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Only print from the main process
    if rank == 0:
        print(f"Starting evaluation at step {global_step}")

    w, h = args.w, args.h
    sample_steps = args.sampling_steps
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1).to(device)
    sigma_schedule = timesteps_shift(args.shift, sigma_schedule)

    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE    # This list will only be populated on rank 0
    all_eval_results_for_logging = []

    # Set a max_eval_num for efficiency
    max_eval_num = 1

    # Some shared variables for the evaluation
    image_ids = prepare_latent_image_ids(latent_h // 2, latent_w // 2, device, torch.bfloat16)


    all_eval_results_for_logging = []

    with torch.no_grad():
        dataloader_progress = tqdm(
            test_dataloader,
            desc="Evaluation Batches", 
            disable=(rank != 0)
        )
        for eval_batch in dataloader_progress:
            (
                encoder_hidden_states_batch, 
                pooled_prompt_embeds_batch, 
                text_ids_batch,
                eval_prompts,
            ) = eval_batch


            # Enough reaults
            if len(all_eval_results_for_logging) >= max_eval_num:
                break

            num_prompts_in_batch = len(eval_prompts)
            total_samples_to_generate = num_prompts_in_batch * num_eval_samples

            generator = torch.Generator(device=device)
            if args.seed is not None:
                generator.manual_seed(args.seed + rank)

            batched_encoder_hidden_states = encoder_hidden_states_batch.repeat_interleave(num_eval_samples, dim=0)
            batched_pooled_prompt_embeds = pooled_prompt_embeds_batch.repeat_interleave(num_eval_samples, dim=0)
            
            input_latents = torch.randn(
                (total_samples_to_generate, IN_CHANNELS, latent_h, latent_w),
                device=device,
                dtype=torch.bfloat16,
                generator=generator
            )

            text_ids = text_ids_batch[0].unsqueeze(0)

            input_latents_packed = pack_latents(input_latents, total_samples_to_generate, IN_CHANNELS, latent_h, latent_w)

            progress_bar = tqdm(range(0, args.sampling_steps), desc="Batched Sampling for Eval", disable=(rank!=0))
            with torch.autocast("cuda", torch.bfloat16):
                sigma_schedule = timesteps_shift(args.shift, torch.linspace(1, 0, args.sampling_steps + 1).to(device))
                z, latents, _, _ = sample_diffusion_trajectory(
                    args,
                    input_latents_packed,
                    progress_bar,
                    sigma_schedule,
                    transformer,
                    batched_encoder_hidden_states,
                    batched_pooled_prompt_embeds,
                    text_ids,
                    image_ids,
                    grpo_sample=True,
                    determistic=[True] * args.sampling_steps,
                )

            # --- Batched VAE Decoding and Reward Computation ---
            vae.enable_tiling()
            image_processor = VaeImageProcessor(do_resize=True)
            
            with torch.inference_mode():
                with torch.autocast("cuda", torch.bfloat16):
                    latents_unpacked = unpack_latents(latents, h, w, 8)
                    latents_unpacked = (latents_unpacked / 0.3611) + 0.1159
                    # Decode all images in one go
                    images_tensor = vae.decode(latents_unpacked, return_dict=False)[0]
                    # decoded_images = [image_processor.postprocess(img)[0] for img in images_tensor]
                    decoded_images = image_processor.postprocess(images_tensor, output_type='pil')

            # Compute rewards for the entire batch
            batched_prompts = [prompt for prompt in eval_prompts for _ in range(num_eval_samples)]
            all_rewards, success = compute_reward(
                decoded_images,
                batched_prompts,
                reward_model
            )

            # --- Gather and Log Results
            # Restructure flattened rewards and images for logging
            for prompt_idx, prompt in enumerate(eval_prompts):
                start_idx = prompt_idx * num_eval_samples
                end_idx = start_idx + num_eval_samples
                
                prompt_results_to_log = {
                    "prompt": prompt,
                    "images": decoded_images[start_idx:end_idx],
                    "rewards": all_rewards[start_idx:end_idx]
                }

                all_eval_results_for_logging.append(prompt_results_to_log)
        
        if rank == 0:
            dataloader_progress.set_postfix({
                'prompts_processed': num_prompts_in_batch,
                'total_results': len(all_eval_results_for_logging)
            })

    # All processes wait here until evaluation is done on all of them.
    if dist.is_initialized():
        dist.barrier()


    # Log on wandb for the main process
    if rank == 0:

        # Gather all all_eval_results_for_logging from all GPUs ( Realy slow, why???)
        # gathered_eval_results = gather_objects(all_eval_results_for_logging, rank, world_size)

        # gathered_eval_results = [item for sublist in gathered_eval_results for item in sublist]


        log_start_time = time.time()
        with tempfile.TemporaryDirectory() as tmpdir:
            wandb_log_dict = {
                "eval/prompt_details": []
            }
            
            for prompt_idx, result in enumerate(all_eval_results_for_logging):
                prompt = result["prompt"]
                images = result["images"]
                rewards = result["rewards"]
                
                wandb_images = []

                for img_idx, (img, reward) in enumerate(zip(images, rewards)):
                    if img_idx > 0: # Only one image per prompt for simplicity
                        break

                    prompt_len = len(prompt)
                    img_name = f"{rank}_{prompt_idx}_{img_idx}.jpg"
                    img.save(
                        os.path.join(tmpdir, img_name),
                        format="JPEG",
                        quality=90,
                        optimize=True
                    )
                    wandb_images.append(wandb.Image(
                        os.path.join(tmpdir, img_name),
                        caption=f"Prompt {prompt[:min(prompt_len, 50)]}, Sample {img_idx}, Reward: {reward:.4f}"
                    ))
            
                wandb_log_dict[f"eval/images"] = wandb_images

                wandb_log_dict["eval/prompt_details"].append({
                    'prompt': prompt,
                    "avg_reward": np.mean(all_rewards[start_idx:end_idx])
                })
            

            wandb_log_dict["eval/global_reward_mean"] = np.mean(all_rewards)
            wandb_log_dict["eval/global_reward_std"] = np.std(all_rewards)
        
            wandb.log(wandb_log_dict, step=global_step)
            log_end_time = time.time()
            print(f"\n\nEvaluation completed. Global average reward: {np.mean(all_rewards):.4f}")
            print(f"Time taken for logging: {log_end_time - log_start_time:.2f} seconds")
    
    if dist.is_initialized():
        dist.barrier()

def main(args):
    ############################# Init #############################
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    dist.init_process_group(backend="nccl")

    initialize_sequence_parallel_state(args.sp_size)

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(f"{args.output_dir}/{args.training_strategy}_{args.experiment_name}", exist_ok=True)
        args_dict = vars(args)
        run_id = wandb.util.generate_id()
        args_dict["wandb_id"] = run_id
        with open(f"{args.output_dir}/{args.training_strategy}_{args.experiment_name}/args.json", "w") as f:
            json.dump(args_dict, f, indent=4)
    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required
    
    ############################# Build reward models #############################
    if args.reward_model == "hpsv2":
        reward_model = HPSClipRewardModel(
            device=device,
            clip_ckpt_path=args.hps_clip_path,
            hps_ckpt_path=args.hps_path,
        )
    elif args.reward_model == "image_reward":
        reward_model = ImageRewardModel(
            model_name=args.image_reward_path,
            device=device,
            med_config=args.image_reward_med_config,
            http_proxy=args.image_reward_http_proxy,
            https_proxy=args.image_reward_https_proxy,
        )
    elif args.reward_model == "clip_score":
        reward_model = CLIPScoreRewardModel(
            clip_model_path=args.clip_score_path,
            device=device,
        )
    elif args.reward_model == "pick_score":
        reward_model = PickScoreRewardModel(
            device=device,
            http_proxy=args.pick_score_http_proxy,
            https_proxy=args.pick_score_https_proxy,
        )
    elif args.reward_model == "ocr_score":
        reward_model = OcrRewardModel(
            device_id=device
        )
    elif args.reward_model == "unified_reward":
        unified_reward_urls = args.unified_reward_url.split(",")
        
        if isinstance(unified_reward_urls, list):
            num_urls = len(unified_reward_urls)
            ur_url_idx = rank % num_urls
            ur_url = unified_reward_urls[ur_url_idx]
            print(f"Rank {rank} using unified-reward URL: {ur_url}")
        reward_model = UnifiedRewardModel(
            api_url=ur_url,
            default_question_type=args.unified_reward_default_question_type,
            num_workers=args.unified_reward_num_workers,
        )
    else:
        raise ValueError(f"Unsupported reward model: {args.reward_model}")


    ############################# Build FLUX #############################
    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")
    # keep the master weight to float32

    transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype = torch.float32
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype = torch.bfloat16,
    ).to(device)

    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        False,
        args.use_cpu_offload,
        args.master_weight_type,
    )
    
    transformer = FSDP(transformer, **fsdp_kwargs)

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(
            transformer, no_split_modules, args.selective_checkpointing
        )


    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}"
    )
    # Load the reference model
    main_print(f"--> model loaded")

    # Set model as trainable.
    transformer.train()

    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))


    #-----------------------------Optimizer------------------------
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )


    # ---------------------------Prepare data------------------------------
    train_dataset = LatentDataset(args.data_json_path, args.num_latent_t, args.cfg)
    test_dataset = LatentDataset(args.test_data_json_path, args.num_latent_t, args.cfg)


    train_sampler = DistributedKRepeatSampler(
        train_dataset,
        rank=rank,
        num_replicas=world_size,
        shuffle=True,
        seed=args.sampler_seed,
        k=args.num_generations
    )

    test_sampler = DistributedKRepeatSampler(
        test_dataset,
        rank=rank,
        num_replicas=world_size,
        shuffle=False,
        seed=args.sampler_seed,
        k=6
    )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )
    train_dataloader = iter(train_dataloader)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    # ---------------------Init wandb---------------------
    if rank <= 0:
        project = "MixGRPO-Flux"
        wandb_run = wandb.init(
            project=project, 
            config=args, 
            name=args.experiment_name,
            id=run_id,
        )

    # wait for wandb init
    if dist.is_initialized():
        dist.barrier()

    # --------------------------Print Parameters--------------------------
    total_batch_size = args.train_batch_size * world_size * args.gradient_accumulation_steps
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(
        f"  Total train batch size = train_batch_size * world_size * gradient_acc_steps = {args.train_batch_size} * {world_size} * {args.gradient_accumulation_steps} = {total_batch_size}"
    )
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps per epoch = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    # print dtype
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        assert NotImplementedError("resume_from_checkpoint is not supported now.")

    progress_bar = tqdm(
        range(0, 100000),
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )

    step_times = deque(maxlen=100)

    # -------------------------MixGRPO States-------------------------
    if args.training_strategy == "part":
        grpo_states = GRPOTrainingStates(
            iters_per_group=args.iters_per_group,
            window_size=args.window_size,
            max_timesteps=args.sampling_steps-2,  # Because the max timestep index is args.sampling_steps - 2
            cur_timestep=0,
            cur_iter_in_group=0,
            sample_strategy=args.sample_strategy,
            prog_overlap=args.prog_overlap,
            prog_overlap_step=args.prog_overlap_step,
            max_iters_per_group=args.max_iters_per_group,
            min_iters_per_group=args.min_iters_per_group,
            roll_back=args.roll_back,
        )

    # -------------------Begin training loop-------------------
    global_step = -1
    for epoch in range(1000000):
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch) # Crucial for distributed shuffling per epoch

        for step in range(init_steps+1, args.max_train_steps+1):
            global_step += 1
            # Do evaluation every args.eval_steps
            if args.enable_eval and global_step % args.eval_steps == 0:
                evaluate(
                    args,
                    device,
                    transformer,
                    vae,
                    reward_model,
                    global_step,
                    test_dataloader,
                    num_eval_samples=args.num_eval_samples
                )

            if step % args.checkpointing_steps == 0:
                # Save at most 2 latest checkpoints
                checkpoint_saving_dir = f"{args.output_dir}/{args.training_strategy}_{args.experiment_name}"
                os.makedirs(checkpoint_saving_dir, exist_ok=True)
                existing_checkpoints = [
                    checkpoint_dir for checkpoint_dir in os.listdir(checkpoint_saving_dir)
                    if os.path.isdir(checkpoint_dir) and 'checkpoint' in checkpoint_dir
                ]
                if len(existing_checkpoints) >= 2:
                    # Remove the oldest checkpoint directory
                    existing_checkpoints.sort()
                    shutil.rmtree(os.path.join(checkpoint_saving_dir, existing_checkpoints[0]))

                save_checkpoint(transformer, rank, checkpoint_saving_dir, step, epoch)

                if dist.is_initialized():
                    dist.barrier()

            start_time = time.time()
            if args.training_strategy == "part":
                timesteps_train = grpo_states.get_current_timesteps()
                grpo_states.update_iteration()
            elif args.training_strategy == "all":
                timesteps_train = [ti for ti in range(args.sampling_steps)]

            train_res = train_one_step(
                args,
                device, 
                transformer,
                vae,
                reward_model,
                optimizer,
                lr_scheduler,
                train_dataloader,
                args.max_grad_norm,
                timesteps_train,
                global_step
            )
    
            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)
    
            progress_bar.set_postfix(train_res)
            progress_bar.update(1)
            if rank == 0:
                
                log_dict = {
                    "train_loss": train_res['loss'],
                    "policy_loss": train_res['policy_loss'],
                    "kl_loss": train_res['kl_loss'],
                    "clip_frac": train_res['clip_frac'],
                    "cur_timesteps": grpo_states.cur_timestep if args.training_strategy == "part" else 0,
                    "cur_iter_in_group": grpo_states.cur_iter_in_group if args.training_strategy == "part" else 0,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "step_time": step_time,
                    "avg_step_time": avg_step_time,
                    "grad_norm": train_res['grad_norm'],
                    "epoch": epoch,
                    "reward_avg": train_res['reward']
                }
                wandb.log(log_dict, step=global_step)
        
            if dist.is_initialized():
                dist.barrier()

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument('--test_data_json_path', type=str, required=True)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the sampling dataloader.",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the test dataloader.",
    )
    parser.add_argument(
        "--num_latent_t",
        type=int,
        default=1,
        help="number of latent frames",
    )
    # text encoder & vae & diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, default=None, help="vae model.")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument(
        "--precondition_outputs",
        action="store_true",
        help="Whether to precondition the outputs of the model.",
    )

    # validation & logs
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    # optimizer & scheduler & Training
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--max_grad_norm", default=2.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )
    parser.add_argument(
        "--test_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel testing",
    )

    parser.add_argument("--fsdp_sharding_startegy", default="full")

    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
        choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to apply."
    )
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )

    #GRPO training
    parser.add_argument(
        "--h",
        type=int,
        default=None,   
        help="video height",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=None,   
        help="video width",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=None,   
        help="video length",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=None,   
        help="sampling steps",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,   
        help="noise eta",
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=None,   
        help="seed of sampler",
    )
    parser.add_argument(
        "--loss_coef",
        type=float,
        default=1.0,   
        help="the global loss should be divided by",
    )
    parser.add_argument(
        "--use_group",
        action="store_true",
        default=False,
        help="whether compute advantages for each prompt",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=16,   
        help="num_generations per prompt",
    )
    parser.add_argument(
        "--ignore_last",
        action="store_true",
        default=False,
        help="whether ignore last step of mdp",
    )
    parser.add_argument(
        "--init_same_noise",
        action="store_true",
        default=False,
        help="whether use the same noise within each prompt",
    )
    parser.add_argument(
        "--shift",
        type = float,
        default=1.0,
        help="shift for timestep scheduler",
    )
    parser.add_argument(
        "--timestep_fraction",
        type = float,
        default=1.0,
        help="timestep downsample ratio",
    )
    parser.add_argument(
        "--clip_range",
        type = float,
        default=1e-4,
        help="clip range for grpo",
    )
    parser.add_argument(
        "--adv_clip_max",
        type = float,
        default=5.0,
        help="clipping advantage",
    )
    parser.add_argument(
        "--advantage_rerange_strategy",
        type=str,
        default="null",
        choices=["random", "balance", "null"],
        help="Rerange strategy for advantages when computing loss"
    )

    #################### MixGRPO ####################
    parser.add_argument(
        "--flow_grpo_sampling",
        action="store_true",
        default=False,
        help="whether to use flow grpo sampling, True for MixGRPO, False for DanceGRPO",
    )
    parser.add_argument(
        "--drop_last_sample",
        action="store_true",
        default=False,
        help="whether to drop the last sample in the batch if it is not complete, True for DanceGRPO but False for MixGRPO",
    )
    parser.add_argument(
        "--trimmed_ratio",
        type=float,
        default=0.0,
        help="ratio of trimmed for advantage computation, now is no used",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="test",
        help="experiment name, used for saving images and logs",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="all",
        choices=["part", "all"],
        help="training strategy, part means MixGRPO, all means DanceGRPO",
    )
    parser.add_argument(
        "--frozen_init_timesteps",
        type=int,
        default=-1,
        help="when training_strategy is 'all' and frozen_init_timesteps >0, it is used for freezing timesteps"
    )
    parser.add_argument(
        "--kl_coeff",
        type=float,
        default=0.01,
        help="coefficient for KL loss",
    )
    
    # Sliding Window
    parser.add_argument(
        "--iters_per_group",
        type=int,
        default=25,
        help="shift interval, moving the window after iters_per_group iterations",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=4,
        help="sliding window size",
    )
    parser.add_argument(
        "--sample_strategy",
        type=str,
        default="progressive",
        choices=["progressive", "random", "decay", "exp_decay"],
        help="Scheduling policy for optimized timesteps",
    )
    parser.add_argument(
        "--prog_overlap",
        action="store_true",
        default=False,
        help="Whether to overlap when moving the window"
    )
    parser.add_argument(
        "--prog_overlap_step",
        type=int,
        default=1,
        help="the window stride when prog_overlap is True",
    )
    parser.add_argument(
        "--max_iters_per_group",
        type=int,
        default=10,
        help="maximum shift interval in 'decay' strategy",
    )
    parser.add_argument(
        "--min_iters_per_group",
        type=int,
        default=1,
        help="minimum shift interval in 'decay' strategy",
    )
    parser.add_argument(
        "--roll_back",
        action="store_true",
        default=False,
        help="whether to roll back (restart) the sliding window",
    )
    #################### Reward ####################
    parser.add_argument(
        "--reward_model",
        type=str,
        default="hpsv2",
        choices=["hpsv2", "clip_score", "image_reward", "ocr_score", "pick_score", "unified_reward", "hpsv2_clip_score", "multi_reward"],
        help="reward model to use"
    )
    parser.add_argument(
        "--hps_path",
        type=str,
        default="hps_ckpt/HPS_v2.1_compressed.pt",
        help="path to load hps reward model",
    )
    parser.add_argument(
        "--hps_clip_path",
        type=str,
        default="hps_ckpt/open_clip_pytorch_model.bin",
        help="path to load hps clip model",
    )
    parser.add_argument(
        "--clip_score_path",
        type=str,
        default="hf-hub:apple/DFN5B-CLIP-ViT-H-14-384",
        help="clip model type"
    )
    parser.add_argument(
        "--image_reward_path",
        type=str,
        default="./image_reward_ckpt/ImageReward.pt",
        help="path to load image reward model",
    )
    parser.add_argument(
        "--image_reward_med_config",
        type=str,
        default="./image_reward_ckpt/med_config.json",
        help="path to load image reward model med config",
    )
    parser.add_argument(
        "--image_reward_http_proxy",
        type=str,
        default=None,
        help="http proxy for image reward model",
    )
    parser.add_argument(
        "--image_reward_https_proxy",
        type=str,
        default=None,
        help="https proxy for image reward model",
    )
    parser.add_argument(
        "--pick_score_http_proxy",
        type=str,
        default=None,
        help="http proxy for pick score reward model",
    )
    parser.add_argument(
        "--pick_score_https_proxy",
        type=str,
        default=None,
        help="https proxy for pick score reward model",
    )
    parser.add_argument(
        "--unified_reward_url",
        type=str,
        default=None,
        help="API URL for the unified reward model",
    )
    parser.add_argument(
        "--unified_reward_default_question_type",
        type=str,
        default=None,
        help="Default question type for the unified reward model",
    )
    parser.add_argument(
        "--unified_reward_num_workers",
        type=int,
        default=1,
        help="Number of workers for the unified reward model",
    )
    parser.add_argument(
        "--multi_reward_mix",
        type=str,
        default="advantage_aggr",
        choices=["advantage_aggr", "reward_aggr"],
        help="How to mix multiple rewards",
    )
    parser.add_argument(
        "--hps_weight",
        type=float,
        default=1.0,
        help="weight for hps reward model",
    )
    parser.add_argument(
        "--clip_score_weight",
        type=float,
        default=1.0,
        help="weight for clip score reward model",
    )
    parser.add_argument(
        "--image_reward_weight",
        type=float,
        default=1.0,
        help="weight for image reward model",
    )
    parser.add_argument(
        "--pick_score_weight",
        type=float,
        default=1.0,
        help="weight for pick score reward model",
    )
    parser.add_argument(
        "--unified_reward_weight",
        type=float,
        default=1.0,
        help="weight for unified reward model",
    )

    #################### Sampling ####################
    parser.add_argument(
        "--dpm_algorithm_type",
        type=str,
        default="null",
        choices=["null", "dpmsolver", "dpmsolver++"],
        help="null means no DPM-Solver, dpmsolver means DPM-Solver, dpmsolver++ means DPM-Solver++",
    )
    parser.add_argument(
        "--dpm_apply_strategy",
        type=str,
        default="post",
        choices=["post", "all"],
        help="post means apply DPM-Solver the ODE sampling process after SDE, all means apply DPM-Solver to all timesteps",
    )
    parser.add_argument(
        "--dpm_post_compress_ratio",
        type=float,
        default=0.4,
        help="when dpm_apply_strategy is post, the timesteps for ODE aftet SDE is compressed by this ratio",
    )
    parser.add_argument(
        "--dpm_solver_order",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Order of the DPM-Solver method. 1 is default DDIM (FM Sampling)",
    )
    parser.add_argument(
       "--dpm_solver_type",
        type=str,
        default="heun",
        choices=["heun", "midpoint"],
        help="when dpm_solver_order is 2, the type of DPM-Solver method.",
    )

    #################### Eval #####################
    parser.add_argument(
        "--enable_eval",
        action="store_true",
        help="Enable evaluation during training",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=10,
        help="Number of steps between evaluations",
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=6,
        help="Number of samples to use for evaluation",
    )

    #################### Wandb ####################
    parser.add_argument(
        "--wandb_key",
        type=str,
        default=None,
        help="Wandb API key for logging. If not provided, will not log to Wandb.",
    )

    args = parser.parse_args()
    wandb.login(key=args.wandb_key)

    if args.image_reward_http_proxy == "None":
        args.image_reward_http_proxy = None
    if args.image_reward_https_proxy == "None":
        args.image_reward_https_proxy = None
    if args.pick_score_http_proxy == "None":
        args.pick_score_http_proxy = None
    if args.pick_score_https_proxy == "None":
        args.pick_score_https_proxy = None
    if args.unified_reward_url == "None":
        args.unified_reward_url = None

    main(args)
