# Copied from https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/diffusers_patch/ddim_with_logprob.py
# We adapt it from flow to flow matching.

import math
from typing import Optional, Union
import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

def sde_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
):
    """
    Predict the sample from the previous timestep by **reversing** the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned flow model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
    """
    # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
    model_output=model_output.float()
    sample=sample.float()
    if prev_sample is not None:
        prev_sample=prev_sample.float()

    step_index = [self.index_for_timestep(t) for t in timestep]
    prev_step_index = [step + 1 for step in step_index]
    sigma = self.sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1))) # sigma is a decreasing sequence from 1 to 0, sigma=1 means pure noise, sigma=0 means pure data
    sigma_prev = self.sigmas[prev_step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_max = self.sigmas[1].item() # =1 here
    dt = sigma_prev - sigma # dt is negative

    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level
    
    # our sde
    # Equation (9):
    #              sigma <-> t
    #        noise_level <-> a below Equation (9) - gives sigma_t = sqrt(t/(1-t))*a in the paper - corresponsds to std_dev_t = sqrt(sigma/(1-sigma))*noise_level here
    #                 dt <-> -\delta_t
    #       model_output <-> v_\theta(x_t, t)
    #             sample <-> x_t
    #        prev_sample <-> x_{t+\delta_t}
    #          std_dev_t <-> sigma_t

    prev_sample_mean = sample * (1 + std_dev_t**2 / (2 * sigma) * dt) + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
    
    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        # Last term of Equation (9)
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
        - torch.log(std_dev_t * torch.sqrt(-1 * dt))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    

    # Returns x_{t+\delta_t}, log_prob, x_{t+\delta_t} mean, sigma_t
    return prev_sample, log_prob, prev_sample_mean, std_dev_t