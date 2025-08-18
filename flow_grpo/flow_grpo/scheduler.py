from typing import Optional
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from typing import Optional
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from typing import Optional
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

class FlowMatchSlidingWindowScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(
        self,
        noise_level : float = 0.7,
        window_size: int = 1000,
        iters_per_group: int = 25,
        left_boundary : int = 0,
        sample_strategy: str = "progressive",
        prog_overlap_step: int = 1,
        roll_back: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._window_size = min(window_size, self.config.num_train_timesteps)
        self.noise_level = noise_level
        self.iters_per_group = iters_per_group
        self.left_boundary = left_boundary
        self.sample_strategy = sample_strategy
        self.prog_overlap_step = prog_overlap_step
        self.roll_back = roll_back

        assert self.noise_level >= 0 and self.noise_level <= 1, "Noise level must be between 0 and 1."
        assert self._window_size > 0, "Window size must be greater than 0."
        assert self.left_boundary >= 0, "Left boundary must be non-negative."
        assert self.prog_overlap_step < self._window_size, "Progressive overlap step must be less than window size."
        assert self.sample_strategy in ["progressive", "random"], f"Sample strategy must be one of ['progressive', 'random']. {sample_strategy} is not supported."

        self.cur_timestep = self.left_boundary
        self.cur_iter_in_group = 0

    @property
    def window_size(self):
        if self._window_size > self.config.num_train_timesteps:
            self._window_size = self.config.num_train_timesteps
        
        return self._window_size

    def update_iteration(self, seed=None):
        self.cur_iter_in_group += 1
        if self.sample_strategy == "progressive":
            if self.prog_overlap_step > 0:
                self.cur_timestep += self.prog_overlap_step
            else:
                self.cur_timestep += self.window_size
        if self.cur_timestep > self.right_boundary:
            if self.roll_back:
                self.cur_timestep = self.left_boundary
            else:
                self.cur_timestep = self.right_boundary
        elif self.sample_strategy == "random":
            generator = torch.Generator()
            if seed is not None:
                generator.manual_seed(seed)
            self.cur_timestep = torch.randint(0, self.config.num_train_timesteps - self.window_size + 1, (1,), generator=generator).item()

    def get_window_timesteps(self) -> torch.Tensor:
        start = self.cur_timestep
        end = min(self.cur_timestep + self.window_size, self.right_boundary)
        return self.timesteps[start:end]

    def get_window_sigmas(self) -> torch.Tensor:
        start = self.cur_timestep
        end = min(self.cur_timestep + self.window_size, self.right_boundary)
        return self.sigmas[start:end]

    def get_noise_levels(self) -> torch.Tensor:
        """ Returns noise levels on all timesteps, where noise level is non-zero only within the current window. """
        window_indices = [self.index_for_timestep(t) for t in self.get_window_timesteps()]
        noise_levels = torch.zeros(self.config.num_train_timesteps)
        noise_levels[window_indices] = self.noise_level
        return noise_levels
    
    def get_noise_level_for_timestep(self, time_step) -> float:
        """
            Return the noise level for a specific timestep.
        """
        if self.left_boundary <= self.index_for_timestep(time_step) <= self.right_boundary:
            return self.noise_level

        return 0.0

    def is_training_complete(self):
        if self.cur_iter_in_group >= self.iters_per_group:
            return True

        return False