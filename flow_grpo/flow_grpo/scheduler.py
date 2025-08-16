from typing import Optional
import torch
from diffusers.schedulers.scheduling_utils import FlowMatchEulerDiscreteScheduler

class FlowMatchSlidingWindowScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(
        self,
        window_size: int,
        iters_per_group: int = 25,
        left_boundary : int = 0,
        right_boundary : Optional[int] = None,
        sample_strategy: str = "progressive",
        prog_overlap_step: int = 1,
        roll_back: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.iters_per_group = iters_per_group
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary if right_boundary is not None else self.num_train_timesteps
        self.sample_strategy = sample_strategy
        self.prog_overlap_step = prog_overlap_step
        self.roll_back = roll_back

        assert self.window_size > 0, "Window size must be greater than 0."
        assert self.left_boundary >= 0, "Left boundary must be non-negative."
        assert self.right_boundary > self.left_boundary, "Right boundary must be greater than left boundary."
        assert self.prog_overlap_step < self.window_size, "Progressive overlap step must be less than window size."
        assert self.sample_strategy in ["progressive", "random"], f"Sample strategy must be one of ['progressive', 'random']. {sample_strategy} is not supported."

        self.cur_timestep = 0
        self.cur_iter_in_group = 0
        # Record original sigmas
        self._sigmas = self.sigmas.clone()
        # Record original timesteps
        self._timesteps = self.timesteps.clone()
        self._sigmas_min = self._sigmas[-1].item()
        self._sigmas_max = self._sigmas[0].item()

    # Overwrite the properties within the sliding window
    @property
    def timesteps(self):
        return torch.tensor(self.get_window_timesteps(), device=self.device)
    
    @property
    def sigmas(self):
        return torch.tensor(self.get_window_sigmas(), device=self.device)

    @property
    def sigmas_min(self):
        return self.sigmas[-1].item()

    @property
    def sigmas_max(self):
        return self.sigmas[0].item()

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
            self.cur_timestep = torch.randint(0, self.num_train_timesteps - self.window_size + 1, (1,), generator=generator).item()

    def get_window_timesteps(self) -> list[int]:
        return self._timesteps[self.cur_timestep:min(self.cur_timestep + self.window_size, self.right_boundary)].tolist()

    def get_window_sigmas(self) -> list[float]:
        return self._sigmas[self.cur_timestep:min(self.cur_timestep + self.window_size, self.right_boundary)].tolist()

    def is_training_complete(self):
        if self.cur_iter_in_group >= self.iters_per_group:
            return True

        return False