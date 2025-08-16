import torch
from diffusers.schedulers.scheduling_utils import FlowMatchEulerDiscreteScheduler

class FlowMatchSlidingWindowScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(
        self,
        window_size: int,
        iters_per_group: int = 25,
        sample_strategy: str = "progressive",
        prog_overlap: bool = False,
        prog_overlap_step: int = 1,
        max_iters_per_group: int = None,
        min_iters_per_group: int = None,
        roll_back: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.iters_per_group = iters_per_group
        self.sample_strategy = sample_strategy
        self.prog_overlap = prog_overlap
        self.prog_overlap_step = prog_overlap_step
        self.max_iters_per_group = max_iters_per_group or iters_per_group
        self.min_iters_per_group = min_iters_per_group or max(1, iters_per_group // 4)
        self.roll_back = roll_back

        self.cur_timestep = 0
        self.cur_iter_in_group = 0
        self.init_timestep = 0
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

    def get_dynamic_iters_per_group(self):
        return self.iters_per_group

    def get_exp_decay_iters_per_group(self):
        return self.iters_per_group

    def update_iteration(self, seed=None):
        if self.sample_strategy == "progressive":
            self.cur_iter_in_group += 1
            if self.cur_iter_in_group >= self.iters_per_group:
                self.cur_iter_in_group = 0
                if self.prog_overlap:
                    self.cur_timestep += self.prog_overlap_step
                else:
                    self.cur_timestep += self.window_size
            if self.cur_timestep > self.num_train_timesteps:
                if self.roll_back:
                    self.cur_timestep = self.init_timestep
                else:
                    self.cur_timestep = self.num_train_timesteps
        elif self.sample_strategy == "random":
            self.cur_timestep = torch.randint(0, self.num_train_timesteps - self.window_size + 1, (1,)).item()
        elif self.sample_strategy == "decay":
            self.cur_iter_in_group += 1
            current_iters = self.get_dynamic_iters_per_group()
            if self.cur_iter_in_group >= current_iters:
                self.cur_iter_in_group = 0
                if self.prog_overlap:
                    self.cur_timestep += self.prog_overlap_step
                else:
                    self.cur_timestep += self.window_size
            if self.cur_timestep > self.num_train_timesteps:
                if self.roll_back:
                    self.cur_timestep = self.init_timestep
                else:
                    self.cur_timestep = self.num_train_timesteps
        elif self.sample_strategy == "exp_decay":
            self.cur_iter_in_group += 1
            current_iters = self.get_exp_decay_iters_per_group()
            if self.cur_iter_in_group >= current_iters:
                self.cur_iter_in_group = 0
                if self.prog_overlap:
                    self.cur_timestep += self.prog_overlap_step
                else:
                    self.cur_timestep += self.window_size
            if self.cur_timestep > self.num_train_timesteps:
                if self.roll_back:
                    self.cur_timestep = self.init_timestep
                else:
                    self.cur_timestep = self.num_train_timesteps
        else:
            raise ValueError(f"Invalid sample strategy: {self.sample_strategy}")

    def get_window_timesteps(self) -> list[int]:
        return self._timesteps[self.cur_timestep:min(self.cur_timestep + self.window_size, self.num_train_timesteps)].tolist()

    def get_window_sigmas(self) -> list[float]:
        return self._sigmas[self.cur_timestep:min(self.cur_timestep + self.window_size, self.num_train_timesteps)].tolist()

    def is_training_complete(self):
        if self.sample_strategy in ["progressive", "decay"]:
            return self.cur_timestep >= self.num_train_timesteps
        return False