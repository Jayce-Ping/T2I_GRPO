import os
from typing import Union, List
from PIL import Image
import torch

class RewardModel():
    """
    Base class for reward models that evaluate image-text pairs.
    """
    def __init__(self, device : Union[str, int]):
        """
        Initializes the base reward model.

        Args:
            device (str or torch.device): The device to load the model on (e.g., 0, 'cuda:0', 'cpu').
        """
        self.device = device
        self.model = None
        self.build_reward_model()

    def build_reward_model(self):
        """
        Abstract method to build and load the reward model.
        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the build_reward_model method.")
    
    @torch.no_grad()
    def __call__(self, images: Union[Image.Image, List[Image.Image]], texts: Union[str, List[str]]) -> List[float]:
        """
        Abstract method for calculating rewards for image-text pairs.
        This method must be implemented by subclasses.

        Args:
            images (Union[Image.Image, List[Image.Image]]): A single image or a list of images.
            texts (Union[str, List[str]]): A single text prompt or a list of text prompts.

        Returns:
            List[float]: A list of reward scores.
        """
        raise NotImplementedError("Subclasses must implement the __call__ method.")