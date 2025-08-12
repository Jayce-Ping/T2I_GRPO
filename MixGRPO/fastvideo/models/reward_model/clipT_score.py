import torch
import clip
from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image
import torch.nn.functional as F
from typing import List, Tuple, Union
import os
import re


def divide_image(image, grid_info : tuple[int, int]):
    assert len(grid_info) == 2, "grid_info must be a tuple of two integers (a, b)"

    a, b = grid_info
    width, height = image.size

    grid_cells = []
    cell_width = width // a
    cell_height = height // b

    for i in range(a):
        for j in range(b):
            left = i * cell_width
            upper = j * cell_height
            right = left + cell_width
            lower = upper + cell_height
            grid_cells.append(image.crop((left, upper, right, lower)))

    return grid_cells

def extract_grid_info(prompt) -> tuple[int, int]:
    # Grid can be represented as int x int, or int ⨉ int. ⨉ has unicode \u2a09
    match = re.findall(r'(\d+)\s*[x⨉]\s*(\d+)', prompt)
    if len(match) == 0:
        return (1, 1)

    return (int(match[0][0]), int(match[0][1]))

@torch.no_grad()
def calculate_clip_score(prompts, images, clip_model, device):
    """
        Assume we have images=[im_1, im_2, ..., im_s] where s = m x n. (m, n) is the grid size.   
        Denote s_{i,j} := sim(im_i, im_j) is the clip similarity between image i,j.
        score = min(s_{i,i+1}) for i in m x n.

    """

    # texts = clip.tokenize(prompts, truncate=True).to(device=device)
    
    scores = []
    for prompt, image in zip(prompts, images):
        grid_info = extract_grid_info(prompt)
        sub_images = divide_image(image, grid_info)

        if len(sub_images) == 1:
            scores.append(1.0)
            continue

        image_proc = clip_model.preprocess(sub_images)

        image_features = clip_model.model.encode_image(image_proc)
        image_features = F.normalize(image_features, dim=-1)

        clip_score = min(image_features[:-1] @ image_features[1:].T)

        scores.append(clip_score.item())

    return scores


class CLIPTScoreRewardModel():
    def __init__(self, clip_model_path, device, http_proxy=None, https_proxy=None, clip_model_type='ViT-H-14'):
        super().__init__()
        if http_proxy:
            os.environ["http_proxy"] = http_proxy
        if https_proxy:
            os.environ["https_proxy"] = https_proxy
        self.clip_model_path = clip_model_path
        self.clip_model_type = clip_model_type
        self.device = device
        self.load_model()

    def load_model(self, logger=None):
        self.model, self.preprocess = create_model_from_pretrained(self.clip_model_path)
        self.tokenizer = get_tokenizer(self.clip_model_type)
        self.model.to(self.device)

    # calculate clip score directly, such as for rerank
    @torch.no_grad()
    def __call__(
        self, 
        prompts: Union[str, List[str]], 
        images: List[Image.Image]
    ) -> List[float]:
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        if len(prompts) != len(images):
            raise ValueError("prompts must have the same length as images")
        
        
        scores = []
        for prompt, image in zip(prompts, images):
            grid_info = extract_grid_info(prompt)
            sub_images = divide_image(image, grid_info)
            
            if len(sub_images) == 1:
                scores.append(1.0)
                continue

            image_proc = self.preprocess(sub_images)

            image_features = self.model.encode_image(image_proc)

            clip_score = min(F.cosine_similarity(image_features[:-1], image_features[1:]))

            scores.append(clip_score.item())

        return scores