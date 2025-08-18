import os
import json
from typing import List, Tuple, Union
import os
import re
from io import BytesIO
import base64
import logging

import torch
import numpy as np
from openai import OpenAI
from PIL import Image

# VLLM log filter
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)


def pil_image_to_base64(image, format="JPEG"):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_qwen = f"data:image/{format.lower()};base64,{encoded_image_text}"
    return base64_qwen

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



class ConsistencyScorer:
    def __init__(self, api_key='dummy_key', base_url='http://127.0.0.1:8000/v1', model_name='QwenVL2.5-7B-Instruct'):
        self.openai_api_key = api_key
        self.openai_base_url = base_url
        self.model_name = model_name

        self.client = OpenAI(
            api_key=self.openai_api_key,
            base_url=self.openai_base_url
        )

        with open(criteria_path, 'r') as f:
            self.criteria_data = json.load(f)


    @torch.no_grad()
    def __call__(self, images : list[Image.Image], prompts : list[str]) -> list[float]:
        assert len(prompts) == len(images), "Length of prompts and images must match"

        dimension_scores = {
            "Style": {"scores": [], "criteria": []},
            "Identity": {"scores": [], "criteria": []},
            "Logic": {"scores": [], "criteria": []}
        }
        for dimension in ["Style", "Identity", "Logic"]:
            # Get criteria for this dimension
            dimension_criteria = case_criteria[dimension][0]  # Get the first (and only) dictionary in the list
            dimension_scores[dimension]["criteria"] = list(dimension_criteria.values())

            for prompt, image in zip(prompts, images):
                grid_info = extract_grid_info(prompt)
                sub_images = divide_image(image, grid_info)

                # Compute each pair of neighbors
                for i in range(len(sub_images) - 1):
                    for j in range(i + 1, len(sub_images)):
                        img1 = sub_images[i]
                        img2 = sub_images[j]
                        
                        score = self.compute_image_consistency(img1, img2, dimension_criteria)
                        

        return scores

    def compute_image_consistency(
            self,
            image1 : Image.Image,
            image2 : Image.Image,
            criterion_text: str,
            top_logprobs: int = 5
        ) -> float:
        """
        Compute the consistency score between two images.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": pil_image_to_base64(image1)}},
                    {"type": "image_url", "image_url": {"url": pil_image_to_base64(image2)}},
                    {"type": "text", "text": f"Do images meet the following criteria? {criterion_text} Please answer Yes or No."},
                ],
            }
        ]

        # TODO: finish it
        completion = self.client.chat.completions.create(
            model_name=self.model_name,
            messages=messages,
            temperature=0.0, # Deterministic result,
            max_completion_tokens=1,
            logprobs=True,
            top_logprobs=top_logprobs
        )
        log_probs = completion.choices[0].logprobs
        if log_probs:
            token_probs = {t.token.lower(): float(np.exp(t.logprob)) for t in log_probs.content[0].top_logprobs}
            score = token_probs.get('yes', 0.0) # Other method to measure score?
        else:
            # log_prob cannot be derived here. How to calculate?
            # TODO
            score = 0.0

        return score