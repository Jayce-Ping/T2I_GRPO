import os
import re
import json
from typing import List, Tuple, Union
from io import BytesIO
import base64
import logging
import asyncio
from itertools import combinations

import torch
import numpy as np
import openai
from openai import OpenAI, AsyncOpenAI
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


def get_score_from_completion(completion : openai.ChatCompletion) -> float:
    logprobs = completion.choices[0].logprobs
    if logprobs:
        token_probs = {t.token.lower(): float(np.exp(t.logprob)) for t in logprobs.content[0].top_logprobs}
        score = token_probs.get('yes', 0.0) # Other method to measure score?
    else:
        # log_prob cannot be derived here. How to calculate?
        # TODO
        score = 0.0

    return score


class ConsistencyScorer:
    def __init__(
            self,
            api_key='dummy_key',
            base_url='http://127.0.0.1:8000/v1',
            model_name='QwenVL2.5-7B-Instruct',
            criteria_path='prompt_consistency_criterion.json',
            async_mode=True
        ):
        self.openai_api_key = api_key
        self.openai_base_url = base_url
        self.model_name = model_name
        self.async_model = async_mode


        if async_mode:
            self.client = AsyncOpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url
            )
        else:
            self.client = OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url
            )


        with open(criteria_path, 'r') as f:
            self.criteria_data = json.load(f)


    @torch.no_grad()
    def __call__(self, images : list[Image.Image], prompts : list[str], metadatas : list[dict]) -> list[float]:
        assert len(prompts) == len(images), "Length of prompts and images must match"

        final_scores = []
        for prompt, image, metadata in zip(prompts, images, metadatas):
            criteria_info = self.criteria_data[metadata['idx']]

            dimensions = criteria_info.keys()
            dimension_scores = {k:0.0 for k in dimensions}
            # Compute scores for each prompt-image pair from different dimensions
            for dimension in dimensions:
                # Get criteria for this dimension
                dimension_criteria = criteria_info[dimension][0]  # Get the first (and only) dictionary in the list
                criteria_texts = list(dimension_criteria.values())

                # [criteria1_scores : list[float], criteria2_scores : list[float], ...]
                criterion_scores = [self.compute_image_consistency(prompt, image, ct) for ct in criteria_texts]

                # Compute the average score within each criterion
                # [criteria1_avg_score, criteria2_avg_score, ...]
                criterion_scores = [sum(scores) / len(scores) if scores else 0.0 for scores in criterion_scores]

                # Compute the overall score for this dimension
                overall_score = sum(criterion_scores) / len(criterion_scores) if criterion_scores else 0.0
                dimension_scores[dimension] = overall_score

            # Compute average scores from each dimension
            final_scores.append(sum(dimension_scores.values()) / len(dimension_scores))

        return final_scores
    

    def compute_image_consistency(
            self,
            prompt : str,
            image : Image.Image,
            criteria_text : str,
            top_logprobs: int = 5
        ):
        if self.async_model:
            return asyncio.run(self._async_compute_image_consistency(prompt, image, criteria_text, top_logprobs))
        else:
            return self._sync_compute_image_consistency(prompt, image, criteria_text, top_logprobs)

    async def _async_compute_image_consistency(
            self,
            prompt : str,
            image : Image.Image,
            criteria_text : str,
            top_logprobs: int = 5
        ):
        """
        Async version of compute_image_consistency.
        """
        completions = []
        grid_info = extract_grid_info(prompt)
        sub_images = divide_image(image, grid_info)
        for image1, image2 in combinations(sub_images, 2):
            messages = [
                {
                    "role": "user",
                    "content":
                    [
                        {"type": "image_url", "image_url": {"url": pil_image_to_base64(image1)}},
                        {"type": "image_url", "image_url": {"url": pil_image_to_base64(image2)}},
                        {"type": "text", "text": f"Do images meet the following criteria? {criteria_text} Please answer Yes or No."},
                    ]
                }
            ]

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0, # Deterministic result,
                max_completion_tokens=1,
                logprobs=True,
                top_logprobs=top_logprobs
            )

            completions.append(completion)


        res = await asyncio.gather(*completions)

        return [get_score_from_completion(c) for c in res]


    def _sync_compute_image_consistency(
            self,
            prompt : str,
            image : Image.Image,
            criteria_text : str,
            top_logprobs: int = 5
        ) -> list[float]:
        """
        Compute the consistency score of a image, for a given criterion.
        """
        completions = []
        grid_info = extract_grid_info(prompt)
        sub_images = divide_image(image, grid_info)
        for image1, image2 in combinations(sub_images, 2):
            messages = [
                {
                    "role": "user",
                    "content":
                    [
                        {"type": "image_url", "image_url": {"url": pil_image_to_base64(image1)}},
                        {"type": "image_url", "image_url": {"url": pil_image_to_base64(image2)}},
                        {"type": "text", "text": f"Do images meet the following criteria? {criteria_text} Please answer Yes or No."},
                    ]
                }
            ]

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0, # Deterministic result,
                max_completion_tokens=1,
                logprobs=True,
                top_logprobs=top_logprobs
            )

            completions.append(completion)


        return [get_score_from_completion(c) for c in completions]