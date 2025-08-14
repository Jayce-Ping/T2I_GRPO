import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import GenerationConfig
from PIL import Image
import torch.nn.functional as F
from typing import List, Tuple, Union
import os
import re
from io import BytesIO
import base64

def PIL_image_to_base64(image: Image.Image, format='JPEG') -> str:
    buffered = BytesIO()

    image.save(buffered, format=format)

    img_bytes = buffered.getvalue()

    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    return img_base64


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



def load_model(model_path):
    # Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    
    # default processer
    processor = AutoProcessor.from_pretrained(model_path)
    
    return model, processor



class ConsistencyScore:
    def __init__(self, qwen_model_path, criteria_path, device):
        self.model_path = qwen_model_path
        self.device = device
        self.model, self.processor = load_model(self.model_path)

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

                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": f"data:image;base64,{PIL_image_to_base64(img1)}", "resized_height": 512, "resized_width": 512},
                                    {"type": "image", "image": f"data:image;base64,{PIL_image_to_base64(img2)}", "resized_height": 512, "resized_width": 512},
                                    {"type": "text", "text": f"Do images meet the following criteria? {criterion_text} Please answer Yes or No."},
                                ],
                            }
                        ]

                        # Prepare for inference
                        text = self.processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        image_inputs, video_inputs = process_vision_info(messages)
                        inputs = self.processor(
                            text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        )
                        inputs = inputs.to("cuda")

                        generated_ids = self.model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True, output_logits=True)
                        logits = generated_ids.logits

                        no_logits = logits[0][0][2753]
                        yes_logits = logits[0][0][9454]

                        # Calculate softmax
                        logits = torch.tensor([no_logits, yes_logits])
                        softmax = torch.nn.functional.softmax(logits, dim=0)
                        yes_softmax = softmax[1].item()

                        dimension_scores[dimension]["scores"]



        return scores