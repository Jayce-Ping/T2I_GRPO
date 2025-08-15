# The code is modified from Flow_GRPO https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/ocr.py
from paddleocr import PaddleOCR
import torch
import numpy as np
from Levenshtein import distance
from typing import List, Union
from PIL import Image
import re
from fastvideo.reward.reward_model import RewardModel

class OcrRewardModel(RewardModel):
    def __init__(self, device: str):
        """
        OCR reward calculator
        device : str - device to load PaddleOCR model
        """
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            device=f'gpu:{device_id}' if torch.cuda.is_available() else 'cpu'
        )

    @torch.no_grad()
    def __call__(self, 
                images: Union[List[Image.Image], List[np.ndarray]], 
                prompts: List[str]):
        """
        Calculate OCR reward
        :param images: List of input images (PIL or numpy format)
        :param prompts: Corresponding target text list
        :return: Reward tensor (CPU)
        """
        prompts = [prompt.split('"')[1] for prompt in prompts]
        # match_content = re.compile(r"\"(.*?)\"", re.DOTALL)
        # prompts = [match_content.findall(p)[0] if match_content.findall(p) else p for p in prompts]
        rewards = []
        # Ensure input lengths are consistent
        assert len(images) == len(prompts), "Images and prompts must have the same length"
        for img, prompt in zip(images, prompts):
            # Convert image format
            if isinstance(img, Image.Image):
                img = np.array(img)
            
            try:
                # OCR recognition
                result = self.ocr.ocr(img)
                result = result[0]['rec_texts']
                # Extract recognized text (handle possible multi-line results)
                recognized_text = ' '.join([text for text in result if text])
                
                # Canonicalize two texts for comparison
                recognized_text = recognized_text.replace(' ', '').lower()
                prompt = prompt.replace(' ', '').lower()
                if prompt in recognized_text:
                    dist = 0
                else:
                    dist = distance(recognized_text, prompt)
                # Recognized many unrelated characters, only add one character penalty
                if dist > len(prompt):
                    dist = len(prompt)
                
            except Exception as e:
                # Error handling (e.g., OCR parsing failure)
                print(f"OCR processing failed: {str(e)}")
                dist = len(prompt)  # Maximum penalty
            reward = 1 - dist / (len(prompt))
            rewards.append(reward)

        return rewards

if __name__ == "__main__":
    example_image_path = "media_images_eval_images_499_ef42de47b8ec98892954.jpg"
    example_image = Image.open(example_image_path)
    example_prompt = 'New York Skyline with "Hello World" written with fireworks on the sky'
    # Instantiate scorer
    scorer = OcrScorer(use_gpu=False)

    # Call scorer and print result
    reward = scorer([example_image], [example_prompt])
    print(f"OCR Reward: {reward}")