from typing import Tuple

import torch
from torchvision import transforms

class Augmentor:
    def __init__(self, image_size: Tuple[int, int]):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
        ])

    def __call__(self, img):
        return self.transform(img)
