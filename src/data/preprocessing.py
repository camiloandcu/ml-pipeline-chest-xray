from dataclasses import dataclass
from typing import Tuple

import torch
from torchvision import transforms
from PIL import Image


@dataclass(frozen=True)
class NormalizationConfig:
    mean: float
    std: float


@dataclass(frozen=True)
class PreprocessingConfig:
    image_size: Tuple[int, int]
    normalization: NormalizationConfig
    augment: bool


class Preprocessor:
    """
    Deterministic preprocessing pipeline for chest X-ray images. 
    Normalization statistics are fixed and computed on training data to avoid data leakage.
    """
    def __init__(self, config: PreprocessingConfig):
        self.transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[config.normalization.mean],
                std=[config.normalization.std],
            ),
        ])

    def __call__(self, image_path: str):
        img = Image.open(image_path).convert("L")
        return self.transform(img)
