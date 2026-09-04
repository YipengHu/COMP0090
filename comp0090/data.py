"""Deterministic, download-free teaching datasets."""
import random
import numpy as np
import torch

def seed_everything(seed: int = 7) -> np.random.Generator:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    return np.random.default_rng(seed)

def make_shape_images(n: int = 600, size: int = 12, centred: bool = False, seed: int = 7):
    """Return noisy images containing either a horizontal or vertical bar."""
    rng = np.random.default_rng(seed)
    images = rng.normal(0, 0.12, (n, 1, size, size)).astype("float32")
    labels = rng.integers(0, 2, n)
    positions = np.full(n, size // 2) if centred else rng.integers(2, size - 2, n)
    for image, label, position in zip(images, labels, positions):
        image[0, position, 2:-2] += label == 0
        image[0, 2:-2, position] += label == 1
    return images, labels.astype("int64")
