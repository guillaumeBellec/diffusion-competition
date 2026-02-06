"""
Example Agent for CIFAR-10 Image Generation Competition

This is a dummy agent that generates random noise images.
Students should replace this with a proper generative model (e.g., diffusion model, GAN, VAE).
"""

import numpy as np
import torch
from torch import nn


class Agent:
    def __init__(self):
        """
        Initialize your generative model here.
        Load pretrained weights, set up the model architecture, etc.
        """
        pass

    def generate(self, class_ids: np.ndarray) -> np.ndarray:
        """
        Generate CIFAR-10 images for the requested classes.

        Args:
            class_ids: Array of shape (n_images,) with dtype np.int32.
                       Each value is in [0, 9] representing CIFAR-10 classes:
                       0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer,
                       5: dog, 6: frog, 7: horse, 8: ship, 9: truck

        Returns:
            images: Array of shape (n_images, 3, 32, 32) with dtype np.uint8.
                    Pixel values should be in [0, 255].
        """
        n_images = len(class_ids)

        # Dummy implementation: random noise
        # TODO: Replace with your generative model
        images = torch.rand(n_images, 3, 32, 32)
        #
        images = nn.functional.avg_pool2d(images, kernel_size=3, stride=1, padding=1)
        images = nn.functional.avg_pool2d(images, kernel_size=3, stride=1, padding=1)
        images = (images * 256).int().cpu().numpy()

        return images
