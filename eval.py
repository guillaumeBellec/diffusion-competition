"""
Evaluation metrics for CIFAR-10 image generation.

Two FID variants:
  - InceptionFID: Standard FID using InceptionV3 (2048-dim, resizes to 299x299). Slow on CPU.
  - ResNetFID: Lightweight FID using ResNet18 (512-dim, native 32x32). Fast on CPU.
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from scipy.linalg import sqrtm


def frechet_distance(mu1, sigma1, mu2, sigma2):
    """Compute Fréchet distance between two multivariate Gaussians."""
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    # Numerical stability: discard imaginary part from sqrtm
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


class ResNetFID:
    """Lightweight FID using ResNet18 features (512-dim, works at 32x32 natively)."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove the final FC layer, keep adaptive avg pool -> 512-dim
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model.eval().to(self.device)
        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def extract_features(self, images_uint8):
        """Extract features from uint8 images (N, 3, 32, 32)."""
        x = images_uint8.to(self.device).float() / 255.0
        x = (x - self.mean) / self.std
        return self.model(x).squeeze(-1).squeeze(-1)  # (N, 512)

    def compute_stats(self, features):
        """Compute mean and covariance from features."""
        features = features.cpu().numpy()
        return features.mean(axis=0), np.cov(features, rowvar=False)

    def compute_fid(self, real_images, fake_images, batch_size=64):
        """Compute FID between real and fake uint8 image tensors."""
        real_feats, fake_feats = [], []
        for i in range(0, len(real_images), batch_size):
            real_feats.append(self.extract_features(real_images[i:i+batch_size]))
        for i in range(0, len(fake_images), batch_size):
            fake_feats.append(self.extract_features(fake_images[i:i+batch_size]))

        mu1, sigma1 = self.compute_stats(torch.cat(real_feats))
        mu2, sigma2 = self.compute_stats(torch.cat(fake_feats))
        return frechet_distance(mu1, sigma1, mu2, sigma2)


class InceptionFID:
    """Standard FID using torchmetrics InceptionV3 (2048-dim, resizes to 299x299)."""

    def __init__(self, device="cpu"):
        from torchmetrics.image.fid import FrechetInceptionDistance
        self.device = torch.device(device)
        self.fid = FrechetInceptionDistance(feature=2048, normalize=False).to(self.device)

    def compute_fid(self, real_images, fake_images, batch_size=16):
        """Compute FID between real and fake uint8 image tensors."""
        self.fid.reset()
        for i in range(0, len(real_images), batch_size):
            self.fid.update(real_images[i:i+batch_size].to(self.device), real=True)
        for i in range(0, len(fake_images), batch_size):
            self.fid.update(fake_images[i:i+batch_size].to(self.device), real=False)
        return self.fid.compute().item()
