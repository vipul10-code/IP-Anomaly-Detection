"""
losses.py
---------
Combined MSE + SSIM reconstruction loss.

Plain per-pixel MSE tends to produce blurry reconstructions that under-penalize
small structural defects (a fine crack on leather can get "averaged away").
Adding a Structural Similarity (SSIM) term, which compares local luminance,
contrast and structure over sliding windows, pushes the network to preserve
fine texture -- which in turn makes the (1-SSIM) error map a much sharper
anomaly heatmap at test time. This combined loss is a well established choice
for autoencoder-based MVTec baselines (Bergmann et al., "Improving Unsupervised
Defect Segmentation by Applying Structural Similarity to Autoencoders", 2018).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window_2d = g.t() @ g
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


class SSIMLoss(nn.Module):
    """Differentiable 1 - SSIM, averaged over the batch. Also exposes
    `ssim_map` (last call) for building the pixel-level anomaly heatmap."""

    def __init__(self, window_size: int = 11, channels: int = 3, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.sigma = sigma
        self.register_buffer("window", torch.zeros(1))
        self._built = False

    def _ensure_window(self, x):
        if not self._built or self.window.shape[0] != x.shape[1]:
            self.window = _gaussian_window(
                self.window_size, self.sigma, x.shape[1], x.device, x.dtype
            )
            self._built = True

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        self._ensure_window(img1)
        pad = self.window_size // 2
        C = img1.shape[1]

        mu1 = F.conv2d(img1, self.window, padding=pad, groups=C)
        mu2 = F.conv2d(img2, self.window, padding=pad, groups=C)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=pad, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=pad, groups=C) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=pad, groups=C) - mu1_mu2

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        self.last_ssim_map = ssim_map.mean(dim=1, keepdim=True)  # B,1,H,W
        return 1.0 - ssim_map.mean()


class ReconstructionLoss(nn.Module):
    def __init__(self, ssim_weight: float = 0.6, channels: int = 3):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.ssim = SSIMLoss(channels=channels)
        self.mse = nn.MSELoss()

    def forward(self, recon: torch.Tensor, target: torch.Tensor):
        ssim_loss = self.ssim(recon, target)
        mse_loss = self.mse(recon, target)
        total = self.ssim_weight * ssim_loss + (1 - self.ssim_weight) * mse_loss
        return total, {"ssim_loss": ssim_loss.item(), "mse_loss": mse_loss.item()}
