"""
draem.py
--------
Implementation of DRAEM (Discriminatively Trained Reconstruction Anomaly Embedding Model,
Zavrtanik et al., ICCV 2021).
Designed specifically for surface and texture defect detection (e.g. leather).

Key components:
  1. Synthetic Anomaly Generation: synthesizes pseudo-defects on-the-fly from normal samples
     using random geometric masks and perturbed texture sources.
  2. Reconstructive Subnetwork: U-Net autoencoder trained to reconstruct clean images from
     synthetically corrupted inputs.
  3. Discriminative Subnetwork: U-Net segmentation network trained to localize anomalies from
     the joint representation of corrupted image and its reconstruction.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from src.losses import SSIMLoss


def generate_synthetic_anomaly(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    img: (3, H, W) float in [0, 1] on CPU
    returns: (corrupted_img, ground_truth_mask)
    """
    _, H, W = img.shape
    mask = np.zeros((H, W), dtype=np.float32)

    # 1. Random geometric blobs / cuts / scratches
    num_components = np.random.randint(1, 4)
    for _ in range(num_components):
        kind = np.random.choice(["blob", "cut", "scratch"])
        cx, cy = np.random.randint(15, W - 15), np.random.randint(15, H - 15)
        if kind == "blob":
            rx, ry = np.random.randint(8, 35), np.random.randint(8, 35)
            cv2.ellipse(mask, (cx, cy), (rx, ry), np.random.randint(0, 180), 0, 360, 1.0, -1)
        elif kind == "cut":
            x2, y2 = cx + np.random.randint(-40, 40), cy + np.random.randint(-40, 40)
            thickness = np.random.randint(2, 6)
            cv2.line(mask, (cx, cy), (x2, y2), 1.0, thickness)
        else:  # scratch / poke
            pts = np.array([
                [cx, cy],
                [cx + np.random.randint(-15, 15), cy + np.random.randint(-15, 15)],
                [cx + np.random.randint(-25, 25), cy + np.random.randint(-25, 25)],
            ], np.int32)
            cv2.polylines(mask, [pts], isClosed=False, color=1.0, thickness=np.random.randint(1, 4))

    # Smooth the mask boundary
    mask = gaussian_filter(mask, sigma=np.random.uniform(0.8, 2.5))
    mask = np.clip(mask * 1.5, 0.0, 1.0)

    # 2. Anomaly texture source (shifted image patch + random noise)
    img_np = img.numpy()
    shift_y = np.random.randint(-50, 50)
    shift_x = np.random.randint(-50, 50)
    shifted = np.roll(np.roll(img_np, shift_y, axis=1), shift_x, axis=2)

    noise = np.random.uniform(0.0, 1.0, (3, H, W)).astype(np.float32)
    alpha = np.random.uniform(0.2, 0.7)
    anomaly_source = (1.0 - alpha) * shifted + alpha * noise
    anomaly_source = np.clip(anomaly_source, 0.0, 1.0)

    # 3. Alpha blend corrupted image
    mask_t = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
    anom_t = torch.from_numpy(anomaly_source)     # (3, H, W)
    corrupted = (1.0 - mask_t) * img + mask_t * anom_t

    return corrupted, mask_t


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Compact U-Net architecture for reconstruction and segmentation."""

    def __init__(self, in_channels: int, out_channels: int, base_ch: int = 32):
        super().__init__()
        b = base_ch
        self.inc = DoubleConv(in_channels, b)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b, b * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b * 2, b * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b * 4, b * 8))

        self.up1 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(b * 8, b * 4)

        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(b * 4, b * 2)

        self.up3 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(b * 2, b)

        self.outc = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        u1 = self.up1(x4)
        u1 = torch.cat([u1, x3], dim=1)
        u1 = self.conv_up1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.conv_up2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, x1], dim=1)
        u3 = self.conv_up3(u3)

        return self.outc(u3)


class DRAEM(nn.Module):
    """
    Discriminatively trained Reconstruction Anomaly Embedding Model.
    """

    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Reconstructive network: R(corrupted) -> clean_recon
        self.reconstructor = UNet(in_channels=3, out_channels=3, base_ch=24)
        # Discriminative network: S(concat[corrupted, clean_recon]) -> anomaly_mask
        self.segmentor = UNet(in_channels=6, out_channels=1, base_ch=24)

        self.ssim_loss = SSIMLoss()
        self.to(self.device)

    def train_step(
        self,
        clean_batch: torch.Tensor,
        optimizer_recon: torch.optim.Optimizer,
        optimizer_seg: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Takes a batch of defect-free images, generates synthetic anomalies,
        and trains reconstructor and segmentor jointly.
        """
        B = clean_batch.shape[0]
        corrupted_list = []
        mask_list = []

        for i in range(B):
            corr, mask = generate_synthetic_anomaly(clean_batch[i].cpu())
            corrupted_list.append(corr)
            mask_list.append(mask)

        corrupted = torch.stack(corrupted_list).to(self.device)
        masks = torch.stack(mask_list).to(self.device)
        clean = clean_batch.to(self.device)

        # 1. Train Reconstructor
        recon = torch.sigmoid(self.reconstructor(corrupted))
        l2_loss = F.mse_loss(recon, clean)
        ssim = self.ssim_loss(recon, clean)
        loss_recon = l2_loss + ssim

        optimizer_recon.zero_grad()
        loss_recon.backward(retain_graph=True)
        optimizer_recon.step()

        # 2. Train Segmentor
        concat_input = torch.cat([corrupted, recon.detach()], dim=1)
        pred_mask_logits = self.segmentor(concat_input)
        pred_mask = torch.sigmoid(pred_mask_logits)

        # Combined BCE + Focal + Dice loss
        bce = F.binary_cross_entropy_with_logits(pred_mask_logits, masks)
        dice = 1.0 - (2.0 * (pred_mask * masks).sum() + 1e-5) / (pred_mask.sum() + masks.sum() + 1e-5)
        loss_seg = bce + dice

        optimizer_seg.zero_grad()
        loss_seg.backward()
        optimizer_seg.step()

        return {
            "loss_recon": float(loss_recon.item()),
            "loss_seg": float(loss_seg.item()),
            "total_loss": float((loss_recon + loss_seg).item()),
        }

    @torch.no_grad()
    def predict(self, imgs: torch.Tensor) -> Dict[str, Any]:
        """
        Inference step: reconstructs input and predicts segmentation mask.
        """
        self.eval()
        imgs = imgs.to(self.device)
        recon = torch.sigmoid(self.reconstructor(imgs))
        concat = torch.cat([imgs, recon], dim=1)
        mask_logits = self.segmentor(concat)
        anomaly_map = torch.sigmoid(mask_logits)  # (B, 1, H, W)

        return {
            "anomaly_maps": anomaly_map.squeeze(1).cpu().numpy(),
            "reconstructions": recon.cpu(),
        }

    def save_checkpoint(self, path: str, extra_meta: Optional[dict] = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "reconstructor_state": self.reconstructor.state_dict(),
            "segmentor_state": self.segmentor.state_dict(),
            "meta": extra_meta or {},
        }, path)
        print(f"[DRAEM] Checkpoint saved to: {path}")

    def load_checkpoint(self, path: str) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")
        data = torch.load(path, map_location=self.device)
        self.reconstructor.load_state_dict(data["reconstructor_state"])
        self.segmentor.load_state_dict(data["segmentor_state"])
        self.eval()
        print(f"[DRAEM] Checkpoint loaded from: {path}")
