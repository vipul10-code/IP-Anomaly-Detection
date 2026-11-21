"""
efficientad.py
--------------
Implementation of EfficientAD (Batzner et al., 2023 / "Accurate Visual Anomaly Detection at Millisecond-Level Latencies").
Architecture:
  1. Teacher: Pretrained frozen backbone extracting intermediate features.
  2. Student: Feedforward CNN trained to match Teacher representations on defect-free images.
  3. Feature Autoencoder: Trained to compress and reconstruct student features to capture structural anomalies.
  4. Scoring: Multi-scale anomaly map combining Student-Teacher distance and Student-Autoencoder distance.
"""

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class TeacherNetwork(nn.Module):
    """Frozen pretrained teacher network based on WideResNet-50."""

    def __init__(self, out_channels: int = 128):
        super().__init__()
        weights = models.Wide_ResNet50_2_Weights.DEFAULT
        bb = models.wide_resnet50_2(weights=weights)
        self.conv1 = bb.conv1
        self.bn1 = bb.bn1
        self.relu = bb.relu
        self.maxpool = bb.maxpool
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        self.proj = nn.Conv2d(512, out_channels, kernel_size=1)

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        return self.proj(x)


class StudentNetwork(nn.Module):
    """Trainable student network matching teacher features."""

    def __init__(self, in_channels: int = 3, out_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureAutoencoder(nn.Module):
    """Autoencoder operating in the feature domain to detect logical/structural defects."""

    def __init__(self, in_channels: int = 128, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class EfficientAD(nn.Module):
    """
    Combined EfficientAD model containing Teacher, Student, and Feature Autoencoder.
    """

    def __init__(self, feature_dim: int = 128, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.feature_dim = feature_dim

        self.teacher = TeacherNetwork(out_channels=feature_dim).to(self.device)
        self.student = StudentNetwork(out_channels=feature_dim).to(self.device)
        self.autoencoder = FeatureAutoencoder(in_channels=feature_dim).to(self.device)

        self.to(self.device)

    def train_step(self, x: torch.Tensor, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        self.student.train()
        self.autoencoder.train()
        x = x.to(self.device)

        with torch.no_grad():
            f_t = self.teacher(x)

        f_s = self.student(x)
        f_ae = self.autoencoder(f_s.detach())

        loss_st = F.mse_loss(f_s, f_t)
        loss_ae = F.mse_loss(f_ae, f_s.detach())
        total_loss = loss_st + loss_ae

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return {
            "total_loss": float(total_loss.item()),
            "loss_st": float(loss_st.item()),
            "loss_ae": float(loss_ae.item()),
        }

    @torch.no_grad()
    def predict(self, imgs: torch.Tensor) -> Dict[str, Any]:
        """
        Inference step: computes combined Student-Teacher and Student-Autoencoder anomaly maps.
        """
        self.eval()
        imgs = imgs.to(self.device)
        B, _, H_orig, W_orig = imgs.shape

        f_t = self.teacher(imgs)
        f_s = self.student(imgs)
        f_ae = self.autoencoder(f_s)

        # Distance maps (mean squared channel difference)
        d_st = ((f_t - f_s) ** 2).mean(dim=1, keepdim=True)  # (B, 1, 32, 32)
        d_ae = ((f_s - f_ae) ** 2).mean(dim=1, keepdim=True)  # (B, 1, 32, 32)

        # Combine local texture distance (ST) + structural distance (AE)
        combined_map = d_st + 0.5 * d_ae

        # Bilinear upsample to input resolution
        anomaly_maps = F.interpolate(combined_map, size=(H_orig, W_orig), mode="bilinear", align_corners=False)
        anomaly_maps = anomaly_maps.squeeze(1).cpu().numpy()  # (B, H_orig, W_orig)

        return {"anomaly_maps": anomaly_maps}

    def save_checkpoint(self, path: str, extra_meta: Optional[dict] = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "student_state": self.student.state_dict(),
            "autoencoder_state": self.autoencoder.state_dict(),
            "meta": extra_meta or {},
        }, path)
        print(f"[EfficientAD] Checkpoint saved to: {path}")

    def load_checkpoint(self, path: str) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")
        data = torch.load(path, map_location=self.device)
        self.student.load_state_dict(data["student_state"])
        self.autoencoder.load_state_dict(data["autoencoder_state"])
        self.eval()
        print(f"[EfficientAD] Checkpoint loaded from: {path}")
