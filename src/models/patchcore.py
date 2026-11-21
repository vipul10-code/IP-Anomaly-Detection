"""
patchcore.py
------------
Direct PyTorch implementation of PatchCore (Roth et al., CVPR 2022).
Key components:
  1. Pretrained, frozen feature backbone (WideResNet-50-2 by default).
  2. Multi-scale feature extraction from intermediate layers (layer2 + layer3).
  3. Local neighborhood aggregation via spatial average pooling.
  4. Memory bank construction with greedy minimax (k-center) coreset subsampling.
  5. kNN patch anomaly scoring with distance-based anomaly map generation.
  6. Bilinear upsampling to input resolution.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from scipy.ndimage import gaussian_filter


class PatchCore(nn.Module):
    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        coreset_sampling_ratio: float = 0.01,
        target_coreset_size: int = 1500,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.target_coreset_size = target_coreset_size
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Load backbone
        if backbone_name == "wide_resnet50_2":
            weights = models.Wide_ResNet50_2_Weights.DEFAULT
            bb = models.wide_resnet50_2(weights=weights)
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            bb = models.resnet50(weights=weights)
        elif backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            bb = models.resnet18(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.conv1 = bb.conv1
        self.bn1 = bb.bn1
        self.relu = bb.relu
        self.maxpool = bb.maxpool
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3

        # Freeze all backbone parameters
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        # Neighborhood aggregation
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # Memory bank buffer (registered as buffer so it moves with the model)
        self.register_buffer("memory_bank", torch.empty(0))
        self.is_fitted = False
        self.to(self.device)

    @torch.no_grad()
    def _extract_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts multi-scale patch features from layer2 and layer3,
        aggregates neighborhood with average pooling, and concatenates.
        Input: (B, 3, H, W)
        Output: (B, H_p, W_p, D) where D = C2 + C3 (e.g. 1536 for WideResNet50)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)  # (B, 512, H/8, W/8)
        x3 = self.layer3(x2)  # (B, 1024, H/16, W/16)

        # Neighborhood aggregation
        p2 = self.avg_pool(x2)
        p3 = self.avg_pool(x3)

        # Upsample layer3 features to match layer2 spatial resolution
        p3_upsampled = F.interpolate(p3, size=p2.shape[-2:], mode="bilinear", align_corners=False)

        # Concatenate features along channel dimension: (B, C2+C3, H_p, W_p)
        features = torch.cat([p2, p3_upsampled], dim=1)
        B, C, H_p, W_p = features.shape

        # Reshape to (B, H_p, W_p, C)
        features = features.permute(0, 2, 3, 1).contiguous()
        return features

    @torch.no_grad()
    def fit(self, dataloader, max_candidates: int = 25000, seed: int = 42) -> None:
        """
        Extracts patch features from all defect-free training images,
        and builds the coreset memory bank using greedy k-center clustering.
        """
        print(f"[PatchCore] Extracting patch embeddings from training images...")
        t0 = time.time()
        patch_list = []

        for batch in dataloader:
            if isinstance(batch, dict):
                imgs = batch["image"]
            elif isinstance(batch, (list, tuple)):
                imgs = batch[0]
            else:
                imgs = batch
            imgs = imgs.to(self.device)
            feats = self._extract_patch_features(imgs)  # (B, H_p, W_p, D)
            B, H_p, W_p, D = feats.shape
            flat_feats = feats.reshape(-1, D)
            patch_list.append(flat_feats.cpu())

        all_patches = torch.cat(patch_list, dim=0)  # (Total_Patches, D)
        total_p, feat_dim = all_patches.shape
        print(f"[PatchCore] Collected {total_p} patch vectors of dimension {feat_dim} in {time.time() - t0:.2f}s")

        # Determine target coreset size
        target_size = min(
            self.target_coreset_size,
            max(500, int(total_p * self.coreset_sampling_ratio)),
        )

        # Subsample candidates if pool is very large for fast greedy selection
        rng = np.random.RandomState(seed)
        if total_p > max_candidates:
            cand_idx = rng.choice(total_p, size=max_candidates, replace=False)
            cand_patches = all_patches[cand_idx]
        else:
            cand_patches = all_patches

        # Fast Johnson-Lindenstrauss random projection for greedy distance calculations
        proj_dim = 128
        torch.manual_seed(seed)
        proj_matrix = torch.randn(feat_dim, proj_dim) / (proj_dim ** 0.5)
        proj_patches = cand_patches @ proj_matrix

        print(f"[PatchCore] Selecting {target_size} coreset patches using minimax k-center...")
        t_core = time.time()
        
        # Greedy k-center algorithm
        selected_indices = [0]
        min_distances = torch.cdist(proj_patches, proj_patches[0:1]).squeeze(1)

        for step in range(target_size - 1):
            farthest_idx = int(torch.argmax(min_distances))
            selected_indices.append(farthest_idx)
            new_dist = torch.cdist(proj_patches, proj_patches[farthest_idx : farthest_idx + 1]).squeeze(1)
            min_distances = torch.min(min_distances, new_dist)

        coreset_vectors = cand_patches[selected_indices].to(self.device)
        self.memory_bank = coreset_vectors
        self.is_fitted = True
        print(f"[PatchCore] Memory bank created with {len(coreset_vectors)} vectors in {time.time() - t_core:.2f}s")

    @torch.no_grad()
    def predict(self, imgs: torch.Tensor) -> Dict[str, Any]:
        """
        Runs kNN inference on test batch.
        Input: (B, 3, H, W)
        Output: dict with 'anomaly_maps': (B, H, W)
        """
        if not self.is_fitted or self.memory_bank.numel() == 0:
            raise RuntimeError("PatchCore model is not fitted yet. Call fit() or load_checkpoint() first.")

        imgs = imgs.to(self.device)
        B, _, H_orig, W_orig = imgs.shape

        feats = self._extract_patch_features(imgs)  # (B, H_p, W_p, D)
        B, H_p, W_p, D = feats.shape
        flat_feats = feats.reshape(B * H_p * W_p, D)

        # Batch Euclidean distance to memory bank: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x y^T
        mb = self.memory_bank
        x2 = (flat_feats ** 2).sum(dim=1, keepdim=True)
        y2 = (mb ** 2).sum(dim=1, keepdim=True).t()
        dist_sq = x2 + y2 - 2 * (flat_feats @ mb.t())
        dist = torch.sqrt(torch.clamp(dist_sq, min=0.0))

        # Nearest neighbor distance per patch: (B * H_p * W_p,)
        min_dist, _ = torch.min(dist, dim=1)
        patch_scores = min_dist.reshape(B, 1, H_p, W_p)

        # Bilinear upsample to original resolution (B, 1, H_orig, W_orig)
        anomaly_maps = F.interpolate(patch_scores, size=(H_orig, W_orig), mode="bilinear", align_corners=False)
        anomaly_maps = anomaly_maps.squeeze(1).cpu().numpy()  # (B, H_orig, W_orig)

        return {"anomaly_maps": anomaly_maps}

    def save_checkpoint(self, path: str, extra_meta: Optional[dict] = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "backbone_name": self.backbone_name,
            "memory_bank": self.memory_bank.cpu(),
            "meta": extra_meta or {},
        }
        torch.save(data, path)
        print(f"[PatchCore] Checkpoint saved to: {path}")

    def load_checkpoint(self, path: str) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")
        data = torch.load(path, map_location=self.device)
        self.memory_bank = data["memory_bank"].to(self.device)
        self.is_fitted = True
        print(f"[PatchCore] Checkpoint loaded from: {path} (Memory bank size: {len(self.memory_bank)})")
