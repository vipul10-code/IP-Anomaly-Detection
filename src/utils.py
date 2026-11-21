"""
utils.py
--------
Shared helpers: reproducibility, anomaly-map computation, heatmap overlay
rendering, and threshold selection.
"""

import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_curve


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def compute_anomaly_maps(recon: torch.Tensor, target: torch.Tensor, ssim_module=None):
    """
    Build the per-pixel anomaly map from a reconstruction.

    Combines:
      - L2 error map (mean absolute channel-wise squared error), sensitive to
        color/intensity anomalies.
      - (1 - SSIM) map from the same SSIM module used in training, sensitive
        to structural/textural anomalies (cracks, cuts, scratches).

    Returns: anomaly_map (B,1,H,W) in [0, ~1], already min-max normalized
    per-image for visualization; also returns the raw (unnormalized) map for
    scoring, which should NOT be per-image normalized when comparing images.
    """
    l2_map = ((recon - target) ** 2).mean(dim=1, keepdim=True)  # B,1,H,W

    if ssim_module is not None:
        _ = ssim_module(recon, target)  # populates ssim_module.last_ssim_map
        ssim_err_map = 1.0 - ssim_module.last_ssim_map
        raw_map = 0.5 * l2_map + 0.5 * ssim_err_map
    else:
        raw_map = l2_map

    return raw_map  # B,1,H,W, unnormalized (comparable across images)


def image_anomaly_score(anomaly_map: np.ndarray, top_k_frac: float = 0.01) -> float:
    """Aggregate a pixel-level map into a single image-level anomaly score
    using the mean of the top-k% highest-error pixels (more robust than max,
    more sensitive than plain mean, for small localized defects)."""
    flat = anomaly_map.flatten()
    k = max(1, int(len(flat) * top_k_frac))
    top_vals = np.partition(flat, -k)[-k:]
    return float(top_vals.mean())


def normalize_map(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float32)
    lo, hi = m.min(), m.max()
    if hi - lo < 1e-8:
        return np.zeros_like(m)
    return (m - lo) / (hi - lo)


def overlay_heatmap(image_rgb_uint8: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.45,
                     colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """image_rgb_uint8: HxWx3 uint8 RGB. anomaly_map: HxW float, any range (gets normalized)."""
    norm_map = (normalize_map(anomaly_map) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(norm_map, colormap)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb_uint8, 1 - alpha, heat, alpha, 0)
    return overlay


def find_best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Youden's J statistic (max of TPR - FPR) over the ROC curve to pick an
    operating threshold for the binary anomaly/normal decision."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return float(thresholds[best_idx])


def save_image(path: str, img_rgb_uint8: np.ndarray):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR))


def to_uint8_img(tensor_chw: torch.Tensor) -> np.ndarray:
    """CHW float [0,1] torch tensor -> HWC uint8 numpy RGB image."""
    arr = tensor_chw.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr
