"""
infer.py
--------
Run the trained autoencoder on ONE new image (not part of the MVTec test
set) and report whether it looks anomalous, plus save a heatmap panel.

Usage:
    python main.py predict --category leather --image path/to/your_photo.jpg

What it does:
  1. Loads the checkpoint trained for that category (outputs/checkpoints/<category>_best.pt).
  2. Preprocesses the image the same way as training/evaluation (resize, normalize).
  3. Reconstructs it, builds the MSE+SSIM anomaly map (src/utils.compute_anomaly_maps).
  4. Aggregates it into a single image-level anomaly score (top-1% pixel mean).
  5. Compares that score against the decision threshold saved during evaluation
     (outputs/results/<category>/summary_metrics.json -> "decision_threshold").
     If you haven't run `evaluate` yet, there is no threshold on file yet --
     the script will still print the raw score and heatmap, just without a
     normal/anomalous verdict.
  6. Saves [original | reconstruction | heatmap overlay] to
     outputs/results/<category>/predictions/<image_name>_panel.png
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from scipy.ndimage import gaussian_filter
from src.evaluator import compute_image_score
from src.model import build_model
from src.losses import SSIMLoss
from src.utils import (
    get_device,
    compute_anomaly_maps,
    overlay_heatmap,
    to_uint8_img,
    save_image,
)


def _load_model(cfg: dict, category: str, device):
    ckpt_path = Path(cfg["train"]["checkpoint_dir"]) / f"{category}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}. Train first: "
            f"python main.py train --category {category}"
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _load_threshold(cfg: dict, category: str):
    summary_path = Path(cfg["evaluate"]["results_dir"]) / category / "summary_metrics.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f).get("decision_threshold")


def preprocess_image(image_path: str, img_size: int) -> torch.Tensor:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
    return tensor.unsqueeze(0)  # 1,C,H,W


@torch.no_grad()
def predict_image(cfg: dict, category: str, image_path: str):
    device = get_device(cfg["train"]["device"])
    d_cfg = cfg["data"]
    e_cfg = cfg["evaluate"]

    model = _load_model(cfg, category, device)
    ssim_module = SSIMLoss(channels=d_cfg["channels"]).to(device)

    img_tensor = preprocess_image(image_path, d_cfg["img_size"]).to(device)
    recon = model(img_tensor)
    anomaly_map = compute_anomaly_maps(recon, img_tensor, ssim_module=ssim_module)
    amap_np = anomaly_map.squeeze(0).squeeze(0).cpu().numpy()  # H,W

    sigma = float(e_cfg.get("gaussian_sigma", 4.0))
    top_k_cfg = e_cfg.get("top_k_frac", 0.01)
    top_k = top_k_cfg.get(category, 0.01) if isinstance(top_k_cfg, dict) else float(top_k_cfg)

    if sigma > 0:
        amap_np = gaussian_filter(amap_np, sigma=sigma)

    score = compute_image_score(amap_np, top_k_frac=top_k)
    threshold = _load_threshold(cfg, category)

    verdict = None
    if threshold is not None:
        verdict = "ANOMALOUS" if score >= threshold else "NORMAL"

    orig = to_uint8_img(img_tensor[0])
    rec = to_uint8_img(recon[0])
    overlay = overlay_heatmap(orig, amap_np, alpha=e_cfg["overlay_alpha"])
    panel = np.concatenate([orig, rec, overlay], axis=1)

    out_dir = Path(e_cfg["results_dir"]) / category / "predictions"
    out_path = out_dir / f"{Path(image_path).stem}_panel.png"
    save_image(out_path, panel)

    print(f"\n[{category}] prediction for: {image_path}")
    print(f"  anomaly score        : {score:.5f}")
    if threshold is not None:
        print(f"  decision threshold    : {threshold:.5f}  (from last evaluate run)")
        print(f"  verdict               : {verdict}")
    else:
        print(f"  decision threshold    : not available yet -- run "
              f"'python main.py evaluate --category {category}' first for a NORMAL/ANOMALOUS verdict")
    print(f"  heatmap panel saved to: {out_path}")

    return {"score": float(score), "threshold": threshold, "verdict": verdict, "panel_path": str(out_path)}
