"""
evaluate.py
-----------
Runs evaluation over the MVTec test split using the unified evaluation layer:
  - Image-level AUROC & Average Precision (AP)
  - Pixel-level AUROC & AUPRO (up to FPR 0.3)
  - Zero-leakage thresholding (validation-split 99th percentile)
  - Oracle Youden test-set threshold for historical comparison
  - Per-defect-type detection rates
  - Visual heatmap overlays

Outputs are saved under outputs/results/cae/<category>/ and synced to
outputs/results/<category>/ for backward compatibility.
"""

import json
import shutil
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch

from src.model import build_model
from src.losses import SSIMLoss
from src.utils import get_device, compute_anomaly_maps
from src.evaluator import evaluate_model


def _load_model(cfg: dict, category: str, device):
    ckpt_path = Path(cfg["train"]["checkpoint_dir"]) / f"{category}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}. Run training first: "
            f"python main.py train --category {category}"
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[{category}] loaded checkpoint from epoch {ckpt['epoch']} (loss {ckpt['loss']:.4f})")
    return model


@torch.no_grad()
def evaluate_category(cfg: dict, category: str) -> Tuple[dict, pd.DataFrame]:
    device = get_device(cfg["train"]["device"])
    d_cfg = cfg["data"]

    model = _load_model(cfg, category, device)
    ssim_module = SSIMLoss(channels=d_cfg["channels"]).to(device)

    def predict_fn(imgs: torch.Tensor):
        recon = model(imgs)
        amaps = compute_anomaly_maps(recon, imgs, ssim_module=ssim_module)
        return {"anomaly_maps": amaps, "reconstructions": recon}

    # Run shared evaluation layer
    summary = evaluate_model(
        model_name="cae",
        category=category,
        cfg=cfg,
        predict_fn=predict_fn,
        device=device,
    )

    # Maintain backward compatibility with legacy output paths (outputs/results/<category>/)
    results_base = Path(cfg.get("evaluate", {}).get("results_dir", "./outputs/results"))
    cae_dir = results_base / "cae" / category
    legacy_dir = results_base / category
    legacy_dir.mkdir(parents=True, exist_ok=True)

    # Sync summary_metrics.json with legacy aliases (e.g. decision_threshold)
    legacy_summary = dict(summary)
    legacy_summary["decision_threshold"] = summary["validation_threshold"]
    with open(legacy_dir / "summary_metrics.json", "w") as f:
        json.dump(legacy_summary, f, indent=2)

    for fname in ["per_image_scores.csv", "per_defect_type_breakdown.csv", "roc_and_score_distribution.png"]:
        src_file = cae_dir / fname
        if src_file.exists():
            shutil.copy2(src_file, legacy_dir / fname)

    # Sync heatmaps
    legacy_heatmap_dir = legacy_dir / "heatmaps"
    legacy_heatmap_dir.mkdir(parents=True, exist_ok=True)
    for p in (cae_dir / "heatmaps").glob("*.png"):
        shutil.copy2(p, legacy_heatmap_dir / p.name)

    per_type_df = pd.read_csv(cae_dir / "per_defect_type_breakdown.csv")
    return summary, per_type_df
