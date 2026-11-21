"""
evaluator.py
------------
Shared evaluation layer for all anomaly detection models (CAE, PatchCore, EfficientAD, DRAEM).
Guarantees identical test set, identical metrics, identical validation threshold protocol,
and standardized output artifact structure across all models:
  outputs/results/<model_name>/<category>/
    ├── summary_metrics.json
    ├── per_image_scores.csv
    ├── per_defect_type_breakdown.csv
    ├── roc_and_score_distribution.png
    └── heatmaps/*.png
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, Optional, Callable, List

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from src.dataset import MVTecDataset, test_collate, scan_split, _load_image
from src.metrics import compute_all_metrics, find_youden_threshold
from src.utils import to_uint8_img, overlay_heatmap, save_image


def get_val_split(root_dir: str, category: str, val_ratio: float = 0.2, seed: int = 42) -> List[str]:
    """Deterministically hold out ~20% of train/good images for validation thresholding."""
    train_samples = scan_split(root_dir, category, "train")
    train_good = [s.path for s in train_samples if s.label == 0]
    train_good.sort()
    
    rng = np.random.RandomState(seed)
    n_val = max(1, int(len(train_good) * val_ratio))
    val_indices = rng.choice(len(train_good), size=n_val, replace=False)
    val_paths = [train_good[i] for i in sorted(val_indices)]
    return val_paths


def compute_image_score(anomaly_map: np.ndarray, top_k_frac: float = 0.01) -> float:
    """Compute image-level anomaly score from the 2D anomaly map."""
    flat = anomaly_map.flatten()
    k = max(1, int(len(flat) * top_k_frac))
    top_vals = np.partition(flat, -k)[-k:]
    return float(top_vals.mean())


def evaluate_model(
    model_name: str,
    category: str,
    cfg: dict,
    predict_fn: Callable[[torch.Tensor], Dict[str, Any]],
    device: torch.device,
    custom_top_k: Optional[float] = None,
    custom_sigma: Optional[float] = None,
    custom_val_percentile: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluates any anomaly detection model with strict zero-leakage thresholding.

    predict_fn: Callable accepting (B, 3, H, W) torch.Tensor -> dict with:
      - "anomaly_maps": np.ndarray or torch.Tensor of shape (B, H, W)
      - "reconstructions": optional np.ndarray or torch.Tensor of shape (B, 3, H, W)
    """
    d_cfg = cfg["data"]
    e_cfg = cfg.get("evaluate", {})
    img_size = d_cfg.get("img_size", 256)
    
    # Configuration resolution
    sigma = custom_sigma if custom_sigma is not None else float(e_cfg.get("gaussian_sigma", 4.0))
    
    top_k_cfg = e_cfg.get("top_k_frac", 0.01)
    if isinstance(top_k_cfg, dict):
        default_top_k = top_k_cfg.get(category, 0.01)
    else:
        default_top_k = float(top_k_cfg)
    top_k = custom_top_k if custom_top_k is not None else default_top_k
    
    val_percentile = (
        custom_val_percentile
        if custom_val_percentile is not None
        else float(e_cfg.get("val_percentile", 99.0))
    )
    val_ratio = float(e_cfg.get("val_ratio", 0.2))
    seed = int(cfg.get("train", {}).get("seed", 42))

    # Output directory
    results_dir = Path(e_cfg.get("results_dir", "./outputs/results")) / model_name / category
    heatmap_dir = results_dir / "heatmaps"
    results_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    # 1. Compute Validation Normal Scores (Zero Test Leakage Thresholding)
    val_paths = get_val_split(d_cfg["root_dir"], category, val_ratio=val_ratio, seed=seed)
    val_scores = []
    
    # Process validation in batches
    batch_size = d_cfg.get("batch_size", 16)
    for i in range(0, len(val_paths), batch_size):
        batch_paths = val_paths[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            img = _load_image(p, img_size)
            tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
            imgs.append(tensor)
        batch_tensor = torch.stack(imgs).to(device)
        
        with torch.no_grad():
            out = predict_fn(batch_tensor)
            maps = out["anomaly_maps"]
            if isinstance(maps, torch.Tensor):
                maps = maps.detach().cpu().numpy()
            if maps.ndim == 4 and maps.shape[1] == 1:
                maps = maps.squeeze(1)
        
        for amap in maps:
            if sigma > 0:
                amap = gaussian_filter(amap, sigma=sigma)
            val_scores.append(compute_image_score(amap, top_k_frac=top_k))

    val_threshold = float(np.percentile(val_scores, val_percentile))
    print(f"[{model_name}][{category}] Validation normal threshold ({val_percentile}th percentile): {val_threshold:.5f}")

    # 2. Evaluate on Test Split
    test_ds = MVTecDataset(d_cfg["root_dir"], category, "test", img_size)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # avoid multiprocessing overhead during eval
        collate_fn=test_collate,
    )

    all_scores, all_labels, all_defect_types, all_paths = [], [], [], []
    all_anomaly_maps, all_gt_masks = [], []
    all_recons, all_orig_imgs = [], []
    saved_count = defaultdict(int)
    n_visual = e_cfg.get("n_visual_samples_per_class", 6)
    alpha = float(e_cfg.get("overlay_alpha", 0.45))

    for batch in test_loader:
        imgs = batch["image"].to(device)
        masks = batch["mask"].numpy()  # B, 1, H, W
        labels = batch["label"].numpy()
        defect_types = batch["defect_type"]
        paths = batch["path"]

        with torch.no_grad():
            out = predict_fn(imgs)
            maps = out["anomaly_maps"]
            recons = out.get("reconstructions", None)
            
            if isinstance(maps, torch.Tensor):
                maps = maps.detach().cpu().numpy()
            if maps.ndim == 4 and maps.shape[1] == 1:
                maps = maps.squeeze(1)
                
            if recons is not None and isinstance(recons, torch.Tensor):
                recons = recons.detach().cpu()

        for j in range(len(labels)):
            amap = maps[j]
            if sigma > 0:
                amap = gaussian_filter(amap, sigma=sigma)
            score = compute_image_score(amap, top_k_frac=top_k)

            all_scores.append(score)
            all_labels.append(int(labels[j]))
            all_defect_types.append(defect_types[j])
            all_paths.append(paths[j])
            all_anomaly_maps.append(amap)
            all_gt_masks.append(masks[j, 0])

            # Qualitative heatmap export
            dtype = defect_types[j]
            if saved_count[dtype] < n_visual:
                orig = to_uint8_img(imgs[j])
                overlay = overlay_heatmap(orig, amap, alpha=alpha)
                gt_overlay = orig.copy()
                mask_np = masks[j, 0]
                if mask_np.max() > 0:
                    gt_overlay[mask_np > 0.5] = [255, 0, 0]

                if recons is not None:
                    rec = to_uint8_img(recons[j])
                    panel = np.concatenate([orig, rec, overlay, gt_overlay], axis=1)
                else:
                    panel = np.concatenate([orig, overlay, gt_overlay], axis=1)

                fname = f"{dtype}_{Path(paths[j]).stem}.png"
                save_image(heatmap_dir / fname, panel)
                saved_count[dtype] += 1

    all_scores = np.array(all_scores, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=int)
    all_anomaly_maps = np.stack(all_anomaly_maps)
    all_gt_masks = np.stack(all_gt_masks)

    # 3. Compute Metrics
    oracle_threshold = find_youden_threshold(all_labels, all_scores)
    metrics = compute_all_metrics(
        y_true=all_labels,
        y_scores=all_scores,
        val_threshold=val_threshold,
        oracle_threshold=oracle_threshold,
        gt_masks=all_gt_masks,
        anomaly_maps=all_anomaly_maps,
        defect_types=all_defect_types,
        img_paths=all_paths,
    )

    # 4. Save Artifacts
    df = pd.DataFrame({
        "score": all_scores,
        "label": all_labels,
        "defect_type": all_defect_types,
        "path": all_paths,
        "val_pred": (all_scores >= val_threshold).astype(int),
        "oracle_pred": (all_scores >= oracle_threshold).astype(int),
    })
    df.to_csv(results_dir / "per_image_scores.csv", index=False)

    defect_breakdown = metrics.pop("defect_breakdown")
    per_type_df = pd.DataFrame(defect_breakdown)
    per_type_df.to_csv(results_dir / "per_defect_type_breakdown.csv", index=False)

    summary = {
        "model": model_name,
        "category": category,
        "n_test_images": len(df),
        "hyperparameters": {
            "gaussian_sigma": sigma,
            "top_k_frac": top_k,
            "val_percentile": val_percentile,
            "val_ratio": val_ratio,
        },
        **metrics,
    }

    with open(results_dir / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save ROC and histogram plot
    _plot_roc_and_distribution(
        all_labels,
        all_scores,
        val_threshold,
        oracle_threshold,
        results_dir,
        category,
        model_name,
    )

    print(f"\n[{model_name}][{category}] === EVALUATION SUMMARY ===")
    print(f"  Image AUROC          : {summary['image_level_roc_auc']:.4f}")
    print(f"  Image Average Prec   : {summary['image_level_average_precision']:.4f}")
    print(f"  Pixel AUROC          : {summary['pixel_level_roc_auc']:.4f}" if summary['pixel_level_roc_auc'] is not None else "  Pixel AUROC: N/A")
    print(f"  Pixel AUPRO          : {summary['pixel_level_aupro']:.4f}" if summary['pixel_level_aupro'] is not None else "  Pixel AUPRO: N/A")
    print(f"  Val Threshold ({val_percentile}%): {summary['validation_threshold']:.5f}")
    print(f"  Val F1 / Prec / Rec  : {summary['f1_score']:.4f} / {summary['precision']:.4f} / {summary['recall']:.4f}")
    print(f"  Oracle (Youden) F1   : {summary['oracle_f1_score']:.4f} (Th: {summary['oracle_threshold']:.5f})")
    print(f"[{model_name}][{category}] per-defect-type breakdown:\n{per_type_df.to_string(index=False)}")
    print(f"[{model_name}][{category}] Heatmaps saved to: {heatmap_dir}")

    return summary


def _plot_roc_and_distribution(
    labels: np.ndarray,
    scores: np.ndarray,
    val_th: float,
    orc_th: float,
    results_dir: Path,
    category: str,
    model_name: str,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    fpr, tpr, _ = roc_curve(labels, scores)
    axes[0].plot(fpr, tpr, color="crimson", lw=2, label=f"ROC (AUC = {np.trapezoid(tpr, fpr):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="chance")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC — {model_name.upper()} ({category})")
    axes[0].legend(loc="lower right")

    axes[1].hist(scores[labels == 0], bins=25, alpha=0.6, color="navy", label="normal (good)")
    axes[1].hist(scores[labels == 1], bins=25, alpha=0.6, color="darkorange", label="anomalous")
    axes[1].axvline(val_th, color="green", linestyle="-", lw=2, label=f"Val Th: {val_th:.4f}")
    axes[1].axvline(orc_th, color="purple", linestyle="--", lw=2, label=f"Oracle Th: {orc_th:.4f}")
    axes[1].set_xlabel("Anomaly score")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Score Distribution — {category}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(results_dir / "roc_and_score_distribution.png", dpi=150)
    plt.close()
