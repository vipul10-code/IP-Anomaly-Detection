"""
metrics.py
----------
Standardized metrics for unsupervised anomaly detection and localization on MVTec AD:
  - Image-level AUROC (ROC-AUC)
  - Image-level Average Precision (AP / PR-AUC)
  - Pixel-level AUROC
  - Pixel-level AUPRO (Per-Region Overlap up to FPR=0.30)
  - Precision, Recall, F1, Confusion Matrix at validation threshold
  - Oracle threshold (Youden's J on test set) for comparison
  - Per-defect-type detection rates
"""

import numpy as np
from scipy.ndimage import label as nd_label
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
)


def find_youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Find threshold maximizing TPR - FPR (Youden's J) on the given set."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return float(thresholds[best_idx])


def compute_aupro(
    gt_masks: np.ndarray,
    anomaly_maps: np.ndarray,
    max_fpr: float = 0.3,
    num_thresholds: int = 100,
) -> float:
    """
    Compute Area Under Per-Region Overlap (AUPRO) metric up to max_fpr (default 0.30).
    gt_masks: binary masks (N, H, W) or list of (H, W) where 1 indicates anomaly
    anomaly_maps: float maps (N, H, W) or list of (H, W)
    """
    gt_masks = np.asarray(gt_masks, dtype=bool)
    anomaly_maps = np.asarray(anomaly_maps, dtype=np.float32)

    # Filter only masks that contain at least one defect pixel
    defect_indices = [i for i, m in enumerate(gt_masks) if m.sum() > 0]
    if len(defect_indices) == 0:
        return 0.0

    gt_masks = gt_masks[defect_indices]
    anomaly_maps = anomaly_maps[defect_indices]

    # Pre-identify connected components (regions) for each ground-truth mask
    regions = []
    total_normal_pixels = 0
    for mask in gt_masks:
        labeled, num_features = nd_label(mask)
        img_regions = []
        for feat_idx in range(1, num_features + 1):
            region_mask = (labeled == feat_idx)
            img_regions.append(region_mask)
        regions.append(img_regions)
        total_normal_pixels += (~mask).sum()

    if total_normal_pixels == 0:
        return 0.0

    # Collect threshold candidates from anomaly map values
    min_val = float(anomaly_maps.min())
    max_val = float(anomaly_maps.max())
    if max_val - min_val < 1e-8:
        return 0.0

    # Percentile-based thresholds for better resolution across the score distribution
    thresholds = np.linspace(max_val, min_val, num_thresholds)

    fpr_list = []
    pro_list = []

    for th in thresholds:
        # Binary prediction map
        pred_binary = (anomaly_maps >= th)

        # 1. Compute FPR across all non-defect pixels
        fp_count = 0
        for i, mask in enumerate(gt_masks):
            fp_count += (pred_binary[i] & (~mask)).sum()
        fpr = fp_count / float(total_normal_pixels)

        # 2. Compute mean Per-Region Overlap across all ground truth regions
        all_overlaps = []
        for i, img_regions in enumerate(regions):
            pred_img = pred_binary[i]
            for r_mask in img_regions:
                r_size = r_mask.sum()
                if r_size > 0:
                    overlap = (pred_img & r_mask).sum() / float(r_size)
                    all_overlaps.append(overlap)

        mean_pro = float(np.mean(all_overlaps)) if all_overlaps else 0.0

        fpr_list.append(fpr)
        pro_list.append(mean_pro)

        if fpr >= max_fpr:
            break

    fpr_arr = np.array(fpr_list)
    pro_arr = np.array(pro_list)

    # Sort by FPR ascending
    sort_idx = np.argsort(fpr_arr)
    fpr_arr = fpr_arr[sort_idx]
    pro_arr = pro_arr[sort_idx]

    # Remove duplicates
    fpr_arr, unique_idx = np.unique(fpr_arr, return_index=True)
    pro_arr = pro_arr[unique_idx]

    # Clip to max_fpr
    valid = fpr_arr <= max_fpr
    fpr_arr = fpr_arr[valid]
    pro_arr = pro_arr[valid]

    if len(fpr_arr) == 0:
        return 0.0

    # Ensure starts at 0
    if fpr_arr[0] > 0:
        fpr_arr = np.concatenate([[0.0], fpr_arr])
        pro_arr = np.concatenate([[0.0], pro_arr])

    # Extend to max_fpr if needed
    if fpr_arr[-1] < max_fpr:
        fpr_arr = np.concatenate([fpr_arr, [max_fpr]])
        pro_arr = np.concatenate([pro_arr, [pro_arr[-1]]])

    # Integrate using trapezoidal rule and normalize by max_fpr
    aupro = float(np.trapezoid(pro_arr, fpr_arr) / max_fpr)
    return max(0.0, min(1.0, aupro))


def compute_all_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    val_threshold: float,
    oracle_threshold: float,
    gt_masks: np.ndarray = None,
    anomaly_maps: np.ndarray = None,
    defect_types: list = None,
    img_paths: list = None,
) -> dict:
    """
    Computes complete standardized metrics for a model on a dataset.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=np.float32)

    # Image-level ROC-AUC and Average Precision
    image_auroc = float(roc_auc_score(y_true, y_scores))
    image_ap = float(average_precision_score(y_true, y_scores))

    # Metrics at validation threshold (strictly honest, zero test leakage)
    val_preds = (y_scores >= val_threshold).astype(int)
    val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(
        y_true, val_preds, average="binary", zero_division=0
    )
    val_cm = confusion_matrix(y_true, val_preds).tolist()

    # Oracle metrics at Youden test threshold (for comparison with legacy v1)
    oracle_preds = (y_scores >= oracle_threshold).astype(int)
    orc_prec, orc_rec, orc_f1, _ = precision_recall_fscore_support(
        y_true, oracle_preds, average="binary", zero_division=0
    )
    orc_cm = confusion_matrix(y_true, oracle_preds).tolist()

    # Pixel-level metrics if masks and maps provided
    pixel_auroc = None
    pixel_aupro = None
    if gt_masks is not None and anomaly_maps is not None and len(gt_masks) > 0:
        # Compute pixel AUROC on subsampled pixels across defect images
        gt_bool = np.asarray(gt_masks > 0.5, dtype=bool)
        defect_idxs = [i for i, m in enumerate(gt_bool) if m.sum() > 0]
        if defect_idxs:
            sub_gts = []
            sub_maps = []
            for idx in defect_idxs:
                flat_gt = gt_bool[idx].flatten()
                flat_map = np.asarray(anomaly_maps[idx], dtype=np.float32).flatten()
                sample_k = min(2000, len(flat_gt))
                choice = np.random.choice(len(flat_gt), size=sample_k, replace=False)
                sub_gts.append(flat_gt[choice])
                sub_maps.append(flat_map[choice])
            pixel_auroc = float(roc_auc_score(np.concatenate(sub_gts), np.concatenate(sub_maps)))
            pixel_aupro = float(compute_aupro(gt_bool, anomaly_maps))

    # Per-defect-type breakdown
    defect_breakdown = []
    if defect_types is not None:
        import pandas as pd
        df = pd.DataFrame({
            "defect_type": defect_types,
            "score": y_scores,
            "label": y_true,
            "val_pred": val_preds,
            "oracle_pred": oracle_preds,
            "path": img_paths if img_paths is not None else [""] * len(y_true),
        })
        for dtype, group in df.groupby("defect_type"):
            if dtype == "good":
                val_det_rate = float((group["val_pred"] == 0).mean())
                orc_det_rate = float((group["oracle_pred"] == 0).mean())
            else:
                val_det_rate = float((group["val_pred"] == 1).mean())
                orc_det_rate = float((group["oracle_pred"] == 1).mean())
            defect_breakdown.append({
                "defect_type": dtype,
                "n_images": len(group),
                "mean_anomaly_score": float(group["score"].mean()),
                "detection_rate_val_th": val_det_rate,
                "detection_rate_oracle_th": orc_det_rate,
            })

    return {
        "image_level_roc_auc": image_auroc,
        "image_level_average_precision": image_ap,
        "pixel_level_roc_auc": pixel_auroc,
        "pixel_level_aupro": pixel_aupro,
        "validation_threshold": float(val_threshold),
        "precision": float(val_prec),
        "recall": float(val_rec),
        "f1_score": float(val_f1),
        "confusion_matrix": {"labels": ["normal", "anomalous"], "matrix": val_cm},
        "oracle_threshold": float(oracle_threshold),
        "oracle_precision": float(orc_prec),
        "oracle_recall": float(orc_rec),
        "oracle_f1_score": float(orc_f1),
        "oracle_confusion_matrix": {"labels": ["normal", "anomalous"], "matrix": orc_cm},
        "defect_breakdown": defect_breakdown,
    }
