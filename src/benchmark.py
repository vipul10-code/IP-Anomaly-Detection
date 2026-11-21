"""
benchmark.py
------------
Unified benchmark generator for all anomaly detection models across MVTec AD categories.
Gathers:
  - Image AUROC, Image AP
  - Pixel AUROC, Pixel AUPRO
  - Validation Threshold F1, Precision, Recall
  - Oracle Youden F1
  - Inference Latency (ms per image on CPU/GPU)
  - Model Checkpoint Size (MB)
  - Detailed Per-Defect-Type Detection Rates

Writes:
  outputs/results/benchmark.json
  outputs/results/benchmark.csv
  outputs/results/defect_benchmark.csv
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import yaml

from src.utils import get_device


def measure_latency_and_size(model_name: str, category: str, cfg: dict, device: torch.device) -> Dict[str, Any]:
    """Measures model checkpoint size in MB and per-image inference latency in ms."""
    ckpt_dir = Path(cfg.get("train", {}).get("checkpoint_dir", "./outputs/checkpoints"))
    
    if model_name == "cae":
        ckpt_path = ckpt_dir / f"{category}_best.pt"
    else:
        ckpt_path = ckpt_dir / f"{model_name}_{category}.pt"

    if not ckpt_path.exists():
        return {"latency_ms": None, "checkpoint_mb": None}

    size_mb = ckpt_path.stat().st_size / (1024 * 1024)

    # Instantiate model and measure inference time
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    try:
        if model_name == "cae":
            from src.model import build_model
            from src.losses import SSIMLoss
            from src.utils import compute_anomaly_maps
            ckpt = torch.load(ckpt_path, map_location=device)
            m = build_model(cfg).to(device)
            m.load_state_dict(ckpt["model_state"])
            m.eval()
            ssim_m = SSIMLoss().to(device)
            fn = lambda x: compute_anomaly_maps(m(x), x, ssim_module=ssim_m)
        elif model_name == "patchcore":
            from src.models.patchcore import PatchCore
            m = PatchCore(device=device)
            m.load_checkpoint(str(ckpt_path))
            fn = lambda x: m.predict(x)["anomaly_maps"]
        elif model_name == "efficientad":
            from src.models.efficientad import EfficientAD
            m = EfficientAD(device=device)
            m.load_checkpoint(str(ckpt_path))
            fn = lambda x: m.predict(x)["anomaly_maps"]
        elif model_name == "draem":
            from src.models.draem import DRAEM
            m = DRAEM(device=device)
            m.load_checkpoint(str(ckpt_path))
            fn = lambda x: m.predict(x)["anomaly_maps"]
        else:
            return {"latency_ms": None, "checkpoint_mb": size_mb}

        # Warm-up
        with torch.no_grad():
            for _ in range(3):
                _ = fn(dummy_input)

        # Timed runs
        n_iters = 15
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iters):
                _ = fn(dummy_input)
        elapsed = time.perf_counter() - t0
        latency_ms = (elapsed / n_iters) * 1000.0

        return {"latency_ms": round(latency_ms, 2), "checkpoint_mb": round(size_mb, 2)}
    except Exception as e:
        print(f"[benchmark] Warning: failed to measure latency for {model_name} {category}: {e}")
        return {"latency_ms": None, "checkpoint_mb": round(size_mb, 2)}


def run_benchmark(cfg: dict) -> pd.DataFrame:
    results_base = Path(cfg.get("evaluate", {}).get("results_dir", "./outputs/results"))
    device = get_device(cfg.get("train", {}).get("device", "cuda"))
    models = ["cae", "patchcore", "efficientad", "draem"]
    categories = cfg.get("data", {}).get("categories", ["leather", "transistor"])

    benchmark_rows = []
    defect_rows = []

    for model_name in models:
        for cat in categories:
            # DRAEM is evaluated on leather specifically
            if model_name == "draem" and cat != "leather":
                continue

            summary_file = results_base / model_name / cat / "summary_metrics.json"
            if not summary_file.exists():
                # Check legacy path for CAE
                if model_name == "cae":
                    summary_file = results_base / cat / "summary_metrics.json"

            if not summary_file.exists():
                print(f"[benchmark] Skipping {model_name} on {cat} (summary_metrics.json not found)")
                continue

            with open(summary_file) as f:
                summary = json.load(f)

            # Measure latency and size
            perf = measure_latency_and_size(model_name, cat, cfg, device)

            row = {
                "model": model_name.upper(),
                "category": cat,
                "image_auroc": summary.get("image_level_roc_auc"),
                "image_ap": summary.get("image_level_average_precision"),
                "pixel_auroc": summary.get("pixel_level_roc_auc"),
                "pixel_aupro": summary.get("pixel_level_aupro"),
                "val_threshold": summary.get("validation_threshold") or summary.get("decision_threshold"),
                "val_f1": summary.get("f1_score"),
                "val_precision": summary.get("precision"),
                "val_recall": summary.get("recall"),
                "oracle_f1": summary.get("oracle_f1_score"),
                "inference_ms": perf["latency_ms"],
                "model_size_mb": perf["checkpoint_mb"],
            }
            benchmark_rows.append(row)

            # Read defect breakdown
            defect_file = results_base / model_name / cat / "per_defect_type_breakdown.csv"
            if not defect_file.exists() and model_name == "cae":
                defect_file = results_base / cat / "per_defect_type_breakdown.csv"

            if defect_file.exists():
                d_df = pd.read_csv(defect_file)
                for _, d_row in d_df.iterrows():
                    defect_type = d_row.get("defect_type")
                    det_rate = d_row.get("detection_rate_val_th")
                    if det_rate is None or pd.isna(det_rate):
                        det_rate = d_row.get("detection_rate")
                    defect_rows.append({
                        "model": model_name.upper(),
                        "category": cat,
                        "defect_type": defect_type,
                        "n_images": d_row.get("n_images"),
                        "detection_rate": det_rate,
                    })

    bench_df = pd.DataFrame(benchmark_rows)
    defect_df = pd.DataFrame(defect_rows)

    # Save benchmark outputs
    out_dir = results_base
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_df.to_csv(out_dir / "benchmark.csv", index=False)
    bench_df.to_json(out_dir / "benchmark.json", orient="records", indent=2)

    if not defect_df.empty:
        defect_df.to_csv(out_dir / "defect_benchmark.csv", index=False)

    print("\n" + "=" * 95)
    print("                      UNIFIED ANOMALY DETECTION BENCHMARK")
    print("=" * 95)
    display_cols = [
        "model", "category", "image_auroc", "image_ap", "pixel_auroc", "pixel_aupro",
        "val_f1", "oracle_f1", "inference_ms", "model_size_mb"
    ]
    print(bench_df[display_cols].to_string(index=False))
    print("=" * 95)
    print(f"Benchmark saved to:\n  {out_dir / 'benchmark.csv'}\n  {out_dir / 'benchmark.json'}\n  {out_dir / 'defect_benchmark.csv'}")

    return bench_df


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    run_benchmark(cfg)
