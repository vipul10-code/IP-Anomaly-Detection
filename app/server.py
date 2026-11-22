"""
server.py
---------
FastAPI application backend for visual anomaly detection inspection, benchmarking,
and model comparison dashboard.
"""

import base64
import datetime
import io
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from scipy.ndimage import gaussian_filter

from src.utils import get_device, overlay_heatmap, to_uint8_img, normalize_map
from src.evaluator import compute_image_score

app = FastAPI(title="Manufacturing Defect Anomaly Detection Dashboard", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "results"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
HISTORY_FILE = OUTPUTS_DIR / "inspection_history.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# In-memory model cache: (model_name, category) -> model instance
MODEL_CACHE: Dict[str, Any] = {}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_cached_model(model_name: str, category: str, device: torch.device):
    """Lazily loads and caches models on the appropriate device."""
    key = f"{model_name}_{category}".lower()
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    cfg = load_config()

    if model_name.lower() == "cae":
        ckpt_path = CHECKPOINTS_DIR / f"{category}_best.pt"
        if not ckpt_path.exists():
            raise HTTPException(status_code=404, detail=f"Checkpoint not found for CAE on {category}: {ckpt_path.name}")
        from src.model import build_model
        from src.losses import SSIMLoss
        ckpt = torch.load(ckpt_path, map_location=device)
        model = build_model(cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        ssim_m = SSIMLoss().to(device)
        MODEL_CACHE[key] = {"model": model, "ssim": ssim_m, "type": "cae"}

    elif model_name.lower() == "patchcore":
        ckpt_path = CHECKPOINTS_DIR / f"patchcore_{category}.pt"
        if not ckpt_path.exists():
            raise HTTPException(status_code=404, detail=f"Checkpoint not found for PatchCore on {category}: {ckpt_path.name}")
        from src.models.patchcore import PatchCore
        model = PatchCore(device=device)
        model.load_checkpoint(str(ckpt_path))
        MODEL_CACHE[key] = {"model": model, "type": "patchcore"}

    elif model_name.lower() == "efficientad":
        ckpt_path = CHECKPOINTS_DIR / f"efficientad_{category}.pt"
        if not ckpt_path.exists():
            raise HTTPException(status_code=404, detail=f"Checkpoint not found for EfficientAD on {category}: {ckpt_path.name}")
        from src.models.efficientad import EfficientAD
        model = EfficientAD(device=device)
        model.load_checkpoint(str(ckpt_path))
        MODEL_CACHE[key] = {"model": model, "type": "efficientad"}

    elif model_name.lower() == "draem":
        if category != "leather":
            raise HTTPException(status_code=400, detail="DRAEM is specifically trained on the leather category.")
        ckpt_path = CHECKPOINTS_DIR / f"draem_{category}.pt"
        if not ckpt_path.exists():
            raise HTTPException(status_code=404, detail=f"Checkpoint not found for DRAEM on {category}: {ckpt_path.name}")
        from src.models.draem import DRAEM
        model = DRAEM(device=device)
        model.load_checkpoint(str(ckpt_path))
        MODEL_CACHE[key] = {"model": model, "type": "draem"}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model name: {model_name}")

    return MODEL_CACHE[key]


def numpy_to_base64_png(arr_rgb_uint8: np.ndarray) -> str:
    """Converts HWC uint8 RGB numpy image to base64 data URI."""
    arr_bgr = cv2.cvtColor(arr_rgb_uint8, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", arr_bgr)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ---------------- API ENDPOINTS ----------------


@app.get("/api/models")
def get_available_models():
    """Lists supported models and categories, with checkpoint existence status."""
    models_info = [
        {"id": "patchcore", "name": "PatchCore (WideResNet-50)", "categories": ["leather", "transistor"]},
        {"id": "efficientad", "name": "EfficientAD (Distillation)", "categories": ["leather", "transistor"]},
        {"id": "cae", "name": "Convolutional Autoencoder (v1)", "categories": ["leather", "transistor"]},
        {"id": "draem", "name": "DRAEM (Synthetic Embedding)", "categories": ["leather"]},
    ]
    for m in models_info:
        m["available"] = {}
        for c in m["categories"]:
            if m["id"] == "cae":
                p = CHECKPOINTS_DIR / f"{c}_best.pt"
            else:
                p = CHECKPOINTS_DIR / f"{m['id']}_{c}.pt"
            m["available"][c] = p.exists()
    return {"models": models_info}


@app.get("/api/benchmark")
def get_benchmark_data():
    """Returns aggregated benchmark metrics across all models and categories."""
    bench_file = RESULTS_DIR / "benchmark.json"
    if not bench_file.exists():
        from src.benchmark import run_benchmark
        cfg = load_config()
        run_benchmark(cfg)

    if bench_file.exists():
        with open(bench_file) as f:
            return json.load(f)
    return []


@app.get("/api/results/{model}/{category}")
def get_model_results(model: str, category: str):
    """Returns detailed summary metrics for a specific model and category."""
    model = model.lower()
    category = category.lower()

    summary_file = RESULTS_DIR / model / category / "summary_metrics.json"
    if not summary_file.exists() and model == "cae":
        summary_file = RESULTS_DIR / category / "summary_metrics.json"

    if not summary_file.exists():
        raise HTTPException(status_code=404, detail=f"Metrics not found for {model} on {category}")

    with open(summary_file) as f:
        return json.load(f)


@app.get("/api/defects/{model}/{category}")
def get_defect_breakdown(model: str, category: str):
    """Returns per-defect-type detection breakdown table."""
    model = model.lower()
    category = category.lower()

    csv_file = RESULTS_DIR / model / category / "per_defect_type_breakdown.csv"
    if not csv_file.exists() and model == "cae":
        csv_file = RESULTS_DIR / category / "per_defect_type_breakdown.csv"

    if not csv_file.exists():
        raise HTTPException(status_code=404, detail=f"Defect breakdown not found for {model} on {category}")

    df = pd.read_csv(csv_file)
    return df.to_dict(orient="records")


@app.get("/api/heatmaps/{model}/{category}")
def list_heatmaps(model: str, category: str):
    """Lists saved heatmap panel images for qualitative browsing."""
    model = model.lower()
    category = category.lower()

    heatmap_dir = RESULTS_DIR / model / category / "heatmaps"
    if not heatmap_dir.exists() and model == "cae":
        heatmap_dir = RESULTS_DIR / category / "heatmaps"

    if not heatmap_dir.exists():
        return {"heatmaps": []}

    panels = []
    for p in sorted(heatmap_dir.glob("*.png")):
        parts = p.stem.split("_")
        defect_type = parts[0] if parts else "unknown"
        panels.append({
            "filename": p.name,
            "defect_type": defect_type,
            "url": f"/heatmaps/{model}/{category}/{p.name}",
        })
    return {"heatmaps": panels}


@app.get("/heatmaps/{model}/{category}/{filename}")
def serve_heatmap_image(model: str, category: str, filename: str):
    """Serves individual heatmap PNG images."""
    path = RESULTS_DIR / model.lower() / category.lower() / "heatmaps" / filename
    if not path.exists() and model.lower() == "cae":
        path = RESULTS_DIR / category.lower() / "heatmaps" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Heatmap image not found")
    return FileResponse(path, media_type="image/png")


@app.post("/api/predict")
async def predict_sample(
    image: UploadFile = File(...),
    model: str = Form(...),
    category: str = Form(...),
):
    """
    Runs real-time anomaly inference on an uploaded image.
    Returns anomaly score, NORMAL / DEFECT verdict, base64 heatmap overlay,
    predicted mask, and latency.
    """
    cfg = load_config()
    d_cfg = cfg.get("data", {})
    e_cfg = cfg.get("evaluate", {})
    img_size = d_cfg.get("img_size", 256)
    device = get_device(cfg.get("train", {}).get("device", "cuda"))

    # Read and decode image
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file uploaded.")

    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # Tensor input
    img_tensor = (
        torch.from_numpy(resized_rgb.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    # Load model
    t_start = time.perf_counter()
    model_entry = get_cached_model(model, category, device)
    m_type = model_entry["type"]

    recon_np = None
    with torch.no_grad():
        if m_type == "cae":
            from src.utils import compute_anomaly_maps
            m = model_entry["model"]
            ssim_m = model_entry["ssim"]
            recon = m(img_tensor)
            raw_map = compute_anomaly_maps(recon, img_tensor, ssim_module=ssim_m).squeeze(1).cpu().numpy()[0]
            recon_np = to_uint8_img(recon[0])
        elif m_type == "draem":
            m = model_entry["model"]
            out = m.predict(img_tensor)
            raw_map = out["anomaly_maps"][0]
            if "reconstructions" in out and out["reconstructions"] is not None:
                recon_np = to_uint8_img(out["reconstructions"][0])
        else:
            m = model_entry["model"]
            out = m.predict(img_tensor)
            raw_map = out["anomaly_maps"][0]

    # Gaussian smoothing
    sigma = float(e_cfg.get("gaussian_sigma", 4.0))
    if sigma > 0:
        smoothed_map = gaussian_filter(raw_map, sigma=sigma)
    else:
        smoothed_map = raw_map

    # Top-k image score
    top_k_cfg = e_cfg.get("top_k_frac", 0.01)
    top_k = top_k_cfg.get(category, 0.01) if isinstance(top_k_cfg, dict) else float(top_k_cfg)
    score = compute_image_score(smoothed_map, top_k_frac=top_k)

    latency_ms = (time.perf_counter() - t_start) * 1000.0

    # Decision threshold from summary metrics
    threshold = None
    summary_path = RESULTS_DIR / model.lower() / category.lower() / "summary_metrics.json"
    if not summary_path.exists() and model.lower() == "cae":
        summary_path = RESULTS_DIR / category.lower() / "summary_metrics.json"

    if summary_path.exists():
        with open(summary_path) as f:
            meta = json.load(f)
            threshold = meta.get("validation_threshold") or meta.get("decision_threshold")

    verdict = "DEFECT" if (threshold is not None and score >= threshold) else "NORMAL"

    # Heatmap rendering
    alpha = float(e_cfg.get("overlay_alpha", 0.45))
    overlay = overlay_heatmap(resized_rgb, smoothed_map, alpha=alpha)

    # Predicted binary mask using threshold (or normalized map 70% threshold)
    norm_map = normalize_map(smoothed_map)
    mask_uint8 = (norm_map > 0.4).astype(np.uint8) * 255
    mask_rgb = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2RGB)

    # Save to history log
    history_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": image.filename,
        "model": model.upper(),
        "category": category,
        "score": round(float(score), 5),
        "threshold": round(float(threshold), 5) if threshold is not None else None,
        "verdict": verdict,
        "latency_ms": round(latency_ms, 2),
    }

    history = []
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE) as f:
                history = json.load(f)
        except Exception:
            history = []
    history.insert(0, history_entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history[:100], f, indent=2)

    return {
        "score": float(score),
        "threshold": float(threshold) if threshold is not None else None,
        "verdict": verdict,
        "latency_ms": round(latency_ms, 1),
        "stats": {
            "min": float(smoothed_map.min()),
            "max": float(smoothed_map.max()),
            "mean": float(smoothed_map.mean()),
            "std": float(smoothed_map.std()),
        },
        "original_b64": numpy_to_base64_png(resized_rgb),
        "overlay_b64": numpy_to_base64_png(overlay),
        "mask_b64": numpy_to_base64_png(mask_rgb),
        "reconstruction_b64": numpy_to_base64_png(recon_np) if recon_np is not None else None,
    }


@app.get("/api/history")
def get_inspection_history():
    """Returns logged inspection history."""
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


# Serve MVTec sample images so the dashboard's "try a test image" buttons work
DATA_DIR = PROJECT_ROOT / "data"
if DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")

# Mount static frontend
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")
