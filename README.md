# Visual Anomaly & Defect Detection for Manufacturing (MVTec AD)

A production-grade, unsupervised visual anomaly detection and defect localization pipeline evaluated on the industry-standard **MVTec Anomaly Detection (MVTec AD)** benchmark for surface textures (`leather`) and rigid assemblies (`transistor`).

---

## 1. Executive Summary & Master Results

This repository implements, evaluates, and benchmarks four distinct anomaly detection paradigms using a **strict zero-leakage thresholding protocol**:

1. **Convolutional Autoencoder (CAE v1 baseline)**: Symmetric encoder-decoder with composite MSE + SSIM reconstruction loss.
2. **PatchCore (SOTA Benchmark)**: Pretrained frozen WideResNet-50-2 feature extractor with multi-scale neighborhood aggregation, minimax k-center coreset subsampling, and exact kNN distance scoring.
3. **EfficientAD (Fast Distillation Benchmark)**: Frozen teacher network distilled into an uninformed student network paired with a feature autoencoder for ultra-low latency detection.
4. **DRAEM (Surface Texture Benchmark)**: Generates synthetic pseudo-anomalies on-the-fly and trains a reconstructive network alongside a discriminative segmentation network.

### Performance Comparison Matrix

| Model | Category | Image AUROC | Image AP | Pixel AUROC | Pixel AUPRO (FPR 0.3) | Val F1 (99% Th) | Oracle F1 (Youden) | Latency (CPU) | Checkpoint Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CAE (v1 Baseline)** | Leather | 0.7874 | 0.9250 | 0.8458 | 0.7174 | 0.8173 | 0.8049 | 89.7 ms | 8.43 MB |
| **CAE (v1 Baseline)** | Transistor | 0.6392 | 0.6568 | 0.7784 | 0.5990 | 0.4776 | 0.4906 | 87.1 ms | 8.43 MB |
| **PatchCore** | **Leather** | **0.9956** | **0.9986** | **0.9875** | **0.9469** | **0.9340** | **0.9890** | 128.3 ms | 8.79 MB |
| **PatchCore** | **Transistor** | **0.9979** | **0.9969** | **0.9390** | **0.8669** | **0.9195** | **0.9756** | 136.0 ms | 8.79 MB |
| **EfficientAD** | Leather | 0.9001 | 0.9691 | 0.9893 | 0.9698 | 0.8517 | 0.8941 | **52.3 ms** | **2.47 MB** |
| **EfficientAD** | Transistor | 0.7546 | 0.7298 | 0.7414 | 0.5614 | 0.4314 | 0.6667 | **47.8 ms** | **2.47 MB** |
| **DRAEM** | Leather | 0.8621 | 0.9365 | 0.8726 | 0.7354 | 0.8364 | 0.8639 | 99.0 ms | 8.37 MB |

### Per-Defect-Type Detection Rates (Validation Threshold)

| Category | Defect Type | CAE (v1) | EfficientAD | PatchCore | DRAEM |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Leather** | `color` | 100.0% | 94.7% | **100.0%** | 26.3% |
| | `cut` | 84.2% | 94.7% | **100.0%** | **100.0%** |
| | `fold` | 100.0% | 100.0% | **100.0%** | 94.1% |
| | `glue` | 78.9% | 94.7% | **100.0%** | 68.4% |
| | `poke` | 100.0% | 100.0% | **100.0%** | 88.9% |
| | `good` (Normal Spec.) | 3.1% | 12.5% | **59.4%** | **87.5%** |
| **Transistor** | `bent_lead` | 50.0% | 40.0% | **100.0%** | — |
| | `cut_lead` | 30.0% | 0.0% | **100.0%** | — |
| | `damaged_case` | 0.0% | 0.0% | **100.0%** | — |
| | `misplaced` | 80.0% | 70.0% | **100.0%** | — |
| | `good` (Normal Spec.) | 81.7% | **100.0%** | 88.3% | — |

---

## 2. Evaluation Protocol & Zero Test Leakage

### The Flaw in Legacy v1
In baseline implementations, the decision threshold was often determined using Youden's J statistic ($J = \text{TPR} - \text{FPR}$) directly on the test set. Because the threshold was chosen after observing test ground-truth labels, reported precision, recall, and F1 scores were unrealistically optimistic.

### The Corrected Validation-Based Protocol
To ensure realistic manufacturing evaluation without test leakage:
1. **Validation Split**: A deterministic ~20% subset of defect-free (`train/good`) images is held out (fixed random seed = 42).
2. **Normal Calibration**: The model computes anomaly scores on these normal validation samples.
3. **Threshold Selection**: The decision threshold is selected as the **99.0th percentile** of normal validation anomaly scores (configurable in `config.yaml`).
4. **Honest Testing**: Test images are evaluated against this frozen threshold without peeking at test labels.
5. **Oracle Reference**: The Youden test-set threshold is retained strictly as an upper-bound reference (`oracle_threshold` and `oracle_f1`).

---

## 3. Repository Structure

```
anomaly_detection_project/
├── config.yaml                    # Master hyperparameters and evaluation configuration
├── requirements.txt               # Locked dependencies
├── main.py                        # Unified CLI entry point
├── app/                           # Interactive Web Inspection Dashboard
│   ├── server.py                  # FastAPI application backend
│   └── static/                    # Plain HTML, CSS, Vanilla JS frontend
│       ├── index.html
│       ├── style.css
│       └── app.js
├── data/
│   └── mvtec/
│       ├── leather/{train,test,ground_truth}/
│       └── transistor/{train,test,ground_truth}/
├── docs/
│   └── PROGRESS.md                # Phase-by-phase implementation and metrics log
├── src/
│   ├── dataset.py                 # MVTec AD dataset loaders and augmentation
│   ├── evaluator.py               # Shared zero-leakage evaluation engine
│   ├── metrics.py                 # AUROC, AP, AUPRO, F1, Youden's J
│   ├── benchmark.py               # Master cross-model benchmark generator
│   ├── infer.py                   # Single-image prediction CLI
│   ├── losses.py                  # Differentiable SSIM and MSE losses
│   ├── model.py                   # Baseline Convolutional Autoencoder architecture
│   ├── train.py                   # CAE training routine
│   ├── evaluate.py                # Legacy CAE evaluation adapter
│   ├── utils.py                   # Heatmaps, overlays, device management
│   ├── run_patchcore.py           # PatchCore CLI runner
│   ├── run_efficientad.py         # EfficientAD CLI runner
│   ├── run_draem.py               # DRAEM CLI runner
│   └── models/
│       ├── __init__.py
│       ├── patchcore.py           # Pure PyTorch PatchCore implementation
│       ├── efficientad.py         # Pure PyTorch EfficientAD implementation
│       └── draem.py               # Pure PyTorch DRAEM implementation
└── outputs/
    ├── checkpoints/               # Trained model weights (*.pt)
    └── results/                   # Standardized evaluation outputs
        ├── benchmark.json
        ├── benchmark.csv
        ├── defect_benchmark.csv
        └── <model>/<category>/
            ├── summary_metrics.json
            ├── per_image_scores.csv
            ├── per_defect_type_breakdown.csv
            ├── roc_and_score_distribution.png
            └── heatmaps/*.png
```

---

## 4. Setup & Installation

Ensure Python 3.10+ (tested on Python 3.13 64-bit) is installed:

```bash
# Optional: create virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 5. Running the Models

### 5.1 Baseline CAE (v1)
```bash
# Train CAE
python main.py train --category leather
python main.py train --category transistor

# Evaluate CAE using unified evaluation layer
python main.py evaluate --category leather
python main.py evaluate --category transistor
```

### 5.2 PatchCore
```bash
# Fit memory bank & evaluate PatchCore
python -m src.run_patchcore --category leather --mode all
python -m src.run_patchcore --category transistor --mode all

# Or run stages independently:
python -m src.run_patchcore --category leather --mode evaluate
```

### 5.3 EfficientAD
```bash
# Train student-autoencoder & evaluate
python -m src.run_efficientad --category leather --mode all --epochs 12
python -m src.run_efficientad --category transistor --mode all --epochs 12
```

### 5.4 DRAEM (Surface Textures)
```bash
# Train synthetic anomaly reconstruction & evaluate on leather
python -m src.run_draem --category leather --mode all --epochs 8
```

### 5.5 Single Image CLI Prediction
```bash
# Run prediction on any image using the baseline autoencoder
python main.py predict --category leather --image data/mvtec/leather/test/cut/000.png
```

---

## 6. Running the Master Benchmark

To regenerate the cross-architecture benchmark table, latency profiles, and defect matrices:

```bash
python main.py benchmark
```

Outputs will be saved directly to:
- `outputs/results/benchmark.csv`
- `outputs/results/benchmark.json`
- `outputs/results/defect_benchmark.csv`

---

## 7. Starting the Interactive Web Dashboard

A zero-build frontend powered by a FastAPI backend:

```bash
python -m uvicorn app.server:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### Dashboard Capabilities
1. **Overview & Benchmark**: Interactive model matrix with best metrics highlighted and grouped AUROC / localization charts.
2. **Live Inspection**: Drag-and-drop or select sample test images; choose detection architecture and category; displays real-time score, validation threshold, NORMAL / DEFECT status, and side-by-side original | heatmap overlay | predicted mask.
3. **Defect Breakdown**: Bar charts and tables comparing detection rates across fine-grained error categories.
4. **Heatmap Gallery**: Browse saved test evaluation panels across defect categories.
5. **Inspection History**: Persisted audit log of all inspected parts with one-click CSV export.

---

## 8. Limitations & Engineering Trade-offs

1. **Memory Bank Scaling (PatchCore)**: While PatchCore achieves near-perfect AUROC (>0.995), memory bank footprint scales linearly with the number of training patches. Greedy k-center coreset subsampling mitigates this, but running inference across thousands of parts per minute requires keeping the memory bank below ~2,000 vectors or using GPU FAISS indexing.
2. **Rigid vs. Texture Inductive Biases**: Reconstructive autoencoders (CAE, DRAEM) excel on stochastic textures (`leather`), where local reconstruction errors clearly highlight cuts and pokes. However, they struggle with rigid, oriented assemblies (`transistor`), where slight positional misalignment creates large reconstruction residuals. Deep feature embedding methods (PatchCore, EfficientAD) resolve this by comparing semantic features invariant to micro-misalignments.
3. **CPU vs. GPU Latency**: Distillation architectures like EfficientAD run at ~48 ms per image on standard CPU and drop below 5 ms on a GPU, making them the preferred candidate for edge and real-time production line deployment.
