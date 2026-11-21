# Project Progress & Experiment Log

## Phase 0: Pipeline Audit & Baseline Verification
- Verified MVTec AD dataset presence for `leather` (245 train, 124 test, 92 ground truth) and `transistor` (213 train, 100 test, 40 ground truth).
- Current v1 pipeline uses a Convolutional Autoencoder (symmetric encoder-decoder down to 8x8 latent space).
- Trained strictly on defect-free ("good") samples with composite 0.6*(1-SSIM) + 0.4*MSE reconstruction loss.
- Anomaly maps are computed as 0.5*L2 + 0.5*(1-SSIM); image anomaly score is the mean of top 1% pixels.
- Decision threshold is selected via Youden's J statistic directly on test labels (optimistic oracle threshold).
- Baseline evaluation confirmed end-to-end on Python 3.13:
  - Leather: Image AUROC 0.7137, Pixel AUROC 0.7719, F1 0.7389 (Fold 88.2%, Poke 72.2%, Cut 57.9%, Glue 52.6%, Color 47.4%).
  - Transistor: Image AUROC 0.5950, Pixel AUROC 0.7260, F1 0.4643 (Misplaced 80.0%, Bent Lead 40.0%, Cut Lead 10.0%, Damaged Case 0.0%).
- Files audited: `src/model.py`, `src/dataset.py`, `src/losses.py`, `src/train.py`, `src/evaluate.py`, `src/utils.py`, `src/infer.py`, `config.yaml`, `main.py`.
- How to reproduce: `python main.py evaluate --category leather` and `python main.py evaluate --category transistor`.


## Phase 1: Evaluation Protocol & Anomaly Map Refinements (CAE v1.1)
- **What Changed**:
  - Implemented zero test-leakage validation thresholding protocol: holds out ~20% of `train/good` images (deterministic seed=42) and computes the 99.0th percentile of normal validation anomaly scores as the operating decision threshold.
  - Retained test-set Youden's J statistic threshold as a clearly labeled `oracle_threshold` and `oracle_f1_score` for historical reference.
  - Added configurable 2D Gaussian smoothing ($\sigma=4.0$) on anomaly maps prior to image score computation.
  - Made `top_k_frac` category-configurable (`0.005` for fine-grained textures like `leather`, `0.02` for rigid objects like `transistor`) following validation/sensitivity analysis.
  - Implemented comprehensive evaluation metrics in `src/metrics.py`: Image AUROC, Image Average Precision (AP), Pixel AUROC, Pixel AUPRO (Per-Region Overlap up to FPR=0.30), Precision/Recall/F1 at validation threshold, and per-defect detection rates.
  - Created modular shared evaluation engine in `src/evaluator.py` capable of evaluating any future architecture (PatchCore, EfficientAD, DRAEM) through a unified interface.
- **Files Touched**:
  - `src/metrics.py` (New: AUROC, AP, AUPRO, F1, Youden's J)
  - `src/evaluator.py` (New: shared zero-leakage evaluation engine and artifact generation)
  - `src/evaluate.py` (Updated to leverage shared evaluation layer and sync outputs)
  - `src/infer.py` (Updated to apply Gaussian smoothing and category-specific `top_k_frac`)
  - `config.yaml` (Updated with validation threshold, sigma, and category top-k parameters)
- **Metrics Before vs After (CAE)**:

| Category | Metric | Before (Phase 0 / Baseline) | After (Phase 1) | Delta |
| :--- | :--- | :--- | :--- | :--- |
| **Leather** | Image AUROC | 0.7137 | **0.7874** | +0.0737 |
| | Image AP | N/A | **0.9250** | — |
| | Pixel AUROC | 0.7719 | **0.8458** | +0.0739 |
| | Pixel AUPRO (FPR 0.3) | N/A | **0.7174** | — |
| | Validation F1 (Val Th: 0.3078) | N/A (leaked test Youden) | **0.8173** (Prec: 0.733, Rec: 0.924) | — |
| | Oracle Youden F1 | 0.7389 | **0.8049** | +0.0660 |
| | Defect Detection (Val Th) | Cut 58%, Fold 88%, Poke 72% | **Cut 84%, Fold 100%, Poke 100%, Color 100%, Glue 79%** | Significant gain |
| **Transistor**| Image AUROC | 0.5950 | **0.6392** | +0.0442 |
| | Image AP | N/A | **0.6568** | — |
| | Pixel AUROC | 0.7260 | **0.7784** | +0.0524 |
| | Pixel AUPRO (FPR 0.3) | N/A | **0.5990** | — |
| | Validation F1 (Val Th: 0.2890) | N/A (leaked test Youden) | **0.4776** (Prec: 0.593, Rec: 0.400) | — |
| | Oracle Youden F1 | 0.4643 | **0.4906** | +0.0263 |
| | Defect Detection (Val Th) | Bent 40%, Cut 10%, Damaged 0% | **Bent 50%, Cut 30%, Damaged 0%, Misplaced 80%** | Gains on small leads |

- **How to Reproduce**:
  `python main.py evaluate --category leather`
  `python main.py evaluate --category transistor`


## Phase 2: PatchCore Implementation & Benchmark
- **What Changed**:
  - Implemented full PatchCore architecture (`src/models/patchcore.py`) natively in PyTorch:
    - Pretrained, frozen WideResNet-50-2 backbone.
    - Multi-scale patch feature extraction at `layer2` (stride 8) and `layer3` (stride 16) with spatial average pooling neighborhood aggregation ($3 \times 3$, stride 1, padding 1).
    - Bilinear upsampling of `layer3` to match `layer2` spatial dimensions and channel concatenation ($D = 512 + 1024 = 1536$).
    - Greedy minimax k-center coreset subsampling via Johnson-Lindenstrauss random projection to build an efficient memory bank ($\approx 1500$ patch vectors).
    - Exact nearest-neighbor $L_2$ patch scoring, spatial map reshaping, and bilinear interpolation back to $256 \times 256$ input resolution.
  - Implemented CLI runner `src/run_patchcore.py` with `--mode {train,evaluate,all}`.
  - Integrated with the unified zero-leakage evaluation layer (`src/evaluator.py`).
  - Extracted features and fitted memory banks for `leather` (250,880 total patches $\to$ 1,500 coreset vectors) and `transistor` (218,112 total patches $\to$ 1,500 coreset vectors).
  - Saved checkpoints under `outputs/checkpoints/patchcore_{category}.pt` and full evaluation artifacts under `outputs/results/patchcore/{category}/`.
- **Files Touched**:
  - `src/models/patchcore.py` (New: pure PyTorch PatchCore implementation)
  - `src/run_patchcore.py` (New: PatchCore CLI training/eval runner)
  - `outputs/checkpoints/patchcore_leather.pt` & `patchcore_transistor.pt`
  - `outputs/results/patchcore/leather/` & `outputs/results/patchcore/transistor/`
- **Metrics Comparison: CAE v1 Baseline vs PatchCore**:

| Category | Metric | Baseline CAE (Phase 0) | CAE Refined (Phase 1) | **PatchCore (Phase 2)** | Delta vs Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Leather** | Image AUROC | 0.7137 | 0.7874 | **0.9956** | **+0.2819** |
| | Image AP | N/A | 0.9250 | **0.9986** | — |
| | Pixel AUROC | 0.7719 | 0.8458 | **0.9875** | **+0.2156** |
| | Pixel AUPRO | N/A | 0.7174 | **0.9469** | — |
| | Val F1 Score | N/A (leaked) | 0.8173 | **0.9340** | — |
| | Oracle F1 Score | 0.7389 | 0.8049 | **0.9890** | **+0.2501** |
| **Transistor** | Image AUROC | 0.5950 | 0.6392 | **0.9979** | **+0.4029** |
| | Image AP | N/A | 0.6568 | **0.9969** | — |
| | Pixel AUROC | 0.7260 | 0.7784 | **0.9390** | **+0.2130** |
| | Pixel AUPRO | N/A | 0.5990 | **0.8669** | — |
| | Val F1 Score | N/A (leaked) | 0.4776 | **0.9195** | — |
| | Oracle F1 Score | 0.4643 | 0.4906 | **0.9756** | **+0.5113** |

- **Per-Defect-Type Detection Breakdown (Val Threshold)**:

| Category | Defect Type | Baseline CAE | CAE Refined | **PatchCore** |
| :--- | :--- | :--- | :--- | :--- |
| **Leather** | `color` | 47.4% | 100.0% | **100.0%** |
| | `cut` | 57.9% | 84.2% | **100.0%** |
| | `fold` | 88.2% | 100.0% | **100.0%** |
| | `glue` | 52.6% | 78.9% | **100.0%** |
| | `poke` | 72.2% | 100.0% | **100.0%** |
| **Transistor** | `bent_lead` | 40.0% | 50.0% | **100.0%** |
| | `cut_lead` | 10.0% | 30.0% | **100.0%** |
| | `damaged_case` | 0.0% | 0.0% | **100.0%** |
| | `misplaced` | 80.0% | 80.0% | **100.0%** |

- **How to Reproduce**:
  `python -m src.run_patchcore --category leather --mode all`
  `python -m src.run_patchcore --category transistor --mode all`


## Phase 3: Second Benchmark (EfficientAD Implementation & Evaluation)
- **What Changed**:
  - Implemented EfficientAD (`src/models/efficientad.py`):
    - Frozen pretrained WideResNet-50 feature extractor Teacher network.
    - Trainable feedforward Student network predicting teacher representations on normal images.
    - Feature Autoencoder network operating on student features to detect structural and logical anomalies.
    - Combined anomaly map combining Student-Teacher feature divergence ($D_{ST}$) with Feature-Autoencoder reconstruction divergence ($D_{AE}$).
  - Built CLI runner `src/run_efficientad.py` with cosine annealing learning rate schedule and checkpoint persistence.
  - Evaluated on `leather` and `transistor` through the shared zero-leakage evaluation layer (`src/evaluator.py`).
  - Saved checkpoints under `outputs/checkpoints/efficientad_{category}.pt` and artifacts under `outputs/results/efficientad/{category}/`.
- **Files Touched**:
  - `src/models/efficientad.py` (New: Teacher-Student + Feature Autoencoder architecture)
  - `src/run_efficientad.py` (New: EfficientAD CLI training/eval runner)
  - `outputs/checkpoints/efficientad_leather.pt` & `efficientad_transistor.pt`
  - `outputs/results/efficientad/leather/` & `outputs/results/efficientad/transistor/`
- **Benchmark Comparison across Models**:

| Category | Metric | Baseline CAE | CAE Refined | **EfficientAD** | **PatchCore** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Leather** | Image AUROC | 0.7137 | 0.7874 | **0.9001** | **0.9956** |
| | Image AP | N/A | 0.9250 | **0.9691** | **0.9986** |
| | Pixel AUROC | 0.7719 | 0.8458 | **0.9893** | **0.9875** |
| | Pixel AUPRO | N/A | 0.7174 | **0.9698** | **0.9469** |
| | Val F1 Score | N/A | 0.8173 | **0.8517** | **0.9340** |
| | Oracle F1 Score| 0.7389 | 0.8049 | **0.8941** | **0.9890** |
| **Transistor** | Image AUROC | 0.5950 | 0.6392 | **0.7546** | **0.9979** |
| | Image AP | N/A | 0.6568 | **0.7298** | **0.9969** |
| | Pixel AUROC | 0.7260 | 0.7784 | **0.7414** | **0.9390** |
| | Pixel AUPRO | N/A | 0.5990 | **0.5614** | **0.8669** |
| | Val F1 Score | N/A | 0.4776 | **0.4314** | **0.9195** |
| | Oracle F1 Score| 0.4643 | 0.4906 | **0.6667** | **0.9756** |

- **How to Reproduce**:
  `python -m src.run_efficientad --category leather --mode all`
  `python -m src.run_efficientad --category transistor --mode all`


## Phase 4: DRAEM on Leather (Synthetic Anomaly Embedding Benchmark)
- **What Changed**:
  - Implemented DRAEM architecture (`src/models/draem.py`):
    - On-the-fly synthetic pseudo-defect generation: random geometric masks (blobs, cuts, scratches) blended with color-jittered, shifted texture sources.
    - Reconstructive U-Net subnetwork $\mathcal{R}$ trained to restore corrupted inputs to defect-free images via $L_2 + \text{SSIM}$ loss.
    - Discriminative U-Net segmentation subnetwork $\mathcal{S}$ trained on $[I_{corrupted}, \hat{I}_{recon}]$ via combined BCE and Dice loss.
  - Implemented CLI runner `src/run_draem.py` and trained 8 epochs on `leather`.
  - Evaluated on `leather` through the shared zero-leakage evaluation layer (`src/evaluator.py`).
  - Saved checkpoint under `outputs/checkpoints/draem_leather.pt` and evaluation artifacts under `outputs/results/draem/leather/`.
- **Files Touched**:
  - `src/models/draem.py` (New: Synthetic anomaly generator + Reconstructive/Discriminative U-Nets)
  - `src/run_draem.py` (New: DRAEM CLI runner)
  - `outputs/checkpoints/draem_leather.pt`
  - `outputs/results/draem/leather/`
- **Results on Leather**:
  - Image AUROC: **0.8621** (Baseline CAE: 0.7137, CAE Refined: 0.7874)
  - Image Average Precision: **0.9365**
  - Pixel AUROC: **0.8726** (Baseline CAE: 0.7719, CAE Refined: 0.8458)
  - Pixel AUPRO: **0.7354**
  - Validation F1: **0.8364** (Precision: 0.9452, Recall: 0.7500)
  - Defect detection (Val Th): `cut` 100%, `fold` 94.1%, `poke` 88.9%, `glue` 68.4%, `color` 26.3%.
- **How to Reproduce**:
  `python -m src.run_draem --category leather --mode evaluate`


## Phase 5: Unified Cross-Model Benchmark
- **What Changed**:
  - Implemented `src/benchmark.py` and connected CLI entry point `python main.py benchmark`.
  - Aggregated performance across all 4 architectures (`CAE`, `PatchCore`, `EfficientAD`, `DRAEM`) on `leather` and `transistor`.
  - Added automated benchmarking of inference latency (ms per image averaged over multiple runs on CPU) and checkpoint disk footprint (MB).
  - Generated persistent standardized outputs:
    - `outputs/results/benchmark.json`
    - `outputs/results/benchmark.csv`
    - `outputs/results/defect_benchmark.csv` (complete per-defect-type detection matrix)
- **Files Touched**:
  - `src/benchmark.py` (New: benchmark aggregation and latency profiling)
  - `main.py` (Updated to expose `benchmark` CLI mode)
  - `outputs/results/benchmark.json`, `benchmark.csv`, `defect_benchmark.csv`
- **Master Benchmark Results**:

| Model | Category | Image AUROC | Image AP | Pixel AUROC | Pixel AUPRO | Val F1 | Oracle F1 | Inference Latency | Model Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CAE** | Leather | 0.7874 | 0.9250 | 0.8458 | 0.7174 | 0.8173 | 0.8049 | 89.7 ms | 8.43 MB |
| **CAE** | Transistor| 0.6392 | 0.6568 | 0.7784 | 0.5990 | 0.4776 | 0.4906 | 87.1 ms | 8.43 MB |
| **PatchCore** | **Leather** | **0.9956** | **0.9986** | **0.9875** | **0.9469** | **0.9340** | **0.9890** | 128.3 ms | 8.79 MB |
| **PatchCore** | **Transistor**| **0.9979** | **0.9969** | **0.9390** | **0.8669** | **0.9195** | **0.9756** | 136.0 ms | 8.79 MB |
| **EfficientAD** | Leather | 0.9001 | 0.9691 | 0.9893 | 0.9698 | 0.8517 | 0.8941 | **52.3 ms** | **2.47 MB** |
| **EfficientAD** | Transistor| 0.7546 | 0.7298 | 0.7414 | 0.5614 | 0.4314 | 0.6667 | **47.8 ms** | **2.47 MB** |
| **DRAEM** | Leather | 0.8621 | 0.9365 | 0.8726 | 0.7354 | 0.8364 | 0.8639 | 99.0 ms | 8.37 MB |

- **How to Reproduce**:
  `python main.py benchmark`


## Phase 6: Interactive Web Inspection Dashboard (FastAPI + Vanilla JS)
- **What Changed**:
  - Implemented backend FastAPI service in `app/server.py`:
    - `GET /api/models`: list of architectures and categories with checkpoint readiness flags.
    - `GET /api/benchmark`: unified master benchmark dataset.
    - `GET /api/results/{model}/{category}`: individual model evaluation metrics.
    - `GET /api/defects/{model}/{category}`: defect-type breakdown statistics.
    - `GET /api/heatmaps/{model}/{category}`: saved evaluation heatmaps with direct image serving.
    - `POST /api/predict`: multipart upload supporting any model/category, returning real-time score, validation threshold, NORMAL / DEFECT verdict, base64 original/overlay/mask images, and millisecond latency.
    - `GET /api/history`: audit log of inspected parts.
    - Lazy model caching with memory management and graceful 404 handling for missing checkpoints.
  - Built zero-build frontend in `app/static/`:
    - `index.html`, `style.css`, `app.js` with responsive dark-slate industrial UI.
    - Chart.js integration for grouped AUROC comparisons and per-defect detection rate charts.
    - Drag-and-drop file uploader with quick MVTec sample buttons.
    - Side-by-side inspection viewer (Original, Heatmap Overlay, Predicted Mask).
    - Inspection history log with client-side CSV export.
- **Files Touched**:
  - `app/server.py` (New: FastAPI application server)
  - `app/static/index.html` (New: Dashboard structure & layout)
  - `app/static/style.css` (New: Theme, responsive grid, badges)
  - `app/static/app.js` (New: Client-side logic, API calls, Chart.js)
  - `requirements.txt` (Added `fastapi`, `uvicorn`, `python-multipart`, `scipy`)
- **Verification**:
  - Tested all endpoints (`/api/models`, `/api/benchmark`, `/api/predict`, etc.) via live HTTP requests.
  - Confirmed real-time inference on `cut_000` (verdict: DEFECT) and `good_000` (verdict: NORMAL).
- **How to Reproduce**:
  `python -m uvicorn app.server:app --reload --port 8000`
  Navigate to `http://localhost:8000` in browser.


## Phase 7: Documentation, System Polish & Deliverables Cleanup
- **What Changed**:
  - Rewrote and updated `README.md` with full architectural guides, zero-leakage thresholding methodology, master benchmark comparison tables, reproduction commands, dashboard setup, and engineering limitations.
  - Added `.gitignore` to prevent Python bytecode, virtual environments, and cache clutter.
  - Purged all `__pycache__` artifacts from the repository.
  - Verified all acceptance criteria across the repository:
    - Legacy `python main.py` CLI works seamlessly for CAE v1.
    - Added `python main.py benchmark` CLI command.
    - Every model (`CAE`, `PatchCore`, `EfficientAD`, `DRAEM`) evaluated with the identical zero-leakage protocol and unified metric layout.
    - Dashboard tested and confirmed operational on `http://localhost:8000`.
- **Files Touched**:
  - `README.md` (Completely rewritten and updated)
  - `.gitignore` (Created)
  - `docs/PROGRESS.md` (Updated)
