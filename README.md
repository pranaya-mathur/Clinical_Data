# 🔬 LipoSpec Clinical Plaque Predictor

![Python](https://img.shields.io/badge/python-3.13+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)
![XGBoost](https://img.shields.io/badge/xgboost-2.0+-green.svg)
![Next.js](https://img.shields.io/badge/next.js-16-black.svg)
![FastAPI](https://img.shields.io/badge/fastapi-0.136+-009688.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

A full-stack, production-ready AI pipeline for cardiovascular plaque risk prediction using **LipoSpec electropherogram spectral data** (HDL/LDL curves) and clinical metadata. Features a multimodal ensemble model and a beautiful medical-grade web dashboard.

---

## 🏗️ Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│                     Clinical Data Repo                     │
│                                                            │
│  ┌──────────────┐        ┌──────────────┐                  │
│  │  AI/ML Core  │        │   Web App    │                  │
│  │  (Python)    │        │  (Monorepo)  │                  │
│  │              │        │              │                  │
│  │  data/       │◄──────►│  backend/    │◄─── FastAPI API  │
│  │  models/     │        │  (FastAPI)   │                  │
│  │  src/        │        │              │                  │
│  └──────────────┘        │  frontend/   │◄─── Next.js 16   │
│                          │  (Next.js)   │                  │
│                          └──────────────┘                  │
└────────────────────────────────────────────────────────────┘
```

The ensemble model averages predictions from two expert models:
- **XGBoost** — processes tabular clinical metadata
- **1D-CNN + Tabular Fusion** — processes raw 1024-point HDL/LDL spectral curves + metadata

**Ensemble Performance (Synthetic Dataset, n=1200):**
| Metric | XGBoost | 1D-CNN | **Ensemble** |
|:---|:---:|:---:|:---:|
| Accuracy | 94.6% | 98.8% | **99.2%** |
| AUC-ROC | 0.989 | 0.997 | **1.000** |

---

## 📂 Full Repository Structure

```text
Clinical_Data/
│
├── 📁 data/                              # Dataset storage
│   ├── hdl_curves.npy                   # 1,200 × 1024 HDL spectral curves
│   ├── ldl_curves.npy                   # 1,200 × 1024 LDL spectral curves
│   ├── synthetic_lipospec_dataset_metadata.csv  # Clinical metadata (15 features)
│   └── synthetic_lipospec_full_dataset.npz      # Compressed full dataset
│
├── 📁 models/                            # Trained model weights
│   ├── plaque_predictor_xgboost.pkl     # Saved XGBoost classifier
│   └── best_plaque_1dcnn_improved.pth   # Saved 1D-CNN PyTorch model
│
├── 📁 src/                               # Core AI/ML pipeline
│   ├── data_generator.py                # Synthetic LipoSpec data generator
│   ├── train_tabular.py                 # XGBoost training pipeline
│   ├── train_cnn.py                     # 1D-CNN + Tabular Fusion training
│   ├── predictor.py                     # 🚀 Production Inference Engine
│   ├── cross_validate.py                # 5-Fold Stratified Cross-Validation
│   ├── error_analysis.py                # Misclassification visualization
│   ├── import_real_data.py              # Template for real data integration
│   └── verify_system.py                 # Full system health check
│
├── 📁 web_app/                           # Full-Stack Web Application (Monorepo)
│   ├── package.json                     # Root scripts for dev/build
│   │
│   ├── 📁 backend/                       # Python FastAPI Inference Server
│   │   ├── main.py                      # API endpoints (/predict, /samples)
│   │   ├── predictor.py                 # Inference engine (copied from src/)
│   │   ├── requirements.txt             # FastAPI, uvicorn, etc.
│   │   ├── data/                        # Data files for demo inference
│   │   └── models/                      # Model weights for inference
│   │
│   └── 📁 frontend/                      # Next.js 16 Medical Dashboard
│       ├── src/
│       │   ├── app/
│       │   │   ├── page.tsx             # Main diagnostic dashboard page
│       │   │   ├── layout.tsx           # Root layout with Inter font
│       │   │   └── globals.css          # Global styles (Tailwind v4)
│       │   └── components/
│       │       ├── CurveChart.tsx       # Interactive Recharts HDL/LDL viewer
│       │       └── PredictionGauge.tsx  # Animated circular risk gauge
│       └── tailwind.config.ts           # Tailwind v4 config
│
├── main.py                              # Master pipeline orchestrator
├── generate_synthetic_lipospec_1200_with_metadata.py  # Standalone data script
├── ensemble_plaque_predictor.py         # Standalone ensemble evaluator
├── requirements.txt                     # Python ML dependencies
├── .gitignore                           # Excludes venv/, models/, data/ etc.
└── README.md                            # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.13+
- Node.js 18+
- macOS (Apple Silicon supported with MPS acceleration)

### 1. Clone & Setup

```bash
git clone https://github.com/pranaya-mathur/Clinical_Data.git
cd Clinical_Data
```

### 2. Setup Python Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Generate Data & Train Models

```bash
python main.py
```

This will:
1. Generate 1,200 synthetic LipoSpec patient records
2. Train the XGBoost tabular model
3. Train the 1D-CNN spectral model
4. Run ensemble verification

---

## 🌐 Running the Web Application

The web app lives in `web_app/` and requires **two terminals**.

### Terminal 1 — Backend (FastAPI)

```bash
source venv/bin/activate   # From repo root
cd web_app
npm run dev:backend
# → API running at http://localhost:8000
```

### Terminal 2 — Frontend (Next.js)

```bash
cd web_app
npm run dev
# → Dashboard at http://localhost:3000
```

> **First time only:** Run `npm run install:all` inside `web_app/` to install frontend packages.

---

## 💻 Web App Features

| Feature | Status |
|:---|:---:|
| Upload HDL/LDL `.npy` curve files | ✅ |
| Interactive side-by-side electropherogram charts | ✅ |
| SHAP-style highlighted regions on curves | ✅ |
| Animated circular risk gauge (green/yellow/red) | ✅ |
| Clinical metadata sidebar (age, BMI, comorbidities) | ✅ |
| Clinical summary table with lipoprotein subfractions | ✅ |
| Downloadable PDF report | ✅ |
| FastAPI inference backend | ✅ |
| Apple Silicon MPS acceleration | ✅ |
| Dark medical theme | ✅ |

---

## 🧬 Using the Inference Engine in Your Code

```python
from src.predictor import PlaquePredictor
import numpy as np

# Load the ensemble (auto-detects model paths)
predictor = PlaquePredictor()

# Prepare inputs
hdl_curve = np.load("data/hdl_curves.npy")[0]   # shape: (1024,)
ldl_curve = np.load("data/ldl_curves.npy")[0]   # shape: (1024,)
patient_meta = {
    "age": 58, "sex_male": 1, "bmi": 29.1,
    "diabetes": 1, "hypertension": 1, "smoking": 0,
    "hdl2b_percent": 0.22, "hdl2a_percent": 0.18,
    "hdl3a_percent": 0.20, "hdl3b_percent": 0.22,
    "hdl3c_percent": 0.18, "sdldl_percent": 0.40,
    "total_hdl": 40, "total_ldl": 135, "hdl_ldl_ratio": 0.30
}

result = predictor.predict(hdl_curve, ldl_curve, patient_meta)
print(result)
# → {'has_plaque': 1, 'probability': 0.843, 'xgb_score': 0.81, 'cnn_score': 0.87}
```

---

## 🔬 Integrating Your Real Data

Edit `src/import_real_data.py` to map your columns and file formats, then run it. The script guides you through:
1. Loading your Excel/CSV metadata
2. Mapping column names to the 15 required features
3. Loading raw spectral files (`.txt`, `.csv`, or `.npy`)
4. Interpolating curves to 1024 points if needed

---

## 🍎 Mac / Apple Silicon Notes

- **GPU Training**: Uses `torch.backends.mps` automatically on M1/M2/M3 chips
- **Stability**: `KMP_DUPLICATE_LIB_OK=TRUE` prevents XGBoost/PyTorch library conflicts
- **Import Order**: XGBoost is always imported before PyTorch throughout the codebase

---

## 🛠️ Development Scripts

| Command | Location | Description |
|:---|:---|:---|
| `python main.py` | repo root | Full training pipeline |
| `python src/verify_system.py` | repo root | Health check all components |
| `python src/cross_validate.py` | repo root | 5-fold CV robustness test |
| `python src/error_analysis.py` | repo root | Visualize misclassifications |
| `npm run dev` | `web_app/` | Start frontend (port 3000) |
| `npm run dev:backend` | `web_app/` | Start FastAPI (port 8000) |
| `npm run build` | `web_app/` | Production build |

---

## 📄 License

MIT License — see `LICENSE` for details.
