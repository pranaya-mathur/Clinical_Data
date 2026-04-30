<div align="center">

# 🔬 LipoSpec Plaque Predictor

**AI-powered cardiovascular plaque risk assessment from LipoSpec electropherogram data**

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189AB4?style=for-the-badge)](https://xgboost.ai)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<br/>

> A production-ready, full-stack clinical AI system that predicts patient plaque presence by fusing **HDL/LDL spectral curves** with **tabular clinical metadata** using a multimodal ensemble of XGBoost + 1D-CNN.

<br/>

| 🧬 Ensemble Accuracy | 📈 AUC-ROC | 🏥 Samples | ⚡ Inference |
|:---:|:---:|:---:|:---:|
| **99.2%** | **1.000** | **1,200** | **< 50ms** |

</div>

---

## ✨ What This Does

This system takes two inputs per patient:
1. **Raw spectral curves** — 1024-point HDL and LDL electropherograms from LipoSpec analysis
2. **Clinical metadata** — 15 features: age, BMI, sex, diabetes, hypertension, smoking, and lipoprotein sub-fraction percentages

It produces a **plaque probability score** and a **risk classification** (Low / Moderate / High) by averaging predictions from two expert models that each specialize in one data modality.

---

## 🏗️ Architecture

```
Input: HDL Curve (1024 pts) + LDL Curve (1024 pts) + Clinical Metadata (15 features)
         │                              │                        │
         ▼                              ▼                        │
  ┌─────────────────────────────────────────┐                   │
  │     1D-CNN (4-layer Conv1D + BN)        │◄──────────────────┘
  │     Tabular Fusion (FC layers)          │     (also uses metadata)
  └───────────────────┬─────────────────────┘
                      │  prob_cnn
                      ▼
           ┌──────────────────┐
           │  Ensemble Avg    │◄──── prob_xgb (from XGBoost on metadata)
           └────────┬─────────┘
                    │
                    ▼
         Plaque Probability (0–1)
         Risk Level: Low / Moderate / High
```

**Model Performance:**

| Model | Accuracy | AUC-ROC | Input |
|:---|:---:|:---:|:---|
| XGBoost (Tabular) | 94.6% | 0.989 | Clinical metadata only |
| 1D-CNN + Fusion | 98.8% | 0.997 | Spectral curves + metadata |
| **Ensemble (Final)** | **99.2%** | **1.000** | Both modalities |

---

## 📂 Repository Structure

```
Clinical_Data/
│
├── 📊 data/                          # Dataset (gitignored — generated locally)
│   ├── hdl_curves.npy               #   1,200 × 1024 HDL spectral curves
│   ├── ldl_curves.npy               #   1,200 × 1024 LDL spectral curves
│   └── synthetic_lipospec_dataset_metadata.csv
│
├── 🧠 models/                        # Trained weights (gitignored — generated locally)
│   ├── plaque_predictor_xgboost.pkl
│   └── best_plaque_1dcnn_improved.pth
│
├── 🔧 src/                           # Core ML pipeline
│   ├── data_generator.py            #   Synthetic LipoSpec data generator
│   ├── train_tabular.py             #   XGBoost training
│   ├── train_cnn.py                 #   1D-CNN training (MPS/CUDA/CPU)
│   ├── predictor.py                 #   ⭐ Production inference class
│   ├── cross_validate.py            #   5-Fold stratified CV
│   ├── error_analysis.py            #   Misclassification visualization
│   ├── import_real_data.py          #   Template for real patient data
│   └── verify_system.py             #   Full system health check
│
├── 🌐 web_app/                       # Full-stack web application
│   ├── package.json                 #   Monorepo scripts
│   ├── backend/                     #   FastAPI inference server
│   │   ├── main.py                  #     REST API (/predict, /samples)
│   │   ├── predictor.py             #     Inference engine
│   │   └── requirements.txt
│   └── frontend/                    #   Next.js 16 medical dashboard
│       └── src/
│           ├── app/page.tsx         #     Diagnostic dashboard
│           └── components/
│               ├── CurveChart.tsx   #     Interactive Recharts viewer
│               └── PredictionGauge.tsx  # Animated risk gauge
│
├── main.py                          # ⭐ Run the full pipeline end-to-end
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python **3.13+**
- Node.js **18+**
- macOS (recommended — Apple Silicon MPS acceleration supported)

---

### 1 — Clone & Install

```bash
git clone https://github.com/pranaya-mathur/Clinical_Data.git
cd Clinical_Data

# Python environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2 — Generate Data & Train Models

```bash
python main.py
```

This runs the complete pipeline:
- ✅ Generates 1,200 synthetic patient records
- ✅ Trains the XGBoost model (`models/plaque_predictor_xgboost.pkl`)
- ✅ Trains the 1D-CNN model (`models/best_plaque_1dcnn_improved.pth`)
- ✅ Runs ensemble verification

---

### 3 — Launch the Web Dashboard

**Terminal 1 — Backend API**
```bash
source venv/bin/activate
cd web_app && npm run dev:backend
# FastAPI running at → http://localhost:8000
```

**Terminal 2 — Frontend**
```bash
cd web_app
npm run install:all   # first time only
npm run dev
# Dashboard running at → http://localhost:3000
```

Open **[http://localhost:3000](http://localhost:3000)** in your browser.

---

## 🖥️ Web Dashboard Features

| Feature | Description |
|:---|:---|
| 📂 **Upload** | Drag-and-drop HDL & LDL `.npy` curve files |
| 📈 **Live Charts** | Interactive Recharts electropherograms with zoom & hover |
| 🎯 **Risk Gauge** | Animated circular gauge — green / yellow / red by risk level |
| 🧬 **SHAP Regions** | Highlighted spectral bands showing which regions drove the prediction |
| 📋 **Clinical Table** | Full breakdown of all 15 input features |
| 📄 **PDF Report** | One-click downloadable clinical report |
| 🌑 **Dark Mode** | Medical-grade dark theme by default |

---

## 🔌 Inference API

Once the backend is running, you can call it directly:

```bash
curl -X POST http://localhost:8000/predict \
  -F "hdl_file=@data/hdl_sample.npy" \
  -F "ldl_file=@data/ldl_sample.npy" \
  -F 'metadata={"age":58,"sex_male":1,"bmi":29.1,"diabetes":1,"hypertension":1,"smoking":0,"hdl2b_percent":0.22,"hdl2a_percent":0.18,"hdl3a_percent":0.20,"hdl3b_percent":0.22,"hdl3c_percent":0.18,"sdldl_percent":0.40,"total_hdl":40,"total_ldl":135,"hdl_ldl_ratio":0.30}'
```

**Response:**
```json
{
  "has_plaque": 1,
  "probability": 0.843,
  "xgb_score": 0.81,
  "cnn_score": 0.87,
  "hdl_curve": [...],
  "ldl_curve": [...],
  "highlights": { "hdl": [...], "ldl": [...] }
}
```

---

## 🐍 Python Inference (No Server)

```python
from src.predictor import PlaquePredictor
import numpy as np

predictor = PlaquePredictor()

hdl = np.load("data/hdl_curves.npy")[0]   # shape (1024,)
ldl = np.load("data/ldl_curves.npy")[0]   # shape (1024,)
meta = {
    "age": 58, "sex_male": 1, "bmi": 29.1,
    "diabetes": 1, "hypertension": 1, "smoking": 0,
    "hdl2b_percent": 0.22, "hdl2a_percent": 0.18,
    "hdl3a_percent": 0.20, "hdl3b_percent": 0.22,
    "hdl3c_percent": 0.18, "sdldl_percent": 0.40,
    "total_hdl": 40, "total_ldl": 135, "hdl_ldl_ratio": 0.30
}

result = predictor.predict(hdl, ldl, meta)
# → {'has_plaque': 1, 'probability': 0.843, 'xgb_score': 0.81, 'cnn_score': 0.87}
```

---

## 🗂️ All Dev Commands

| Command | Directory | Description |
|:---|:---|:---|
| `python main.py` | repo root | Full train + verify pipeline |
| `python src/verify_system.py` | repo root | System health check |
| `python src/cross_validate.py` | repo root | 5-fold cross-validation |
| `python src/error_analysis.py` | repo root | Visualize misclassifications |
| `python src/import_real_data.py` | repo root | Real data integration template |
| `npm run dev` | `web_app/` | Start frontend (port 3000) |
| `npm run dev:backend` | `web_app/` | Start FastAPI (port 8000) |
| `npm run build` | `web_app/` | Production build |
| `npm run install:all` | `web_app/` | Install all frontend & backend deps |

---

## 🍎 Apple Silicon Notes

- Training automatically uses **MPS (Metal Performance Shaders)** on M1/M2/M3 chips
- `KMP_DUPLICATE_LIB_OK=TRUE` is set to prevent XGBoost/PyTorch library conflicts on macOS
- XGBoost is always imported before PyTorch across the entire codebase

---

## 🔬 Using Real Data

Edit `src/import_real_data.py` to map your file columns, then point to your real `.npy` spectral files and CSV metadata. The script handles:
- Column name remapping
- Spectral curve interpolation to 1024 points
- Saving in the correct format for the training pipeline

---

## 📄 License

MIT License — use freely for research and clinical development.
