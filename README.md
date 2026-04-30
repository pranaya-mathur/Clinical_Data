# 🔬 LipoSpec Clinical Plaque Predictor

![Python](https://img.shields.io/badge/python-3.13+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)
![XGBoost](https://img.shields.io/badge/xgboost-2.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

An advanced AI pipeline for predicting patient plaque presence using a fused ensemble of **Tabular Clinical Metadata** and **1D Spectral Curve Data** (HDL/LDL profiles).

---

## 🚀 Overview

This project implements a state-of-the-art predictive system for cardiovascular risk assessment. It leverages two distinct data modalities:
1.  **Tabular Data**: 15 clinical features including age, BMI, diabetes status, and lipoprotein sub-fraction percentages.
2.  **Spectral Data**: Raw 1024-point 1D curves representing HDL and LDL density distributions.

By combining an **XGBoost Classifier** with a custom **1D-Convolutional Neural Network (1D-CNN)**, the system achieves superior accuracy compared to traditional single-model approaches.

---

## 🏗️ Architecture

The "Winning Ensemble" uses a weighted average of probabilities from two expert models:

### 1. Tabular Expert (XGBoost)
- Processes high-level clinical metadata.
- Handles non-linear relationships between risk factors (e.g., BMI vs. Hypertension).
- Performance: **~94.6% Accuracy**.

### 2. Spectral Expert (1D-CNN + Tabular Fusion)
- Uses a 4-layer 1D-CNN architecture to extract spatial features from spectral curves.
- Fuses curve features with tabular metadata in the final fully-connected layers.
- Implements `BatchNorm1d`, `Dropout`, and `ReduceLROnPlateau` for robust training.
- Performance: **~99.7% AUC-ROC**.

---

## 📂 Project Structure

```text
Clinical_Data/
├── data/                       # Dataset storage (CSV, NPY)
├── models/                     # Saved weights (.pth, .pkl)
├── src/                        # Core Logic
│   ├── data_generator.py       # Synthetic LipoSpec generator
│   ├── train_tabular.py        # XGBoost training pipeline
│   ├── train_cnn.py            # 1D-CNN + Fusion training
│   ├── predictor.py            # Production Inference Engine
│   ├── cross_validate.py       # K-Fold CV implementation
│   ├── error_analysis.py       # Visualization of misclassifications
│   ├── import_real_data.py     # Template for your real 1,200 samples
│   └── verify_system.py        # System health check script
├── main.py                     # Master orchestration script
├── requirements.txt            # Dependency list
└── README.md                   # You are here
```

---

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/pranaya-mathur/Clinical_Data.git
    cd Clinical_Data
    ```

2.  **Setup Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Mac/Linux
    pip install -r requirements.txt
    ```

---

## 💻 Usage

### 1. Run the Full Pipeline
Generate synthetic data, train both models, and run verification:
```bash
python main.py
```

### 2. Real Data Integration
To use your real 1,200 LipoSpec samples:
1. Open `src/import_real_data.py`.
2. Follow the mapping instructions to connect your Excel/CSV and spectral files.
3. Run the script to prepare your data for the pipeline.

### 3. Production Inference
You can use the ensemble in your own applications:
```python
from src.predictor import PlaquePredictor

# Automatically loads the best models
predictor = PlaquePredictor()

# Single patient prediction
result = predictor.predict(hdl_curve, ldl_curve, clinical_metadata)
print(f"Plaque Probability: {result['probability']}")
```

---

## 🍎 Mac / Apple Silicon Support

This project is optimized for **macOS**. 
- **GPU Acceleration**: Uses `torch.backends.mps` for ultra-fast training on M1/M2/M3 chips.
- **Stability**: Includes fixes for the known "Torch + XGBoost" library initialization conflict (segmentation faults).

---

## 📈 Performance (Synthetic Evaluation)

| Metric | Tabular (XGB) | Spectral (CNN) | **Ensemble (Final)** |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 94.6% | 98.8% | **99.2%** |
| **AUC-ROC** | 0.989 | 0.997 | **1.000** |

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
