# Clinical Data Plaque Prediction

This repository contains a full pipeline for predicting clinical plaque using a fused ensemble of tabular XGBoost and 1D-CNN spectral models.

## Folder Structure

- `data/`: Contains synthetic dataset files (`.csv`, `.npy`).
- `models/`: Contains trained model weights (`.pkl`, `.pth`).
- `src/`: Core logic for data generation, training, and the production predictor.
- `main.py`: Master script to run the entire pipeline.
- `requirements.txt`: Python dependencies.

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run Full Pipeline
To generate data, train both models, and verify the ensemble:
```bash
python main.py
```

### Use the Predictor in your code
```python
from src.predictor import PlaquePredictor

predictor = PlaquePredictor()
result = predictor.predict(hdl_curve, ldl_curve, metadata_dict)
print(result['has_plaque'])
```

## Technology Stack
- **XGBoost**: For tabular clinical metadata.
- **PyTorch (1D-CNN)**: For spectral curve feature extraction.
- **Ensemble**: Weighted average of probabilities for maximum robustness.
- **MPS Support**: Automatically uses Apple Silicon GPU for faster CNN training.
