import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import xgboost as xgb # Must be before torch
import sys

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_generator import run_generation
from train_tabular import train_tabular_model
from train_cnn import train_cnn_model
from predictor import PlaquePredictor
import numpy as np
import pandas as pd

def main():
    print("====================================================")
    print("🔬 CLINICAL DATA PLAQUE PREDICTION PIPELINE")
    print("====================================================")

    # 1. Generate Data
    if not os.path.exists("data/synthetic_lipospec_dataset_metadata.csv"):
        run_generation(output_dir="data")
    else:
        print("⏭️ Data already exists, skipping generation.")

    # 2. Train Tabular Model
    if not os.path.exists("models/plaque_predictor_xgboost.pkl"):
        train_tabular_model(data_dir="data", model_dir="models")
    else:
        print("⏭️ Tabular model already exists, skipping training.")

    # 3. Train CNN Model
    if not os.path.exists("models/best_plaque_1dcnn_improved.pth"):
        train_cnn_model(data_dir="data", model_dir="models", epochs=20)
    else:
        print("⏭️ CNN model already exists, skipping training.")

    # 4. Verify with Predictor
    print("\n🔍 Verifying Final Ensemble...")
    predictor = PlaquePredictor()
    
    # Load a test sample
    meta_df = pd.read_csv("data/synthetic_lipospec_dataset_metadata.csv")
    hdl_curves = np.load("data/hdl_curves.npy")
    ldl_curves = np.load("data/ldl_curves.npy")
    
    # Predict on the first sample
    sample_idx = 0
    sample_meta = meta_df.iloc[sample_idx].to_dict()
    hdl = hdl_curves[sample_idx]
    ldl = ldl_curves[sample_idx]
    
    result = predictor.predict(hdl, ldl, sample_meta)
    
    print("\n--- TEST PREDICTION ---")
    print(f"Sample ID:      {sample_idx}")
    print(f"True Label:     {sample_meta['has_plaque']}")
    print(f"Predicted:      {result['has_plaque']}")
    print(f"Confidence:     {result['probability']*100:.2f}%")
    print(f"Ensemble split: XGB({result['xgb_score']:.2f}) | CNN({result['cnn_score']:.2f})")
    print("-----------------------")

    print("\n🎉 Pipeline complete! Your models are ready in 'models/'.")

if __name__ == "__main__":
    main()
