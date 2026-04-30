import os
# Fix for Mac segmentation faults
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import xgboost as xgb
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from predictor import PlaquePredictor

def run_error_analysis(data_dir="data"):
    print("🕵️ Starting Error Analysis...")
    
    # 1. Load data
    meta_df = pd.read_csv(os.path.join(data_dir, "synthetic_lipospec_dataset_metadata.csv"))
    hdl_curves = np.load(os.path.join(data_dir, "hdl_curves.npy"))
    ldl_curves = np.load(os.path.join(data_dir, "ldl_curves.npy"))
    
    # Use the test set (last 240 samples)
    test_indices = range(len(meta_df) - 240, len(meta_df))
    
    predictor = PlaquePredictor()
    
    errors = []
    
    print("🔍 Scanning test set for misclassifications...")
    for i in test_indices:
        meta = meta_df.iloc[i].to_dict()
        hdl = hdl_curves[i]
        ldl = ldl_curves[i]
        
        result = predictor.predict(hdl, ldl, meta)
        
        true_label = meta['has_plaque']
        if result['has_plaque'] != true_label:
            errors.append({
                "index": i,
                "true_label": true_label,
                "predicted_label": result['has_plaque'],
                "probability": result['probability'],
                "xgb_score": result['xgb_score'],
                "cnn_score": result['cnn_score'],
                "hdl": hdl,
                "ldl": ldl,
                "meta": meta
            })

    print(f"✅ Found {len(errors)} misclassifications.")

    # 2. Visualize Errors
    if not errors:
        print("🎉 No errors found! The model is perfect on this test set.")
        return

    for idx, err in enumerate(errors):
        plt.figure(figsize=(12, 5))
        
        # Plot HDL
        plt.subplot(1, 2, 1)
        plt.plot(err['hdl'], color='blue', label='HDL Curve')
        plt.title(f"Sample {err['index']} - HDL\nTrue: {err['true_label']}, Pred: {err['predicted_label']}")
        plt.legend()
        
        # Plot LDL
        plt.subplot(1, 2, 2)
        plt.plot(err['ldl'], color='red', label='LDL Curve')
        plt.title(f"Sample {err['index']} - LDL\nProb: {err['probability']:.4f} (XGB: {err['xgb_score']}, CNN: {err['cnn_score']})")
        plt.legend()
        
        plt.tight_layout()
        save_path = f"error_analysis_sample_{err['index']}.png"
        plt.savefig(save_path)
        print(f"📊 Saved error visualization to {save_path}")
        plt.close()

if __name__ == "__main__":
    # Ensure src is in path for imports
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "."))
    run_error_analysis()
