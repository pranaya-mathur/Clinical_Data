import os
# Fix for Mac segmentation faults - MUST be at the very top
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import xgboost as xgb # Import XGB before Torch
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib

class ImprovedPlaque1DCNN(nn.Module):
    def __init__(self, num_tab=15):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=11, padding=5), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=7, padding=3), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(128, 256, kernel_size=5, padding=2), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(256, 512, kernel_size=3, padding=1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(512 + num_tab, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, curves, tabular):
        cnn_out = self.cnn(curves).squeeze(-1)
        combined = torch.cat([cnn_out, tabular], dim=1)
        return self.fc(combined)

class PlaquePredictor:
    def __init__(self, xgb_path="models/plaque_predictor_xgboost.pkl", cnn_path="models/best_plaque_1dcnn_improved.pth"):
        print("📦 Loading ensemble models...")
        self.xgb_model = joblib.load(xgb_path)
        self.cnn_model = ImprovedPlaque1DCNN(num_tab=15)
        self.cnn_model.load_state_dict(torch.load(cnn_path, map_location="cpu"))
        self.cnn_model.eval()
        print("✅ Models loaded successfully.")

    def predict(self, hdl_curve, ldl_curve, metadata_dict):
        feature_cols = ["age","sex_male","bmi","diabetes","hypertension","smoking",
                        "hdl2b_percent","hdl2a_percent","hdl3a_percent","hdl3b_percent",
                        "hdl3c_percent","sdldl_percent","total_hdl","total_ldl","hdl_ldl_ratio"]
        
        tab_features = np.array([[metadata_dict[col] for col in feature_cols]]).astype(np.float32)
        curves = np.stack([hdl_curve, ldl_curve], axis=0).astype(np.float32)
        curves_t = torch.tensor(curves).unsqueeze(0)
        tab_t = torch.tensor(tab_features)

        xgb_proba = self.xgb_model.predict_proba(tab_features)[0, 1]
        with torch.no_grad():
            cnn_logits = self.cnn_model(curves_t, tab_t)
            cnn_proba = torch.softmax(cnn_logits, dim=1)[0, 1].item()

        final_proba = (xgb_proba + cnn_proba) / 2
        prediction = 1 if final_proba > 0.5 else 0
        
        return {
            "has_plaque": int(prediction),
            "probability": round(float(final_proba), 4),
            "xgb_score": round(float(xgb_proba), 4),
            "cnn_score": round(float(cnn_proba), 4)
        }

# Example usage:
if __name__ == "__main__":
    # Initialize with default paths
    predictor = PlaquePredictor(
        xgb_path="models/plaque_predictor_xgboost.pkl", 
        cnn_path="models/best_plaque_1dcnn_improved.pth"
    )
    
    # Load one real sample from our dataset
    meta_df = pd.read_csv("data/synthetic_lipospec_dataset_metadata.csv")
    hdl_curves = np.load("data/hdl_curves.npy")
    ldl_curves = np.load("data/ldl_curves.npy")
    
    sample_idx = 0
    meta = meta_df.iloc[sample_idx].to_dict()
    hdl = hdl_curves[sample_idx]
    ldl = ldl_curves[sample_idx]
    
    result = predictor.predict(hdl, ldl, meta)
    print(f"\n--- STANDALONE TEST PREDICTION ---")
    print(f"Result: {result}")
    print("----------------------------------")
