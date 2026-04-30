import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import torch
import joblib
import sys

# Add src to path
sys.path.append(os.path.dirname(__file__))
from train_cnn import ImprovedPlaque1DCNN, PlaqueDataset, DEVICE
from torch.utils.data import DataLoader

def cross_validate_pipeline(n_splits=5):
    print(f"🔄 Starting {n_splits}-Fold Cross-Validation...")
    
    # Load Data
    metadata_df = pd.read_csv("data/synthetic_lipospec_dataset_metadata.csv")
    hdl_curves = np.load("data/hdl_curves.npy")
    ldl_curves = np.load("data/ldl_curves.npy")
    
    feature_cols = ["age","sex_male","bmi","diabetes","hypertension","smoking",
                    "hdl2b_percent","hdl2a_percent","hdl3a_percent","hdl3b_percent",
                    "hdl3c_percent","sdldl_percent","total_hdl","total_ldl","hdl_ldl_ratio"]
    
    X_tab = metadata_df[feature_cols].values
    X_curves = np.stack([hdl_curves, ldl_curves], axis=1)
    y = metadata_df["has_plaque"].values
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_tab, y)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        
        # Split
        X_tab_train, X_tab_val = X_tab[train_idx], X_tab[val_idx]
        X_cur_train, X_cur_val = X_curves[train_idx], X_curves[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 1. Train XGBoost
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        xgb_model.fit(X_tab_train, y_train)
        xgb_proba = xgb_model.predict_proba(X_tab_val)[:, 1]
        
        # 2. Train CNN (Small number of epochs for CV)
        cnn_model = ImprovedPlaque1DCNN(num_tab=15).to(DEVICE)
        train_loader = DataLoader(PlaqueDataset(X_cur_train, X_tab_train, y_train), batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(10): # Reduced epochs for CV speed
            cnn_model.train()
            for c, t, l in train_loader:
                c, t, l = c.to(DEVICE), t.to(DEVICE), l.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(cnn_model(c, t), l)
                loss.backward()
                optimizer.step()
        
        cnn_model.eval()
        with torch.no_grad():
            c_val = torch.tensor(X_cur_val, dtype=torch.float32).to(DEVICE)
            t_val = torch.tensor(X_tab_val, dtype=torch.float32).to(DEVICE)
            cnn_logits = cnn_model(c_val, t_val)
            cnn_proba = torch.softmax(cnn_logits, dim=1)[:, 1].cpu().numpy()
            
        # 3. Ensemble
        ensemble_proba = (xgb_proba + cnn_proba) / 2
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        acc = accuracy_score(y_val, ensemble_pred)
        auc = roc_auc_score(y_val, ensemble_proba)
        
        print(f"Fold Result: Accuracy={acc:.4f}, AUC={auc:.4f}")
        results.append(auc)

    print(f"\n✅ Mean AUC-ROC across folds: {np.mean(results):.4f} (+/- {np.std(results):.4f})")

if __name__ == "__main__":
    cross_validate_pipeline()
