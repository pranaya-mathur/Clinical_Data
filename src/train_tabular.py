import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import os

def train_tabular_model(data_dir="data", model_dir="models"):
    print("📈 Training Tabular XGBoost Model...")
    os.makedirs(model_dir, exist_ok=True)
    
    metadata_df = pd.read_csv(os.path.join(data_dir, "synthetic_lipospec_dataset_metadata.csv"))
    
    feature_cols = [
        "age", "sex_male", "bmi", "diabetes", "hypertension", "smoking",
        "hdl2b_percent", "hdl2a_percent", "hdl3a_percent", "hdl3b_percent", "hdl3c_percent",
        "sdldl_percent", "total_hdl", "total_ldl", "hdl_ldl_ratio"
    ]
    
    X = metadata_df[feature_cols]
    y = metadata_df["has_plaque"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, eval_metric="auc", random_state=42
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"✅ AUC-ROC:  {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    model_path = os.path.join(model_dir, "plaque_predictor_xgboost.pkl")
    joblib.dump(model, model_path)
    print(f"💾 Model saved to '{model_path}'")

if __name__ == "__main__":
    train_tabular_model()
