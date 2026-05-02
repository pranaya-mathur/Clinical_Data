from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json
import os
from predictor import PlaquePredictor
import pandas as pd

app = FastAPI(title="Lipospec Plaque Predictor API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        print("📦 Initializing Plaque Predictor (first time)...")
        _predictor = PlaquePredictor(
            xgb_path="models/plaque_predictor_xgboost.pkl",
            cnn_path="models/best_plaque_1dcnn_improved.pth"
        )
    return _predictor

@app.get("/")
def read_root():
    return {"message": "Lipospec Plaque Predictor API is running"}

def generate_insight(prob: float, metadata: dict) -> str:
    risk = "High" if prob > 0.6 else "Moderate" if prob > 0.3 else "Low"
    
    insights = []
    if metadata.get("sdldl_percent", 0) > 0.35:
        insights.append("elevated Small Dense LDL (sdLDL)")
    if metadata.get("hdl2b_percent", 0) < 0.20:
        insights.append("low HDL-2b cardioprotective sub-fractions")
    if metadata.get("diabetes"):
        insights.append("co-existing diabetic metabolic state")
    
    if not insights:
        insights.append("balanced lipoprotein sub-fraction profiles")
    
    insight_text = f"The patient presents with {risk} risk. key drivers include " + ", ".join(insights) + "."
    return insight_text

from typing import Optional

@app.post("/predict")
async def predict(
    hdl_file: Optional[UploadFile] = File(None),
    ldl_file: Optional[UploadFile] = File(None),
    metadata: str = Form(...)
):
    print(f"📥 Received prediction request. Metadata: {metadata[:100]}...")
    try:
        predictor = get_predictor()
        meta_dict = json.loads(metadata)
        
        # Check if it's a demo sample request
        if meta_dict.get("is_sample") or not hdl_file or not ldl_file:
            print("🧪 Processing as demo sample...")
            hdl_curves = np.load("data/hdl_curves.npy")
            ldl_curves = np.load("data/ldl_curves.npy")
            # Use the first sample for the demo
            hdl_curve = hdl_curves[0]
            ldl_curve = ldl_curves[0]
        else:
            # Load from uploaded files
            hdl_curve = np.load(hdl_file.file)
            ldl_curve = np.load(ldl_file.file)
        
        # Parse metadata
        meta_dict = json.loads(metadata)
        
        # Run prediction
        result = predictor.predict(hdl_curve, ldl_curve, meta_dict)
        
        # Prepare SHAP-style mock data for demo
        shap_highlights = {
            "hdl": [
                {"start": 150, "end": 250, "color": "rgba(239, 68, 68, 0.2)", "label": "HDL-2b Peak"},
                {"start": 400, "end": 500, "color": "rgba(59, 130, 246, 0.2)", "label": "HDL-3a Peak"}
            ],
            "ldl": [
                {"start": 650, "end": 750, "color": "rgba(239, 68, 68, 0.2)", "label": "sdLDL Region"}
            ]
        }
        
        return {
            **result,
            "hdl_curve": hdl_curve.tolist(),
            "ldl_curve": ldl_curve.tolist(),
            "highlights": shap_highlights,
            "clinical_insight": generate_insight(result["probability"], meta_dict)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/samples")
def get_samples():
    """Returns sample data for demo mode including actual curves"""
    try:
        meta_df = pd.read_csv("data/synthetic_lipospec_dataset_metadata.csv")
        hdl_curves = np.load("data/hdl_curves.npy")
        ldl_curves = np.load("data/ldl_curves.npy")
        
        samples = []
        # Return 5 samples with full data
        for i in range(min(5, len(meta_df))):
            sample = meta_df.iloc[i].to_dict()
            samples.append({
                "metadata": sample,
                "hdl_curve": hdl_curves[i].tolist(),
                "ldl_curve": ldl_curves[i].tolist()
            })
        return samples
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
