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

@app.post("/predict")
async def predict(
    hdl_file: UploadFile = File(...),
    ldl_file: UploadFile = File(...),
    metadata: str = Form(...)
):
    try:
        predictor = get_predictor()
        # Load curves
        hdl_curve = np.load(hdl_file.file)
        ldl_curve = np.load(ldl_file.file)
        
        # Parse metadata
        meta_dict = json.loads(metadata)
        
        # Run prediction
        result = predictor.predict(hdl_curve, ldl_curve, meta_dict)
        
        # Prepare SHAP-style mock data for demo (highlighting top regions)
        # In a real app, we'd run SHAP, but for now we'll simulate the "colored bands"
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
            "highlights": shap_highlights
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/samples")
def get_samples():
    """Returns sample data for demo mode"""
    try:
        meta_df = pd.read_csv("data/synthetic_lipospec_dataset_metadata.csv")
        return meta_df.head(5).to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
