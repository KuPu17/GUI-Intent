"""
inference.py

The live FastAPI microservice for the UI Behavior Engine.
Downloads model weights from the Hugging Face CDN on startup and 
serves real-time multimodal predictions on incoming UI telemetry.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from huggingface_hub import hf_hub_download
import torch
import joblib
import numpy as np
import pandas as pd

# Import the neural architecture we defined earlier
from architectures import IntentForecasterGRU_V2

# ==========================================
# 1. PYDANTIC SCHEMAS (Data Validation)
# ==========================================
class ClickEvent(BaseModel):
    timestamp: float
    x: float
    y: float
    zone_id: int  # The UI zone tokenized by the frontend/GMM

class TelemetryPayload(BaseModel):
    user_id: str
    session_id: str
    events: List[ClickEvent]

# ==========================================
# 2. GLOBAL SYSTEM STATE
# ==========================================
app = FastAPI(title="UI Behavior Engine", version="2.0")

# Device configuration (Auto-detects GPU if deployed on heavy compute)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables to hold the models in RAM
friction_model = None
intent_model = None

# HF Repository Info (Change this to your actual HF username)
HF_REPO_ID = "your-username/ui-behavior-engine"

# ==========================================
# 3. SERVER LIFESPAN (The Boot Sequence)
# ==========================================
@app.on_event("startup")
async def load_models():
    """Downloads weights from HF CDN and initializes the engines."""
    global friction_model, intent_model
    
    print("Booting AI Systems Engine...")
    
    try:
        # A. Ignite Branch A: Friction Engine
        print("Fetching Friction Engine (LightGBM)...")
        friction_path = hf_hub_download(repo_id=HF_REPO_ID, filename="friction_lightgbm.pkl")
        friction_model = joblib.load(friction_path)
        
        # B. Ignite Branch B: Intent Engine
        print("Fetching Intent Engine (Multimodal GRU)...")
        intent_path = hf_hub_download(repo_id=HF_REPO_ID, filename="intent_gru_v2.pth")
        
        # Initialize the architecture and load the weights
        intent_model = IntentForecasterGRU_V2(num_zones=12, embedding_dim=32, hidden_dim=64, num_classes=6).to(device)
        intent_model.load_state_dict(torch.load(intent_path, map_location=device))
        intent_model.eval() # Lock weights for inference
        
        print("Engines online and ready for telemetry.")
        
    except Exception as e:
        print(f"CRITICAL BOOT FAILURE: {str(e)}")
        raise e

# ==========================================
# 4. THE INFERENCE PIPELINE
# ==========================================
@app.post("/predict")
async def predict_behavior(payload: TelemetryPayload):
    """
    Ingests a raw array of coordinates, computes kinematics, 
    and returns psychological friction and workflow intent.
    """
    if len(payload.events) < 2:
        raise HTTPException(status_code=400, detail="Not enough data to calculate kinematics.")
    
    # --- Feature Extraction Pipeline ---
    df = pd.DataFrame([event.dict() for event in payload.events])
    df = df.sort_values('timestamp')
    
    # Physics Calculations
    df['dt'] = df['timestamp'].diff().fillna(0.0).clip(lower=0.0)
    df['dx'] = df['x'].diff().fillna(0.0)
    df['dy'] = df['y'].diff().fillna(0.0)
    df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
    
    # Temporal Processing for GRU
    df['log_time'] = np.log1p(df['dt'])
    
    # 1. Friction Features (Branch A)
    total_distance = df['distance'].sum()
    euclidean = np.sqrt((df['x'].iloc[-1] - df['x'].iloc[0])**2 + (df['y'].iloc[-1] - df['y'].iloc[0])**2)
    tortuosity = (total_distance / euclidean) if euclidean > 0 else 1.0
    
    rage_clicks = ((df['dt'] < 1.0) & (df['distance'] < 0.05)).sum()
    max_hesitation = df['dt'].max()
    
    kinematic_vector = [[max_hesitation, rage_clicks, total_distance, tortuosity]]
    
    # 2. Intent Tensors (Branch B)
    # Pad sequences to max length or pass directly if batch size is 1
    zone_tensor = torch.tensor(df['zone_id'].tolist(), dtype=torch.long).unsqueeze(0).to(device)
    time_tensor = torch.tensor(df['log_time'].tolist(), dtype=torch.float).unsqueeze(0).to(device)

    # --- INFERENCE ---
    with torch.no_grad():
        # Predict Friction (1 = Frustrated, 0 = Smooth)
        friction_pred = friction_model.predict(kinematic_vector)[0]
        
        # Predict Intent
        intent_logits = intent_model(zone_tensor, time_tensor)
        predicted_class = torch.argmax(intent_logits, dim=1).item()
        
    # Map class integer back to human-readable string
    intent_map = {
        0: 'Information Retrieval', 1: 'E-Commerce Flow', 
        2: 'Media & Entertainment', 3: 'Productivity & Creation', 
        4: 'Social & Communication', 5: 'System Navigation'
    }

    # --- RETURN JSON RESPONSE ---
    return {
        "session_id": payload.session_id,
        "friction_detected": bool(friction_pred),
        "predicted_intent": intent_map.get(predicted_class, "Unknown"),
        "telemetry_processed": len(df)
    }

if __name__ == "__main__":
    import uvicorn
    # Starts the ASGI server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)