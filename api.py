from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

# --- Load Model ---
model = joblib.load("xgb_model.pkl")

app = FastAPI(
    title="ZenithMind AI - Burnout Predictor API",
    description="Predicts student burnout risk using academic & lifestyle features",
    version="1.0.0"
)

# --- Input Schema ---
class StudentInput(BaseModel):
    cgpa: float                  # 0.0 - 10.0
    sleep_hours: float           # 0 - 12
    exam_pressure: int           # 1 - 5 scale
    anxiety_level: int           # 1 - 5 scale
    screen_time: float           # hours per day
    study_hours: float           # hours per day
    extracurricular: int         # 0 = No, 1 = Yes

# --- Root Endpoint ---
@app.get("/")
def home():
    return {
        "message": "Welcome to ZenithMind AI API 🧠",
        "author": "Yash",
        "docs": "/docs"
    }

# --- Predict Endpoint ---
@app.post("/predict")
def predict_burnout(data: StudentInput):
    
    input_features = np.array([[
        data.cgpa,
        data.sleep_hours,
        data.exam_pressure,
        data.anxiety_level,
        data.screen_time,
        data.study_hours,
        data.extracurricular
    ]])
    
    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0]
    
    # Risk Label
    risk_map = {0: "Low", 1: "Medium", 2: "High"}
    risk_label = risk_map.get(int(prediction), "Unknown")
    
    confidence = round(float(max(probability)) * 100, 2)
    
    return {
        "burnout_risk": risk_label,
        "confidence_percent": confidence,
        "input_received": data.dict(),
        "advice": get_advice(risk_label)
    }

# --- Advice Function ---
def get_advice(risk: str):
    advice_map = {
        "Low": "Great balance! Keep maintaining healthy study and sleep habits.",
        "Medium": "Warning signs detected. Reduce screen time and take regular breaks.",
        "High": "High burnout risk! Please talk to a counselor and prioritize rest immediately."
    }
    return advice_map.get(risk, "Stay consistent and take care of yourself.")

# --- Run Locally ---
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)