"""
Bike Demand Prediction API

FastAPI endpoint for hourly demand forecasting.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

app = FastAPI(
    title="Bike Demand Forecasting API",
    description="Predict hourly bike demand using XGBoost model trained on TfL Santander Cycles data (2019-2021)",
    version="1.0.0"
)

# Load model on startup
MODEL_PATH = Path("models/xgboost_demand.joblib")
model = None

@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. Run train_model.py first.")


class PredictionRequest(BaseModel):
    """Input features for demand prediction"""
    hour: int = Field(ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    month: int = Field(ge=1, le=12, description="Month (1-12)")
    is_weekend: bool = Field(default=False, description="Is it a weekend?")
    
    # Optional lag features (use historical averages if not provided)
    demand_lag_1h: float | None = Field(default=None, description="Demand from 1 hour ago")
    demand_lag_24h: float | None = Field(default=None, description="Demand from 24 hours ago")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "hour": 8,
                "day_of_week": 0,
                "month": 6,
                "is_weekend": False
            }
        }
    }


class PredictionResponse(BaseModel):
    """Prediction output"""
    predicted_demand: int
    confidence: str
    input_features: dict


def create_features(request: PredictionRequest) -> pd.DataFrame:
    """Transform request into model features"""
    
    # Historical averages (from training data) for missing lag features
    AVG_HOURLY_DEMAND = 1150
    
    features = {
        'hour': request.hour,
        'day_of_week': request.day_of_week,
        'day_of_month': 15,  # mid-month default
        'month': request.month,
        'year': datetime.now().year,
        'week_of_year': 26,  # mid-year default
        'is_weekend': int(request.is_weekend),
        'is_rush_hour': int(request.hour in [7, 8, 9, 17, 18, 19]),
        
        # Cyclical encoding
        'hour_sin': np.sin(2 * np.pi * request.hour / 24),
        'hour_cos': np.cos(2 * np.pi * request.hour / 24),
        'dow_sin': np.sin(2 * np.pi * request.day_of_week / 7),
        'dow_cos': np.cos(2 * np.pi * request.day_of_week / 7),
        'month_sin': np.sin(2 * np.pi * request.month / 12),
        'month_cos': np.cos(2 * np.pi * request.month / 12),
        
        # Lag features (use provided or defaults)
        'demand_lag_1h': request.demand_lag_1h or AVG_HOURLY_DEMAND,
        'demand_lag_24h': request.demand_lag_24h or AVG_HOURLY_DEMAND,
        'demand_lag_168h': AVG_HOURLY_DEMAND,
        'demand_rolling_24h': AVG_HOURLY_DEMAND,
        'demand_rolling_7d': AVG_HOURLY_DEMAND,
    }
    
    return pd.DataFrame([features])


@app.get("/")
def root():
    """API health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "docs": "/docs"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_demand(request: PredictionRequest):
    """
    Predict hourly bike demand.
    
    Returns predicted number of journeys for the specified hour.
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Run train_model.py first."
        )
    
    # Create features
    features_df = create_features(request)
    
    # Predict
    prediction = model.predict(features_df)[0]
    predicted_demand = max(0, int(round(prediction)))
    
    # Simple confidence based on whether lag features were provided
    confidence = "high" if request.demand_lag_1h else "medium"
    
    return PredictionResponse(
        predicted_demand=predicted_demand,
        confidence=confidence,
        input_features=request.model_dump()
    )


@app.get("/predict/peak-hours")
def get_peak_hours():
    """Get typical peak demand hours based on training data analysis"""
    return {
        "weekday_peaks": {
            "morning": {"hours": [7, 8, 9], "typical_demand": "1800-2500 journeys/hour"},
            "evening": {"hours": [17, 18, 19], "typical_demand": "2000-2800 journeys/hour"}
        },
        "weekend_peaks": {
            "afternoon": {"hours": [12, 13, 14, 15], "typical_demand": "1200-1600 journeys/hour"}
        },
        "low_demand": {"hours": [0, 1, 2, 3, 4, 5], "typical_demand": "50-200 journeys/hour"}
    }


@app.get("/model/info")
def get_model_info():
    """Get model metadata and performance metrics"""
    return {
        "model_type": "XGBoost Regressor",
        "training_data": {
            "source": "TfL Santander Cycles",
            "period": "2019-2021",
            "total_journeys": "31M",
            "hourly_observations": "~26,000"
        },
        "performance": {
            "test_r2": 0.966,
            "test_mae": 199.7,
            "test_rmse": 285.3,
            "validation_method": "Time-based split (train: 2019-2020, test: 2021)"
        },
        "features": {
            "temporal": ["hour", "day_of_week", "month", "is_weekend", "is_rush_hour"],
            "cyclical": ["hour_sin/cos", "dow_sin/cos", "month_sin/cos"],
            "lag": ["demand_lag_1h", "demand_lag_24h", "demand_lag_168h"],
            "rolling": ["demand_rolling_24h", "demand_rolling_7d"]
        },
        "model_loaded": model is not None,
        "last_updated": "2026-02"
    }


@app.get("/predict/day")
def predict_full_day(month: int = 6, day_of_week: int = 0):
    """
    Predict demand for all 24 hours of a given day.
    
    - month: 1-12
    - day_of_week: 0=Monday, 6=Sunday
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not (1 <= month <= 12):
        raise HTTPException(status_code=400, detail="Month must be 1-12")
    if not (0 <= day_of_week <= 6):
        raise HTTPException(status_code=400, detail="day_of_week must be 0-6")
    
    is_weekend = day_of_week >= 5
    predictions = []
    
    for hour in range(24):
        request = PredictionRequest(
            hour=hour,
            day_of_week=day_of_week,
            month=month,
            is_weekend=is_weekend
        )
        features_df = create_features(request)
        pred = model.predict(features_df)[0]
        predictions.append({
            "hour": hour,
            "predicted_demand": max(0, int(round(pred)))
        })
    
    return {
        "month": month,
        "day_of_week": day_of_week,
        "day_name": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][day_of_week],
        "is_weekend": is_weekend,
        "hourly_predictions": predictions,
        "peak_hour": max(predictions, key=lambda x: x["predicted_demand"]),
        "total_daily_demand": sum(p["predicted_demand"] for p in predictions)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
