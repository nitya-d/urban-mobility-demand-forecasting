"""
Train and save the XGBoost model for API deployment.

Run this script before starting the API:
    python train_model.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from xgboost import XGBRegressor


def create_features(df):
    """Create temporal features for demand forecasting."""
    df = df.copy()
    
    # Basic temporal features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_month'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['week_of_year'] = df['datetime'].dt.isocalendar().week.astype(int)
    
    # Binary features
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Lag features
    df['demand_lag_1h'] = df['demand'].shift(1)
    df['demand_lag_24h'] = df['demand'].shift(24)
    df['demand_lag_168h'] = df['demand'].shift(168)
    
    # Rolling averages
    df['demand_rolling_24h'] = df['demand'].shift(1).rolling(window=24).mean()
    df['demand_rolling_7d'] = df['demand'].shift(1).rolling(window=168).mean()
    
    return df


def main():
    print("Loading data...")
    df = pd.read_parquet("data/journeys_2019_2020_2021.parquet")
    
    # Remove outliers
    df = df[df['Duration'] < 1440].copy()
    
    # Aggregate to hourly demand
    df['hour_bucket'] = df['Start Date'].dt.floor('h')
    hourly_demand = df.groupby('hour_bucket').size().reset_index(name='demand')
    hourly_demand.columns = ['datetime', 'demand']
    
    print(f"Created {len(hourly_demand):,} hourly observations")
    
    # Create features
    hourly_demand = create_features(hourly_demand)
    hourly_demand = hourly_demand.dropna()
    
    # Define features
    feature_cols = [
        'hour', 'day_of_week', 'day_of_month', 'month', 'year', 'week_of_year',
        'is_weekend', 'is_rush_hour',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
        'demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h',
        'demand_rolling_24h', 'demand_rolling_7d'
    ]
    
    X = hourly_demand[feature_cols]
    y = hourly_demand['demand']
    
    # Train on all data for deployment
    print("Training XGBoost model...")
    model = XGBRegressor(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=1
    )
    model.fit(X, y)
    
    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "xgboost_demand.joblib"
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    # Quick validation
    sample_pred = model.predict(X.head(1))[0]
    print(f"✓ Sample prediction: {sample_pred:.0f} journeys/hour")
    print("\nReady to start API: uvicorn app:app --reload")


if __name__ == "__main__":
    main()
