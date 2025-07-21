# tft_snapshot_generator.py
# Generate CSV snapshots when TFT predicts HCDP > 5°C

import os
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from datetime import datetime
import torch

# Force CPU usage to avoid MPS float64 issues (same as tft_model.py)
torch.set_default_device('cpu')

# Configuration
INPUT_WINDOW = 96
OUTPUT_HORIZON = 30
TARGET = "Sensor 1"
THRESHOLD = 5.40  # HCDP threshold in °C

def load_data():
    """Load test data"""
    test_df = pd.read_csv("data/processed/test_raw.csv")
    return test_df

def load_tft_model():
    """Load trained TFT model"""
    model_path = "results/tft_model.pkl"
    if os.path.exists(model_path):
        return TFTModel.load(model_path)
    else:
        print("❌ TFT model not found. Train the model first.")
        return None

def generate_snapshots():
    """Generate CSV snapshots for HCDP violations"""
    print("🔍 Loading data and model...")
    
    # Load data
    test_df = load_data()
    model = load_tft_model()
    
    if model is None:
        return
    
    # Convert to TimeSeries with same dtype as training
    series = TimeSeries.from_dataframe(test_df, time_col="timestamp", fill_missing_dates=True, freq="T").astype(np.float32)
    
    # Separate target and covariates
    target = series[TARGET]
    covariates = series.drop_columns([TARGET])
    
    # Scale data
    target_scaler = Scaler()
    covariates_scaler = Scaler()
    
    target_scaled = target_scaler.fit_transform(target)
    covariates_scaled = covariates_scaler.fit_transform(covariates)
    
    print("📊 Generating TFT forecasts...")
    
    # Generate forecasts
    forecasts = model.historical_forecasts(
        series=target_scaled,
        past_covariates=covariates_scaled,
        future_covariates=covariates_scaled,
        forecast_horizon=OUTPUT_HORIZON,
        stride=OUTPUT_HORIZON,
        retrain=False,
        verbose=True
    )
    
    # Unscale forecasts
    forecasts_unscaled = target_scaler.inverse_transform(forecasts)
    
    # Get P50 (median) predictions
    if forecasts_unscaled.n_samples > 1:
        forecast_p50 = forecasts_unscaled.quantile_timeseries(0.5)
    else:
        forecast_p50 = forecasts_unscaled
    
    print("🔍 Checking for threshold violations...")
    
    # Find violations
    violations = []
    forecast_values = forecast_p50.values().flatten()
    forecast_times = forecast_p50.time_index
    
    for i, (timestamp, pred_value) in enumerate(zip(forecast_times, forecast_values)):
        if pred_value > THRESHOLD:
            violations.append({
                'index': i,
                'timestamp': timestamp,
                'predicted_hcdp': pred_value
            })
    
    print(f"✅ Found {len(violations)} violations")
    
    # Create snapshots directory
    os.makedirs("snapshots", exist_ok=True)
    
    # Generate CSV for each violation
    for i, violation in enumerate(violations):
        print(f"📄 Creating snapshot {i+1}/{len(violations)}...")
        
        # Calculate time windows
        forecast_start = violation['index'] * OUTPUT_HORIZON
        input_start = max(0, forecast_start - INPUT_WINDOW)
        
        # Extract historical data (input window)
        historical_data = []
        for j in range(input_start, forecast_start):
            if j < len(test_df):
                row = test_df.iloc[j].copy()
                row['data_type'] = 'historical'
                historical_data.append(row)
        
        # Extract forecast data
        forecast_data = []
        for j in range(OUTPUT_HORIZON):
            if forecast_start + j < len(test_df):
                row = test_df.iloc[forecast_start + j].copy()
                row['data_type'] = 'forecasted'
                
                # Replace target with predicted value
                if j < len(forecast_values):
                    row[TARGET] = forecast_values[j]
                
                forecast_data.append(row)
        
        # Combine data
        combined_data = historical_data + forecast_data
        snapshot_df = pd.DataFrame(combined_data)
        
        # Save CSV
        filename = f"snapshot_{i+1:03d}_hcdp_{violation['predicted_hcdp']:.1f}.csv"
        filepath = os.path.join("snapshots", filename)
        snapshot_df.to_csv(filepath, index=False)
        
        print(f"   ✅ Saved: {filename}")
    
    print(f"\n🎉 Generated {len(violations)} snapshots in 'snapshots/' directory")
    print("📁 Files ready for LLM analysis")

if __name__ == "__main__":
    generate_snapshots() 