# generate_graphs.py
# Generate TFT graphs from existing trained model (no retraining)

import os
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import TFTModel
from darts.metrics import mae, mse, mape
from darts.dataprocessing.transformers import Scaler
import matplotlib.pyplot as plt
from datetime import datetime
import torch

# Force CPU usage to avoid MPS float64 issues (same as tft_model.py)
torch.set_default_device('cpu')

# Import shared configuration
from config import TFTConfig

config = TFTConfig()

def generate_graphs():
    """Generate TFT graphs from existing trained model"""
    print("📊 Loading existing TFT model and generating graphs...")
    
    # Load data
    val_df = pd.read_csv(os.path.join(config.PROC_DIR, "val_raw.csv"))
    
    # Convert to TimeSeries with same dtype as training
    series_val = TimeSeries.from_dataframe(val_df, time_col="timestamp", freq="min").astype(np.float32)
    
    # Separate target and covariates
    target_val = series_val[config.TARGET]
    covariates_val = series_val.drop_columns([config.TARGET])
    
    # Scale data
    target_scaler = Scaler()
    covariates_scaler = Scaler()
    
    target_val_scaled = target_scaler.fit_transform(target_val)
    covariates_val_scaled = covariates_scaler.fit_transform(covariates_val)
    
    # Load existing model
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print("❌ No trained TFT model found. Run tft_model.py first.")
        return
    
    print("🔍 Loading trained model...")
    model = TFTModel.load(config.MODEL_SAVE_PATH)
    
    # Generate forecasts
    print("📈 Generating forecasts...")
    forecast = model.historical_forecasts(
        series=target_val_scaled,
        past_covariates=covariates_val_scaled,
        future_covariates=covariates_val_scaled,
        forecast_horizon=config.OUTPUT_HORIZON,
        stride=config.OUTPUT_HORIZON,
        retrain=False,
        verbose=True
    )
    
    # Align ground truth
    y_true = target_val_scaled.slice_intersect(forecast)
    
    # Unscale for plotting
    y_true_unscaled = target_scaler.inverse_transform(y_true)
    forecast_unscaled = target_scaler.inverse_transform(forecast)
    
    # Extract quantiles
    if forecast_unscaled.n_samples > 1:
        forecast_p10 = forecast_unscaled.quantile_timeseries(0.1)
        forecast_p50 = forecast_unscaled.quantile_timeseries(0.5)
        forecast_p90 = forecast_unscaled.quantile_timeseries(0.9)
    else:
        forecast_p10 = forecast_unscaled
        forecast_p50 = forecast_unscaled
        forecast_p90 = forecast_unscaled

    # --- Quick Diagnostic Block ---
    print("Forecast P50 time index (first 10):", forecast_p50.time_index[:10])
    print("True time index (first 10):", y_true_unscaled.time_index[:10])

    print("Forecast P50 values (first 10):", forecast_p50.values()[:10].flatten())
    print("True values (first 10):", y_true_unscaled.values()[:10].flatten())

    # Check if time indices are equal
    if np.array_equal(forecast_p50.time_index[:len(y_true_unscaled)], y_true_unscaled.time_index):
        print("✅ Time indices are aligned.")
    else:
        print("❌ Time indices are NOT aligned!")

    # Check the spread between quantiles
    print("P90 - P10 spread (max, mean):", 
          np.max(forecast_p90.values() - forecast_p10.values()), 
          np.mean(forecast_p90.values() - forecast_p10.values()))
    # --- End Diagnostic Block ---
    
    print("🎨 Generating plots...")
    
    # Plot 1: Main forecast with uncertainty bands
    plt.figure(figsize=(16, 8))
    
    y_true_unscaled.plot(label="True", linewidth=2, color='black')
    forecast_p50.plot(label="TFT P50 (Median)", linestyle="--", color='blue', linewidth=2)
    
    # Add uncertainty bands
    plt.fill_between(
        forecast_p10.time_index, 
        forecast_p10.values().flatten(), 
        forecast_p90.values().flatten(), 
        alpha=0.3, 
        color='lightblue', 
        label='P10-P90 Uncertainty Band'
    )
    
    plt.legend(fontsize=12)
    plt.title("TFT Probabilistic Forecast vs True (Validation Set - Unscaled)", fontsize=14, fontweight='bold')
    plt.ylabel("Sensor 1 Value (°C)", fontsize=12)
    plt.xlabel("Time", fontsize=12)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "pred_vs_actual_tft.png"), dpi=300, bbox_inches='tight')
    print("✅ Saved: pred_vs_actual_tft.png")
    
    # Plot 2: Detailed quantile plot
    plt.figure(figsize=(16, 10))
    
    y_true_unscaled.plot(label="True", linewidth=3, color='black')
    forecast_p10.plot(label="P10 (Lower Bound)", linestyle="-", alpha=0.8, color='red', linewidth=2)
    forecast_p50.plot(label="P50 (Median)", linestyle="-", color='blue', linewidth=3)
    forecast_p90.plot(label="P90 (Upper Bound)", linestyle="-", alpha=0.8, color='green', linewidth=2)
    
    # Add uncertainty bands
    plt.fill_between(
        forecast_p10.time_index, 
        forecast_p10.values().flatten(), 
        forecast_p90.values().flatten(), 
        alpha=0.4, 
        color='lightblue', 
        label='P10-P90 Uncertainty Range'
    )
    
    plt.legend(fontsize=12, loc='upper left')
    plt.title("TFT Quantile Predictions vs True (Validation Set)", fontsize=14, fontweight='bold')
    plt.ylabel("Sensor 1 Value (°C)", fontsize=12)
    plt.xlabel("Time", fontsize=12)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "pred_vs_actual_tft_quantiles.png"), dpi=300, bbox_inches='tight')
    print("✅ Saved: pred_vs_actual_tft_quantiles.png")
    
    # Calculate and print metrics
    mae_val = mae(y_true_unscaled, forecast_p50)
    mse_val = mse(y_true_unscaled, forecast_p50)
    mape_val = mape(y_true_unscaled, forecast_p50)
    
    print(f"\n📊 Quick Metrics:")
    print(f"MAE: {mae_val:.4f}")
    print(f"MSE: {mse_val:.4f}")
    print(f"MAPE: {mape_val:.2f}%")
    
    print(f"\n🎉 Graphs generated successfully!")
    print(f"📁 Files saved in: {config.RESULTS_DIR}/")

if __name__ == "__main__":
    generate_graphs() 