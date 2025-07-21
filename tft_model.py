# tft_model.py
# ------------------------------------------------------
# Temporal Fusion Transformer (TFT) for HCDP Forecasting

import os
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import TFTModel
from darts.metrics import mae, mse, mape
from darts.utils.likelihood_models import QuantileRegression
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import logging
import joblib
import time
from darts.dataprocessing.transformers import Scaler

import torch
# Force CPU usage to avoid MPS float64 issues
torch.set_default_device('cpu')

# Import shared configuration
from config import TFTConfig

logging.basicConfig(filename=TFTConfig().LOG_PATH, level=logging.INFO)
logging.info(f"[{datetime.now()}] Starting tft_model.py")

start_time = time.time()

config = TFTConfig()

print("Loading data...")
train_df = pd.read_csv(os.path.join(config.PROC_DIR, "train_raw.csv"))
val_df = pd.read_csv(os.path.join(config.PROC_DIR, "val_raw.csv"))
test_df = pd.read_csv(os.path.join(config.PROC_DIR, "test_raw.csv"))

print("Converting to TimeSeries...")
series_train = TimeSeries.from_dataframe(train_df, time_col="timestamp", fill_missing_dates=True, freq="T").astype(np.float32)
series_val = TimeSeries.from_dataframe(val_df, time_col="timestamp", fill_missing_dates=True, freq="T").astype(np.float32)
series_test = TimeSeries.from_dataframe(test_df, time_col="timestamp", fill_missing_dates=True, freq="T").astype(np.float32)

print("Separating target and covariates...")
target_train = series_train[config.TARGET]
target_val = series_val[config.TARGET]
target_test = series_test[config.TARGET]

covariates_train = series_train.drop_columns([config.TARGET])
covariates_val = series_val.drop_columns([config.TARGET])
covariates_test = series_test.drop_columns([config.TARGET])

print("Scaling target and covariates...")
# Scale target series
target_scaler = Scaler()
target_train_scaled = target_scaler.fit_transform(target_train)
target_val_scaled = target_scaler.transform(target_val)

# Scale covariates (critical for TFT performance)
covariates_scaler = Scaler()
covariates_train_scaled = covariates_scaler.fit_transform(covariates_train)
covariates_val_scaled = covariates_scaler.transform(covariates_val)

print("Instantiating model...")
from pytorch_lightning.callbacks import EarlyStopping
model = TFTModel(
    add_relative_index=True,
    input_chunk_length=config.INPUT_WINDOW,
    output_chunk_length=config.OUTPUT_HORIZON,
    hidden_size=config.HIDDEN_SIZE,
    lstm_layers=config.LSTM_LAYERS,
    num_attention_heads=config.NUM_ATTENTION_HEADS,
    dropout=config.DROPOUT,
    batch_size=config.BATCH_SIZE,
    n_epochs=config.MAX_EPOCHS,
    likelihood=QuantileRegression(quantiles=config.QUANTILES),
    optimizer_kwargs={"lr": config.LEARNING_RATE},
    random_state=42,
    force_reset=True,
    pl_trainer_kwargs={
        "accelerator": "mps",
        "gradient_clip_val": 0.5,  # Reduced from 1.0
        "enable_progress_bar": True,
        "enable_model_summary": True,
        "callbacks": [
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ],
    }
)

print("Training model...")
tqdm.write("[INFO] Starting TFT model training...")
tqdm.write("[INFO] This may take several minutes depending on data size and system performance.")
logging.info("Training TFT model...")
fit_start = time.time()

# TFT requires both past and future covariates for optimal performance
model.fit(
    series=target_train_scaled, 
    past_covariates=covariates_train_scaled,
    future_covariates=covariates_train_scaled,  # Use same covariates as future
    val_series=target_val_scaled, 
    val_past_covariates=covariates_val_scaled,
    val_future_covariates=covariates_val_scaled
)

# Get actual epochs completed and reason for stopping
actual_epochs = getattr(model, 'epochs_trained', config.MAX_EPOCHS)
early_stopping_reason = "Completed all epochs"

# Check if early stopping occurred
if actual_epochs < config.MAX_EPOCHS:
    # Try to get early stopping reason from trainer
    try:
        trainer = getattr(model, 'trainer', None)
        if trainer and hasattr(trainer, 'early_stopping_callback'):
            early_stopping_reason = "Early stopping triggered (validation loss stopped improving)"
        else:
            early_stopping_reason = f"Stopped early at epoch {actual_epochs} (reason unknown)"
    except:
        early_stopping_reason = f"Stopped early at epoch {actual_epochs} (early stopping likely)"
else:
    early_stopping_reason = "Completed all planned epochs"

print("Predicting...")
fit_end = time.time()
training_time_minutes = (fit_end - fit_start) / 60
training_time_hours = training_time_minutes / 60
tqdm.write(f"[INFO] Training completed in {training_time_minutes:.2f} minutes ({training_time_hours:.2f} hours)")
logging.info(f"Training completed in {training_time_minutes:.2f} minutes ({training_time_hours:.2f} hours)")
logging.info(f"Actual epochs completed: {actual_epochs}")
logging.info(f"Stopping reason: {early_stopping_reason}")

forecast = model.historical_forecasts(
    series=target_val_scaled,
    past_covariates=covariates_val_scaled,
    future_covariates=covariates_val_scaled,  # Required for TFT
    forecast_horizon=config.OUTPUT_HORIZON,
    stride=config.OUTPUT_HORIZON,
    retrain=False,
    verbose=True
)

# Ground truth: align with forecast timestamps using slice_intersect
y_true = target_val_scaled.slice_intersect(forecast)

print("Evaluating...")
# Unscale for meaningful metrics comparison with other models
y_true_unscaled = target_scaler.inverse_transform(y_true)
forecast_unscaled = target_scaler.inverse_transform(forecast)

# Extract quantiles (P10, P50, P90) from probabilistic forecast
print("Forecast sample shape:", forecast_unscaled.values().shape)
print("Number of samples:", forecast_unscaled.n_samples)
print("Extracting quantile predictions...")
forecast_p10 = forecast_unscaled.quantile_timeseries(0.1)
forecast_p50 = forecast_unscaled.quantile_timeseries(0.5)
forecast_p90 = forecast_unscaled.quantile_timeseries(0.9)
# --- Quick Diagnostic Block ---
print("Forecast P50 length:", len(forecast_p50))
print("True length:", len(y_true_unscaled))
print("Forecast P50 time index (first 10):", forecast_p50.time_index[:10])
print("True time index (first 10):", y_true_unscaled.time_index[:10])

if np.array_equal(forecast_p50.time_index[:len(y_true_unscaled)], y_true_unscaled.time_index):
    print("✅ Time indices are aligned.")
else:
    print("❌ Time indices are NOT aligned!")

# Check the spread between quantiles
print("P90 - P10 spread (max, mean):", 
      np.max(forecast_p90.values() - forecast_p10.values()), 
      np.mean(forecast_p90.values() - forecast_p10.values()))
# --- End Diagnostic Block ---

# Calculate metrics for each quantile
mae_p10 = mae(y_true_unscaled, forecast_p10)
mse_p10 = mse(y_true_unscaled, forecast_p10)
mape_p10 = mape(y_true_unscaled, forecast_p10)

mae_p50 = mae(y_true_unscaled, forecast_p50)  # P50 (median) - most important
mse_p50 = mse(y_true_unscaled, forecast_p50)
mape_p50 = mape(y_true_unscaled, forecast_p50)

mae_p90 = mae(y_true_unscaled, forecast_p90)
mse_p90 = mse(y_true_unscaled, forecast_p90)
mape_p90 = mape(y_true_unscaled, forecast_p90)

# Use P50 (median) as the main metric for comparison with other models
mae_val = mae_p50
mse_val = mse_p50
mape_val = mape_p50

print("Saving metrics...")
print(f"\nTFT Hyperparameters:")
print(f"  Input Window: {config.INPUT_WINDOW}, Output Horizon: {config.OUTPUT_HORIZON}")
print(f"  Hidden Size: {config.HIDDEN_SIZE}, LSTM Layers: {config.LSTM_LAYERS}")
print(f"  Dropout: {config.DROPOUT}")
print(f"  Batch Size: {config.BATCH_SIZE}, Epochs: {config.MAX_EPOCHS}")
print(f"  Actual Epochs Completed: {actual_epochs}")
print(f"  Training Time: {training_time_minutes:.2f} minutes ({training_time_hours:.2f} hours)")
print(f"  Stopping Reason: {early_stopping_reason}")
print(f"  Quantiles: {config.QUANTILES}")

print(f"\nTFT Quantile Results:")
print(f"P10 - MAE: {mae_p10:.4f}, MSE: {mse_p10:.4f}, MAPE: {mape_p10:.2f}%")
print(f"P50 - MAE: {mae_p50:.4f}, MSE: {mse_p50:.4f}, MAPE: {mape_p50:.2f}%")
print(f"P90 - MAE: {mae_p90:.4f}, MSE: {mse_p90:.4f}, MAPE: {mape_p90:.2f}%")

logging.info(f"TFT Hyperparameters - Input: {config.INPUT_WINDOW}, Output: {config.OUTPUT_HORIZON}, Hidden: {config.HIDDEN_SIZE}, LSTM: {config.LSTM_LAYERS}, Dropout: {config.DROPOUT}, Batch: {config.BATCH_SIZE}, Epochs: {config.MAX_EPOCHS}")
logging.info(f"TFT P10 - MAE: {mae_p10:.4f}, MSE: {mse_p10:.4f}, MAPE: {mape_p10:.2f}%")
logging.info(f"TFT P50 - MAE: {mae_p50:.4f}, MSE: {mse_p50:.4f}, MAPE: {mape_p50:.2f}%")
logging.info(f"TFT P90 - MAE: {mae_p90:.4f}, MSE: {mse_p90:.4f}, MAPE: {mape_p90:.2f}%")

# Create or update TFT-specific metrics file - one row per run
try:
    results_df = pd.read_csv(config.METRICS_FILE)
except FileNotFoundError:
    results_df = pd.DataFrame(columns=[
        "Model", "timestamp",
        "MAE_P10", "MSE_P10", "MAPE_P10",
        "MAE_P50", "MSE_P50", "MAPE_P50", 
        "MAE_P90", "MSE_P90", "MAPE_P90",
        "input_chunk_length", "output_chunk_length", "hidden_size", 
        "lstm_layers", "dropout", "batch_size", "n_epochs", "actual_epochs", 
        "training_time_minutes", "training_time_hours", "stopping_reason", "quantiles"
    ])

# Create single row with all metrics and hyperparameters
new_run = {
    "Model": "TFT",
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "MAE_P10": mae_p10, "MSE_P10": mse_p10, "MAPE_P10": mape_p10,
    "MAE_P50": mae_p50, "MSE_P50": mse_p50, "MAPE_P50": mape_p50,
    "MAE_P90": mae_p90, "MSE_P90": mse_p90, "MAPE_P90": mape_p90,
    "input_chunk_length": config.INPUT_WINDOW,
    "output_chunk_length": config.OUTPUT_HORIZON, 
    "hidden_size": config.HIDDEN_SIZE,
    "lstm_layers": config.LSTM_LAYERS,
    "dropout": config.DROPOUT,
    "batch_size": config.BATCH_SIZE,
    "n_epochs": config.MAX_EPOCHS,
    "actual_epochs": actual_epochs,
    "training_time_minutes": training_time_minutes,
    "training_time_hours": training_time_hours,
    "stopping_reason": early_stopping_reason,
    "quantiles": str(config.QUANTILES)
}

results_df = pd.concat([results_df, pd.DataFrame([new_run])], ignore_index=True)
results_df.to_csv(config.METRICS_FILE, index=False)

print("Generating plot with uncertainty bands...")
plt.figure(figsize=(16, 8))

# Plot unscaled values for better interpretability
y_true_unscaled.plot(label="True", linewidth=2, color='black')
forecast_p50.plot(label="TFT P50 (Median)", linestyle="--", color='blue', linewidth=2)

# Add uncertainty bands with better visibility
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

# Add grid for better readability
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(config.RESULTS_DIR, "pred_vs_actual_tft.png"), dpi=300, bbox_inches='tight')

# Also create a detailed quantile plot
plt.figure(figsize=(16, 10))

# Plot all quantiles with better visibility
y_true_unscaled.plot(label="True", linewidth=3, color='black')
forecast_p10.plot(label="P10 (Lower Bound)", linestyle="-", alpha=0.8, color='red', linewidth=2)
forecast_p50.plot(label="P50 (Median)", linestyle="-", color='blue', linewidth=3)
forecast_p90.plot(label="P90 (Upper Bound)", linestyle="-", alpha=0.8, color='green', linewidth=2)

# Add uncertainty bands with higher alpha for visibility
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

# Add grid for better readability
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(config.RESULTS_DIR, "pred_vs_actual_tft_quantiles.png"), dpi=300, bbox_inches='tight')

# Save the trained model for later use
model.save(config.MODEL_SAVE_PATH)
logging.info(f"Saved trained TFT model to {config.MODEL_SAVE_PATH}")
logging.info("Saved TFT prediction plots and quantile metrics.")

elapsed_time = time.time() - start_time
print(f"\n✅ Finished in {elapsed_time / 60:.2f} minutes")
print(f"TFT Results - MAE: {mae_val:.4f}, MSE: {mse_val:.4f}, MAPE: {mape_val:.2f}%")
