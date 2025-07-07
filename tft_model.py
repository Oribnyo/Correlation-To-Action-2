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

class TFTConfig:
    def __init__(self):
        self.INPUT_WINDOW = 96
        self.OUTPUT_HORIZON = 30
        self.HIDDEN_SIZE = 32
        self.LSTM_LAYERS = 2
        self.NUM_ATTENTION_HEADS = 4
        self.DROPOUT = 0.1

        self.BATCH_SIZE = 128
        self.MAX_EPOCHS = 30
        self.LEARNING_RATE = 1e-3
        self.WEIGHT_DECAY = 1e-4

        self.TARGET = "Sensor 1"
        self.QUANTILES = [0.1, 0.5, 0.9]
        self.VALIDATION_SPLIT = 0.2

        self.PROC_DIR = "data/processed"
        self.RESULTS_DIR = "results"
        self.LOG_PATH = "log.txt"
        self.MODEL_SAVE_PATH = os.path.join(self.RESULTS_DIR, "tft_model.pkl")
        self.METRICS_FILE = os.path.join(self.RESULTS_DIR, "metrics_tft.csv")

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
    random_state=42,
    force_reset=True,
    pl_trainer_kwargs={
        "accelerator": "cpu",
        "gradient_clip_val": 1.0,
        "enable_progress_bar": True,
        "enable_model_summary": True,
        "callbacks": [
            EarlyStopping(monitor="val_loss", patience=7, mode="min"),
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

print("Predicting...")
fit_end = time.time()
tqdm.write(f"[INFO] Training completed in {(fit_end - fit_start)/60:.2f} minutes")

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
print("Extracting quantile predictions...")
forecast_p10 = forecast_unscaled.quantile_timeseries(0.1) if forecast_unscaled.n_samples > 1 else forecast_unscaled
forecast_p50 = forecast_unscaled.quantile_timeseries(0.5) if forecast_unscaled.n_samples > 1 else forecast_unscaled  # Median
forecast_p90 = forecast_unscaled.quantile_timeseries(0.9) if forecast_unscaled.n_samples > 1 else forecast_unscaled

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
        "lstm_layers", "dropout", "batch_size", "n_epochs", "quantiles"
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
    "quantiles": str(config.QUANTILES)
}

results_df = pd.concat([results_df, pd.DataFrame([new_run])], ignore_index=True)
results_df.to_csv(config.METRICS_FILE, index=False)

print("Generating plot with uncertainty bands...")
plt.figure(figsize=(14, 6))

# Plot unscaled values for better interpretability
y_true_unscaled.plot(label="True", linewidth=2, color='black')
forecast_p50.plot(label="TFT P50 (Median)", linestyle="--", color='blue')

# Add uncertainty bands
plt.fill_between(
    forecast_p10.time_index, 
    forecast_p10.values().flatten(), 
    forecast_p90.values().flatten(), 
    alpha=0.2, 
    color='blue', 
    label='P10-P90 Uncertainty Band'
)

plt.legend()
plt.title("TFT Probabilistic Forecast vs True (Validation Set - Unscaled)")
plt.ylabel("Sensor 1 Value")
plt.xlabel("Time")
plt.tight_layout()
plt.savefig(os.path.join(config.RESULTS_DIR, "pred_vs_actual_tft.png"))

# Also create a detailed quantile plot
plt.figure(figsize=(14, 8))
y_true_unscaled.plot(label="True", linewidth=2, color='black')
forecast_p10.plot(label="P10", linestyle=":", alpha=0.7, color='red')
forecast_p50.plot(label="P50 (Median)", linestyle="--", color='blue')
forecast_p90.plot(label="P90", linestyle=":", alpha=0.7, color='green')
plt.fill_between(
    forecast_p10.time_index, 
    forecast_p10.values().flatten(), 
    forecast_p90.values().flatten(), 
    alpha=0.1, 
    color='gray', 
    label='P10-P90 Range'
)
plt.legend()
plt.title("TFT Quantile Predictions vs True (Validation Set)")
plt.ylabel("Sensor 1 Value")
plt.xlabel("Time")
plt.tight_layout()
plt.savefig(os.path.join(config.RESULTS_DIR, "pred_vs_actual_tft_quantiles.png"))

logging.info("Saved TFT prediction plots and quantile metrics.")

elapsed_time = time.time() - start_time
print(f"\n✅ Finished in {elapsed_time / 60:.2f} minutes")
print(f"TFT Results - MAE: {mae_val:.4f}, MSE: {mse_val:.4f}, MAPE: {mape_val:.2f}%")
