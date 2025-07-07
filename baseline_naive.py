# baseline_naive.py
# ------------------------------------------------------
# Implements naive forecasting baselines: last value, rolling mean, linear trend

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from datetime import datetime

PROC_DIR = "data/processed"
RESULTS_DIR = "results"
LOG_PATH = "log.txt"
os.makedirs(RESULTS_DIR, exist_ok=True)

HORIZON = 30
TARGET = "Sensor 1"

def log(message):
    with open(LOG_PATH, "a") as f:
        f.write(f"[{datetime.now()}] {message}\n")

log("Starting baseline_naive.py")

# Load test set (use raw unscaled data for meaningful metrics)
df = pd.read_csv(os.path.join(PROC_DIR, "test_raw.csv"))
timestamps = pd.to_datetime(df["timestamp"])
sensor_df = df.drop(columns=["timestamp"])

true_vals = sensor_df[TARGET].rolling(HORIZON).apply(lambda x: x[-1], raw=True).shift(-HORIZON)

# Naive: repeat last known value
last_value_pred = sensor_df[TARGET].shift(1)

# Naive: rolling mean
rolling_pred = sensor_df[TARGET].rolling(HORIZON).mean().shift(1)

# Naive: linear trend forecast (simple extrapolation)
def linear_forecast(series, window):
    preds = []
    for i in range(len(series)):
        if i < window:
            preds.append(np.nan)
            continue
        y = series[i - window:i]
        x = np.arange(window)
        coef = np.polyfit(x, y, 1)
        pred = coef[0] * window + coef[1]  # next step
        preds.append(pred)
    return pd.Series(preds, index=series.index)

linear_pred = linear_forecast(sensor_df[TARGET], HORIZON)

# Trim NaNs (intersection of all valid indices)
valid_idx = true_vals.dropna().index
valid_idx = valid_idx.intersection(last_value_pred.dropna().index)
valid_idx = valid_idx.intersection(rolling_pred.dropna().index)
valid_idx = valid_idx.intersection(linear_pred.dropna().index)

true_vals = true_vals.loc[valid_idx]
last_value_pred = last_value_pred.loc[valid_idx]
rolling_pred = rolling_pred.loc[valid_idx]
linear_pred = linear_pred.loc[valid_idx]

# Using raw data, so threshold is the original value
threshold_unscaled = 5.4

# Evaluate and save results
def evaluate(y_true, y_pred, model):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    log(f"{model} - MAE: {mae:.4f}, MSE: {mse:.4f}, MAPE: {mape:.2f}%")
    return {"Model": model, "MAE": mae, "MSE": mse, "MAPE": mape}

results = [
    evaluate(true_vals, last_value_pred, "Naive_LastValue"),
    evaluate(true_vals, rolling_pred, "Naive_RollingMean"),
    evaluate(true_vals, linear_pred, "Naive_LinearTrend")
]

# Save metrics
pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, "metrics_naive.csv"), index=False)

# Plot predictions vs actual
plt.figure(figsize=(14, 6))
plt.plot(true_vals.values[:300], label="True", linewidth=2)
plt.plot(last_value_pred.values[:300], label="Naive_LastValue", linestyle="--")
plt.plot(rolling_pred.values[:300], label="Naive_RollingMean", linestyle=":")
plt.plot(linear_pred.values[:300], label="Naive_LinearTrend", linestyle="-.")
plt.axhline(y=threshold_unscaled, color="red", linestyle="--", label="Threshold 5.4")
plt.legend()
plt.title("Naive Model Forecasts vs True HCDP (Unscaled)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "pred_vs_actual_naive.png"))
plt.close()
log("Saved naive prediction plot and metrics.")
