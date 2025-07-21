# baseline_ml.py
# ------------------------------------------------------
# Machine Learning baseline using RandomForestRegressor (mini test version)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import load
from datetime import datetime
from tqdm import tqdm

PROC_DIR = "data/processed"
RESULTS_DIR = "results"
LOG_PATH = "log.txt"
METRICS_FILE = os.path.join(RESULTS_DIR, "metrics_naive.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

INPUT_WINDOW = 60
OUTPUT_STEP = 30
TARGET = "Sensor 1"

def log(message):
    with open(LOG_PATH, "a") as f:
        f.write(f"[{datetime.now()}] {message}\n")

log("Starting baseline_ml.py (mini test version)")
log("Loading train and test datasets...")

# Load data and limit size for testing
train_df = pd.read_csv(os.path.join(PROC_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(PROC_DIR, "test.csv"))

features = [col for col in train_df.columns if col != "timestamp"]

# Generate sliding window features
def make_supervised(df, input_window, output_step):
    X, y = [], []
    data = df[features].values
    for i in tqdm(range(input_window, len(data) - output_step), desc="Building windows"):
        X.append(data[i - input_window:i].flatten())
        y.append(data[i + output_step - 1, features.index(TARGET)])
    return np.array(X), np.array(y)

log("Generating sliding windows for training data...")
X_train, y_train = make_supervised(train_df, INPUT_WINDOW, OUTPUT_STEP)
log("Generating sliding windows for test data...")
X_test, y_test = make_supervised(test_df, INPUT_WINDOW, OUTPUT_STEP)

log(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Train model (light config)
model = RandomForestRegressor(n_estimators=10, n_jobs=-1, random_state=42)
log("Training RandomForest model...")
model.fit(X_train, y_train)
log("Random Forest trained.")

# Predict
log("Predicting test set...")
from tqdm import tqdm

y_pred = []
for x in tqdm(X_test, desc="Predicting test set", unit="samples"):
    y_pred.append(model.predict(x.reshape(1, -1))[0])
y_pred = np.array(y_pred)

# Evaluate with unscaling for meaningful metrics
def evaluate(y_true, y_pred, model):
    # Load scaler and unscale for meaningful metrics
    scaler = load(os.path.join(PROC_DIR, "scalers.pkl"))
    sensor1_index = list(scaler.feature_names_in_).index(TARGET)
    
    # Create dummy arrays with all features for inverse_transform
    n_features = len(scaler.feature_names_in_)
    y_true_full = np.zeros((len(y_true), n_features))
    y_pred_full = np.zeros((len(y_pred), n_features))
    
    y_true_full[:, sensor1_index] = y_true
    y_pred_full[:, sensor1_index] = y_pred
    
    y_true_unscaled = scaler.inverse_transform(y_true_full)[:, sensor1_index]
    y_pred_unscaled = scaler.inverse_transform(y_pred_full)[:, sensor1_index]
    
    mae = mean_absolute_error(y_true_unscaled, y_pred_unscaled)
    mse = mean_squared_error(y_true_unscaled, y_pred_unscaled)
    mape = np.mean(np.abs((y_true_unscaled - y_pred_unscaled) / y_true_unscaled)) * 100
    log(f"{model} - MAE: {mae:.4f}, MSE: {mse:.4f}, MAPE: {mape:.2f}%")
    return {"Model": model, "MAE": mae, "MSE": mse, "MAPE": mape}

# Load existing metrics and append new results
results_df = pd.read_csv(METRICS_FILE)
results_df = pd.concat([results_df, pd.DataFrame([evaluate(y_test, y_pred, "RandomForest_Mini")])], ignore_index=True)
results_df.to_csv(METRICS_FILE, index=False)

# Plot
plt.figure(figsize=(14, 6))
plt.plot(y_test, label="True", linewidth=2)
plt.plot(y_pred, label="RandomForest_Mini", linestyle="--")
# Using raw data, so threshold is the original value
plt.axhline(y=5.4, color="red", linestyle="--", label="Threshold 5.4")
plt.legend()
plt.title("RandomForest Mini Forecast vs True HCDP (Scaled)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "pred_vs_actual_ml.png"))
plt.close()
log("Saved ML prediction plot and metrics.")
