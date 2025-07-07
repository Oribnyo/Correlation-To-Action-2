# lstm_model.py
# ------------------------------------------------------
# LSTM baseline model with logging and visual tracking

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import load
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROC_DIR = "data/processed"
RESULTS_DIR = "results"
LOG_PATH = "log.txt"
METRICS_FILE = os.path.join(RESULTS_DIR, "metrics_naive.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

INPUT_WINDOW = 60
OUTPUT_STEP = 30
TARGET = "Sensor 1"
EPOCHS = 5
BATCH_SIZE = 64


def log(message):
    with open(LOG_PATH, "a") as f:
        f.write(f"[{datetime.now()}] {message}\n")

log("Starting lstm_model.py")
log("Loading train and test datasets...")

train_df = pd.read_csv(os.path.join(PROC_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(PROC_DIR, "test.csv"))
features = [col for col in train_df.columns if col != "timestamp"]


def make_supervised_tensor(df):
    X, y = [], []
    data = df[features].values
    for i in range(INPUT_WINDOW, len(data) - OUTPUT_STEP):
        X.append(data[i - INPUT_WINDOW:i])
        y.append(data[i + OUTPUT_STEP - 1, features.index(TARGET)])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

log("Generating supervised tensors...")
X_train, y_train = make_supervised_tensor(train_df)
X_test, y_test = make_supervised_tensor(test_df)
log(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


model = LSTMModel(input_dim=len(features))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

log("Training LSTM model...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    log(f"Epoch {epoch+1}: Loss = {running_loss / len(train_loader):.4f}")

log("Predicting test set...")
model.eval()
with torch.no_grad():
    y_pred = model(X_test).numpy()
    y_true = y_test.numpy()

# Unscale for meaningful metrics
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
log(f"LSTM - MAE: {mae:.4f}, MSE: {mse:.4f}, MAPE: {mape:.2f}%")

results_df = pd.read_csv(METRICS_FILE)
results_df = pd.concat([results_df, pd.DataFrame([{"Model": "LSTM", "MAE": mae, "MSE": mse, "MAPE": mape}])], ignore_index=True)
results_df.to_csv(METRICS_FILE, index=False)

plt.figure(figsize=(14, 6))
plt.plot(y_true, label="True", linewidth=2)
plt.plot(y_pred, label="LSTM", linestyle="--")
scaler = load(os.path.join(PROC_DIR, "scalers.pkl"))
sensor1_index = list(scaler.feature_names_in_).index(TARGET)
threshold_scaled = (5.4 - scaler.mean_[sensor1_index]) / scaler.scale_[sensor1_index]
plt.axhline(y=threshold_scaled, color="red", linestyle="--", label="Threshold 5.4")
plt.legend()
plt.title("LSTM Forecast vs True HCDP (Scaled)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "pred_vs_actual_lstm.png"))

# Plot unscaled prediction vs true with proper time alignment
plt.figure(figsize=(14, 6))

# Use the correctly unscaled values from metrics calculation above

# Get the actual timestamps for the test period (accounting for windowing)
test_raw_df = pd.read_csv(os.path.join(PROC_DIR, "test_raw.csv"))
# The predictions correspond to indices starting at INPUT_WINDOW + OUTPUT_STEP - 1
start_idx = INPUT_WINDOW + OUTPUT_STEP - 1  # 89
end_idx = start_idx + len(y_true_unscaled)
test_timestamps = test_raw_df["timestamp"].iloc[start_idx:end_idx]

plt.plot(test_timestamps, y_true_unscaled, label="True (Test Data)", linewidth=2)
plt.plot(test_timestamps, y_pred_unscaled, label="LSTM Predictions", linestyle="--", alpha=0.8)
plt.axhline(y=5.4, color="red", linestyle="--", label="Threshold 5.4")
plt.legend()
plt.title("LSTM Forecast vs True Values (Test Set - Unscaled)")
plt.xlabel("Time")
plt.ylabel("Sensor 1")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "pred_vs_actual_lstm_unscaled.png"))

# Also create a plot showing the FULL test data for comparison
plt.figure(figsize=(14, 6))
full_timestamps = pd.to_datetime(test_raw_df["timestamp"])
plt.plot(full_timestamps, test_raw_df[TARGET], 
         label="Full Test Data", alpha=0.7, color='lightblue')
pred_timestamps = pd.to_datetime(test_timestamps)
plt.plot(pred_timestamps, y_true_unscaled, label="True (Prediction Period)", linewidth=2, color='blue')
plt.plot(pred_timestamps, y_pred_unscaled, label="LSTM Predictions", linestyle="--", alpha=0.8, color='orange')
plt.axhline(y=5.4, color="red", linestyle="--", label="Threshold 5.4")
plt.legend()
plt.title("LSTM Predictions in Context of Full Test Dataset")
plt.xlabel("Time")
plt.ylabel("Sensor 1")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "pred_vs_actual_lstm_full_context.png"))
plt.close()
log("Saved LSTM prediction plot and metrics.")
