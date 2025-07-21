import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from joblib import load
import os

PROC_DIR = "data/processed"
INPUT_WINDOW = 60
OUTPUT_STEP = 30
TARGET = "Sensor 1"

# Load the data
test_df = pd.read_csv(os.path.join(PROC_DIR, "test.csv"))
test_raw_df = pd.read_csv(os.path.join(PROC_DIR, "test_raw.csv"))
features = [col for col in test_df.columns if col != "timestamp"]

# Recreate the windowing function
def make_supervised_tensor(df):
    X, y = [], []
    data = df[features].values
    for i in range(INPUT_WINDOW, len(data) - OUTPUT_STEP):
        X.append(data[i - INPUT_WINDOW:i])
        y.append(data[i + OUTPUT_STEP - 1, features.index(TARGET)])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

# Generate test data
X_test, y_test = make_supervised_tensor(test_df)
print(f"Test tensor shapes: X={X_test.shape}, y={y_test.shape}")

# Load scaler
scaler = load(os.path.join(PROC_DIR, "scalers.pkl"))
sensor1_index = list(scaler.feature_names_in_).index(TARGET)

# Convert y_test back to numpy and unscale
y_true_scaled = y_test.numpy()
y_true_unscaled = y_true_scaled * scaler.scale_[sensor1_index] + scaler.mean_[sensor1_index]

print(f"\n=== Ground Truth Analysis ===")
print(f"First 10 scaled y_true: {y_true_scaled[:10]}")
print(f"First 10 unscaled y_true: {y_true_unscaled[:10]}")

# Check what the corresponding raw values should be
start_idx = INPUT_WINDOW + OUTPUT_STEP - 1  # This is where y_true should align
print(f"\n=== Raw Data at Target Indices ===")
print(f"Start index: {start_idx}")
for i in range(10):
    target_idx = start_idx + i
    if target_idx < len(test_raw_df):
        raw_value = test_raw_df[TARGET].iloc[target_idx]
        unscaled_value = y_true_unscaled[i]
        print(f"Index {target_idx}: Raw={raw_value:.6f}, Unscaled={unscaled_value:.6f}, Diff={abs(raw_value - unscaled_value):.10f}")

# Check the range of values
print(f"\n=== Value Ranges ===")
print(f"Raw test data range: {test_raw_df[TARGET].min():.6f} to {test_raw_df[TARGET].max():.6f}")
print(f"Unscaled y_true range: {y_true_unscaled.min():.6f} to {y_true_unscaled.max():.6f}")

# Check if there are any issues with the data
print(f"\n=== Data Quality ===")
print(f"Raw data has NaN: {test_raw_df[TARGET].isna().sum()}")
print(f"Scaled data has NaN: {test_df[TARGET].isna().sum()}")
print(f"y_true has NaN: {np.isnan(y_true_unscaled).sum()}")

# Sample some timestamps to see alignment
print(f"\n=== Timestamp Alignment Check ===")
for i in [0, 100, 500, 1000]:
    if i < len(y_true_unscaled):
        prediction_idx = start_idx + i
        if prediction_idx < len(test_raw_df):
            timestamp = test_raw_df['timestamp'].iloc[prediction_idx]
            raw_val = test_raw_df[TARGET].iloc[prediction_idx]
            unscaled_val = y_true_unscaled[i]
            print(f"Prediction {i}: Time={timestamp}, Raw={raw_val:.6f}, Unscaled={unscaled_val:.6f}") 