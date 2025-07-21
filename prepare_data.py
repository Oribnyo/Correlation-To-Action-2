# prepare_data.py
# ------------------------------------------------------
# Merge, clean, normalize, and window raw sensor data for forecasting

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import matplotlib.pyplot as plt


RAW_DIR = "data/raw"
PROC_DIR = "data/processed"
LOG_PATH = "log.txt"
TARGET_COLUMN = "Sensor 1"
TIMESTAMP_COL = "timestamp"

os.makedirs(PROC_DIR, exist_ok=True)

def log(message):
    with open(LOG_PATH, "a") as f:
        f.write(f"[{datetime.now()}] {message}\n")

log("Starting data preparation.")

# Load and merge
files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(".csv")])
df_list = []
for f in files:
    path = os.path.join(RAW_DIR, f)
    df = pd.read_csv(path)
    df_list.append(df)
    log(f"Loaded {f} with shape {df.shape}")

full_df = pd.concat(df_list).reset_index(drop=True)
full_df[TIMESTAMP_COL] = pd.to_datetime(full_df[TIMESTAMP_COL], dayfirst=True)
full_df = full_df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

# Clean non-numeric
sensor_cols = full_df.columns.drop(TIMESTAMP_COL)
full_df[sensor_cols] = full_df[sensor_cols].apply(pd.to_numeric, errors='coerce') #fill in NaNs
full_df[sensor_cols] = full_df[sensor_cols].ffill().bfill() #fill NanS with values from previous/next row
log("Cleaned non-numeric values and filled NaNs, filled the NaNs with values from previous/next row.")

# Plot Sensor 1 for each original file
os.makedirs(PROC_DIR, exist_ok=True)
for f in files:
    base_date = f.split('_')[0]  # extract year prefix
    subset = full_df[full_df[TIMESTAMP_COL].dt.year == int(base_date)].copy()
    plt.figure(figsize=(12, 4))
    plt.plot(subset[TIMESTAMP_COL], subset[TARGET_COLUMN], label=TARGET_COLUMN)
    plt.axhline(y=5.4, color='red', linestyle='--', label='Threshold 5.4')
    plt.title(f"{TARGET_COLUMN} over time - {f}")
    plt.xlabel("Time")
    plt.ylabel(TARGET_COLUMN)
    plt.tight_layout()
    plot_path = os.path.join(PROC_DIR, f"sensor1_{f.replace('.csv', '')}.png")
    plt.savefig(plot_path)
    plt.close()
    log(f"Saved plot to {plot_path}")


# Train/val/test split by time (Apr-May-June 2025, 80/10/10)
apr_may_june_df = full_df[(full_df[TIMESTAMP_COL] >= "2025-01-01") & (full_df[TIMESTAMP_COL] < "2025-06-30")].copy()
total_len = len(apr_may_june_df)
train_end_idx = int(total_len * 0.8)
val_end_idx = int(total_len * 0.9)

train_df = apr_may_june_df.iloc[:train_end_idx]
val_df = apr_may_june_df.iloc[train_end_idx:val_end_idx]
test_df = apr_may_june_df.iloc[val_end_idx:]

log(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")

# Save unscaled raw versions
train_df.to_csv(os.path.join(PROC_DIR, "train_raw.csv"), index=False)
val_df.to_csv(os.path.join(PROC_DIR, "val_raw.csv"), index=False)
test_df.to_csv(os.path.join(PROC_DIR, "test_raw.csv"), index=False)
log("Saved unscaled raw train/val/test splits.")

# Plot Sensor 1 for train/val/test raw data
for split_name, df in zip(["train_raw", "val_raw", "test_raw"], [train_df, val_df, test_df]):
    plt.figure(figsize=(12, 4))
    plt.plot(df[TIMESTAMP_COL], df[TARGET_COLUMN], label=TARGET_COLUMN)
    plt.axhline(y=5.4, color='red', linestyle='--', label='Threshold 5.4')
    plt.title(f"{TARGET_COLUMN} over time - {split_name}")
    plt.xlabel("Time")
    plt.ylabel(TARGET_COLUMN)
    plt.tight_layout()
    plot_path = os.path.join(PROC_DIR, f"sensor1_{split_name}.png")
    plt.savefig(plot_path)
    plt.close()
    log(f"Saved plot to {plot_path}")

# Fit scaler on train only
scaler = StandardScaler()
train_scaled = train_df.copy()
val_scaled = val_df.copy()
test_scaled = test_df.copy()

train_scaled[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])
val_scaled[sensor_cols] = scaler.transform(val_df[sensor_cols])
test_scaled[sensor_cols] = scaler.transform(test_df[sensor_cols])

joblib.dump(scaler, os.path.join(PROC_DIR, "scalers.pkl"))
log("StandardScaler fitted on train and applied to all splits.")

# Save preprocessed CSVs
train_scaled.to_csv(os.path.join(PROC_DIR, "train.csv"), index=False)
val_scaled.to_csv(os.path.join(PROC_DIR, "val.csv"), index=False)
test_scaled.to_csv(os.path.join(PROC_DIR, "test.csv"), index=False)
log("Saved train/val/test CSVs to processed directory.")
log("Data preparation complete.")
