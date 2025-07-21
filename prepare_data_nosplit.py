# prepare_data_nosplit.py
# ------------------------------------------------------
# Merge, clean, normalize, and window raw sensor data for inference (no split)

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import argparse

RAW_DIR = "data/raw"
PROC_DIR = "data/processed"
LOG_PATH = "log.txt"
TARGET_COLUMN = "Sensor 1"
TIMESTAMP_COL = "timestamp"

os.makedirs(PROC_DIR, exist_ok=True)

def log(message):
    with open(LOG_PATH, "a") as f:
        f.write(f"[{datetime.now()}] {message}\n")

log("Starting data preparation (no split).")

parser = argparse.ArgumentParser(description="Prepare data for inference (no split).")
parser.add_argument("--file", type=str, default=None, help="Specific CSV file in data/raw to process (default: all files)")
args = parser.parse_args()

if args.file:
    files = [args.file]
else:
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

# Save unscaled full data
full_df.to_csv(os.path.join(PROC_DIR, "full_raw.csv"), index=False)
log("Saved unscaled full data as full_raw.csv.")

# Fit scaler on full data
scaler = StandardScaler()
full_scaled = full_df.copy()
full_scaled[sensor_cols] = scaler.fit_transform(full_df[sensor_cols])

joblib.dump(scaler, os.path.join(PROC_DIR, "scalers.pkl"))
log("StandardScaler fitted on full data.")

# Save preprocessed full data
full_scaled.to_csv(os.path.join(PROC_DIR, "full.csv"), index=False)
log("Saved scaled full data as full.csv.")
log("Data preparation (no split) complete.")
