# prepare_data_filling.py
# ------------------------------------------------------
# Like prepare_data.py, but ensures regular time index and fills missing dates

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

log("Starting data preparation with missing date filling.")

# Load and merge
files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(".csv")])
df_list = []
for f in files:
    path = os.path.join(RAW_DIR, f)
    # Read without dtype specification first
    df = pd.read_csv(path, low_memory=False)
    
    # Convert timestamp to string explicitly
    df[TIMESTAMP_COL] = df[TIMESTAMP_COL].astype(str)
    
    # Convert all other columns to numeric, coercing errors to NaN
    sensor_cols = [col for col in df.columns if col != TIMESTAMP_COL]
    for col in sensor_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df_list.append(df)
    log(f"Loaded {f} with shape {df.shape}")

full_df = pd.concat(df_list).reset_index(drop=True)
# Parse datetime with the correct format for your data
try:
    # Try the actual format first: "1/1/22 0:00" 
    full_df[TIMESTAMP_COL] = pd.to_datetime(full_df[TIMESTAMP_COL], format='%m/%d/%y %H:%M', dayfirst=False)
except Exception as e:
    try:
        # Fallback to mixed format inference if the first attempt fails
        full_df[TIMESTAMP_COL] = pd.to_datetime(full_df[TIMESTAMP_COL], format='mixed', dayfirst=True)
        log("Used mixed format parsing for timestamps")
    except Exception as e2:
        print(f"Datetime parsing failed: {e2}")
        log(f"Datetime parsing failed: {e2}")
        raise
full_df = full_df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

# Set timestamp as index and infer frequency
full_df = full_df.set_index(TIMESTAMP_COL)

# Identify and print duplicates immediately after setting index
duplicates = full_df.index.duplicated(keep='first')
duplicates_df = full_df[duplicates]
if not duplicates_df.empty:
    print("Dropped duplicate timestamps (A: timestamp, B: Sensor 1 value):")
    print(duplicates_df[[TARGET_COLUMN]].reset_index().rename(columns={TIMESTAMP_COL: 'A', TARGET_COLUMN: 'B'}))
    for _, row in duplicates_df[[TARGET_COLUMN]].reset_index().iterrows():
        log(f"Dropped duplicate: timestamp={row[TIMESTAMP_COL]}, Sensor 1={row[TARGET_COLUMN]}")
# Drop duplicates, keeping the first occurrence
full_df = full_df[~full_df.index.duplicated(keep='first')]

# Now infer frequency and call asfreq
inferred_freq = pd.infer_freq(full_df.index)
if inferred_freq is None:
    log("Could not infer frequency. Defaulting to minutes ('T').")
    inferred_freq = 'T'  # Change as appropriate for your data
else:
    log(f"Inferred frequency: {inferred_freq}")

full_df = full_df.asfreq(inferred_freq)
# Fill missing sensor values after reindexing
sensor_cols = full_df.columns
full_df[sensor_cols] = full_df[sensor_cols].apply(pd.to_numeric, errors='coerce')
full_df[sensor_cols] = full_df[sensor_cols].ffill().bfill()
log("Filled missing dates and sensor values.")

# Reset index to have timestamp as a column again
full_df = full_df.reset_index()

# Plot Sensor 1 for each original file (using year prefix as before)
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
    plot_path = os.path.join(PROC_DIR, f"sensor1_{f.replace('.csv', '')}_filled.png")
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
train_df.to_csv(os.path.join(PROC_DIR, "train_raw_filled.csv"), index=False)
val_df.to_csv(os.path.join(PROC_DIR, "val_raw_filled.csv"), index=False)
test_df.to_csv(os.path.join(PROC_DIR, "test_raw_filled.csv"), index=False)
log("Saved unscaled raw train/val/test splits (filled).")

# Plot Sensor 1 for train/val/test raw data
for split_name, df in zip(["train_raw_filled", "val_raw_filled", "test_raw_filled"], [train_df, val_df, test_df]):
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

joblib.dump(scaler, os.path.join(PROC_DIR, "scalers_filled.pkl"))
log("StandardScaler fitted on train and applied to all splits (filled).")

# Save preprocessed CSVs
train_scaled.to_csv(os.path.join(PROC_DIR, "train_filled.csv"), index=False)
val_scaled.to_csv(os.path.join(PROC_DIR, "val_filled.csv"), index=False)
test_scaled.to_csv(os.path.join(PROC_DIR, "test_filled.csv"), index=False)
log("Saved train/val/test CSVs to processed directory (filled). Data preparation with missing date filling complete.")