# eda_analysis.py
# ------------------------------------------------------
# Perform exploratory data analysis on both scaled and raw sensor data

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

PROC_DIR = "data/processed"
EDA_DIR = "results/eda"
LOG_PATH = "log.txt"
os.makedirs(EDA_DIR, exist_ok=True)

def log(message):
    with open(LOG_PATH, "a") as f:
        f.write(f"[{datetime.now()}] {message}\n")

log("Starting EDA analysis.")

# Load scaled and raw training data
scaled_df = pd.read_csv(os.path.join(PROC_DIR, "train.csv"))
raw_df = pd.read_csv(os.path.join(PROC_DIR, "train_raw.csv"))

# Sensor columns
sensor_cols = [col for col in raw_df.columns if col != "timestamp"]

# Correlation matrix (scaled)
corr = scaled_df.corr(numeric_only=True)
target_corr = corr[["Sensor 1"]].drop("Sensor 1").sort_values("Sensor 1", ascending=False)
target_corr.to_csv(os.path.join(EDA_DIR, "top_sensor_correlations.csv"))

plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Sensor Correlation Heatmap (Scaled Data)")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "correlation_heatmap_scaled.png"))
plt.close()
log("Saved correlation heatmap and top correlations.")

# Rolling stats of Sensor 1 (raw)
sensor_1 = raw_df["Sensor 1"]
rolling_mean = sensor_1.rolling(60).mean()
rolling_std = sensor_1.rolling(60).std()

plt.figure(figsize=(12, 6))
plt.plot(sensor_1.values, label="Sensor 1", alpha=0.4)
plt.plot(rolling_mean, label="Rolling Mean (60m)")
plt.plot(rolling_std, label="Rolling Std (60m)")
plt.legend()
plt.title("Rolling Statistics of Sensor 1 (Raw Values)")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "rolling_stats_sensor_1_raw.png"))
plt.close()
log("Saved rolling statistics plot (raw).")

# Top correlated scatter plots (raw)
top_sensors = target_corr.index[:3]
for sensor in top_sensors:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=raw_df[sensor], y=raw_df["Sensor 1"], alpha=0.3)
    plt.xlabel(sensor)
    plt.ylabel("Sensor 1")
    plt.title(f"{sensor} vs Sensor 1 (Raw)")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, f"scatter_{sensor}_vs_sensor1_raw.png"))
    plt.close()
log("Saved scatter plots for top correlated sensors (raw).")

log("EDA analysis complete.")
