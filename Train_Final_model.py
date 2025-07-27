#INSTALL DARTS
#pip install darts


# IMPORT LIBRARIES
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae # Import the MAE metric
from darts.models import TFTModel
from darts.utils.likelihood_models.torch import QuantileRegression
from darts.utils.statistics import check_seasonality, plot_acf
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.metrics import mae, mql


warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)


# DEFINING SOME CONSTANTS
num_samples = 100
figsize = (10, 6)
lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99
label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles"

# LOAD AND PREPARE DATA:
# Load the training data
try:
    train_df = pd.read_csv('/Users/oribenyosef/Correlation-To-Action-2/data/my first model split/train_raw.csv')
    print("Training data loaded successfully.")
except FileNotFoundError:
    print("Error: train_raw.csv not found. Please check the file path.")
    train_df = None

# Load the validation data
try:
    val_df = pd.read_csv('/Users/oribenyosef/Correlation-To-Action-2/data/my first model split/val_raw.csv')
    print("Validation data loaded successfully.")
except FileNotFoundError:
    print("Error: val_raw.csv not found. Please check the file path.")
    val_df = None

# Load the test data
try:
    test_df = pd.read_csv('/Users/oribenyosef/Correlation-To-Action-2/data/my first model split/test_raw.csv')
    print("Test data loaded successfully.")
except FileNotFoundError:
    print("Error: test_raw.csv not found. Please check the file path.")
    test_df = None

# Convert DataFrames to TimeSeries objects and concatenate if dataframes were loaded
if train_df is not None and val_df is not None and test_df is not None:
    train_series = TimeSeries.from_dataframe(train_df, 'timestamp')
    val_series = TimeSeries.from_dataframe(val_df, 'timestamp')
    test_series = TimeSeries.from_dataframe(test_df, 'timestamp')

    all_series = train_series.append(val_series).append(test_series)

    print("DataFrames converted to TimeSeries objects and concatenated.")
else:
    print("Could not proceed with TimeSeries conversion and concatenation due to file loading errors.")


# SEPERATE TARGET AND SENSORS
# The data has been loaded and concatenated. The next step is to separate the target variable ('Sensor 1') from the rest of the sensors, which will be used as past covariates.
target_series = all_series['Sensor 1']
past_covariates = all_series.drop_columns('Sensor 1')

print("Target series columns:", target_series.columns)
print("Past covariates columns:", past_covariates.columns)


# DEFINE CUTOFF AND SPLIT
# The target and past covariates have been defined. The next step is to split these series into training, validation, and test sets based on the original data file boundaries.
# Define the cutoff points for splitting data
training_cutoff = train_series.end_time()
validation_cutoff = val_series.end_time()

# Split target series
target_train = target_series.split_after(training_cutoff)[0]
target_val_test = target_series.split_after(training_cutoff)[1]
target_val = target_val_test.split_after(validation_cutoff)[0]
target_test = target_val_test.split_after(validation_cutoff)[1]

# Split past covariates
past_cov_train = past_covariates.split_after(training_cutoff)[0]
past_cov_val_test = past_covariates.split_after(training_cutoff)[1]
past_cov_val = past_cov_val_test.split_after(validation_cutoff)[0]
past_cov_test = past_cov_val_test.split_after(validation_cutoff)[1]


# Print start and end times to verify the split
print("Target Train:", target_train.start_time(), target_train.end_time())
print("Target Val:", target_val.start_time(), target_val.end_time())
print("Target Test:", target_test.start_time(), target_test.end_time())
print("Past Covariates Train:", past_cov_train.start_time(), past_cov_train.end_time())
print("Past Covariates Val:", past_cov_val.start_time(), past_cov_val.end_time())
print("Past Covariates Test:", past_cov_test.start_time(), past_cov_test.end_time())


# SCALE DATA
# The data has been split into training, validation, and test sets for both the target and past covariates. The next step is to scale these time series data using appropriate scalers, ensuring that the scalers are fitted only on the training data.

# Instantiate and fit Scaler for target series on training data, then transform
scaler_target = Scaler()
target_train_scaled = scaler_target.fit_transform(target_train)
target_val_scaled = scaler_target.transform(target_val)
target_test_scaled = scaler_target.transform(target_test)

# Instantiate and fit Scaler for past covariates on training data, then transform
scaler_past_cov = Scaler()
past_cov_train_scaled = scaler_past_cov.fit_transform(past_cov_train)
past_cov_val_scaled = scaler_past_cov.transform(past_cov_val)
past_cov_test_scaled = scaler_past_cov.transform(past_cov_test)
past_covariates_scaled_all = scaler_past_cov.transform(past_covariates)

print("Target and past covariates scaled.")


# CHECK SCALING IMPACT
# Check scaling impact using numpy arrays instead
print("=== BEFORE SCALING ===")
train_values = target_train.values().flatten()
print("Target train min:", train_values.min())
print("Target train max:", train_values.max())
print("Target train mean:", train_values.mean())
print("Target train std:", train_values.std())

print("\n=== AFTER SCALING ===")
scaled_values = target_train_scaled.values().flatten()
print("Target train scaled min:", scaled_values.min())
print("Target train scaled max:", scaled_values.max())
print("Target train scaled mean:", scaled_values.mean())
print("Target train scaled std:", scaled_values.std())

# Check if peaks are getting compressed
original_range = train_values.max() - train_values.min()
scaled_range = scaled_values.max() - scaled_values.min()
print(f"\nRange compression: {original_range:.4f} -> {scaled_range:.4f}")

# Most importantly - check peak values specifically
peak_indices = np.where(train_values > np.percentile(train_values, 95))[0]
print(f"\nOriginal peak values (95th percentile+): {train_values[peak_indices]}")
print(f"Scaled peak values (95th percentile+): {scaled_values[peak_indices]}")

# Define the combined training and validation data
combined_train_val_target = target_train_scaled.append(target_val_scaled)
combined_train_val_past_covariates = past_cov_train_scaled.append(past_cov_val_scaled)


# Define the quantiles for the QuantileRegression likelihood
quantiles = [
     0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99
]
print("quantile defined")


# DEFINE THE MODEL
model = TFTModel(
    # SET HYPERPARAMETERS
    input_chunk_length=96,
    output_chunk_length=12,
    hidden_size=40,
    lstm_layers=1,
    num_attention_heads=4,
    dropout=0.7,
    batch_size=256,
    n_epochs=100,
    add_relative_index=False,
    add_encoders={"cyclic": {"future": ["month"]}},
    optimizer_kwargs={"lr": 1e-4},
    likelihood=QuantileRegression(quantiles=quantiles),
    random_state=42,  
    save_checkpoints=True,  
    pl_trainer_kwargs={
        "accelerator": "cpu",
    }
)


# TRAIN THE MODEL
# Control flag
TRAIN_MODEL = False  # Set to True when you want to train the model
if TRAIN_MODEL:
    model.fit(
            combined_train_val_target,
            past_covariates=combined_train_val_past_covariates,
            verbose=True
    )
    # SAVE THE MODEL
    model.save("/Users/oribenyosef/Correlation-To-Action-2/TFT_FINAL_MODEL_100_EPOCH.pt")
    print("Model saved successfully.")
else:
    # LOAD THE MODEL
    model = TFTModel.load("/Users/oribenyosef/Correlation-To-Action-2/TFT_FINAL_MODEL_100_EPOCH.pt")


# EVALUATE ON TEST DATA
test_pred_scaled = model.predict(
    n=len(target_test_scaled),
    series=combined_train_val_target,
    past_covariates=past_covariates_scaled_all,
    num_samples=100
)


# INVERSE TRANSFORM TO ORIGINAL SCALE
test_pred = scaler_target.inverse_transform(test_pred_scaled)
target_test_original = scaler_target.inverse_transform(target_test_scaled)


# CALCULATE METRICS
test_mae = mae(target_test_original, test_pred)
mql_p50 = mql(target_test_original, test_pred, q=0.5)
mql_p90 = mql(target_test_original, test_pred, q=0.9)


# PLOT WITH P10-P90 BAND AND P50 ON ORIGINAL SCALE
plt.figure(figsize=(12, 6))

# Plot actual (original scale)
target_test_original.plot(label='Actual', color='black', alpha=0.7)

# Plot P10-P90 band (original scale)
test_pred.plot(
    low_quantile=0.1, 
    high_quantile=0.9, 
    label='P10-P90',
    color='blue',
    alpha=0.3
)

# Plot P50 median (original scale)
test_pred.plot(
    low_quantile=0.5,
    high_quantile=0.5,
    label='P50 (median)',
    color='blue',
    linewidth=2
)

# Add metrics as text box on the plot
textstr = f'MAE: {test_mae:.4f}\nMQL P50: {mql_p50:.4f}\nMQL P90: {mql_p90:.4f}'
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.xlabel('Time')
plt.ylabel('Value (Original Scale)')
plt.title('Test Predictions with P10-P90 Band')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# === SAVE EVALUATION DATA FOR SEPARATE TESTING ===
print("💾 Saving evaluation data for separate testing...")

import pickle

evaluation_data = {
    'target_test_scaled': target_test_scaled,
    'combined_train_val_target': combined_train_val_target,
    'past_covariates_scaled_all': past_covariates_scaled_all,
    'scaler_target': scaler_target,
    'scaler_past_cov': scaler_past_cov,
    'target_test_original': target_test_original,
    'test_pred': test_pred,
    'test_pred_scaled': test_pred_scaled,
    # Additional useful data
    'target_train_scaled': target_train_scaled,
    'target_val_scaled': target_val_scaled,
    'past_cov_train_scaled': past_cov_train_scaled,
    'past_cov_val_scaled': past_cov_val_scaled,
    'quantiles': quantiles,
    # Add the missing data for feature analysis
    'past_covariates': past_covariates,  # Original unscaled covariates for column names
    'test_mae': test_mae,
    'test_quantile_losses_per_q': {q: mql(target_test_original, test_pred, q=q) for q in quantiles}
}

# Save to pickle file
with open('/Users/oribenyosef/Correlation-To-Action-2/evaluation_data.pkl', 'wb') as f:
    pickle.dump(evaluation_data, f)

print("✅ Evaluation data saved to: /Users/oribenyosef/Correlation-To-Action-2/evaluation_data.pkl")
print("📊 Available data keys:")
for key in evaluation_data.keys():
    print(f"  - {key}")
print("\n🎯 You can now use Test_Final_Model_Results.py for separate evaluation!")




