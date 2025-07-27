"""
Pipeline: Full HCDP Monitoring and Reporting
Description: Forecasts HCDP, detects exceedance, launches SHAP, scenario simulation, clustering, and triggers LLM report.
"""

import os
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
import torch

# Import modules
from module_scenario_simulation import run_scenario_simulation
from module_regime_clustering import run_regime_clustering
from module_shap_explainer import run_shap_explainer
# from LLM_chat_langchain import generate_llm_report    # Optional if LLM integration needed

# Configuration
MODEL_PATH = "/Users/oribenyosef/Correlation-To-Action-2/TFT_FINAL_MODEL_100_EPOCH.pt"  # UPDATE THIS PATH
CSV_PATH = "/Users/oribenyosef/Correlation-To-Action-2/data/snapshot for full model/19-6-25-raw.csv"
CSV_FORECAST_PATH = "/Users/oribenyosef/Correlation-To-Action-2/data/snapshot for full model/19-6-25-raw-forecast.csv"
OUTPUT_BASE = Path("/Users/oribenyosef/Correlation-To-Action-2/outputs")
THRESHOLD_UNSCALED = 5.4
TARGET_COL = "Sensor 1 [Hydrocarbon_Dew_Point_C]"
FORECAST_HORIZON = 12
INPUT_CHUNK_LENGTH = 96  # Historical window size

def run_pipeline():
    start_time = time.time()
    
    print("🚀 Starting HCDP Monitoring Pipeline...")

    # Create session output folder with short number
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_number = int(time.time()) % 10000  # Short 4-digit number
    session_path = OUTPUT_BASE / f"session_{session_number}_{timestamp}"
    os.makedirs(session_path, exist_ok=True)

    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    df = pd.read_csv(CSV_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    # Remove duplicate timestamps and invalid rows
    print(f"   Original data length: {len(df)}")
    df = df[~df.index.duplicated(keep='first')]
    df = df.dropna(subset=[TARGET_COL])  # Remove rows with missing target values
    print(f"   After removing duplicates and invalid rows: {len(df)}")
    
    # Data already has consistent 1-minute intervals
    print(f"   Data spans from {df.index[0]} to {df.index[-1]}")
    
    # Create target series
    target_series = TimeSeries.from_dataframe(df, value_cols=[TARGET_COL])
    
    # Create past covariates using all other sensors
    sensor_columns = [col for col in df.columns if col != TARGET_COL and 'Sensor' in col]
    past_covariates_df = df[sensor_columns]
    past_covariates_series = TimeSeries.from_dataframe(past_covariates_df)
    
    # Scale data
    print("🔧 Scaling data...")
    target_scaler = Scaler()
    past_cov_scaler = Scaler()
    
    target_scaled = target_scaler.fit_transform(target_series)
    past_covariates_scaled = past_cov_scaler.fit_transform(past_covariates_series)

    # Load model
    print("🤖 Loading model...")
    model = TFTModel.load(MODEL_PATH, map_location='cpu')
    
    # ========== WARM-UP PERIOD ==========
    print("🔥 Warming up model with preliminary predictions...")
    WARMUP_WINDOWS = 10  # More windows since we're moving by 1 minute
    WARMUP_STRIDE = 1    # 1-minute stride for warmup
    
    # Start warmup from beginning of data (after having enough history)
    warmup_start_position = INPUT_CHUNK_LENGTH
    
    for warmup_idx in range(WARMUP_WINDOWS):
        warmup_position = warmup_start_position + (warmup_idx * WARMUP_STRIDE)
        
        if warmup_position + FORECAST_HORIZON > len(df):
            break
            
        # Get warmup window
        warmup_hist_start = warmup_position - INPUT_CHUNK_LENGTH
        warmup_hist_end = warmup_position
        
        warmup_input = target_scaled[warmup_hist_start:warmup_hist_end]
        warmup_covariates = past_covariates_scaled[warmup_hist_start:warmup_hist_end]
        
        # Make warmup prediction
        if warmup_idx % 3 == 0:  # Print every 3rd to avoid clutter
            print(f"   Warmup {warmup_idx + 1}/{WARMUP_WINDOWS} at {df.index[warmup_position].strftime('%H:%M')}...")
        
        warmup_forecast = model.predict(
            n=FORECAST_HORIZON,
            series=warmup_input,
            past_covariates=warmup_covariates,
            num_samples=500  # ⚡ INCREASED to 500
        )
        
        # Check warmup values to see if model is stabilizing
        if warmup_idx == WARMUP_WINDOWS - 1:  # Last warmup
            # Get P99 for warmup
            warmup_p99 = warmup_forecast.quantile(0.99)
            warmup_p99_unscaled = target_scaler.inverse_transform(warmup_p99)
            warmup_values = warmup_p99_unscaled.univariate_values()
            print(f"     Final warmup P99 forecast range: {warmup_values.min():.3f} to {warmup_values.max():.3f}")
    
    print("✅ Model warmup completed!\n")
    
    # ========== MAIN PREDICTION LOOP ==========
    # Initialize prediction columns in the dataframe
    print("🔮 Starting continuous rolling window forecasting (1-minute updates)...")
    df['Forecast_HCDP'] = np.nan  # Keep original name for backward compatibility
    df['Forecast_Median'] = np.nan  # P50 median forecast
    df['Forecast_Conservative'] = np.nan  # P99 ultra-conservative forecast
    df['P10_HCDP'] = np.nan
    df['P50_HCDP'] = np.nan
    df['P90_HCDP'] = np.nan
    df['P99_HCDP'] = np.nan  # ⚡ NEW: P99 column
    df['Uncertainty_Range'] = np.nan
    df['Exceeds_Threshold'] = False
    df['Exceeds_Threshold_Conservative'] = False  # Based on P99
    df['Forecast_Window'] = np.nan
    
    # Rolling window forecasting
    exceedance_detected = False
    exceedance_window = None
    
    # Start at 10:05 (605 minutes from start)
    start_minute = 605
    
    # Extend DataFrame to accommodate future predictions (12 minutes beyond last actual data)
    print("📈 Extending DataFrame for future predictions...")
    last_timestamp = df.index[-1]
    future_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(minutes=1), 
                                     periods=FORECAST_HORIZON, 
                                     freq='1min')
    
    # Create future rows with NaN values
    future_df = pd.DataFrame(index=future_timestamps)
    for col in df.columns:
        future_df[col] = np.nan
    
    # Append future rows to main DataFrame
    df = pd.concat([df, future_df])
    print(f"   ✅ Extended DataFrame to include predictions up to {df.index[-1]}")
    
    # Use 1-minute stride for continuous monitoring
    PREDICTION_STRIDE = 1
    
    # Calculate how many windows we need
    remaining_minutes = len(df) - start_minute
    num_windows = remaining_minutes - FORECAST_HORIZON + 1
    
    print(f"   📈 Will process {num_windows} forecast windows (1 per minute)")
    print(f"   📊 Starting from minute {start_minute} ({df.index[start_minute]})")
    print(f"   ⚡ Continuous monitoring: New 12-minute forecast every minute")
    print(f"   🎯 Using Ultra-Conservative (P99) forecast with 500 samples\n")
    
    window_count = 0
    last_progress_update = 0
    
    for current_position in range(start_minute, len(df), PREDICTION_STRIDE):
        window_count += 1
        
        # Get historical window (96 minutes before current position)
        hist_start = current_position - INPUT_CHUNK_LENGTH
        hist_end = current_position
        
        # Extract input data
        input_series = target_scaled[hist_start:hist_end]
        input_past_covs = past_covariates_scaled[hist_start:hist_end]
        
        # Progress update every 10%
        progress = (window_count / num_windows) * 100
        if progress - last_progress_update >= 10:
            print(f"   Progress: {progress:.0f}% - Window {window_count}/{num_windows} - Time: {df.index[current_position]}")
            last_progress_update = progress
        
        try:
            # Make prediction with MORE samples
            forecast = model.predict(
                n=FORECAST_HORIZON,
                series=input_series,
                past_covariates=input_past_covs,
                num_samples=500  # ⚡ INCREASED from 200 to 500
            )
            
            # Get quantiles including P99
            p10_scaled = forecast.quantile(0.1)
            p50_scaled = forecast.quantile(0.5)
            p90_scaled = forecast.quantile(0.9)
            p99_scaled = forecast.quantile(0.99)  # ⚡ NEW: P99 for extreme safety
            
            p10_unscaled = target_scaler.inverse_transform(p10_scaled)
            p50_unscaled = target_scaler.inverse_transform(p50_scaled)
            p90_unscaled = target_scaler.inverse_transform(p90_scaled)
            p99_unscaled = target_scaler.inverse_transform(p99_scaled)
            
            p10_values = p10_unscaled.univariate_values().flatten()
            p50_values = p50_unscaled.univariate_values().flatten()
            p90_values = p90_unscaled.univariate_values().flatten()
            p99_values = p99_unscaled.univariate_values().flatten()
            
            # ⚡ UPDATED: Use P99 as the ultra-conservative forecast
            forecast_median = p50_values  # The "typical" expected value
            forecast_conservative = p99_values  # ⚡ CHANGED from P90 to P99!
            
            # For backward compatibility, set main forecast to ultra-conservative
            forecast_values = forecast_conservative
            
            # Store predictions for all 12 minutes ahead
            for i in range(min(FORECAST_HORIZON, len(df) - current_position)):
                idx = current_position + i
                # Only store if this cell hasn't been filled yet
                if pd.isna(df.iloc[idx, df.columns.get_loc('Forecast_HCDP')]):
                    df.iloc[idx, df.columns.get_loc('Forecast_HCDP')] = forecast_values[i]
                    df.iloc[idx, df.columns.get_loc('Forecast_Median')] = forecast_median[i]
                    df.iloc[idx, df.columns.get_loc('Forecast_Conservative')] = forecast_conservative[i]
                    df.iloc[idx, df.columns.get_loc('P10_HCDP')] = p10_values[i]
                    df.iloc[idx, df.columns.get_loc('P50_HCDP')] = p50_values[i]
                    df.iloc[idx, df.columns.get_loc('P90_HCDP')] = p90_values[i]
                    df.iloc[idx, df.columns.get_loc('P99_HCDP')] = p99_values[i]
                    df.iloc[idx, df.columns.get_loc('Uncertainty_Range')] = p99_values[i] - p10_values[i]
                    df.iloc[idx, df.columns.get_loc('Exceeds_Threshold')] = forecast_median[i] > THRESHOLD_UNSCALED
                    df.iloc[idx, df.columns.get_loc('Exceeds_Threshold_Conservative')] = forecast_conservative[i] > THRESHOLD_UNSCALED
                    df.iloc[idx, df.columns.get_loc('Forecast_Window')] = window_count
            
            # Debug first real prediction
            if window_count == 1:
                print(f"\n   🔍 First prediction after warmup:")
                print(f"      Time: {df.index[current_position]}")
                print(f"      Actual value: {df[TARGET_COL].iloc[current_position]:.3f}")
                print(f"      Forecast Median (P50): {forecast_median[0]:.3f}")
                print(f"      Forecast P90: {p90_values[0]:.3f}")
                print(f"      Forecast Ultra-Conservative (P99): {forecast_conservative[0]:.3f}")
                print(f"      Spread (P99-P50): {forecast_conservative[0] - forecast_median[0]:.3f}")
                print()
            
            # Check for exceedance using ULTRA-CONSERVATIVE forecast (P99)
            if any(forecast_conservative > THRESHOLD_UNSCALED):
                if not exceedance_detected:  # First detection
                    exceedance_detected = True
                    exceedance_window = window_count
                    
                    # Find when in the forecast the exceedance occurs
                    exceedance_minutes_median = [i for i, v in enumerate(forecast_median) if v > THRESHOLD_UNSCALED]
                    exceedance_minutes_p90 = [i for i, v in enumerate(p90_values) if v > THRESHOLD_UNSCALED]
                    exceedance_minutes_conservative = [i for i, v in enumerate(forecast_conservative) if v > THRESHOLD_UNSCALED]
                    
                    first_exceedance_conservative = exceedance_minutes_conservative[0] + 1 if exceedance_minutes_conservative else None
                    first_exceedance_p90 = exceedance_minutes_p90[0] + 1 if exceedance_minutes_p90 else None
                    first_exceedance_median = exceedance_minutes_median[0] + 1 if exceedance_minutes_median else None
                    
                    print(f"\n🚨 EXCEEDANCE DETECTED!")
                    print(f"   Current time: {df.index[current_position]}")
                    if first_exceedance_conservative:
                        print(f"   Ultra-Conservative (P99) exceedance in {first_exceedance_conservative} minutes")
                    if first_exceedance_p90:
                        print(f"   P90 exceedance in {first_exceedance_p90} minutes")
                    if first_exceedance_median:
                        print(f"   Median (P50) exceedance in {first_exceedance_median} minutes")
                    else:
                        print(f"   Median forecast stays below threshold")
                    print(f"   Max P99 forecast: {np.max(forecast_conservative):.2f}°C")
                    print(f"   Max P90 forecast: {np.max(p90_values):.2f}°C")
                    print(f"   Max median forecast: {np.max(forecast_median):.2f}°C")
                    print(f"   Operator has {first_exceedance_conservative} minutes to take action!")
                    # Don't break here - continue to complete this forecast window
                    # The exceedance will be handled after the loop completes
                
        except Exception as e:
            print(f"Error in window {window_count} at position {current_position}: {e}")
            continue
    
    print(f"\n📊 Continuous monitoring completed!")
    print(f"   Windows processed: {window_count}")
    print(f"   Update frequency: Every 1 minute")
    print(f"   Forecast horizon maintained: 12 minutes ahead")
    print(f"   Samples per forecast: 500")
    
    if exceedance_detected:
        print(f"   ⚠️ Exceedance detected with advance warning")
    else:
        print(f"   ✅ No exceedance detected in any forecast window")

    # Save results
    print("\n💾 Saving results...")
    output_csv_path = session_path / f"data_with_rolling_forecast_{session_number}.csv"
    df.to_csv(output_csv_path)
    print(f"   💾 Data with continuous forecasts saved to: {output_csv_path}")
    
    # Create filtered CSV with only data from prediction start time onwards
    print("📊 Creating filtered forecast CSV...")
    start_minute = 605  # 10:05 AM
    filtered_df = df.iloc[start_minute:].copy()
    
    # Save filtered CSV to the specified location
    filtered_csv_path = Path("/Users/oribenyosef/Correlation-To-Action-2/data/snapshot for full model/19-6-25-raw-forecast.csv")
    filtered_df.to_csv(filtered_csv_path, index=True)
    print(f"   💾 Filtered forecast data saved to: {filtered_csv_path}")
    print(f"   📊 Filtered data: {len(filtered_df)} rows from {filtered_df.index[0]} to {filtered_df.index[-1]}")

    # Display summary statistics
    forecast_mask = ~df['Forecast_Conservative'].isna()
    if forecast_mask.sum() > 0:
        print(f"\n📈 Forecast Summary:")
        print(f"   Total forecast points: {forecast_mask.sum()}")
        print(f"   Ultra-Conservative (P99) Forecasts:")
        print(f"     - Max: {df.loc[forecast_mask, 'Forecast_Conservative'].max():.2f}°C")
        print(f"     - Min: {df.loc[forecast_mask, 'Forecast_Conservative'].min():.2f}°C")
        print(f"     - Exceedances: {(df.loc[forecast_mask, 'Exceeds_Threshold_Conservative'] == True).sum()}")
        print(f"   P90 Forecasts:")
        print(f"     - Max: {df.loc[forecast_mask, 'P90_HCDP'].max():.2f}°C")
        print(f"     - Min: {df.loc[forecast_mask, 'P90_HCDP'].min():.2f}°C")
        print(f"   Median (P50) Forecasts:")
        print(f"     - Max: {df.loc[forecast_mask, 'Forecast_Median'].max():.2f}°C")
        print(f"     - Min: {df.loc[forecast_mask, 'Forecast_Median'].min():.2f}°C")
        print(f"     - Exceedances: {(df.loc[forecast_mask, 'Exceeds_Threshold'] == True).sum()}")
        print(f"   Average uncertainty range (P99-P10): {df.loc[forecast_mask, 'Uncertainty_Range'].mean():.3f}°C")

    if not exceedance_detected:
        print("\n✅ No HCDP exceedance forecasted. Pipeline completed.")
        total_time = time.time() - start_time
        print(f"⏱️ Total runtime: {total_time:.2f} seconds")
        print(f"📁 Results saved to: {session_path}")
        return total_time

    print("\n🚨 HCDP Exceedance Detected! Launching mitigation modules...")

    # 1. Run SHAP Analysis
    print("🔍 Running SHAP explainer...")
    run_shap_explainer(MODEL_PATH, CSV_FORECAST_PATH, str(session_path))


    # 2. Run Regime Clustering
    print("🎯 Running regime clustering...")
    run_regime_clustering(CSV_PATH, str(session_path))

    # 3. Run Scenario Simulation
    print("🎲 Running scenario simulation...")
    run_scenario_simulation(MODEL_PATH, CSV_PATH, str(session_path))

    # 4. Launch LLM Report (Optional)
    # print("📝 Generating LLM report...")
    # generate_llm_report(input_csv=CSV_PATH, output_folder=session_path)

    total_time = time.time() - start_time
    print(f"\n✅ Full pipeline completed in {total_time:.2f} seconds.")
    print(f"📁 Results saved to: {session_path}")
    
    return total_time

if __name__ == "__main__":
    print("=" * 60)
    print("🏭 HCDP Monitoring and Reporting Pipeline")
    print("=" * 60)
    run_pipeline()