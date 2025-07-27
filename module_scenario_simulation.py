"""
Module: Flowrate Scenario Simulation
Description: Find the optimal per-train reduction plan that prevents HCDP exceedance with the least total flowrate reduction.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from itertools import product
from datetime import timedelta

def run_scenario_simulation(model_path, csv_path, output_dir,
                            target_col="Sensor 1 [Hydrocarbon_Dew_Point_C]",
                            threshold_unscaled=5.00,
                            granularity=50,
                            forecast_horizon=12,
                            max_runtime_seconds=300,  # 5 minute timeout
                            train_cols=[
                                "Sensor 2 [Train_1_Gas_FlowRate_MMBTU_HR]",
                                "Sensor 6 [Train_2_Gas_FlowRate_MMBTU_HR]", 
                                "Sensor 10 [Train_3_Gas_FlowRate_MMBTU_HR]", 
                                "Sensor 14 [Train_4_Gas_FlowRate_MMBTU_HR]", 
                                "Sensor 18 [Train_5_Gas_FlowRate_MMBTU_HR]", 
                                "Sensor 22 [Train_6_Gas_FlowRate_MMBTU_HR]", 
                                "Sensor 26 [Train_7_Gas_FlowRate_MMBTU_HR]"]):

    print("🧠 Running optimal flowrate scenario simulation...")
    start_time = time.time()

    # Load and preprocess
    print("📊 Loading data...", end="", flush=True)
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    print(" Done!")
    
    # Create separate scalers for target and covariates
    print("🔧 Preparing scalers...", end="", flush=True)
    target_series = TimeSeries.from_dataframe(df, value_cols=[target_col])
    covariate_cols = [col for col in df.columns if col != target_col]
    past_covariates = TimeSeries.from_dataframe(df, value_cols=covariate_cols)
    
    # Scale target and covariates separately
    target_scaler = Scaler()
    cov_scaler = Scaler()
    
    target_scaled = target_scaler.fit_transform(target_series)
    past_covariates_scaled = cov_scaler.fit_transform(past_covariates)
    print(" Done!")

    # Load model
    print("🤖 Loading model...", end="", flush=True)
    model = TFTModel.load(model_path, map_location='cpu')
    print(" Done!")
    
    # Prepare input for prediction
    input_target = target_scaled[:-forecast_horizon]
    input_past_covariates = past_covariates_scaled[:-forecast_horizon]

    # Initial forecast
    print("🔮 Generating initial forecast...", end="", flush=True)
    forecast = model.predict(
        n=forecast_horizon, 
        series=input_target,
        past_covariates=input_past_covariates
    )
    
    # Inverse transform using target scaler
    forecast_unscaled = target_scaler.inverse_transform(forecast)
    forecast_values = forecast_unscaled.univariate_values()
    print(" Done!")

    # Check if exceedance exists
    max_forecast = np.max(forecast_values)
    if max_forecast <= threshold_unscaled:
        print(f"\n✅ No exceedance detected. Max forecast: {max_forecast:.2f}°C")
        return None

    print(f"\n⚠️ Exceedance detected! Max forecast: {max_forecast:.2f}°C > {threshold_unscaled}°C")
    
    # Find exceedance time steps
    exceeding_steps = np.where(forecast_values > threshold_unscaled)[0]
    print(f"   Exceedance at time steps: {exceeding_steps}")

    # Calculate total scenarios
    reduction_options = list(range(0, 125, granularity))
    total_scenarios = len(reduction_options) ** 7 * forecast_horizon
    print(f"\n🔍 Testing up to {total_scenarios:,} scenarios (timeout: {max_runtime_seconds}s)")
    print("   Reduction options per train: {}\n".format(reduction_options))

    # Initialize search
    base_df = df.copy()
    min_total_reduction = float('inf')
    best_plan = []
    scenarios_tested = 0
    last_update_time = time.time()

    # Progress bar setup
    print("Progress: 0%", end="", flush=True)

    # Main search loop
    for t in range(forecast_horizon):
        timestamp = df.index[-forecast_horizon + t]

        for combination in product(reduction_options, repeat=7):
            # Check timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > max_runtime_seconds:
                print(f"\n\n⏰ Timeout reached after {elapsed_time:.1f}s and {scenarios_tested:,} scenarios")
                break
            
            scenarios_tested += 1
            total_reduction = sum(combination)
            
            # Skip if already worse than best found
            if total_reduction >= min_total_reduction:
                continue
            
            # Update progress every second
            if time.time() - last_update_time > 1.0:
                progress_pct = (scenarios_tested / total_scenarios) * 100
                elapsed = time.time() - start_time
                scenarios_per_sec = scenarios_tested / elapsed
                eta = (total_scenarios - scenarios_tested) / scenarios_per_sec if scenarios_per_sec > 0 else 0
                
                print(f"\rProgress: {progress_pct:.1f}% | Scenarios: {scenarios_tested:,} | "
                      f"Speed: {scenarios_per_sec:.0f}/s | ETA: {eta:.0f}s | "
                      f"Best found: {min_total_reduction if min_total_reduction < float('inf') else 'None'}", 
                      end="", flush=True)
                last_update_time = time.time()
            
            # Apply reductions
            temp_df = base_df.copy()
            for i, sensor in enumerate(train_cols):
                temp_df.loc[timestamp:, sensor] -= combination[i]
                # Ensure non-negative
                temp_df.loc[timestamp:, sensor] = temp_df.loc[timestamp:, sensor].clip(lower=0)

            # Re-create time series with modified values
            sim_target = TimeSeries.from_dataframe(temp_df, value_cols=[target_col])
            sim_covariates = TimeSeries.from_dataframe(temp_df, value_cols=covariate_cols)
            
            # Scale using existing scalers
            sim_target_scaled = target_scaler.transform(sim_target)
            sim_covariates_scaled = cov_scaler.transform(sim_covariates)
            
            # Prepare simulation input
            sim_input_target = sim_target_scaled[:-forecast_horizon]
            sim_input_covariates = sim_covariates_scaled[:-forecast_horizon]
            
            # Predict with modified values
            sim_forecast = model.predict(
                n=forecast_horizon, 
                series=sim_input_target,
                past_covariates=sim_input_covariates
            )
            
            # Inverse transform
            sim_forecast_unscaled = target_scaler.inverse_transform(sim_forecast)
            sim_forecast_values = sim_forecast_unscaled.univariate_values()

            # Check if this prevents exceedance
            if np.max(sim_forecast_values) <= threshold_unscaled:
                if total_reduction < min_total_reduction:
                    best_plan = [(timestamp.strftime("%H:%M"), combination)]
                    min_total_reduction = total_reduction

        # Check if timeout was reached
        if time.time() - start_time > max_runtime_seconds:
            break

    # Clear progress line
    print("\n")
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n📊 Search Summary:")
    print(f"   Total scenarios tested: {scenarios_tested:,}")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Average speed: {scenarios_tested/total_time:.0f} scenarios/s")

    # Save result if found
    if best_plan:
        timestamp, reductions = best_plan[0]
        
        # Create detailed plan
        detailed_plan = {
            "Timestamp": timestamp,
            "Total_Reduction_MMBTU_HR": min_total_reduction,
        }
        
        # Add train-specific reductions
        train_names = ["Train_1", "Train_2", "Train_3", "Train_4", "Train_5", "Train_6", "Train_7"]
        for i, (train_name, reduction) in enumerate(zip(train_names, reductions)):
            detailed_plan[f"{train_name}_Reduction"] = reduction
        
        detailed_df = pd.DataFrame([detailed_plan])
        
        # Save to CSV
        output_path = os.path.join(output_dir, "optimal_flowrate_plan.csv")
        detailed_df.to_csv(output_path, index=False)
        
        # Print results
        print(f"\n✅ Optimal solution found!")
        print(f"   Total reduction: {min_total_reduction} MMBTU/HR")
        print(f"   Apply at: {timestamp}")
        print(f"\n   Reduction breakdown:")
        for train_name, reduction in zip(train_names, reductions):
            if reduction > 0:
                print(f"   - {train_name}: {reduction} MMBTU/HR")
        
        print(f"\n💾 Saved to: {output_path}")
        return detailed_df
    else:
        print("\n⚠️ No effective reduction combination found within limits.")
        print("   Consider:")
        print("   - Increasing max reduction per train")
        print("   - Reducing granularity for finer control")
        print("   - Extending timeout duration")
        return None