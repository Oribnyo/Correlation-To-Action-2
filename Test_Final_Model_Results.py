
# === ALL-IN-ONE TFT MODEL EVALUATION MODULE ===

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import TFTModel
from darts.metrics import mae, mql
from darts.explainability.tft_explainer import TFTExplainer

# === CONFIGURATION ===
model_path = "/Users/oribenyosef/Correlation-To-Action-2/TFT_FINAL_MODEL_100_EPOCH.pt"
results_path = "/Users/oribenyosef/Correlation-To-Action-2/results/final_model/"
os.makedirs(results_path, exist_ok=True)

# === LOAD MODEL AND DATA ===
model = TFTModel.load(model_path)

# Import the variables from your training script
import sys
sys.path.append('/Users/oribenyosef/Correlation-To-Action-2')

# Load the saved scalers and data (you'll need to save these in Train_Final_model.py)
import pickle

# Load saved data (add this saving code to your Train_Final_model.py):
with open('/Users/oribenyosef/Correlation-To-Action-2/evaluation_data.pkl', 'rb') as f:
    data = pickle.load(f)
    # Use the actual keys that exist in the file
    target_test_original = data['target_test_original']  # This is the actual test data
    test_pred = data['test_pred']  # This is the predictions
    quantiles = data['quantiles']  # Quantiles used
    
    print("🎯 Loaded available data from evaluation file:")
    print(f"  - target_test_original: {type(target_test_original)}")
    print(f"  - test_pred: {type(test_pred)}")
    print(f"  - quantiles: {len(quantiles)} quantiles")
    
    # Load additional data if available
    past_covariates_scaled_all = data.get('past_covariates_scaled_all', None)
    past_covariates = data.get('past_covariates', None)
    
    if past_covariates_scaled_all is not None:
        print(f"  - past_covariates_scaled_all: {type(past_covariates_scaled_all)} with {past_covariates_scaled_all.width} features")
    if past_covariates is not None:
        print(f"  - past_covariates: {type(past_covariates)} with columns: {past_covariates.columns[:5].tolist()}...")
    
print("🎯 Loaded available data from evaluation file:")
print(f"  - target_test_original: {type(target_test_original)}")
print(f"  - test_pred: {type(test_pred)}")
print(f"  - quantiles: {len(quantiles)} quantiles")

# === PREDICTION ===
# Skip prediction since we already have test_pred from the saved file
print("✅ Using existing predictions from saved file")

# We already have:
# - target_test_original (actual test values)
# - test_pred (model predictions)
# - quantiles (quantile definitions)

print(f"📊 Test data shape: {target_test_original.values().shape}")
print(f"📊 Predictions shape: {test_pred.values().shape}")

# === METRICS ===
# Use the quantiles from the saved data
print("📊 Calculating metrics...")
pinball_losses = {q: mql(target_test_original, test_pred, q=q) for q in quantiles}
mae_val = mae(target_test_original, test_pred.quantile(0.5))  # Use median prediction
mql_p50_val = mql(target_test_original, test_pred, q=0.5)
mql_p90_val = mql(target_test_original, test_pred, q=0.9)
mean_pinball = np.mean(list(pinball_losses.values()))

print(f"✅ MAE: {mae_val:.4f}")
print(f"✅ MQL P50: {mql_p50_val:.4f}")
print(f"✅ MQL P90: {mql_p90_val:.4f}")

metrics_df = pd.DataFrame({
    "Metric": [f"Pinball Loss q={q:.2f}" for q in quantiles] + ["MAE", "MQL P50", "MQL P90", "Mean Pinball Loss"],
    "Value": list(pinball_losses.values()) + [mae_val, mql_p50_val, mql_p90_val, mean_pinball]
})
metrics_df.to_csv(os.path.join(results_path, "Graph_1_Metrics.csv"), index=False)
print("💾 Metrics saved to Graph_1_Metrics.csv")

# === GRAPH 1 === - Prediction VS test data
plt.figure(figsize=(14, 6))
target_test_original.plot(label="Actual", color="black", linewidth=2)
test_pred.plot(low_quantile=0.01, high_quantile=0.99, label="P1-P99", alpha=0.1, color="lightgray")
test_pred.plot(low_quantile=0.1, high_quantile=0.9, label="P10-P90", alpha=0.4, color="orange")
test_pred.plot(low_quantile=0.5, high_quantile=0.5, label="P50", color="red", linewidth=1.5)

metrics_text = f"MAE: {mae_val:.4f}\nP50 QLoss: {pinball_losses[0.5]:.4f}\nP90 QLoss: {pinball_losses[0.9]:.4f}\nMean QLoss: {mean_pinball:.4f}"
plt.text(0.01, 0.99, metrics_text, transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
plt.title("Graph 1 - Predictions VS Test Data")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_path, "Graph_1_Prediction_Intervals.png"))
plt.close()

# === GRAPH 2 === - Pinball Loss Across Quantiles
plt.figure(figsize=(10, 6))
plt.plot(quantiles, list(pinball_losses.values()), marker='o', color='orange')
plt.title("Graph 2 - Pinball Loss Across Quantiles")
plt.xlabel("Quantile")
plt.ylabel("Pinball Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_path, "Graph_2_Pinball_Loss_Quantiles.png"))
plt.close()

# === GRAPH 3 === - Interval Width (P90 - P10)
# Graph 3 Explanation:
# This plot visualizes the width of the prediction interval between the 90th and 10th quantiles.
# It represents the model's forecast uncertainty over time.
# Narrow widths imply higher confidence in predictions, while wider widths reflect increased uncertainty.
# Monitoring this can reveal patterns, e.g., whether uncertainty spikes during certain events or time windows.
width = test_pred.quantile(0.9) - test_pred.quantile(0.1)
width.plot()
plt.title("Graph 3 - Interval Width (P90 - P10)")
plt.tight_layout()
plt.savefig(os.path.join(results_path, "Graph_3_Interval_Width.png"))
plt.close()

# === GRAPH 4 === - Absolute Error Over Time
# Graph 4 Explanation:
# This plot shows the absolute difference between actual and predicted values over time.
# It helps identify systematic errors or sudden spikes in prediction accuracy.
# Monitoring this can reveal patterns, e.g., whether errors increase during certain events or time windows.
abs_error_vals = np.abs((target_test_original - test_pred.quantile(0.5)).values())
abs_error = TimeSeries.from_times_and_values(target_test_original.time_index, abs_error_vals)
abs_error.plot()
plt.title("Graph 4 - Absolute Error Over Time")
plt.tight_layout()
plt.savefig(os.path.join(results_path, "Graph_4_Absolute_Error.png"))
plt.close()

# === GRAPH 5 === - Coverage Rate (Rolling 24h)
# Graph 5 Explanation:
# This plot shows the percentage of actual values that fall within the prediction interval over time.
# It helps assess the model's coverage rate, i.e., how often the prediction interval contains the actual value.
# Monitoring this can reveal patterns, e.g., whether coverage rates vary significantly over time. 
actual_vals = target_test_original.values()
p10_vals = test_pred.quantile(0.1).values()
p90_vals = test_pred.quantile(0.9).values()
covered = np.logical_and(actual_vals >= p10_vals, actual_vals <= p90_vals).astype(int)
coverage_rate = pd.Series(covered.flatten(), index=target_test_original.time_index)
coverage_rate.rolling(24).mean().plot()
plt.title("Graph 5 - Coverage Rate (Rolling 24h)")
plt.tight_layout()
plt.savefig(os.path.join(results_path, "Graph_5_Coverage_Rate.png"))
plt.close()

# === GRAPH 6 === - Absolute Error by Hour of Day
# Graph 6 Explanation:
# This plot shows the average absolute error by hour of day.
# It helps identify whether errors are more frequent at certain times of day.
# Monitoring this can reveal patterns, e.g., whether errors increase during certain hours of the day.
error_df = pd.DataFrame({
    "timestamp": target_test_original.time_index,
    "abs_error": abs_error.values().flatten()
})
error_df["hour"] = error_df["timestamp"].dt.hour
hourly_mean = error_df.groupby("hour")["abs_error"].mean()
hourly_mean.plot(kind="bar")
plt.title("Graph 6 - Absolute Error by Hour of Day")
plt.tight_layout()
plt.savefig(os.path.join(results_path, "Graph_6_Error_by_Hour.png"))
plt.close()

# === GRAPH 7 === - TFT Feature Importance (Variable Selection Weights)
# Graph 7 Explanation:
# This plot shows the importance of each feature in the TFT model.
# It helps identify which features are most important for the model's predictions.
# Monitoring this can reveal patterns, e.g., whether certain features are more important at certain times.

try:
    print("📊 Generating Graph 7 - TFT Feature Importance...")
    print(f"   Model type: {type(model)}")
    print(f"   Model loaded from: {model_path}")
    
    explainer = TFTExplainer(model)
    print("   ✅ TFTExplainer created successfully")
    
    explanation = explainer.explain()
    print(f"   ✅ Explanation generated: {type(explanation)}")
    print(f"   Available explanation methods: {dir(explanation)}")
    
    # Create a new figure explicitly
    plt.figure(figsize=(12, 8))
    explainer.plot_variable_selection(explanation)
    plt.tight_layout()
    
    # Save with explicit path and format
    graph7_path = os.path.join(results_path, "Graph_7_TFT_Feature_Importance.png")
    plt.savefig(graph7_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Graph 7 saved to: {graph7_path}")
    plt.close()
    
except Exception as e:
    import traceback
    print(f"❌ Error generating Graph 7: {e}")
    print(f"   Full traceback: {traceback.format_exc()}")
    # Create a placeholder plot
    plt.figure(figsize=(12, 8))
    plt.text(0.5, 0.5, f"TFT Feature Importance\nError: {str(e)}", 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
    plt.title("Graph 7 - TFT Feature Importance (Error)")
    plt.savefig(os.path.join(results_path, "Graph_7_TFT_Feature_Importance.png"), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# === GRAPH 8 === - TFT Attention Heatmap
# Graph 8 Explanation:
# This heatmap shows the attention mechanism of the TFT across the input time window.
# It indicates how much the model attends to past time steps when predicting future values.
# High attention scores (darker regions) signal strong influence.
# This helps explain temporal dependencies and the model's focus on long vs short-term history.

try:
    print("📊 Generating Graph 8 - TFT Attention Heatmap...")
    
    # Create a new figure explicitly
    plt.figure(figsize=(14, 10))
    explainer.plot_attention(explanation, plot_type="heatmap")
    plt.tight_layout()
    
    # Save with explicit path and format
    graph8_path = os.path.join(results_path, "Graph_8_TFT_Attention_Heatmap.png")
    plt.savefig(graph8_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Graph 8 saved to: {graph8_path}")
    plt.close()
    
except Exception as e:
    import traceback
    print(f"❌ Error generating Graph 8: {e}")
    print(f"   Full traceback: {traceback.format_exc()}")
    # Create a placeholder plot
    plt.figure(figsize=(14, 10))
    plt.text(0.5, 0.5, f"TFT Attention Heatmap\nError: {str(e)}", 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
    plt.title("Graph 8 - TFT Attention Heatmap (Error)")
    plt.savefig(os.path.join(results_path, "Graph_8_TFT_Attention_Heatmap.png"), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


# === GRAPH 9: Encoder Variable Importance (Custom Bar Plot) ===
# This graph shows the average importance of each encoder variable (feature) across all forecast horizons.

try:
    print("📊 Generating Graph 9 - Encoder Variable Importance (Custom Bar Plot)...")
    print(f"   Available methods: {[m for m in dir(explanation) if 'importance' in m.lower() or 'encoder' in m.lower()]}")
    
    # Try different methods to get feature importances
    fi_result = None
    method_used = None
    
    # Try get_feature_importances first
    try:
        fi_result = explanation.get_feature_importances()
        method_used = "get_feature_importances"
        print(f"   Tried {method_used}: {type(fi_result)}")
    except:
        pass
    
    # Try get_encoder_importance if first method failed
    if fi_result is None:
        try:
            fi_result = explanation.get_encoder_importance()
            method_used = "get_encoder_importance"
            print(f"   Tried {method_used}: {type(fi_result)}")
        except:
            pass
    
    # Try feature_importances attribute
    if fi_result is None:
        try:
            fi_result = explanation.feature_importances
            method_used = "feature_importances"
            print(f"   Tried {method_used}: {type(fi_result)}")
        except:
            pass
    
    if fi_result is not None:
        print(f"   ✅ Got feature importances using {method_used}")
        print(f"   Type: {type(fi_result)}")
        
        # Handle different return types
        if isinstance(fi_result, dict):
            print(f"   Dict keys: {list(fi_result.keys())}")
            fi_matrix = fi_result.get('importances')
            print(f"   Importances value: {type(fi_matrix)} - {fi_matrix}")
            
            if fi_matrix is None:
                # Try other keys in the dictionary
                for key in fi_result.keys():
                    val = fi_result[key]
                    if hasattr(val, 'shape'):
                        print(f"   Found matrix in key '{key}': shape {val.shape}")
                        fi_matrix = val
                        break
            
            if fi_matrix is not None and hasattr(fi_matrix, 'shape'):
                # Try to get real feature names from the data
                try:
                    # Get the actual sensor/column names from the saved data
                    actual_feature_names = None
                    if past_covariates is not None:
                        actual_feature_names = past_covariates.columns.tolist()
                        print(f"   Found {len(actual_feature_names)} actual feature names from past_covariates: {actual_feature_names[:5]}...")
                    elif past_covariates_scaled_all is not None:
                        actual_feature_names = past_covariates_scaled_all.columns.tolist()
                        print(f"   Found {len(actual_feature_names)} actual feature names from past_covariates_scaled_all: {actual_feature_names[:5]}...")
                    
                    if actual_feature_names is not None:
                        # Use actual names if the count matches
                        if len(actual_feature_names) == fi_matrix.shape[1]:
                            feature_names = actual_feature_names
                            print("   ✅ Using actual sensor/column names")
                        else:
                            print(f"   ⚠️ Feature count mismatch: {len(actual_feature_names)} vs {fi_matrix.shape[1]}")
                            feature_names = [f"Feature {i}" for i in range(fi_matrix.shape[1])]
                    else:
                        # Fallback to generic names
                        feature_names = [f"Feature {i}" for i in range(fi_matrix.shape[1])]
                        print("   Using generic feature names (no covariate data found)")
                except Exception as e:
                    print(f"   ⚠️ Could not get actual feature names: {e}")
                    feature_names = [f"Feature {i}" for i in range(fi_matrix.shape[1])]
            else:
                # Try encoder importance method instead
                print("   Trying get_encoder_importance as fallback...")
                try:
                    encoder_result = explanation.get_encoder_importance()
                    print(f"   Encoder importance type: {type(encoder_result)}")
                    if hasattr(encoder_result, 'shape'):
                        fi_matrix = encoder_result
                        # Try to get actual names here too
                        if 'past_covariates_scaled_all' in data and len(data['past_covariates_scaled_all'].columns) == fi_matrix.shape[1]:
                            feature_names = data['past_covariates_scaled_all'].columns.tolist()
                        else:
                            feature_names = [f"Encoder {i}" for i in range(fi_matrix.shape[1])]
                    elif isinstance(encoder_result, dict):
                        print(f"   Encoder dict keys: {list(encoder_result.keys())}")
                        for key, val in encoder_result.items():
                            if hasattr(val, 'shape'):
                                fi_matrix = val
                                # Try to get actual names here too
                                if 'past_covariates_scaled_all' in data and len(data['past_covariates_scaled_all'].columns) == fi_matrix.shape[1]:
                                    feature_names = data['past_covariates_scaled_all'].columns.tolist()
                                else:
                                    feature_names = [f"Encoder {i}" for i in range(fi_matrix.shape[1])]
                                break
                except Exception as ee:
                    print(f"   Encoder importance failed: {ee}")
                    
        elif hasattr(fi_result, 'shape'):
            fi_matrix = fi_result
            feature_names = [f"Feature {i}" for i in range(fi_matrix.shape[1])]
        else:
            print(f"   ⚠️ Unexpected return type: {type(fi_result)}")
            fi_matrix = None
            
        if fi_matrix is not None and hasattr(fi_matrix, 'shape'):
            print(f"   Matrix shape: {fi_matrix.shape}")
            
            # Compute mean importance for each feature
            mean_importance = fi_matrix.mean(axis=0)
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Mean Importance": mean_importance
            }).sort_values("Mean Importance", ascending=True)

            plt.figure(figsize=(10, max(6, len(feature_names) * 0.4)))
            plt.barh(importance_df["Feature"], importance_df["Mean Importance"], color="skyblue")
            plt.xlabel("Mean Importance")
            plt.title("Graph 9 - Encoder Variable Importance (Mean Across Horizons)")
            plt.tight_layout()
            plt.savefig(os.path.join(results_path, "Graph_9_Encoder_Variable_Importance.png"), dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print("✅ Graph 9 saved as Graph_9_Encoder_Variable_Importance.png")

            # Also create fi_df for use in Graph 10
            fi_df = pd.DataFrame(fi_matrix, columns=feature_names)
        else:
            print("   ❌ Could not extract valid matrix from feature importances")
            fi_df = None
    else:
        print("   ❌ Could not get feature importances using any method")
        fi_df = None
        
except Exception as e:
    import traceback
    print(f"❌ Error generating Graph 9: {e}")
    print(f"   Full traceback: {traceback.format_exc()}")
    plt.figure(figsize=(10, 4))
    plt.text(0.5, 0.5, f"Encoder Variable Importance\nError: {str(e)}", 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
    plt.title("Graph 9 - Encoder Variable Importance (Error)")
    plt.savefig(os.path.join(results_path, "Graph_9_Encoder_Variable_Importance.png"), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    fi_df = None  # So later code doesn't break if this fails

# === GRAPH 10: SHAP Feature Importance by Hour of Day ===
# 🗓 Graph 10 Explanation:
# This plot aggregates total feature importance by hour-of-day, revealing cyclic dependencies.
# It shows whether the model leans more on certain features during specific hours (e.g., daytime vs night).

try:
    if fi_df is not None:
        print("📊 Generating Graph 10 - SHAP Feature Importance by Hour of Day...")
        print(f"   fi_df shape: {fi_df.shape}")
        
        if fi_df.shape[0] == 1:
            # Single timestep - create a different visualization showing importance by feature
            print("   Single timestep detected - creating feature ranking instead")
            
            plt.figure(figsize=(12, 8))
            feature_importance = fi_df.iloc[0].sort_values(ascending=True)
            plt.barh(range(len(feature_importance)), feature_importance.values)
            plt.yticks(range(len(feature_importance)), feature_importance.index, rotation=0)
            plt.xlabel("Feature Importance")
            plt.title("Graph 10 - Feature Importance Ranking (Single Timestep)")
            plt.tight_layout()
            plt.savefig(os.path.join(results_path, "Graph_10_Feature_Ranking.png"), dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print("✅ Graph 10 saved as Graph_10_Feature_Ranking.png")
        else:
            # Multiple timesteps - original hourly analysis
            shap_hourly_df = pd.DataFrame({
                "timestamp": target_test_original.time_index,
                "shap_total": fi_df.sum(axis=1).values[:len(target_test_original)]  # truncate to length
            })
            shap_hourly_df["hour"] = shap_hourly_df["timestamp"].dt.hour
            hourly_mean_importance = shap_hourly_df.groupby("hour")["shap_total"].mean()
            
            plt.figure(figsize=(12, 6))
            hourly_mean_importance.plot(kind="bar")
            plt.title("Graph 10 - SHAP Feature Importance by Hour of Day")
            plt.xlabel("Hour of Day")
            plt.ylabel("Mean Feature Importance")
            plt.tight_layout()
            plt.savefig(os.path.join(results_path, "Graph_10_SHAP_by_Hour.png"), dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print("✅ Graph 10 saved as Graph_10_SHAP_by_Hour.png")
    else:
        print("⚠️ Skipping Graph 10 - No feature importance data available")
        # Create placeholder
        plt.figure(figsize=(12, 6))
        plt.text(0.5, 0.5, "SHAP Feature Importance by Hour\nNo feature importance data available", 
                 ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        plt.title("Graph 10 - SHAP Feature Importance by Hour (No Data)")
        plt.savefig(os.path.join(results_path, "Graph_10_SHAP_by_Hour.png"), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
except Exception as e:
    print(f"❌ Error generating Graph 10: {e}")
    plt.figure(figsize=(12, 6))
    plt.text(0.5, 0.5, f"SHAP Feature Importance by Hour\nError: {str(e)}", 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
    plt.title("Graph 10 - SHAP Feature Importance by Hour (Error)")
    plt.savefig(os.path.join(results_path, "Graph_10_SHAP_by_Hour.png"), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# === GRAPH 11: SHAP Heatmap Weighted by Residual Error ===
# 🌊 Graph 11 Explanation:
# This heatmap blends SHAP values with residual errors.
# It highlights which features contribute most during inaccurate predictions (high error points).

try:
    if fi_df is not None and 'fi_matrix' in locals() and 'feature_names' in locals():
        print("📊 Generating Graph 11 - SHAP Heatmap Weighted by Residual Error...")
        print(f"   fi_matrix shape: {fi_matrix.shape}")
        print(f"   residuals shape: {abs_error.values().flatten().shape}")
        
        if fi_matrix.shape[0] == 1:
            # Single timestep - create correlation between feature importance and overall prediction error
            print("   Single timestep detected - creating feature-error correlation")
            
            residuals = abs_error.values().flatten()
            overall_error = residuals.mean()  # Single error value
            
            # Create a bar plot showing feature importance scaled by overall error
            feature_importance = fi_matrix[0] * overall_error
            
            plt.figure(figsize=(12, 8))
            sorted_idx = np.argsort(feature_importance)
            plt.barh(range(len(feature_importance)), feature_importance[sorted_idx], color='coolwarm')
            plt.yticks(range(len(feature_importance)), [feature_names[i] for i in sorted_idx])
            plt.xlabel("Feature Importance × Overall Error")
            plt.title("Graph 11 - Feature Importance Weighted by Overall Prediction Error")
            plt.tight_layout()
            plt.savefig(os.path.join(results_path, "Graph_11_Feature_Error_Weighting.png"), dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print("✅ Graph 11 saved as Graph_11_Feature_Error_Weighting.png")
        else:
            # Multiple timesteps - original heatmap
            import seaborn as sns
            
            residuals = abs_error.values().flatten()
            weighted_shap = fi_matrix[:len(residuals)] * residuals[:, None]
            
            plt.figure(figsize=(14, 8))
            sns.heatmap(pd.DataFrame(weighted_shap, columns=feature_names).T, cmap="coolwarm", xticklabels=100)
            plt.title("Graph 11 - SHAP Weighted by Residual Error")
            plt.xlabel("Time Index")
            plt.ylabel("Features")
            plt.tight_layout()
            plt.savefig(os.path.join(results_path, "Graph_11_Error_Weighted_SHAP_Heatmap.png"), dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print("✅ Graph 11 saved as Graph_11_Error_Weighted_SHAP_Heatmap.png")
    else:
        print("⚠️ Skipping Graph 11 - No feature importance data available")
        plt.figure(figsize=(14, 8))
        plt.text(0.5, 0.5, "SHAP Weighted by Residual Error\nNo feature importance data available", 
                 ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        plt.title("Graph 11 - SHAP Weighted by Residual Error (No Data)")
        plt.savefig(os.path.join(results_path, "Graph_11_Error_Weighted_SHAP_Heatmap.png"), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
except Exception as e:
    print(f"❌ Error generating Graph 11: {e}")
    plt.figure(figsize=(14, 8))
    plt.text(0.5, 0.5, f"SHAP Weighted by Residual Error\nError: {str(e)}", 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
    plt.title("Graph 11 - SHAP Weighted by Residual Error (Error)")
    plt.savefig(os.path.join(results_path, "Graph_11_Error_Weighted_SHAP_Heatmap.png"), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# === GRAPH 12: Estimated Lag Between Features and Target ===
# ⏱ Graph 12 Explanation:
# This analysis estimates how many time steps (lags) it takes for each feature to influence the target.
# It computes lagged correlations up to `max_lag` hours and selects the best-correlated lag for each feature.
# This is useful for understanding the causal delay between inputs and response.

try:
    if fi_df is not None and 'feature_names' in locals() and (past_covariates_scaled_all is not None or past_covariates is not None):
        print("📊 Generating Graph 12 - Estimated Lag Between Features and Target...")
        max_lag = 48  # test up to 48 time steps (e.g., 2 days if hourly)
        feature_lags = {}
        
        # Use the available covariate data
        covariates_to_use = past_covariates_scaled_all if past_covariates_scaled_all is not None else past_covariates
        print(f"   Using covariates: {type(covariates_to_use)} with {len(feature_names)} features")
        
        for feat_name in feature_names:
            if feat_name in covariates_to_use.columns:
                feat_series = covariates_to_use[feat_name].pd_series()
                target_series = target_test_original.pd_series()
                correlations = [target_series.corr(feat_series.shift(lag)) for lag in range(1, max_lag + 1)]
                best_lag = np.argmax(np.abs(correlations)) + 1
                feature_lags[feat_name] = best_lag

        if feature_lags:
            plt.figure(figsize=(12, 8))
            pd.Series(feature_lags).sort_values().plot(kind="bar")
            plt.title("Graph 12 - Estimated Lag Between Features and Target")
            plt.xlabel("Feature")
            plt.ylabel("Lag in Hours")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(results_path, "Graph_12_Lag_Estimates.png"), dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print("✅ Graph 12 saved as Graph_12_Lag_Estimates.png")
        else:
            print("⚠️ No matching features found for lag analysis")
            plt.figure(figsize=(12, 8))
            plt.text(0.5, 0.5, "Estimated Lag Between Features and Target\nNo matching features found", 
                     ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
            plt.title("Graph 12 - Estimated Lag Between Features and Target (No Data)")
            plt.savefig(os.path.join(results_path, "Graph_12_Lag_Estimates.png"), 
                        dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
    else:
        print("⚠️ Skipping Graph 12 - No feature importance or covariate data available")
        print(f"   fi_df: {fi_df is not None}")
        print(f"   feature_names available: {'feature_names' in locals()}")
        print(f"   past_covariates_scaled_all: {past_covariates_scaled_all is not None}")
        print(f"   past_covariates: {past_covariates is not None}")
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, "Estimated Lag Between Features and Target\nNo feature data available", 
                 ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        plt.title("Graph 12 - Estimated Lag Between Features and Target (No Data)")
        plt.savefig(os.path.join(results_path, "Graph_12_Lag_Estimates.png"), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
except Exception as e:
    print(f"❌ Error generating Graph 12: {e}")
    plt.figure(figsize=(12, 8))
    plt.text(0.5, 0.5, f"Estimated Lag Between Features and Target\nError: {str(e)}", 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
    plt.title("Graph 12 - Estimated Lag Between Features and Target (Error)")
    plt.savefig(os.path.join(results_path, "Graph_12_Lag_Estimates.png"), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()




print("🎉 MODEL EVALUATION COMPLETED!")
print("=" * 60)
print("📊 Generated Evaluation Graphs:")
print("  1. ✅ Prediction vs Test Data with Confidence Intervals")
print("  2. ✅ Pinball Loss Across Quantiles") 
print("  3. ✅ Prediction Interval Width (P90-P10)")
print("  4. ✅ Absolute Error Over Time")
print("  5. ✅ Coverage Rate (Rolling 24h)")
print("  6. ✅ Absolute Error by Hour of Day")
print(f"📁 Results saved to: {results_path}")
print("=" * 60)

# === DATASET STATISTICS ===
print("\n=== DATASET STATISTICS ===")

# Helper to compute stats for a TimeSeries
import csv

def compute_stats(ts, name, spike_threshold=5.4):
    n_timestamps = len(ts)
    n_variables = ts.width
    n_measurements = n_timestamps * n_variables
    values = ts.values().flatten()
    n_spikes = np.sum(values > spike_threshold)
    percent_spikes = 100 * n_spikes / n_timestamps if n_timestamps > 0 else 0
    return {
        'Set': name,
        'Number of timestamps': n_timestamps,
        'Number of variables': n_variables,
        'Number of measurements': n_measurements,
        f"Number of time steps 'Sensor 1' > {spike_threshold}": n_spikes,
        f"Time in minutes 'Sensor 1' > {spike_threshold}": n_spikes,  # 1-min resolution
        f"Percent of time 'Sensor 1' > {spike_threshold}": f"{percent_spikes:.2f}%"
    }

# Compute stats for test set
stats_test = compute_stats(target_test_original, 'Test Set')

# Try to load and compute stats for train+val set if available
stats_trainval = None
if 'target_train_scaled' in data and 'target_val_scaled' in data:
    trainval = data['target_train_scaled'].append(data['target_val_scaled'])
    stats_trainval = compute_stats(trainval, 'Combined Train+Validation Set')

# Save stats to CSV
stats_csv_path = os.path.join(results_path, 'dataset_statistics.csv')
with open(stats_csv_path, 'w', newline='') as csvfile:
    fieldnames = list(stats_test.keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    if stats_trainval:
        writer.writerow(stats_trainval)
    writer.writerow(stats_test)

print(f"✅ Dataset statistics saved to: {stats_csv_path}")
