import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from darts.explainability import TFTExplainer
import json
# === SENSOR NAME MAPPING ===
SENSOR_NAME_MAP = {
    'Sensor 1': 'Sensor 1 [Hydrocarbon_Dew_Point_C]',
    'Sensor 2': 'Sensor 2 [Train_1_Gas_FlowRate_MMBTU_HR]',
    'Sensor 3': 'Sensor 3 [Train_1_Gas_Temperature_C]',
    'Sensor 4': 'Sensor 4 [Train_1_Liquid_Out_1_Gallons_Day_Cumulative_zeros_every_24h]',
    'Sensor 5': 'Sensor 5 [Train_1_Liquid_Out_2_Gallons_Day_Cumulative_zeros_every_24h]',
    'Sensor 6': 'Sensor 6 [Train_2_Gas_FlowRate_MMBTU_HR]',
    'Sensor 7': 'Sensor 7 [Train_2_Gas_Temperature_C]',
    'Sensor 8': 'Sensor 8 [Train_2_Liquid_Out_1_Gallons_Day_Cumulative_zeros_every_24h]',
    'Sensor 9': 'Sensor 9 [Train_2_Liquid_Out_2_Gallons_Day_Cumulative_zeros_every_24h]',
    'Sensor 10': 'Sensor 10 [Train_3_Gas_FlowRate_MMBTU_HR]',
    'Sensor 11': 'Sensor 11 [Train_3_Gas_Temperature_C]',
    'Sensor 12': 'Sensor 12 [Train_3_Liquid_Out_1_Gallons_Day_Cumulative_zeros_every_24h]',
    'Sensor 13': 'Sensor 13 [Train_3_Liquid_Out_2_Gallons_Day_Cumulative_zeros_every_24h]',
    'Sensor 14': 'Sensor 14 [Train_4_Gas_FlowRate_MMBTU_HR]',
    'Sensor 15': 'Sensor 15 [Train_4_Gas_Temperature_C]',
    'Sensor 16': 'Sensor 16 [Train_4_Liquid_Out_1_Gallons_Day_Cumulative_zeros_every_24h]',
    'Sensor 17': 'Sensor 17 [Train_4_Liquid_Out_2_Gallons_Day_Cumulative_zeros_every_24h]',
    'Sensor 18': 'Sensor 18 [Train_5_Gas_FlowRate_MMBTU_HR]',
    'Sensor 19': 'Sensor 19 [Train_5_Gas_Temperature_C]',
    'Sensor 20': 'Sensor 20 [Train_5_Liquid_Out_1_Gallons_Day_Cumulative_zeros_every_24h]',
    'Sensor 21': 'Sensor 21 [Train_5_Liquid_Out_2_Gallons_Day_Cumulative_zeros_every_24h]',
    'Sensor 22': 'Sensor 22 [Train_6_Gas_FlowRate_MMBTU_HR]',
    'Sensor 23': 'Sensor 23 [Train_6_Gas_Temperature_C]',
    'Sensor 24': 'Sensor 24 [Train_6_Liquid_Out_1_Gallons_Day_Cumulative_zeros_every_24h]',
    'Sensor 25': 'Sensor 25 [Train_6_Liquid_Out_2_Gallons_Day_Cumulative_zeros_every_24h]',
    'Sensor 26': 'Sensor 26 [Train_7_Gas_FlowRate_MMBTU_HR]',
    'Sensor 27': 'Sensor 27 [Train_7_Gas_Temperature_C]',
    'Sensor 28': 'Sensor 28 [Train_7_Liquid_Out_1_Gallons_Day_Cumulative_zeros_every_24h]',
    'Sensor 29': 'Sensor 29 [Train_7_Liquid_Out_2_Gallons_Day_Cumulative_zeros_every_24h]',
    'Sensor 30': 'Sensor 30 [Entrance_Pressure_psig]',
    'Sensor 31': 'Sensor 31 [Delivery_Pressure_psig]'
}

def rename_with_full_sensor_names(name: str) -> str:
    if name.startswith("Sensor"):
        sensor_part = name.split("_")[0]
        suffix = "_".join(name.split("_")[1:])
        return f"{SENSOR_NAME_MAP.get(sensor_part, sensor_part)}_{suffix}"
    return name


def convert_to_json_serializable(obj):
    """Recursively convert non-serializable objects to JSON-serializable format"""
    if isinstance(obj, dict):
        return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj

def extract_explainer_data(explainer_result):
    """Extract all raw data from TFT explainer result"""
    data = {}
    
    # Map simplified sensor names to detailed descriptions
    sensor_descriptions = {
        'Sensor 1': 'Sensor 1 [Hydrocarbon_Dew_Point_C]',
        'Sensor 2': 'Sensor 2 [Train_1_Gas_FlowRate_MMBTU_HR]',
        'Sensor 3': 'Sensor 3 [Train_1_Gas_Temperature_C]',
        'Sensor 4': 'Sensor 4 [Train_1_Liquid_Out_1_Gallons_Day_Cumulative_zeros_every_24h]',
        'Sensor 5': 'Sensor 5 [Train_1_Liquid_Out_2_Gallons_Day_Cumulative_zeros_every_24h]',
        'Sensor 6': 'Sensor 6 [Train_2_Gas_FlowRate_MMBTU_HR]',
        'Sensor 7': 'Sensor 7 [Train_2_Gas_Temperature_C]',
        'Sensor 8': 'Sensor 8 [Train_2_Liquid_Out_1_Gallons_Day_Cumulative_zeros_every_24h]',
        'Sensor 9': 'Sensor 9 [Train_2_Liquid_Out_2_Gallons_Day_Cumulative_zeros_every_24h]',
        'Sensor 10': 'Sensor 10 [Train_3_Gas_FlowRate_MMBTU_HR]',
        'Sensor 11': 'Sensor 11 [Train_3_Gas_Temperature_C]',
        'Sensor 12': 'Sensor 12 [Train_3_Liquid_Out_1_Gallons_Day_Cumulative_zeros_every_24h]',
        'Sensor 13': 'Sensor 13 [Train_3_Liquid_Out_2_Gallons_Day_Cumulative_zeros_every_24h]',
        'Sensor 14': 'Sensor 14 [Train_4_Gas_FlowRate_MMBTU_HR]',
        'Sensor 15': 'Sensor 15 [Train_4_Gas_Temperature_C]',
        'Sensor 16': 'Sensor 16 [Train_4_Liquid_Out_1_Gallons_Day_Cumulative_zeros_every_24h]',
        'Sensor 17': 'Sensor 17 [Train_4_Liquid_Out_2_Gallons_Day_Cumulative_zeros_every_24h]',
        'Sensor 18': 'Sensor 18 [Train_5_Gas_FlowRate_MMBTU_HR]',
        'Sensor 19': 'Sensor 19 [Train_5_Gas_Temperature_C]',
        'Sensor 20': 'Sensor 20 [Train_5_Liquid_Out_1_Gallons_Day_Cumulative_zeros_every_24h]',
        'Sensor 21': 'Sensor 21 [Train_5_Liquid_Out_2_Gallons_Day_Cumulative_zeros_every_24h]',
        'Sensor 22': 'Sensor 22 [Train_6_Gas_FlowRate_MMBTU_HR]',
        'Sensor 23': 'Sensor 23 [Train_6_Gas_Temperature_C]',
        'Sensor 24': 'Sensor 24 [Train_6_Liquid_Out_1_Gallons_Day_Cumulative_zeros_every_24h]',
        'Sensor 25': 'Sensor 25 [Train_6_Liquid_Out_2_Gallons_Day_Cumulative_zeros_every_24h]',
        'Sensor 26': 'Sensor 26 [Train_7_Gas_FlowRate_MMBTU_HR]',
        'Sensor 27': 'Sensor 27 [Train_7_Gas_Temperature_C]',
        'Sensor 28': 'Sensor 28 [Train_7_Liquid_Out_1_Gallons_Day_Cumulative_zeros_every_24h]',
        'Sensor 29': 'Sensor 29 [Train_7_Liquid_Out_2_Gallons_Day_Cumulative_zeros_every_24h]',
        'Sensor 30': 'Sensor 30 [Entrance_Pressure_psig]',
        'Sensor 31': 'Sensor 31 [Delivery_Pressure_psig]'
    }
    
    # Extract feature importances and map to detailed names
    encoder_importance = explainer_result.get_encoder_importance()
    decoder_importance = explainer_result.get_decoder_importance()
    static_importance = explainer_result.get_static_covariates_importance()
    
    # Map encoder importance to detailed names
    if encoder_importance is not None:
        encoder_dict = encoder_importance.to_dict()
        detailed_encoder_dict = {}
        for key, value in encoder_dict.items():
            if key in sensor_descriptions:
                detailed_encoder_dict[sensor_descriptions[key]] = value
            else:
                detailed_encoder_dict[key] = value
        data['encoder_importance'] = detailed_encoder_dict
    else:
        data['encoder_importance'] = {}
    
    # Map decoder importance to detailed names
    if decoder_importance is not None:
        decoder_dict = decoder_importance.to_dict()
        detailed_decoder_dict = {}
        for key, value in decoder_dict.items():
            if key in sensor_descriptions:
                detailed_decoder_dict[sensor_descriptions[key]] = value
            else:
                detailed_decoder_dict[key] = value
        data['decoder_importance'] = detailed_decoder_dict
    else:
        data['decoder_importance'] = {}
    
    # Map static importance to detailed names
    if static_importance is not None and not static_importance.empty:
        static_dict = static_importance.to_dict()
        detailed_static_dict = {}
        for key, value in static_dict.items():
            if key in sensor_descriptions:
                detailed_static_dict[sensor_descriptions[key]] = value
            else:
                detailed_static_dict[key] = value
        data['static_importance'] = detailed_static_dict
    else:
        data['static_importance'] = {}
    
    # Extract attention as TimeSeries
    attention_ts = explainer_result.get_attention()
    
    # Convert attention TimeSeries to DataFrame
    if hasattr(attention_ts, 'pd_dataframe'):
        attention_df = attention_ts.pd_dataframe()
    else:
        attention_df = attention_ts.to_dataframe()
    
    data['attention'] = attention_df.to_dict()
    data['attention_array'] = attention_df.values
    data['attention_index'] = attention_df.index.tolist()
    data['attention_columns'] = attention_df.columns.tolist()
    
    # Extract horizon names (columns represent different horizons)
    data['horizons'] = list(attention_df.columns)
    
    return data

def create_enhanced_importance_plot(importance_df, title, path, color_scheme='viridis'):
    """Create an enhanced bar plot with gradients and annotations"""
    plt.figure(figsize=(16, 10))  # Increased width to accommodate longer names
    
    # Get the first row of data (should only have one row)
    importance_dict = importance_df.iloc[0].to_dict()
    # Rename sensor names
    importance_dict = {rename_with_full_sensor_names(k): v for k, v in importance_dict.items()}

    
    # Sort by importance
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=False)
    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    # Create color gradient
    colors = plt.cm.get_cmap(color_scheme)(np.linspace(0.3, 0.9, len(features)))
    
    # Create horizontal bar plot
    bars = plt.barh(features, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, value in zip(bars, values):
        plt.text(value + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    # Styling
    plt.xlabel('Importance (%)', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add background gradient
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')
    
    # Adjust y-axis labels for better readability
    ax.tick_params(axis='y', labelsize=10)
    
    # Add more space on the left for long sensor names
    plt.subplots_adjust(left=0.4)
    
    # Highlight top 3 features
    if len(features) >= 3:
        from matplotlib.patches import Rectangle
        for i in range(min(3, len(features))):
            rect = Rectangle((0, len(features)-i-1.4), values[-(i+1)], 0.8, 
                           alpha=0.2, facecolor='gold', edgecolor='orange', linewidth=2)
            ax.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ {title} saved at: {path}")

def create_clean_attention_plot(data, output_dir, model):
    """Create a clean attention plot like the one in the documentation"""
    attention_array = data['attention_array']
    
    # Calculate mean attention across all horizons
    mean_attention = np.mean(attention_array, axis=1)
    
    # Create x-axis positions
    input_chunk_length = model.input_chunk_length
    output_chunk_length = model.output_chunk_length
    x_positions = np.arange(-input_chunk_length, output_chunk_length)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot the attention line
    ax.plot(x_positions, mean_attention, 'k-', linewidth=2, alpha=0.8)
    ax.fill_between(x_positions, mean_attention, alpha=0.2, color='gray')
    
    # Add vertical line at prediction start
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(0.5, 0.95, 'Prediction Start', transform=ax.transAxes, 
            fontsize=10, ha='center', color='red')
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
    
    # Set labels
    ax.set_xlabel('Index relative to first prediction point', fontsize=12)
    ax.set_ylabel('Attention', fontsize=12)
    ax.set_title('Mean Attention', fontsize=14, fontweight='bold')
    
    # Set limits
    ax.set_xlim(-input_chunk_length, output_chunk_length - 1)
    ax.set_ylim(-0.01, max(mean_attention) * 1.1)
    
    # Style the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add text annotations for key peaks
    # Find the highest peak
    max_idx = np.argmax(mean_attention)
    max_pos = x_positions[max_idx]
    max_val = mean_attention[max_idx]
    
    # Annotate the maximum
    ax.annotate(f'Max: {max_val:.3f}\nat position {max_pos}',
                xy=(max_pos, max_val),
                xytext=(max_pos + 5, max_val + 0.01),
                fontsize=9,
                ha='left',
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'attention_mean_clean.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Clean mean attention plot saved at: {path}")

def save_raw_data(data, output_dir):
    """Save raw data to JSON and CSV files"""
    # Save to JSON
    json_path = os.path.join(output_dir, 'explainer_raw_data.json')
    with open(json_path, 'w') as f:
        json_data = {}
        for key, value in data.items():
            if key != 'attention_array':
                json_data[key] = convert_to_json_serializable(value)
        json.dump(json_data, f, indent=2)
    print(f"✅ Raw data saved to: {json_path}")
    
    # Save attention array to CSV with proper column names
    attention_df = pd.DataFrame(data['attention_array'], 
                               columns=[f'horizon_{i+1}' for i in range(data['attention_array'].shape[1])])
    attention_df.index = pd.RangeIndex(start=-len(attention_df), stop=0) + len(data['horizons'])
    attention_df.index.name = 'relative_position'
    
    csv_path = os.path.join(output_dir, 'attention_weights.csv')
    attention_df.to_csv(csv_path)
    print(f"✅ Attention weights saved to: {csv_path}")

def run_enhanced_explainer(model_path, csv_forecast_path, output_dir=None, target_col="Sensor 1 [Hydrocarbon_Dew_Point_C]"):
    """Run TFT explainer with only selected visualizations"""
    print("\n🎯 Running TFT Explainer (Focused Version)...")
    
    if output_dir is None:
        output_dir = "./enhanced_explain"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load and prepare data
        df = pd.read_csv(csv_forecast_path, index_col=0)
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[~df.index.duplicated()]
        df.dropna(subset=[target_col], inplace=True)

        target = TimeSeries.from_dataframe(df, value_cols=[target_col])
        # Get full sensor names from the original dataframe columns
        sensor_columns = [c for c in df.columns if c != target_col and "Sensor" in c]
        past_cov = TimeSeries.from_dataframe(df[sensor_columns])

        target_scaled = Scaler().fit_transform(target)
        past_cov_scaled = Scaler().fit_transform(past_cov)

        # Load model
        model = TFTModel.load(model_path, map_location="cpu")
        
        # Create explainer
        explainer = TFTExplainer(model=model)

        if len(target_scaled) < model.input_chunk_length:
            print(f"⚠️ Need ≥ {model.input_chunk_length} data points")
            return None

        # Get explanations
        trg = target_scaled[-model.input_chunk_length:]
        cov = past_cov_scaled[-model.input_chunk_length:]
        result = explainer.explain(foreground_series=trg, foreground_past_covariates=cov)

        # Extract all data
        data = extract_explainer_data(result)
        
        # Save raw data
        save_raw_data(data, output_dir)
        
        # Create only the requested visualizations
        print("\n🎨 Creating selected visualizations...")
        
        # 1. Enhanced encoder importance plot
        if result.get_encoder_importance() is not None:
            create_enhanced_importance_plot(result.get_encoder_importance(), 
                                          "Enhanced Encoder Feature Importance", 
                                          os.path.join(output_dir, "encoder_importance_enhanced.png"),
                                          color_scheme='viridis')
        
        # 2. Clean mean attention plot
        create_clean_attention_plot(data, output_dir, model)
        
        # Print summary
        print("\n📊 Summary:")
        print(f"   - Input chunk length: {model.input_chunk_length}")
        print(f"   - Output chunk length: {model.output_chunk_length}")
        print(f"   - Files created:")
        print(f"     • attention_mean_clean.png")
        print(f"     • encoder_importance_enhanced.png")
        print(f"     • explainer_raw_data.json")
        print(f"     • attention_weights.csv")
        
        print(f"\n✅ All files saved to: {output_dir}")
        return data, output_dir

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    MODEL_PATH = "/Users/oribenyosef/Correlation-To-Action-2/TFT_FINAL_MODEL_100_EPOCH.pt"
    CSV_FORECAST_PATH = "/Users/oribenyosef/Correlation-To-Action-2/data/snapshot for full model/19-6-25-raw-forecast.csv"
    OUTPUT_DIR = "/Users/oribenyosef/Correlation-To-Action-2/data/snapshot for full model/generated"

    print("🚀 Running TFT Explainer (Focused Version)...")
    data, output_path = run_enhanced_explainer(MODEL_PATH, CSV_FORECAST_PATH, OUTPUT_DIR)