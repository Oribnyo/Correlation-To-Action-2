import pandas as pd
import os

def convert_csv_timestamps(input_file, output_file):
    """Convert CSV timestamps from 12-hour to 24-hour format"""
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Convert timestamp column to datetime with proper parsing
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %I:%M:%S %p')
    
    # Convert back to string in 24-hour format
    df['timestamp'] = df['timestamp'].dt.strftime('%d/%m/%Y %H:%M:%S')
    
    # Save the converted file
    df.to_csv(output_file, index=False)
    
    print(f"✅ Converted timestamps from 12-hour to 24-hour format")
    print(f"📁 Original: {input_file}")
    print(f"📁 Converted: {output_file}")
    
    # Show sample of converted timestamps
    print("\n📅 Sample converted timestamps:")
    print(df['timestamp'].head())

if __name__ == "__main__":
    input_file = "data/snapshot for llm/19-6-25.csv"
    output_file = "data/snapshot for llm/19-6-25_24h.csv"
    
    convert_csv_timestamps(input_file, output_file) 