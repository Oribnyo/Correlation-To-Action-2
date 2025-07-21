# find_duplicates.py
# Find exactly which rows have duplicate timestamps

import pandas as pd
import os

def find_exact_duplicates():
    """Find exactly which rows have duplicate timestamps"""
    
    print("🔍 Finding exact duplicate timestamps...")
    print("="*60)
    
    # Check raw files
    raw_dir = "data/raw"
    if os.path.exists(raw_dir):
        print("\n📁 RAW FILES:")
        for file in sorted(os.listdir(raw_dir)):
            if file.endswith('.csv'):
                filepath = os.path.join(raw_dir, file)
                print(f"\n📄 {file}:")
                print("-" * 40)
                
                df = pd.read_csv(filepath)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Find duplicates
                duplicates = df[df['timestamp'].duplicated(keep=False)]
                
                if len(duplicates) > 0:
                    print(f"Found {len(duplicates)} rows with duplicate timestamps:")
                    
                    # Group by timestamp and show each group
                    for timestamp, group in duplicates.groupby('timestamp'):
                        print(f"\n  Timestamp: {timestamp}")
                        print(f"  Rows: {list(group.index)}")
                        print(f"  Count: {len(group)}")
                        
                        # Show first few values for comparison
                        for i, (idx, row) in enumerate(group.head(3).iterrows()):
                            try:
                                sensor1_val = float(row['Sensor 1'])
                                print(f"    Row {idx}: Sensor 1 = {sensor1_val:.3f}")
                            except Exception:
                                print(f"    Row {idx}: Sensor 1 = {row['Sensor 1']}")
                else:
                    print("  ✅ No duplicates found")
    
    # Check processed files
    proc_dir = "data/processed"
    if os.path.exists(proc_dir):
        print("\n📁 PROCESSED FILES:")
        for file in ['train_raw.csv', 'val_raw.csv', 'test_raw.csv']:
            filepath = os.path.join(proc_dir, file)
            if os.path.exists(filepath):
                print(f"\n📄 {file}:")
                print("-" * 40)
                
                df = pd.read_csv(filepath)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Find duplicates
                duplicates = df[df['timestamp'].duplicated(keep=False)]
                
                if len(duplicates) > 0:
                    print(f"Found {len(duplicates)} rows with duplicate timestamps:")
                    
                    # Group by timestamp and show each group
                    for timestamp, group in duplicates.groupby('timestamp'):
                        print(f"\n  Timestamp: {timestamp}")
                        print(f"  Rows: {list(group.index)}")
                        print(f"  Count: {len(group)}")
                        
                        # Show first few values for comparison
                        for i, (idx, row) in enumerate(group.head(3).iterrows()):
                            try:
                                sensor1_val = float(row['Sensor 1'])
                                print(f"    Row {idx}: Sensor 1 = {sensor1_val:.3f}")
                            except Exception:
                                print(f"    Row {idx}: Sensor 1 = {row['Sensor 1']}")
                else:
                    print("  ✅ No duplicates found")
    
    print("\n" + "="*60)
    print("✅ Duplicate search complete!")

if __name__ == "__main__":
    find_exact_duplicates() 