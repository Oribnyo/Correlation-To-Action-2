# check_duplicates.py
# Check for duplicate timestamps in data files

import pandas as pd
import os

def check_duplicates():
    """Check for duplicate timestamps in all data files"""
    
    print("🔍 Checking for duplicate timestamps...")
    print("="*50)
    
    # Check raw files
    raw_dir = "data/raw"
    if os.path.exists(raw_dir):
        print("\n📁 RAW FILES:")
        for file in sorted(os.listdir(raw_dir)):
            if file.endswith('.csv'):
                filepath = os.path.join(raw_dir, file)
                df = pd.read_csv(filepath)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                duplicates = df['timestamp'].duplicated().sum()
                total_rows = len(df)
                
                print(f"  {file}:")
                print(f"    Total rows: {total_rows}")
                print(f"    Duplicate timestamps: {duplicates}")
                print(f"    Duplicate %: {(duplicates/total_rows)*100:.2f}%")
                
                if duplicates > 0:
                    print(f"    ⚠️  DUPLICATES FOUND!")
                    # Show some examples
                    dup_times = df[df['timestamp'].duplicated(keep=False)]['timestamp'].unique()[:5]
                    print(f"    Example duplicate timestamps: {dup_times}")
                print()
    
    # Check processed files
    proc_dir = "data/processed"
    if os.path.exists(proc_dir):
        print("📁 PROCESSED FILES:")
        for file in ['train_raw.csv', 'val_raw.csv', 'test_raw.csv']:
            filepath = os.path.join(proc_dir, file)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                duplicates = df['timestamp'].duplicated().sum()
                total_rows = len(df)
                
                print(f"  {file}:")
                print(f"    Total rows: {total_rows}")
                print(f"    Duplicate timestamps: {duplicates}")
                print(f"    Duplicate %: {(duplicates/total_rows)*100:.2f}%")
                
                if duplicates > 0:
                    print(f"    ⚠️  DUPLICATES FOUND!")
                    # Show some examples
                    dup_times = df[df['timestamp'].duplicated(keep=False)]['timestamp'].unique()[:3]
                    print(f"    Example duplicate timestamps: {dup_times}")
                print()
    
    print("="*50)
    print("✅ Duplicate check complete!")

if __name__ == "__main__":
    check_duplicates() 