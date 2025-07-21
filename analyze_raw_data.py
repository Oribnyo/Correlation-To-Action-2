# analyze_raw_data.py
# Comprehensive analysis of raw data before any processing

import os
import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

RAW_DIR = "data/raw"

def analyze_dataset(df, year_name):
    """Analyze a single dataset and return statistics"""
    
    # Handle DST changes by converting to UTC
    timestamp_col = df.columns[0]  # Assume first column is timestamp
    
    # Create a copy of the dataframe to avoid modifying the original raw data
    df_processed = df.copy()
    
    # Convert timestamp column to datetime
    df_processed[timestamp_col] = pd.to_datetime(df_processed[timestamp_col], errors='coerce')
    
    # Set timestamp as index for easier processing
    df_processed.set_index(timestamp_col, inplace=True)
    
    # Step 1: Localize to 'Asia/Jerusalem' (Israel time), handling DST properly
    df_processed.index = df_processed.index.tz_localize('Asia/Jerusalem', ambiguous='infer', nonexistent='shift_forward')
    
    # Step 2: Convert to UTC
    df_processed.index = df_processed.index.tz_convert('UTC')
    
    # Reset index to bring timestamp back as a column
    df_processed.reset_index(inplace=True)
    
    # Rename the timestamp column back to original name
    df_processed.rename(columns={df_processed.columns[0]: timestamp_col}, inplace=True)
    
    # Use the processed dataframe for analysis
    df = df_processed
    
    total_cells = df.shape[0] * df.shape[1]
    
    # Column A (Time Steps) - first column
    time_steps_count = df.shape[0]
    time_steps_percent = (time_steps_count / total_cells) * 100
    
    # Row 1 (Column Headers) - first row
    headers_count = df.shape[1]
    headers_percent = (headers_count / total_cells) * 100
    
    # Data cells (excluding column A and row 1)
    data_cells = (df.shape[0] - 1) * (df.shape[1] - 1)
    
    # Analyze data cells
    data_df = df.iloc[1:, 1:]  # Exclude first row and first column
    
    # Count different types
    non_zero_numeric = 0
    zero_numeric = 0
    text_values = 0
    empty_cells = 0
    text_list = []
    
    for col in data_df.columns:
        for val in data_df[col]:
            # Check if it's empty (NaN, None, empty string, or whitespace)
            if pd.isna(val) or (isinstance(val, str) and str(val).strip() == ''):
                empty_cells += 1
            # Check if it's numeric (including scientific notation)
            elif isinstance(val, (int, float)):
                # Already numeric
                if val == 0:
                    zero_numeric += 1
                else:
                    non_zero_numeric += 1
            elif isinstance(val, str):
                # Try to convert string to numeric, including scientific notation
                try:
                    float_val = float(val)
                    if float_val == 0:
                        zero_numeric += 1
                    else:
                        non_zero_numeric += 1
                except (ValueError, TypeError):
                    # It's text
                    text_values += 1
                    if str(val).strip() != '':
                        text_list.append(str(val).strip())
            else:
                # It's text
                text_values += 1
                if str(val).strip() != '':
                    text_list.append(str(val).strip())
    
    # Calculate percentages
    non_zero_percent = (non_zero_numeric / total_cells) * 100
    zero_percent = (zero_numeric / total_cells) * 100
    text_percent = (text_values / total_cells) * 100
    empty_percent = (empty_cells / total_cells) * 100
    
    # Enhanced Timestamp validation (now using UTC timestamps)
    timestamps = pd.to_datetime(df[timestamp_col], errors='coerce')
    
    # Check for missing timestamps
    missing_timestamps = timestamps.isna().sum()
    
    # Check for duplicate timestamps
    duplicate_timestamps = timestamps.duplicated().sum()
    
    # Enhanced timestamp analysis
    timestamps_clean = timestamps.dropna()
    timestamp_analysis = {
        'missing_timestamps': missing_timestamps,
        'duplicate_timestamps': duplicate_timestamps,
        'timestamp_gaps': 0,
        'missing_minutes': 0,
        'excess_timestamps': 0,
        'expected_timestamps': 0,
        'actual_timestamps': len(timestamps_clean),
        'start_date': None,
        'end_date': None,
        'frequency_minutes': 1,
        'problematic_locations': []
    }
    
    if len(timestamps_clean) > 1:
        # Sort timestamps
        timestamps_sorted = timestamps_clean.sort_values()
        timestamp_analysis['start_date'] = timestamps_sorted.iloc[0]
        timestamp_analysis['end_date'] = timestamps_sorted.iloc[-1]
        
        # Calculate expected timestamps based on date range and frequency
        date_range = timestamp_analysis['end_date'] - timestamp_analysis['start_date']
        total_minutes = date_range.total_seconds() / 60
        expected_timestamps = int(total_minutes) + 1  # +1 because we include both start and end
        timestamp_analysis['expected_timestamps'] = expected_timestamps
        
        # Calculate actual frequency from data
        time_diffs = timestamps_sorted.diff().dropna()
        if len(time_diffs) > 0:
            median_freq_minutes = time_diffs.median().total_seconds() / 60
            timestamp_analysis['frequency_minutes'] = median_freq_minutes
        
        # Find gaps larger than expected frequency
        expected_freq = pd.Timedelta(minutes=timestamp_analysis['frequency_minutes'])
        gaps = time_diffs[time_diffs > expected_freq]
        timestamp_analysis['timestamp_gaps'] = len(gaps)
        
        # Calculate total missing minutes due to gaps
        if len(gaps) > 0:
            timestamp_analysis['missing_minutes'] = gaps.sum().total_seconds() / 60
            
            # Find locations of gaps
            gap_indices = time_diffs[time_diffs > expected_freq].index
            for idx in gap_indices:
                gap_start = timestamps_sorted.iloc[idx-1]
                gap_end = timestamps_sorted.iloc[idx]
                gap_duration = (gap_end - gap_start).total_seconds() / 60
                timestamp_analysis['problematic_locations'].append({
                    'type': 'gap',
                    'start_time': gap_start,
                    'end_time': gap_end,
                    'duration_minutes': gap_duration,
                    'row_index': idx
                })
        
        # Check for excess timestamps (more frequent than expected)
        excess_intervals = time_diffs[time_diffs < expected_freq]
        timestamp_analysis['excess_timestamps'] = len(excess_intervals)
        
        # Find locations of excess timestamps
        if len(excess_intervals) > 0:
            excess_indices = excess_intervals.index
            for idx in excess_indices:
                excess_time = timestamps_sorted.iloc[idx]
                interval_duration = excess_intervals.loc[idx].total_seconds() / 60
                timestamp_analysis['problematic_locations'].append({
                    'type': 'excess',
                    'time': excess_time,
                    'interval_minutes': interval_duration,
                    'row_index': idx
                })
        
        # Find locations of duplicate timestamps
        if duplicate_timestamps > 0:
            duplicate_mask = timestamps.duplicated(keep=False)
            duplicate_indices = duplicate_mask[duplicate_mask].index
            duplicate_times = timestamps[duplicate_mask].unique()
            
            for dup_time in duplicate_times:
                dup_indices = timestamps[timestamps == dup_time].index.tolist()
                timestamp_analysis['problematic_locations'].append({
                    'type': 'duplicate',
                    'time': dup_time,
                    'row_indices': dup_indices,
                    'count': len(dup_indices)
                })
    
    return {
        'year': year_name,
        'total_cells': total_cells,
        'time_steps_count': time_steps_count,
        'time_steps_percent': time_steps_percent,
        'headers_count': headers_count,
        'headers_percent': headers_percent,
        'non_zero_numeric_count': non_zero_numeric,
        'non_zero_numeric_percent': non_zero_percent,
        'zero_numeric_count': zero_numeric,
        'zero_numeric_percent': zero_percent,
        'text_values_count': text_values,
        'text_values_percent': text_percent,
        'empty_cells_count': empty_cells,
        'empty_cells_percent': empty_percent,
        'text_list': text_list,
        'missing_timestamps': timestamp_analysis['missing_timestamps'],
        'duplicate_timestamps': timestamp_analysis['duplicate_timestamps'],
        'timestamp_gaps': timestamp_analysis['timestamp_gaps'],
        'missing_minutes': timestamp_analysis['missing_minutes'],
        'excess_timestamps': timestamp_analysis['excess_timestamps'],
        'expected_timestamps': timestamp_analysis['expected_timestamps'],
        'actual_timestamps': timestamp_analysis['actual_timestamps'],
        'start_date': timestamp_analysis['start_date'],
        'end_date': timestamp_analysis['end_date'],
        'frequency_minutes': timestamp_analysis['frequency_minutes'],
        'problematic_locations': timestamp_analysis['problematic_locations']
    }

def main():
    """Main analysis function"""
    print("🔍 Raw Data Analysis - Before Any Processing")
    print("=" * 80)
    
    # Get all CSV files
    files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(".csv")])
    
    if not files:
        print("❌ No CSV files found in data/raw/")
        return
    
    print(f"📁 Found {len(files)} CSV files: {files}")
    print()
    
    # Analyze each year
    results = []
    all_text_values = []
    
    for f in files:
        year = f.split('_')[0]  # Extract year from filename
        filepath = os.path.join(RAW_DIR, f)
        
        print(f"📊 Analyzing {f}...")
        df = pd.read_csv(filepath)
        result = analyze_dataset(df, year)
        results.append(result)
        all_text_values.extend(result['text_list'])
    
    # Create combined analysis
    print(f"\n📊 Creating combined analysis...")
    combined_df = pd.concat([pd.read_csv(os.path.join(RAW_DIR, f)) for f in files], ignore_index=True)
    combined_result = analyze_dataset(combined_df, "ALL YEARS")
    results.append(combined_result)
    
    # Create Table 1: Cell Type Analysis (with separated count and percentage columns)
    print("\n" + "="*80)
    print("📋 TABLE 1: CELL TYPE ANALYSIS")
    print("=" * 80)
    
    table1_data = []
    for result in results:
        table1_data.append({
            'Year': result['year'],
            'Total Cells': f"{result['total_cells']:,}",
            'Time Steps Count': f"{result['time_steps_count']:,}",
            'Time Steps %': f"{result['time_steps_percent']:.2f}%",
            'Column Headers Count': f"{result['headers_count']:,}",
            'Column Headers %': f"{result['headers_percent']:.2f}%",
            'Non-Zero Numeric Count': f"{result['non_zero_numeric_count']:,}",
            'Non-Zero Numeric %': f"{result['non_zero_numeric_percent']:.2f}%",
            'Zero Numeric Count': f"{result['zero_numeric_count']:,}",
            'Zero Numeric %': f"{result['zero_numeric_percent']:.2f}%",
            'Text Values Count': f"{result['text_values_count']:,}",
            'Text Values %': f"{result['text_values_percent']:.2f}%",
            'Empty Cells Count': f"{result['empty_cells_count']:,}",
            'Empty Cells %': f"{result['empty_cells_percent']:.2f}%",
            'Total %': f"{result['time_steps_percent'] + result['headers_percent'] + result['non_zero_numeric_percent'] + result['zero_numeric_percent'] + result['text_values_percent'] + result['empty_cells_percent']:.2f}%"
        })
    
    table1_df = pd.DataFrame(table1_data)
    print(table1_df.to_string(index=False))
    
    # Create Enhanced Table 1.5: Timestamp Validation
    print("\n" + "="*80)
    print("📋 TABLE 1.5: ENHANCED TIMESTAMP VALIDATION")
    print("=" * 80)
    
    timestamp_data = []
    for result in results:
        if result['start_date'] is not None:
            start_str = result['start_date'].strftime('%Y-%m-%d %H:%M')
            end_str = result['end_date'].strftime('%Y-%m-%d %H:%M')
            freq_str = f"{result['frequency_minutes']:.2f} min"
        else:
            start_str = "N/A"
            end_str = "N/A"
            freq_str = "N/A"
        
        timestamp_data.append({
            'Year': result['year'],
            'Start Date': start_str,
            'End Date': end_str,
            'Expected Freq': freq_str,
            'Expected Timestamps': f"{result['expected_timestamps']:,}",
            'Actual Timestamps': f"{result['actual_timestamps']:,}",
            'Missing Timestamps': f"{result['missing_timestamps']:,}",
            'Duplicate Timestamps': f"{result['duplicate_timestamps']:,}",
            'Timestamp Gaps': f"{result['timestamp_gaps']:,}",
            'Missing Minutes': f"{result['missing_minutes']:.1f}",
            'Excess Timestamps': f"{result['excess_timestamps']:,}"
        })
    
    timestamp_df = pd.DataFrame(timestamp_data)
    print(timestamp_df.to_string(index=False))
    
    # Create Table 1.6: Problematic Timestamp Locations
    print("\n" + "="*80)
    print("📋 TABLE 1.6: PROBLEMATIC TIMESTAMP LOCATIONS")
    print("=" * 80)
    
    # Create data for problematic locations export
    problematic_data = []
    
    for result in results[:-1]:  # Exclude combined
        if result['problematic_locations']:
            print(f"\n🔍 {result['year']} - Problematic Timestamps:")
            print("-" * 60)
            
            for i, problem in enumerate(result['problematic_locations'], 1):
                if problem['type'] == 'gap':
                    print(f"{i}. GAP: {problem['start_time']} → {problem['end_time']} "
                          f"(Duration: {problem['duration_minutes']:.1f} min, Row: {problem['row_index']})")
                    problematic_data.append({
                        'Year': result['year'],
                        'Type': 'GAP',
                        'Start Time': problem['start_time'],
                        'End Time': problem['end_time'],
                        'Duration (min)': problem['duration_minutes'],
                        'Row Index': problem['row_index'],
                        'Count': 1
                    })
                elif problem['type'] == 'excess':
                    print(f"{i}. EXCESS: {problem['time']} "
                          f"(Interval: {problem['interval_minutes']:.2f} min, Row: {problem['row_index']})")
                    problematic_data.append({
                        'Year': result['year'],
                        'Type': 'EXCESS',
                        'Start Time': problem['time'],
                        'End Time': problem['time'],
                        'Duration (min)': problem['interval_minutes'],
                        'Row Index': problem['row_index'],
                        'Count': 1
                    })
                elif problem['type'] == 'duplicate':
                    print(f"{i}. DUPLICATE: {problem['time']} "
                          f"(Count: {problem['count']}, Rows: {problem['row_indices']})")
                    problematic_data.append({
                        'Year': result['year'],
                        'Type': 'DUPLICATE',
                        'Start Time': problem['time'],
                        'End Time': problem['time'],
                        'Duration (min)': 0,
                        'Row Index': str(problem['row_indices']),
                        'Count': problem['count']
                    })
        else:
            print(f"\n✅ {result['year']} - No timestamp issues found")
    
    # Create DataFrame for problematic locations
    if problematic_data:
        problematic_df = pd.DataFrame(problematic_data)
        print(f"\n📊 Problematic locations summary:")
        print(problematic_df.groupby(['Year', 'Type']).size().unstack(fill_value=0))
    else:
        problematic_df = pd.DataFrame()
    
    # Create Table 2: Text Value Analysis
    print("\n" + "="*80)
    print("📋 TABLE 2: TEXT VALUE ANALYSIS")
    print("=" * 80)
    
    # Count text values per year
    text_analysis = {}
    
    for result in results:
        year = result['year']
        text_counter = Counter(result['text_list'])
        
        for text, count in text_counter.items():
            if text not in text_analysis:
                text_analysis[text] = {}
            text_analysis[text][year] = count
    
    # Create DataFrame for text analysis
    years = [r['year'] for r in results]
    text_data = []
    
    for text, year_counts in text_analysis.items():
        row = {'Text Value': text}
        for year in years:
            row[year] = year_counts.get(year, 0)
        text_data.append(row)
    
    if text_data:
        table2_df = pd.DataFrame(text_data)
        table2_df = table2_df.sort_values('Text Value')
        print(table2_df.to_string(index=False))
    else:
        print("✅ No text values found in any dataset")
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("=" * 80)
    
    total_text_values = sum(len(r['text_list']) for r in results[:-1])  # Exclude combined
    total_cells_analyzed = sum(r['total_cells'] for r in results[:-1])
    
    print(f"📈 Total cells analyzed: {total_cells_analyzed:,}")
    print(f"📝 Total text values found: {total_text_values:,}")
    print(f"📊 Text percentage: {(total_text_values/total_cells_analyzed)*100:.2f}%")
    
    # Check for potential issues
    print(f"\n⚠️ POTENTIAL ISSUES:")
    issues_found = False
    
    for result in results[:-1]:  # Exclude combined
        if result['text_values_count'] > 0:
            print(f"  • {result['year']}: {result['text_values_count']} text values found")
            issues_found = True
        
        if result['empty_cells_count'] > 0:
            print(f"  • {result['year']}: {result['empty_cells_count']} empty cells found")
            issues_found = True
        
        if result['missing_timestamps'] > 0:
            print(f"  • {result['year']}: {result['missing_timestamps']} missing timestamps")
            issues_found = True
            
        if result['duplicate_timestamps'] > 0:
            print(f"  • {result['year']}: {result['duplicate_timestamps']} duplicate timestamps")
            issues_found = True
            
        if result['timestamp_gaps'] > 0:
            print(f"  • {result['year']}: {result['timestamp_gaps']} timestamp gaps ({result['missing_minutes']:.1f} missing minutes)")
            issues_found = True
            
        if result['excess_timestamps'] > 0:
            print(f"  • {result['year']}: {result['excess_timestamps']} excess timestamps (more frequent than {result['frequency_minutes']:.2f} minutes)")
            issues_found = True
        
        if result['expected_timestamps'] > 0:
            diff = result['actual_timestamps'] - result['expected_timestamps']
            if abs(diff) > 0:
                print(f"  • {result['year']}: {diff:+d} timestamps difference (expected: {result['expected_timestamps']:,}, actual: {result['actual_timestamps']:,})")
                issues_found = True
    
    if not issues_found:
        print("  ✅ No issues found - data appears clean")
    
    # Save results to CSV
    table1_df.to_csv("raw_data_analysis_table1.csv", index=False)
    timestamp_df.to_csv("raw_data_analysis_timestamp_validation.csv", index=False)
    if text_data:
        table2_df.to_csv("raw_data_analysis_table2.csv", index=False)
    if not problematic_df.empty:
        problematic_df.to_csv("raw_data_analysis_problematic_timestamps.csv", index=False)
    
    print(f"\n💾 Results saved to:")
    print(f"  • raw_data_analysis_table1.csv")
    print(f"  • raw_data_analysis_timestamp_validation.csv")
    if text_data:
        print(f"  • raw_data_analysis_table2.csv")
    if not problematic_df.empty:
        print(f"  • raw_data_analysis_problematic_timestamps.csv")

if __name__ == "__main__":
    main() 