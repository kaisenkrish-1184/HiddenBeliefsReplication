"""
Script to append filtered rigid managers to consideration sets dataframe
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def filter_rigid_managers_with_25_plus(df):
    """
    Filter rigid managers from CSV and apply 25+ positive holdings rule
    
    Parameters:
    csv_file (str): Path to finalFileBeforeZeros.csv
    
    Returns:
    pd.DataFrame: Filtered rigid managers with 25+ positive holdings per quarter
    """
    print(f"Loaded {len(df):,} total records")
    
    # Convert rdate to datetime to match parquet format
    df['rdate'] = pd.to_datetime(df['rdate'])
    print("Converted rdate to datetime format")
    
    # Convert numeric columns that might be read as strings
    numeric_columns = [
        'Monthly_RET', 'CRSP_Monthly_PRC', 'CRSP_Monthly_SHROUT', 'CRSP_Monthly_VOL',
        'Market_Equity', 'Annual_Dividends_Common_Total', 'ATQ_LAG1', 'ATQ_LAG2', 
        'CEQQ_LAG1', 'SEQQ_LAG1', 'REVTQ_LAG1', 'COGSQ_LAG1', 'XSGAQ_LAG1', 
        'XINTQ_LAG1', 'NIQ_LAG1', 'CHEQ_LAG1', 'DLCQ_LAG1', 'DLTTQ_LAG1',
        'Log_Market_Cap', 'Book_Value_Equity', 'Log_Book_Value', 'Dividend_to_Book',
        'Operating_Profitability', 'Investment', 'Leverage', 'Cash_Holdings',
        'sigma2', 'beta', 'Position_Value', 'AUM', 'Value_Share', 
        'Volatility_Scaled_Demand', 'NAICS_LAG1', 'shares', 'prc'
    ]
    
    print("Converting numeric columns...")
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print("Numeric conversion completed")
    
    # Filter for rigid managers only
    rigid_df = df[df['Manager_Type'] == 'Rigid'].copy()
    print(f"Found {len(rigid_df):,} rigid manager records")
    
    if len(rigid_df) == 0:
        print("No rigid managers found in the data!")
        return pd.DataFrame()
    
    # Count positive VSD per manager-quarter
    print("Counting positive holdings (VSD > 0) per manager-quarter...")
    positive_vsd = rigid_df[rigid_df['Volatility_Scaled_Demand'] > 0]
    
    # Count positive holdings per manager-quarter
    manager_quarter_counts = positive_vsd.groupby(['mgrno', 'rdate']).size().reset_index(name='positive_count')
    
    print(f"Manager-quarter combinations with positive VSD: {len(manager_quarter_counts):,}")
    
    # Filter for 25+ positive holdings
    qualifying_manager_quarters = manager_quarter_counts[manager_quarter_counts['positive_count'] >= 25]
    print(f"Manager-quarters with 25+ positive holdings: {len(qualifying_manager_quarters):,}")
    
    if len(qualifying_manager_quarters) == 0:
        print("No rigid manager-quarters meet the 25+ positive holdings criterion!")
        return pd.DataFrame()
    
    # Create a set of qualifying (mgrno, rdate) pairs for efficient filtering
    qualifying_pairs = set(zip(qualifying_manager_quarters['mgrno'], qualifying_manager_quarters['rdate']))
    
    # Filter original rigid data to only include qualifying manager-quarters
    print("Filtering rigid managers data for qualifying manager-quarters...")
    filtered_rigid = rigid_df[
        rigid_df.apply(lambda row: (row['mgrno'], row['rdate']) in qualifying_pairs, axis=1)
    ].copy()
    
    print(f"Final filtered rigid manager records: {len(filtered_rigid):,}")
    
    # Show summary statistics
    print(f"\nSummary of filtered rigid managers:")
    print(f"- Unique managers: {filtered_rigid['mgrno'].nunique():,}")
    print(f"- Unique quarters: {filtered_rigid['rdate'].nunique():,}")
    print(f"- Date range: {filtered_rigid['rdate'].min()} to {filtered_rigid['rdate'].max()}")
    
    return filtered_rigid

def append_to_consideration_sets(filtered_rigid_df, parquet_file, output_file):
    """
    Append filtered rigid managers to existing consideration sets parquet
    
    Parameters:
    filtered_rigid_df (pd.DataFrame): Filtered rigid managers data
    parquet_file (str): Path to existing consideration sets parquet
    output_file (str): Path for output parquet file
    """
    
    if len(filtered_rigid_df) == 0:
        print("No rigid managers to append. Copying original file...")
        # Just copy the original file
        original_df = parquet_file
        original_df.to_parquet(output_file, index=False)
        return
    
    print(f"Loading existing consideration sets from {parquet_file}...")
    consideration_sets_df = parquet_file
    print(f"Existing consideration sets: {len(consideration_sets_df):,} records")
    
    # Verify column compatibility
    original_cols = set(consideration_sets_df.columns)
    rigid_cols = set(filtered_rigid_df.columns)
    
    if original_cols != rigid_cols:
        missing_in_rigid = original_cols - rigid_cols
        missing_in_original = rigid_cols - original_cols
        
        if missing_in_rigid:
            print(f"Warning: Columns missing in rigid data: {missing_in_rigid}")
        if missing_in_original:
            print(f"Warning: Extra columns in rigid data: {missing_in_original}")
    
    # Ensure same column order
    filtered_rigid_df = filtered_rigid_df[consideration_sets_df.columns]
    
    # Align data types to match the parquet file
    print("Aligning data types...")
    for col in consideration_sets_df.columns:
        if col in filtered_rigid_df.columns:
            try:
                # Get the dtype from the parquet file
                target_dtype = consideration_sets_df[col].dtype
                current_dtype = filtered_rigid_df[col].dtype
                
                if current_dtype != target_dtype:
                    print(f"Converting {col}: {current_dtype} -> {target_dtype}")
                    
                    # Special handling for different data types
                    if pd.api.types.is_numeric_dtype(target_dtype) and not pd.api.types.is_object_dtype(target_dtype):
                        # Convert to numeric, handling any remaining string values
                        filtered_rigid_df[col] = pd.to_numeric(filtered_rigid_df[col], errors='coerce')
                        # Then convert to the specific numeric type
                        filtered_rigid_df[col] = filtered_rigid_df[col].astype(target_dtype)
                    elif pd.api.types.is_datetime64_any_dtype(target_dtype):
                        # Handle datetime conversion
                        filtered_rigid_df[col] = pd.to_datetime(filtered_rigid_df[col])
                    elif pd.api.types.is_object_dtype(target_dtype):
                        # For object types, keep as object but ensure clean conversion
                        filtered_rigid_df[col] = filtered_rigid_df[col].astype(str).replace('nan', np.nan)
                    else:
                        # For other types, try direct conversion
                        filtered_rigid_df[col] = filtered_rigid_df[col].astype(target_dtype)
                        
            except Exception as e:
                print(f"Warning: Could not convert {col}: {e}")
                # For problematic conversions, let pandas handle it in concat
    
    print("Appending filtered rigid managers to consideration sets...")
    
    # Use concat with the original df first to maintain schema
    combined_df = pd.concat([consideration_sets_df, filtered_rigid_df], ignore_index=True)
    
    # Force the combined dataframe to have the same dtypes as the original
    print("Enforcing original data types on combined dataframe...")
    for col in consideration_sets_df.columns:
        if col in combined_df.columns:
            original_dtype = consideration_sets_df[col].dtype
            if combined_df[col].dtype != original_dtype:
                try:
                    # Skip problematic object conversions that cause PyArrow issues
                    if pd.api.types.is_object_dtype(original_dtype) and pd.api.types.is_numeric_dtype(combined_df[col].dtype):
                        print(f"Skipping conversion for {col}: numeric -> object (potential PyArrow conflict)")
                        continue
                    elif pd.api.types.is_numeric_dtype(original_dtype) and not pd.api.types.is_object_dtype(original_dtype):
                        # Ensure numeric columns stay numeric
                        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').astype(original_dtype)
                        print(f"Final conversion {col}: -> {original_dtype}")
                    else:
                        combined_df[col] = combined_df[col].astype(original_dtype)
                        print(f"Final conversion {col}: -> {original_dtype}")
                except Exception as e:
                    print(f"Final conversion failed for {col}: {e}")
                    # Keep the current dtype if conversion fails
    
    print(f"Combined dataset: {len(combined_df):,} records")
    print(f"- Original consideration sets: {len(consideration_sets_df):,}")
    print(f"- Added rigid managers: {len(filtered_rigid_df):,}")
    
    # Clean data types one final time for PyArrow compatibility
    print("Final data cleaning for PyArrow compatibility...")
    for col in combined_df.columns:
        if combined_df[col].dtype == 'object':
            # Check if object column contains mixed types
            sample_vals = combined_df[col].dropna().head(1000)
            if len(sample_vals) > 0:
                # If it looks like it should be numeric, convert it
                try:
                    pd.to_numeric(sample_vals)
                    print(f"Converting object column {col} to float64")
                    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
                except:
                    # Keep as object if not numeric
                    pass

    # Save combined dataframe with engine specification
    print(f"Saving combined dataset to {output_file}...")
    try:
        # Try with pyarrow first (default)
        combined_df.to_parquet(output_file, index=False, engine='pyarrow')
    except Exception as e:
        print(f"PyArrow failed: {e}")
        print("Trying with fastparquet engine...")
        try:
            combined_df.to_parquet(output_file, index=False, engine='fastparquet')
        except Exception as e2:
            print(f"FastParquet also failed: {e2}")
            print("Saving as CSV instead...")
            output_file_csv = output_file.replace('.parquet', '.csv')
            combined_df.to_csv(output_file_csv, index=False)
            print(f"Saved as CSV: {output_file_csv}")
            return combined_df
    
    # Verify file was saved
    file_size_mb = os.path.getsize(output_file) / (1024*1024)
    print(f"Successfully saved {output_file} ({file_size_mb:.1f} MB)")
    
    # Final summary
    print(f"\nFinal dataset summary:")
    print(f"- Total records: {len(combined_df):,}")
    print(f"- Unique managers: {combined_df['mgrno'].nunique():,}")
    print(f"- Manager types: {combined_df['Manager_Type'].value_counts().to_dict()}")
    
    return combined_df

def main(df, consideration_sets, FOLDER):
    """Main execution function"""
    
    print("=== Appending Rigid Managers to Consideration Sets ===")
    print(f"Start time: {datetime.now()}")
    print()
    
    # File paths
    csv_file = df
    parquet_file = consideration_sets
    output_file = f'{FOLDER}/instrument_construction_DF.parquet'
    
    try:
        # Step 1: Filter rigid managers with 25+ positive holdings
        print("STEP 1: Filtering rigid managers with 25+ positive holdings per quarter")
        print("-" * 60)
        filtered_rigid_df = filter_rigid_managers_with_25_plus(csv_file)
        
        print("\nSTEP 2: Appending to consideration sets")
        print("-" * 60)
        # Step 2: Append to consideration sets
        combined_df = append_to_consideration_sets(filtered_rigid_df, parquet_file, output_file)
        
        print(f"\n✓ Process completed successfully!")
        print(f"Output saved to: {output_file}")
        print(f"End time: {datetime.now()}")
        
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
