import pandas as pd
import numpy as np
from datetime import datetime

def load_and_validate_files(df, instruments):
    """
    Load and validate input files
    """
    print("Loading and validating input files...")
    
    # Validate required columns in main data
    required_cols = ['beta', 'Investment', 'Operating_Profitability', 'Dividend_to_Book', 
                     'Log_Book_Value', 'Log_Market_Cap', 'LPERMNO', 'rdate']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in main data: {missing_cols}")
    
    # Validate required columns in instruments
    required_instrument_cols = ['LPERMNO', 'rdate', 'log_price_instrument']
    missing_instrument_cols = [col for col in required_instrument_cols if col not in instruments.columns]
    if missing_instrument_cols:
        raise ValueError(f"Missing required columns in instruments: {missing_instrument_cols}")
    
    # Convert dates
    df['rdate'] = pd.to_datetime(df['rdate'])
    instruments['rdate'] = pd.to_datetime(instruments['rdate'])
    
    print(f"✓ All required columns present")
    print(f"✓ Date columns converted to datetime")
    
    return df, instruments

def clean_missing_values(df):
    """
    Remove rows with missing values in required columns
    """
    print(f"\nCleaning missing values...")
    
    # Required columns to check
    required_cols = ['beta', 'Investment', 'Operating_Profitability', 'Dividend_to_Book', 
                     'Log_Book_Value', 'Log_Market_Cap']
    
    # Check data types
    print("Checking data types...")
    for col in required_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Warning: {col} is not numeric (type: {df[col].dtype})")
    
    # Count missing values before cleaning
    print("\nMissing values before cleaning:")
    initial_rows = len(df)
    for col in required_cols:
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / initial_rows) * 100
        print(f"  {col}: {missing_count:,} ({missing_pct:.2f}%)")
    
    # Remove rows with any missing values in required columns
    df_clean = df.dropna(subset=required_cols)
    
    # Report results
    final_rows = len(df_clean)
    removed_rows = initial_rows - final_rows
    removed_pct = (removed_rows / initial_rows) * 100
    
    print(f"\nCleaning results:")
    print(f"  Initial rows: {initial_rows:,}")
    print(f"  Final rows: {final_rows:,}")
    print(f"  Removed rows: {removed_rows:,} ({removed_pct:.2f}%)")
    
    return df_clean

def merge_instruments(df_clean, instruments):
    """
    Merge log_price_instrument from instruments file
    """
    print(f"\nMerging instruments...")
    
    # Check for missing instruments
    main_combinations = set(zip(df_clean['LPERMNO'], df_clean['rdate']))
    instrument_combinations = set(zip(instruments['LPERMNO'], instruments['rdate']))
    
    missing_combinations = main_combinations - instrument_combinations
    if missing_combinations:
        sample_missing = list(missing_combinations)[:5]
        raise ValueError(f"Missing instruments for {len(missing_combinations)} LPERMNO-rdate combinations. "
                        f"Examples: {sample_missing}")
    
    # Merge instruments
    df_with_instruments = df_clean.merge(
        instruments[['LPERMNO', 'rdate', 'log_price_instrument']], 
        on=['LPERMNO', 'rdate'], 
        how='left'
    )
    
    # Check for missing instruments after merge
    missing_instruments = df_with_instruments['log_price_instrument'].isna().sum()
    if missing_instruments > 0:
        raise ValueError(f"Failed to merge instruments for {missing_instruments} rows")
    
    print(f"✓ Successfully merged instruments for all {len(df_with_instruments):,} rows")
    
    return df_with_instruments

def process_by_quarter(df_clean, instruments):
    """
    Process data quarter by quarter to report progress
    """
    print(f"\nProcessing by quarter for detailed reporting...")
    
    quarters = sorted(df_clean['rdate'].unique())
    all_quarters_data = []
    
    for i, quarter_date in enumerate(quarters):
        print(f"\nProcessing quarter {i+1}/{len(quarters)}: {quarter_date}")
        
        # Filter data for this quarter
        df_quarter = df_clean[df_clean['rdate'] == quarter_date]
        instruments_quarter = instruments[instruments['rdate'] == quarter_date]
        
        print(f"  Quarter data: {len(df_quarter):,} rows")
        print(f"  Quarter instruments: {len(instruments_quarter):,} instruments")
        
        # Merge instruments for this quarter
        df_quarter_with_instruments = df_quarter.merge(
            instruments_quarter[['LPERMNO', 'rdate', 'log_price_instrument']], 
            on=['LPERMNO', 'rdate'], 
            how='left'
        )
        
        # Check for missing instruments
        missing_instruments = df_quarter_with_instruments['log_price_instrument'].isna().sum()
        if missing_instruments > 0:
            raise ValueError(f"Missing instruments for {missing_instruments} rows in quarter {quarter_date}")
        
        print(f"  ✓ Successfully processed quarter {quarter_date}")
        all_quarters_data.append(df_quarter_with_instruments)
    
    # Combine all quarters
    df_final = pd.concat(all_quarters_data, ignore_index=True)
    print(f"\n✓ Combined all quarters: {len(df_final):,} total rows")
    
    return df_final

def save_results(df_final, folder):
    """
    Save the final cleaned dataset as parquet
    """
    print(f"\nSaving results...")
    
    output_filename = f'{folder}/first_stage_regression_dataframe.parquet'
    df_final.to_parquet(output_filename, index=False)
    
    print(f"✓ Saved to: {output_filename}")
    print(f"✓ Final dataset shape: {df_final.shape}")
    
    return output_filename

def main(df, instruments, folder):
    """
    Main function for data cleaning (all quarters)
    """
    print("=== Data Cleaning Script (All Quarters) ===\n")
    
    # Load and validate files
    df, instruments = load_and_validate_files(df, instruments)
    
    # Clean missing values
    df_clean = clean_missing_values(df)
    
    # Process by quarter with detailed reporting
    df_final = process_by_quarter(df_clean, instruments)
    
    # Save results
    output_filename = save_results(df_final, folder)
    
    print(f"\n=== Final Summary ===")
    print(f"Total quarters processed: {df_final['rdate'].nunique()}")
    print(f"Date range: {df_final['rdate'].min()} to {df_final['rdate'].max()}")
    print(f"Final dataset: {df_final.shape}")
    print(f"Unique stocks: {df_final['LPERMNO'].nunique()}")
    print(f"Unique institutions: {df_final['mgrno'].nunique()}")
    print(f"Output file: {output_filename}")
    
    return df_final