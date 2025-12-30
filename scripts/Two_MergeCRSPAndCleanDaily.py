import pandas as pd
import numpy as np


def merge_crsp_and_clean_daily(
    dataframe, 
    crsp_monthly_csv_path, 
    crsp_daily_csv_path,
    folder
):
    """
    Merge 13F-CRSP panel with CRSP monthly data and clean daily data.
    
    Args:
        dataframe (str): Path to the 13F-CRSP panel CSV file
        crsp_monthly_csv_path (str): Path to CRSP monthly data CSV
        crsp_daily_csv_path (str): Path to CRSP daily data CSV
        output_panel_path (str): Output path for unified panel
        output_daily_path (str): Output path for cleaned daily data
        
    Returns:
        tuple: (merged_panel_df, cleaned_daily_df, summary_dict)
    """
    
    # --- 1. Load datasets ---
    try:
        # Handle both CSV and Parquet files for the panel
        final_13f_crsp_panel = dataframe
        
        crsp_monthly = pd.read_csv(crsp_monthly_csv_path, low_memory=False)
        crsp_daily = pd.read_csv(crsp_daily_csv_path, low_memory=False)
        print(f"Loaded panel: {len(final_13f_crsp_panel)}, monthly: {len(crsp_monthly)}, daily: {len(crsp_daily)} records")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None, None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None
    
    # --- 2. Clean and prepare 13F panel data ---
    final_13f_crsp_panel['rdate'] = pd.to_datetime(final_13f_crsp_panel['rdate'], format='%Y-%m-%d', errors='coerce')
    final_13f_crsp_panel.dropna(subset=['rdate'], inplace=True)
    final_13f_crsp_panel['LPERMNO'] = final_13f_crsp_panel['LPERMNO'].astype(int)
    
    # --- 3. Clean and prepare CRSP monthly data ---
    crsp_monthly.columns = [col.upper() for col in crsp_monthly.columns]
    crsp_monthly['PERMNO'] = crsp_monthly['PERMNO'].astype(int)
    
    # Convert dates
    date_format_crsp = '%Y-%m-%d'
    crsp_monthly['DATE'] = pd.to_datetime(crsp_monthly['DATE'], format=date_format_crsp, errors='coerce')
    crsp_monthly.dropna(subset=['DATE'], inplace=True)
    
    # Calculate Market Equity and apply filters
    crsp_monthly['ME'] = crsp_monthly['PRC'].abs() * crsp_monthly['SHROUT'] / 1000
    crsp_monthly = crsp_monthly[(crsp_monthly['ME'] > 0) & (crsp_monthly['PRC'].abs() > 0)]
    
    # Apply stock filters
    crsp_monthly['SHRCD'] = pd.to_numeric(crsp_monthly['SHRCD'], errors='coerce')
    crsp_monthly['EXCHCD'] = pd.to_numeric(crsp_monthly['EXCHCD'], errors='coerce')
    crsp_monthly.dropna(subset=['SHRCD', 'EXCHCD'], inplace=True)
    
    # Filter for common stocks and major exchanges
    common_stock_shrcds = [10, 11]
    major_exchanges = [1, 2, 3]
    crsp_monthly_filtered = crsp_monthly[
        (crsp_monthly['SHRCD'].isin(common_stock_shrcds)) &
        (crsp_monthly['EXCHCD'].isin(major_exchanges))
    ].copy()
    
    # --- 4. Align to quarter-ends and merge ---
    # Filter for quarter-end months
    crsp_monthly_filtered['month'] = crsp_monthly_filtered['DATE'].dt.month
    crsp_monthly_quarter_ends = crsp_monthly_filtered[
        crsp_monthly_filtered['month'].isin([3, 6, 9, 12])
    ].copy()
    
    # Create quarter-end merge keys
    crsp_monthly_quarter_ends['calendar_quarter_end'] = (
        crsp_monthly_quarter_ends['DATE'].dt.to_period('Q').dt.end_time.dt.normalize()
    )
    final_13f_crsp_panel['calendar_quarter_end'] = final_13f_crsp_panel['rdate'].dt.normalize()
    
    # Perform merge
    crsp_monthly_cols = ['PERMNO', 'calendar_quarter_end', 'RET', 'PRC', 'SHROUT', 'VOL', 'ME', 'SHRCD', 'EXCHCD']
    merged_panel = pd.merge(
        final_13f_crsp_panel,
        crsp_monthly_quarter_ends[crsp_monthly_cols],
        left_on=['LPERMNO', 'calendar_quarter_end'],
        right_on=['PERMNO', 'calendar_quarter_end'],
        how='left'
    )

    merged_panel_cleaned = merged_panel.dropna(subset=['SHRCD']).copy()
    print(f"Merged panel cleaned, dropped {len(merged_panel) - len(merged_panel_cleaned)} rows")
    # Clean up merged data
    merged_panel_cleaned.drop(columns=['PERMNO', 'SHRCD', 'EXCHCD'], inplace=True)
    merged_panel_cleaned.rename(columns={
        'RET': 'Monthly_RET',
        'PRC': 'CRSP_Monthly_PRC',
        'SHROUT': 'CRSP_Monthly_SHROUT',
        'VOL': 'CRSP_Monthly_VOL',
        'ME': 'Market_Equity'
    }, inplace=True)
    
    # --- 5. Clean CRSP daily data ---
    crsp_daily.columns = [col.upper() for col in crsp_daily.columns]
    crsp_daily['PERMNO'] = crsp_daily['PERMNO'].astype(int)
    crsp_daily['DATE'] = pd.to_datetime(crsp_daily['DATE'], format=date_format_crsp, errors='coerce')
    crsp_daily.dropna(subset=['DATE'], inplace=True)
    
    # Filter to only include PERMNOs from filtered monthly data
    crsp_daily_cleaned = crsp_daily[
        crsp_daily['PERMNO'].isin(crsp_monthly_filtered['PERMNO'].unique())
    ].copy()
    
    # --- 6. Generate summary statistics ---
    summary = {
        'original_panel_records': len(final_13f_crsp_panel),
        'crsp_monthly_records': len(crsp_monthly_filtered),
        'crsp_daily_records': len(crsp_daily_cleaned),
        'merged_panel_records': len(merged_panel_cleaned),
        'successful_monthly_links': merged_panel_cleaned['Market_Equity'].notna().sum(),
        'quarter_end_records': len(crsp_monthly_quarter_ends)
    }
    
    # --- 7. Save results ---
    try:
        merged_panel_cleaned.to_parquet(f'{folder}/13F_and_CRSPM.parquet', index=False)
        crsp_daily_cleaned.to_csv(f'{folder}/filteredDaily.csv', index=False)
        print(f"Unified panel saved to '13F_and_CRSPM.csv'")
        print(f"Cleaned daily data saved to 'filteredDaily.csv'")
    except Exception as e:
        print(f"Error saving files: {e}")
    
    # --- 8. Print summary ---
    print(f"\n--- Merge Summary ---")
    print(f"Original panel records: {summary['original_panel_records']}")
    print(f"Final merged records: {summary['merged_panel_records']}")
    print(f"Successful monthly links: {summary['successful_monthly_links']}")
    print(f"CRSP daily records after cleaning: {summary['crsp_daily_records']}")
    
    print(f"\n--- Sample of Unified Panel (first 5 rows) ---")
    print(merged_panel_cleaned.head())
    
    return merged_panel_cleaned, crsp_daily_cleaned, summary
