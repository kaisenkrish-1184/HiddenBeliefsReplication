import pandas as pd
import numpy as np

def merge_keys(crsp_monthly_path, gvkey_link_path, folder):
    """
    Link CRSP monthly data with GVKEY using date-effective matching on PERMNO.
    
    Args:
        crsp_monthly_path (str): Path to CRSP monthly CSV file (with PERMNO, date columns)
        gvkey_link_path (str): Path to GVKEY link CSV file (with LPERMNO, GVKEY, LINKDT, LINKENDDT)
        folder (str): Output folder for saved files
        
    Returns:
        tuple: (linked_crsp_df, summary_dict) where summary_dict contains merge statistics
    """
    
    # --- 1. Load datasets ---
    try:
        crsp_monthly = pd.read_csv(crsp_monthly_path)
        gvkey_link = pd.read_csv(gvkey_link_path)
        print(f"Loaded CRSP monthly: {len(crsp_monthly)} records")
        print(f"Loaded GVKEY link: {len(gvkey_link)} records")
    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Please check your file paths.")
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    # --- 2. Clean CRSP monthly data ---
    # Standardize column names and ensure proper types
    crsp_monthly.columns = [col.upper() for col in crsp_monthly.columns]
    crsp_monthly['PERMNO'] = crsp_monthly['PERMNO'].astype(int)
    crsp_monthly['DATE'] = pd.to_datetime(crsp_monthly['DATE'], errors='coerce')
    crsp_monthly.dropna(subset=['DATE'], inplace=True)
    
    # Apply basic CRSP filters for common stocks on major exchanges
    crsp_monthly['SHRCD'] = pd.to_numeric(crsp_monthly['SHRCD'], errors='coerce')
    crsp_monthly['EXCHCD'] = pd.to_numeric(crsp_monthly['EXCHCD'], errors='coerce')
    crsp_monthly.dropna(subset=['SHRCD', 'EXCHCD'], inplace=True)
    
    # Filter for common stocks (10, 11) on major exchanges (1, 2, 3)
    crsp_filtered = crsp_monthly[
        (crsp_monthly['SHRCD'].isin([10, 11])) &
        (crsp_monthly['EXCHCD'].isin([1, 2, 3]))
    ].copy()
    
    # Calculate market equity if PRC and SHROUT are available
    if all(col in crsp_filtered.columns for col in ['PRC', 'SHROUT']):
        crsp_filtered['ME'] = crsp_filtered['PRC'].abs() * crsp_filtered['SHROUT'] / 1000
        crsp_filtered = crsp_filtered[crsp_filtered['ME'] > 0]
    
    print(f"CRSP after filtering: {len(crsp_filtered)} records")

    # --- 3. Clean GVKEY link data ---
    gvkey_link.columns = [col.upper() for col in gvkey_link.columns]
    gvkey_link['LPERMNO'] = gvkey_link['LPERMNO'].astype(int)
    gvkey_link['GVKEY'] = gvkey_link['GVKEY'].astype(int)
    
    # Handle special date values
    gvkey_link['LINKENDDT'] = gvkey_link['LINKENDDT'].astype(str).replace('99991231', '2099-12-31')
    gvkey_link['LINKENDDT'] = gvkey_link['LINKENDDT'].replace('E', '2099-12-31')
    
    # Convert dates
    gvkey_link['LINKDT'] = pd.to_datetime(gvkey_link['LINKDT'], errors='coerce')
    gvkey_link['LINKENDDT'] = pd.to_datetime(gvkey_link['LINKENDDT'], errors='coerce')
    gvkey_link.dropna(subset=['LINKDT', 'LINKENDDT'], inplace=True)

    # Filter for primary link types
    primary_link_types = ['LC', 'LU']
    gvkey_link_primary = gvkey_link[gvkey_link['LINKTYPE'].isin(primary_link_types)].copy()
    
    print(f"GVKEY link after filtering: {len(gvkey_link_primary)} records")

    # --- 4. Perform date-effective merge ---

    # Strict sorting for merge_asof:
    # 1) sort by the 'on' key globally
    # 2) (optionally) also by the 'by' key so the key is increasing within each group
    crsp_sorted = crsp_filtered.sort_values(['DATE', 'PERMNO']).reset_index(drop=True)
    gvkey_sorted = gvkey_link_primary.sort_values(['LINKDT', 'LPERMNO']).reset_index(drop=True)

    # Perform merge_asof to find the most recent GVKEY link for each CRSP record
    merged_df = pd.merge_asof(
        crsp_sorted,
        gvkey_sorted,
        left_on='DATE',
        right_on='LINKDT',
        left_by='PERMNO',
        right_by='LPERMNO',
        direction='backward',
        allow_exact_matches=True
    )

    # Filter to ensure date is within valid link period
    merged_df_filtered = merged_df[
        (merged_df['DATE'] >= merged_df['LINKDT']) &
        (merged_df['DATE'] <= merged_df['LINKENDDT'])
    ].copy()

    # Track dropped records
    dropped_records = merged_df[
        ~((merged_df['DATE'] >= merged_df['LINKDT']) &
          (merged_df['DATE'] <= merged_df['LINKENDDT']))
    ].copy()

    # Select final columns (keep CRSP data + GVKEY)
    keep_cols = ['PERMNO', 'DATE', 'SHRCD', 'EXCHCD', 'PRC', 'VOL', 'RET', 'SHROUT', 'GVKEY']
    if 'ME' in merged_df_filtered.columns:
        keep_cols.append('ME')
    
    # Only keep columns that exist
    final_cols = [col for col in keep_cols if col in merged_df_filtered.columns]
    crsp_with_gvkey = merged_df_filtered[final_cols].copy()

    # --- 5. Print Summary ---
    print("\n--- Sample of CRSP with GVKEY (first 5 rows) ---")
    print(crsp_with_gvkey.head())
    print(f"\nOriginal CRSP filtered records: {len(crsp_filtered)}")
    print(f"Records after GVKEY linking: {len(crsp_with_gvkey)}")
    print(f"Records dropped during linking: {len(crsp_filtered) - len(crsp_with_gvkey)}")
    print(f"Records dropped due to date criteria: {len(dropped_records)}")
    print(f"Records with GVKEY: {crsp_with_gvkey['GVKEY'].notna().sum()}")

    crsp_with_gvkey.to_parquet(f'{folder}/CRSP_with_GVKEY.parquet', index=False)

    # --- 7. Return results and summary ---
    summary = {
        'original_crsp_records': len(crsp_filtered),
        'final_records': len(crsp_with_gvkey),
        'records_with_gvkey': crsp_with_gvkey['GVKEY'].notna().sum(),
        'dropped_date_criteria': len(dropped_records),
        'gvkey_link_rate': crsp_with_gvkey['GVKEY'].notna().sum() / len(crsp_with_gvkey) if len(crsp_with_gvkey) > 0 else 0
    }

    return crsp_with_gvkey

def filter_crsp_and_clean_daily(
    crsp_with_gvkey_df, 
    crsp_daily_csv_path,
    folder
):
    """
    Filter CRSP monthly data for quarter-ends and clean CRSP daily data.
    
    Args:
        crsp_with_gvkey_df (DataFrame): CRSP monthly data with GVKEY from merge_keys function
        crsp_daily_csv_path (str): Path to CRSP daily data CSV
        folder (str): Output folder for saved files
        
    Returns:
        tuple: (filtered_monthly_df, cleaned_daily_df)
    """
    
    # --- 1. Load CRSP daily data ---
    try:
        crsp_daily = pd.read_csv(crsp_daily_csv_path, low_memory=False)
        print(f"Loaded CRSP daily: {len(crsp_daily)} records")
        print(f"Input CRSP monthly with GVKEY: {len(crsp_with_gvkey_df)} records")
    except FileNotFoundError as e:
        print(f"Error loading daily file: {e}")
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None
    
    # --- 2. Filter CRSP monthly for quarter-ends ---
    crsp_monthly = crsp_with_gvkey_df.copy()
    
    # Ensure proper data types and date handling
    crsp_monthly['DATE'] = pd.to_datetime(crsp_monthly['DATE'], errors='coerce')
    crsp_monthly.dropna(subset=['DATE'], inplace=True)
    
    # Recalculate Market Equity if needed (should already be done in merge_keys)
    if 'ME' not in crsp_monthly.columns and all(col in crsp_monthly.columns for col in ['PRC', 'SHROUT']):
        crsp_monthly['ME'] = crsp_monthly['PRC'].abs() * crsp_monthly['SHROUT'] / 1000
    
    # Apply additional filters for positive market equity and price
    if 'ME' in crsp_monthly.columns:
        crsp_monthly = crsp_monthly[(crsp_monthly['ME'] > 0) & (crsp_monthly['PRC'].abs() > 0)]
    
    # Filter for quarter-end months (March, June, September, December)
    crsp_monthly['month'] = crsp_monthly['DATE'].dt.month
    crsp_monthly_quarter_ends = crsp_monthly[
        crsp_monthly['month'].isin([3, 6, 9, 12])
    ].copy()
    
    # Create calendar quarter-end date for consistency
    crsp_monthly_quarter_ends['calendar_quarter_end'] = (
        crsp_monthly_quarter_ends['DATE'].dt.to_period('Q').dt.end_time.dt.normalize()
    )
    crsp_monthly_quarter_ends['DATE'] = crsp_monthly_quarter_ends['calendar_quarter_end']
    print(f"CRSP monthly after quarter-end filtering: {len(crsp_monthly_quarter_ends)} records")
    
    # --- 3. Clean CRSP daily data ---
    crsp_daily.columns = [col.upper() for col in crsp_daily.columns]
    crsp_daily['PERMNO'] = crsp_daily['PERMNO'].astype(int)
    crsp_daily['DATE'] = pd.to_datetime(crsp_daily['DATE'], errors='coerce')
    crsp_daily.dropna(subset=['DATE'], inplace=True)
    
    # Filter daily data to only include PERMNOs from the filtered monthly data
    valid_permnos = crsp_monthly_quarter_ends['PERMNO'].unique()
    crsp_daily_cleaned = crsp_daily[
        crsp_daily['PERMNO'].isin(valid_permnos)
    ].copy()
    
    # Additional cleaning for daily data - remove invalid returns if needed
    if 'RET' in crsp_daily_cleaned.columns:
        crsp_daily_cleaned['RET'] = pd.to_numeric(crsp_daily_cleaned['RET'], errors='coerce')
    
    print(f"CRSP daily after cleaning: {len(crsp_daily_cleaned)} records")
    print(f"Unique PERMNOs in filtered monthly: {len(valid_permnos):,}")
    print(f"Unique PERMNOs in cleaned daily: {crsp_daily_cleaned['PERMNO'].nunique():,}")
    
    # --- 4. Print sample outputs ---
    print(f"\n--- Sample of Filtered Monthly (first 5 rows) ---")
    print(crsp_monthly_quarter_ends.head())
    
    print(f"\n--- Sample of Cleaned Daily (first 5 rows) ---")
    print(crsp_daily_cleaned.head())

    crsp_monthly_quarter_ends = crsp_monthly_quarter_ends.rename(columns={
        'PERMNO': 'LPERMNO',
        'DATE': 'rdate',
        'GVKEY': 'gvkey',
        'RET': 'Monthly_RET',
        'PRC': 'CRSP_Monthly_PRC',
        'SHROUT': 'CRSP_Monthly_SHROUT',
        'VOL': 'CRSP_Monthly_VOL',
        'ME': 'Market_Equity'
    })

    # --- 5. Save results ---
    # Save filtered monthly data
    crsp_monthly_quarter_ends.to_parquet(f'{folder}/CRSP_monthly_quarter_ends.parquet', index=False)
    print(f"Filtered monthly data saved to {folder}/CRSP_monthly_quarter_ends.parquet")
    
    # Save cleaned daily data
    crsp_daily_cleaned.to_csv(f'{folder}/CRSP_daily_cleaned.csv', index=False)
    print(f"Cleaned daily data saved to {folder}/CRSP_daily_cleaned.csv")
    
    return crsp_monthly_quarter_ends, crsp_daily_cleaned

def merge_and_calculate_vsd(
    characteristics_panel_path,
    variance_beta_path,
    folder
):
    """
    Merge characteristics panel with variance/beta data and calculate Volatility Scaled Demand.
    
    Args:
        characteristics_panel_path (str): Path to panel with firm characteristics
        variance_beta_path (str): Path to variance and beta CSV file
        output_path (str): Path for final output CSV file
        
    Returns:
        tuple: (final_panel_df, summary_dict) containing merged data and statistics
    """
    
    # --- 1. Load unified panel with characteristics ---
    try:
        panel = characteristics_panel_path
        
        # Clean and prepare panel data
        panel['rdate'] = pd.to_datetime(panel['rdate']).dt.normalize()
        panel['LPERMNO'] = panel['LPERMNO'].astype(int)
        
        if 'mgrno' in panel.columns:
            panel['mgrno'] = panel['mgrno'].astype(int)
        else:
            print("Warning: 'mgrno' not found. AUM and Value_Share calculations may be incomplete.")
        
        print(f"Loaded characteristics panel: {len(panel)} records")
    except Exception as e:
        print(f"Error loading characteristics panel: {e}")
        return None, None
    
    # --- 2. Load variance and beta data ---
    try:
        variance_beta = pd.read_csv(variance_beta_path)
        variance_beta['LPERMNO'] = variance_beta['LPERMNO'].astype(int)
        variance_beta['rdate'] = pd.to_datetime(variance_beta['rdate']).dt.normalize()
        print(f"Loaded variance/beta data: {len(variance_beta)} records")
    except FileNotFoundError:
        print("Warning: Variance/beta file not found. Proceeding with NaN values.")
        variance_beta = pd.DataFrame(columns=['LPERMNO', 'rdate', 'sigma2', 'beta'])
    except Exception as e:
        print(f"Error loading variance/beta data: {e}. Proceeding with NaN values.")
        variance_beta = pd.DataFrame(columns=['LPERMNO', 'rdate', 'sigma2', 'beta'])
    
    # --- 3. Merge dataframes ---
    merged_panel = pd.merge(
        panel,
        variance_beta[['LPERMNO', 'rdate', 'sigma2', 'beta']],
        on=['LPERMNO', 'rdate'],
        how='left'
    )
    
    print(f"Records after merging: {len(merged_panel)}")
    print(f"Records with sigma2: {merged_panel['sigma2'].notna().sum()}")
    print(f"Records with beta: {merged_panel['beta'].notna().sum()}")
    
    # --- 7. Save results ---
    try:
        merged_panel.to_parquet(f'{folder}/Universe.parquet', index=False)
        print(f"Final panel saved to 'f'{folder}/Universe.parquet''")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    # Display sample with key columns
    display_cols = ['rdate', 'LPERMNO', 'Market_Equity', 'Log_Market_Cap', 'Log_Book_Value',
                   'Dividend_to_Book', 'Operating_Profitability', 'Investment', 'sigma2', 'beta',
                   'Position_Value', 'AUM', 'Value_Share', 'Volatility_Scaled_Demand']
    available_cols = [col for col in display_cols if col in merged_panel.columns]
    
    print(f"\n--- Sample of Final Panel (first 5 rows) ---")
    print(merged_panel[available_cols].head())
    
    return merged_panel