import pandas as pd
import numpy as np


def merge_keys(holdings_csv_path, link_csv_path, folder):
    """
    Merge 13F holdings data with CUSIP-PERMNO link data using date-effective matching.
    
    Args:
        holdings_csv_path (str): Path to the 13F holdings CSV file
        link_csv_path (str): Path to the CUSIP-PERMNO link CSV file  
        output_path (str): Path for the output CSV file (default: '13F_MergeKeys.csv')
        
    Returns:
        tuple: (final_panel_df, summary_dict) where summary_dict contains merge statistics
    """
    
    # --- 1. Load your datasets (assuming CSVs) ---
    try:
        # Use 'skiprows' to skip the initial metadata and get to the real headers.
        df_13f = pd.read_parquet(holdings_csv_path)
        df_link = pd.read_csv(link_csv_path)
        print("Data loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Please check your file paths.")
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    # --- 2. Data Cleaning and Type Conversion ---

    # Ensure cusips are strings and standardize to 8 digits
    df_13f['cusip'] = df_13f['cusip'].astype(str).str.zfill(8)
    df_link['cusip'] = df_link['CUSIP'].astype(str).str.zfill(8)

    # --- Correctly handle special date values before conversion ---
    # Replace WRDS's common '99991231' and 'E' for NAMEENDT with a future date string
    df_link['SecInfoEndDt'] = df_link['SecInfoEndDt'].astype(str).replace('99991231', '2099-12-31')
    df_link['SecInfoEndDt'] = df_link['SecInfoEndDt'].replace('E', '2099-12-31')

    # Convert date columns to datetime objects for proper comparison
    date_format_13f = '%Y-%m-%d'
    date_format_link = '%Y-%m-%d'

    df_13f['rdate'] = pd.to_datetime(df_13f['rdate'], format=date_format_13f, errors='coerce')
    df_link['SecInfoStartDt'] = pd.to_datetime(df_link['SecInfoStartDt'], format=date_format_link, errors='coerce')
    df_link['SecInfoEndDt'] = pd.to_datetime(df_link['SecInfoEndDt'], format=date_format_link, errors='coerce')

    # Handle potential NaT (Not a Time) values from conversion errors if any
    df_13f.dropna(subset=['rdate'], inplace=True)
    df_link.dropna(subset=['SecInfoStartDt', 'SecInfoEndDt'], inplace=True)

    # Sort both DataFrames by date for pd.merge_asof.
    df_13f_sorted = df_13f.sort_values(by='rdate').copy()
    df_link_primary_sorted = df_link.sort_values(by='SecInfoStartDt').copy()

    # --- 4. Perform the Date-Effective Merge ---
    merged_df = pd.merge_asof(
        df_13f_sorted,
        df_link_primary_sorted,
        left_on='rdate',
        right_on='SecInfoStartDt',
        by='cusip',
        direction='backward',
        allow_exact_matches=True
    )
    # After merge_asof, we need to filter to ensure rdate is actually within DATE and NAMEENDT
    merged_df_filtered = merged_df[
        (merged_df['rdate'] >= merged_df['SecInfoStartDt']) &
        (merged_df['rdate'] <= merged_df['SecInfoEndDt'])
    ].copy()

    # Capture the dropped records (records that don't meet the date criteria)
    dropped_records = merged_df[
        ~((merged_df['rdate'] >= merged_df['SecInfoStartDt']) &
          (merged_df['rdate'] <= merged_df['SecInfoEndDt']))
    ].copy()

    # Select and rename columns as needed
    final_13f_crsp_panel = merged_df_filtered[['mgrno', 'rdate', 'cusip', 'PERMNO','shares', 'prc']].copy()

    # Also capture records that didn't get merged at all (no CUSIP match)
    unmerged_records = df_13f[~df_13f['cusip'].isin(merged_df['cusip'])].copy()

    # --- 5. Print Summary ---
    print("\n--- Sample of Final 13F-CRSP Panel (first 5 rows) ---")
    print(final_13f_crsp_panel.head())
    print(f"\nOriginal 13F records: {len(df_13f)}")
    print(f"Records after linking to PERMNO: {len(final_13f_crsp_panel)}")
    print(f"Records dropped during linking: {len(df_13f) - len(final_13f_crsp_panel)}")
    print(f"Records dropped due to date criteria: {len(dropped_records)}")
    print(f"Records with no CUSIP match: {len(unmerged_records)}")
    print(f"Total dropped: {len(dropped_records) + len(unmerged_records)}")

    # --- 6. Save the successful matches ---
    try:
        final_13f_crsp_panel.to_parquet(f'{folder}/13F_1stMerge.parquet', index=False)
        print(f"Results saved to {folder}/13F_1stMerge.parquet")
    except Exception as e:
        print(f"Error saving results: {e}")

    dropped_records.to_csv(f'{folder}/dropped_records_date_filter.csv', index=False)

    # --- 7. Return results and summary ---
    summary = {
        'original_records': len(df_13f),
        'final_records': len(final_13f_crsp_panel),
        'dropped_date_criteria': len(dropped_records),
        'dropped_no_cusip_match': len(unmerged_records),
        'total_dropped': len(dropped_records) + len(unmerged_records)
    }

    return final_13f_crsp_panel, summary
