import pandas as pd
import numpy as np


def merge_keys(holdings_csv_path, link_csv_path, folder):
    """
    Merge 13F holdings data with CUSIP-PERMNO link data using date-effective matching on PERMNO.
    
    Args:
        holdings_csv_path (str): Path to the 13F holdings CSV file (must contain PERMNO column)
        link_csv_path (str): Path to the CUSIP-PERMNO link CSV file (must contain LPERMNO column)
        folder (str): Output folder for saved files
        
    Returns:
        tuple: (final_panel_df, summary_dict) where summary_dict contains merge statistics
    """
    
    # --- 1. Load your datasets ---
    try:
        df_13f = holdings_csv_path
        df_link = pd.read_csv(link_csv_path)
        print("Data loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Please check your file paths.")
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    # --- 2. Data Cleaning and Type Conversion ---

    # Ensure PERMNOs are integers for proper matching
    df_13f['PERMNO'] = df_13f['PERMNO'].astype(int)
    df_link['LPERMNO'] = df_link['LPERMNO'].astype(int)

    # --- Correctly handle special date values before conversion ---
    # Replace WRDS's common '99991231' and 'E' for LINKENDDT with a future date string
    df_link['LINKENDDT'] = df_link['LINKENDDT'].astype(str).replace('99991231', '2099-12-31')
    df_link['LINKENDDT'] = df_link['LINKENDDT'].replace('E', '2099-12-31')

    # Convert date columns to datetime objects for proper comparison
    date_format_13f = '%Y-%m-%d'
    date_format_link = '%Y-%m-%d'

    df_13f['rdate'] = pd.to_datetime(df_13f['rdate'], format=date_format_13f, errors='coerce')
    df_link['LINKDT'] = pd.to_datetime(df_link['LINKDT'], format=date_format_link, errors='coerce')
    df_link['LINKENDDT'] = pd.to_datetime(df_link['LINKENDDT'], format=date_format_link, errors='coerce')

    # Handle potential NaT (Not a Time) values from conversion errors if any
    df_13f.dropna(subset=['rdate'], inplace=True)
    df_link.dropna(subset=['LINKDT', 'LINKENDDT'], inplace=True)
    
    # Ensure rdate and fdate match
    if 'fdate' in df_13f.columns:
        df_13f['fdate'] = pd.to_datetime(df_13f['fdate'], format=date_format_13f, errors='coerce')
        initial_count = len(df_13f)
        df_13f = df_13f[df_13f['rdate'] == df_13f['fdate']].copy()
        dropped_count = initial_count - len(df_13f)
        if dropped_count > 0:
            print(f"Filtered {dropped_count} records where rdate != fdate. Remaining: {len(df_13f)} records")

    # --- 3. Filter Link Table for Primary Links ---
    primary_link_types = ['LC', 'LU']
    df_link_primary = df_link[df_link['LINKTYPE'].isin(primary_link_types)].copy()

    # Sort both DataFrames by date for pd.merge_asof
    df_13f_sorted = df_13f.sort_values(by='rdate').copy()
    df_link_primary_sorted = df_link_primary.sort_values(by='LINKDT').copy()

    # --- 4. Perform the Date-Effective Merge on PERMNO ---
    merged_df = pd.merge_asof(
        df_13f_sorted,
        df_link_primary_sorted,
        left_on='rdate',
        right_on='LINKDT',
        left_by='PERMNO',
        right_by='LPERMNO',
        direction='backward',
        allow_exact_matches=True
    )

    # After merge_asof, we need to filter to ensure rdate is actually within LINKDT and LINKENDDT
    merged_df_filtered = merged_df[
        (merged_df['rdate'] >= merged_df['LINKDT']) &
        (merged_df['rdate'] <= merged_df['LINKENDDT'])
    ].copy()

    # Capture the dropped records (records that don't meet the date criteria)
    dropped_records = merged_df[
        ~((merged_df['rdate'] >= merged_df['LINKDT']) &
          (merged_df['rdate'] <= merged_df['LINKENDDT']))
    ].copy()

    # Select and rename columns as needed
    final_13f_crsp_panel = merged_df_filtered[['mgrno', 'rdate', 'LPERMNO', 'gvkey', 'shares', 'prc']].copy()

    # Also capture records that didn't get merged at all (no PERMNO match)
    unmerged_records = df_13f[~df_13f['PERMNO'].isin(merged_df['PERMNO'])].copy()

    # --- 5. Print Summary ---
    print("\n--- Sample of Final 13F-CRSP Panel (first 5 rows) ---")
    print(final_13f_crsp_panel.head())
    print(f"\nOriginal 13F records: {len(df_13f)}")
    print(f"Records after linking: {len(final_13f_crsp_panel)}")
    print(f"Records dropped during linking: {len(df_13f) - len(final_13f_crsp_panel)}")
    print(f"Records dropped due to date criteria: {len(dropped_records)}")
    print(f"Records with no PERMNO match: {len(unmerged_records)}")
    print(f"Total dropped: {len(dropped_records) + len(unmerged_records)}")

    # --- 6. Save the successful matches ---
    try:
        final_13f_crsp_panel.to_parquet(f'{folder}/13F_MergeKeys.parquet', index=False)
        print(f"Results saved to {folder}/13F_MergeKeys.parquet")
    except Exception as e:
        print(f"Error saving results: {e}")

    dropped_records.to_csv(f'{folder}/dropped_records_date_filter.csv', index=False)

    # --- 7. Return results and summary ---
    summary = {
        'original_records': len(df_13f),
        'final_records': len(final_13f_crsp_panel),
        'dropped_date_criteria': len(dropped_records),
        'dropped_no_permno_match': len(unmerged_records),
        'total_dropped': len(dropped_records) + len(unmerged_records)
    }

    return final_13f_crsp_panel, summary
