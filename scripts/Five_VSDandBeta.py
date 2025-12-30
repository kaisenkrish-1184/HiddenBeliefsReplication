import pandas as pd
import numpy as np


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
    
    # --- 4. Calculate financial metrics ---
    required_cols = ['shares', 'CRSP_Monthly_PRC', 'mgrno']
    
    if all(col in merged_panel.columns for col in required_cols):
        # Calculate Position Value
        merged_panel['Position_Value'] = (
            merged_panel['shares'] * merged_panel['CRSP_Monthly_PRC'].abs()
        )
        merged_panel['Position_Value'] = merged_panel['Position_Value'].replace([np.inf, -np.inf], np.nan)
        
        # Calculate AUM by institution and quarter
        aum_df = (merged_panel.groupby(['mgrno', 'rdate'])['Position_Value']
                 .sum().reset_index().rename(columns={'Position_Value': 'AUM'}))
        
        # Merge AUM back to main dataframe
        merged_panel = pd.merge(merged_panel, aum_df, on=['mgrno', 'rdate'], how='left')
        
        # Calculate Value Share (portfolio weight)
        merged_panel['Value_Share'] = merged_panel['Position_Value'] / merged_panel['AUM']
        merged_panel['Value_Share'] = merged_panel['Value_Share'].replace([np.inf, -np.inf], np.nan)
        
        print(f"Calculated Position Value, AUM, and Value Share")
        print(f"Records with valid Value_Share: {merged_panel['Value_Share'].notna().sum()}")
    else:
        # Set to NaN if required columns missing
        for col in ['Position_Value', 'AUM', 'Value_Share']:
            merged_panel[col] = np.nan
        print(f"Warning: Cannot calculate financial metrics. Missing columns: {[col for col in required_cols if col not in merged_panel.columns]}")
    
    # --- 5. Calculate Volatility Scaled Demand ---
    if all(col in merged_panel.columns for col in ['Value_Share', 'sigma2']):
        # VSD = Value Share × Idiosyncratic Variance
        merged_panel['Volatility_Scaled_Demand'] = merged_panel['Value_Share'] * merged_panel['sigma2']
        merged_panel['Volatility_Scaled_Demand'] = merged_panel['Volatility_Scaled_Demand'].replace([np.inf, -np.inf], np.nan)
        
        vsd_count = merged_panel['Volatility_Scaled_Demand'].notna().sum()
        print(f"Calculated Volatility Scaled Demand: {vsd_count} records")
    else:
        merged_panel['Volatility_Scaled_Demand'] = np.nan
        print("Warning: Cannot calculate Volatility Scaled Demand. Missing Value_Share or sigma2.")
    
    # --- 6. Generate summary statistics ---
    summary = {
        'original_characteristics_records': len(panel),
        'variance_beta_records': len(variance_beta),
        'final_records': len(merged_panel),
        'records_with_sigma2': merged_panel['sigma2'].notna().sum(),
        'records_with_beta': merged_panel['beta'].notna().sum(),
        'records_with_value_share': merged_panel['Value_Share'].notna().sum(),
        'records_with_vsd': merged_panel['Volatility_Scaled_Demand'].notna().sum(),
        'sigma2_merge_rate': merged_panel['sigma2'].notna().sum() / len(merged_panel) if len(merged_panel) > 0 else 0,
        'vsd_calculation_rate': merged_panel['Volatility_Scaled_Demand'].notna().sum() / len(merged_panel) if len(merged_panel) > 0 else 0
    }
    
    # --- 7. Save results ---
    try:
        merged_panel.to_parquet(f'{folder}/LHSandSixCharacteristics.parquet', index=False)
        print(f"Final panel saved to 'f'{folder}/LHSandSixCharacteristics.parquet''")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    # --- 8. Print summary ---
    print(f"\n--- Merge and VSD Calculation Summary ---")
    print(f"Original characteristics records: {summary['original_characteristics_records']}")
    print(f"Final merged records: {summary['final_records']}")
    print(f"Sigma2 merge rate: {summary['sigma2_merge_rate']:.1%}")
    print(f"VSD calculation rate: {summary['vsd_calculation_rate']:.1%}")
    
    # Display sample with key columns
    display_cols = ['rdate', 'LPERMNO', 'Market_Equity', 'Log_Market_Cap', 'Log_Book_Value',
                   'Dividend_to_Book', 'Operating_Profitability', 'Investment', 'sigma2', 'beta',
                   'Position_Value', 'AUM', 'Value_Share', 'Volatility_Scaled_Demand']
    available_cols = [col for col in display_cols if col in merged_panel.columns]
    
    print(f"\n--- Sample of Final Panel (first 5 rows) ---")
    print(merged_panel[available_cols].head())
    
    return merged_panel, summary