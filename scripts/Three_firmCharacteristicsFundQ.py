import pandas as pd
import numpy as np


def add_firm_characteristics(
    unified_panel_path, 
    fundamentals_quarterly_path,
    folder
):
    """
    Add firm characteristics from Compustat quarterly fundamentals to the unified panel.
    
    Args:
        unified_panel_path (str): Path to the unified panel CSV file (with GVKEY)
        fundamentals_quarterly_path (str): Path to Compustat quarterly fundamentals CSV
        output_path (str): Path for the output CSV file with characteristics
        
    Returns:
        tuple: (final_panel_df, summary_dict) where summary_dict contains merge statistics
    """
    
    # --- 1. Load unified panel ---
    try:
        unified_panel = unified_panel_path
        
        # Clean and prepare unified panel
        unified_panel['rdate'] = pd.to_datetime(unified_panel['rdate']).dt.normalize()
        unified_panel['LPERMNO'] = unified_panel['LPERMNO'].astype(int)
        
        # Convert and format GVKEY
        unified_panel['gvkey'] = pd.to_numeric(unified_panel['gvkey'], errors='coerce')
        unified_panel.dropna(subset=['gvkey'], inplace=True)
        unified_panel['gvkey'] = unified_panel['gvkey'].astype(int)
        unified_panel['GVKEY_formatted'] = unified_panel['gvkey'].astype(str).str.zfill(6)
        
        print(f"Loaded unified panel: {len(unified_panel)} records")
    except Exception as e:
        print(f"Error loading unified panel: {e}")
        return None, None
    
    # --- 2. Load Compustat quarterly fundamentals ---
    try:
        compustat_q = pd.read_parquet(fundamentals_quarterly_path)
        print(f"Loaded fundamentals data: {len(compustat_q)} records")
    except Exception as e:
        print(f"Error loading fundamentals data: {e}")
        return None, None
    
    # --- 3. Clean and prepare Compustat data ---
    compustat_q.columns = [col.upper() for col in compustat_q.columns]
    
    # Format GVKEY
    compustat_q['GVKEY'] = pd.to_numeric(compustat_q['GVKEY'], errors='coerce')
    compustat_q.dropna(subset=['GVKEY'], inplace=True)
    compustat_q['GVKEY'] = compustat_q['GVKEY'].astype(int)
    compustat_q['GVKEY_formatted'] = compustat_q['GVKEY'].astype(str).str.zfill(6)
    
    # Convert dates
    compustat_q['DATADATE'] = pd.to_datetime(compustat_q['DATADATE'], format='%Y-%m-%d', errors='coerce')
    compustat_q.dropna(subset=['DATADATE'], inplace=True)
    
    # Convert accounting variables to numeric
    accounting_vars = ['ATQ', 'CEQQ', 'SEQQ', 'REVTQ', 'COGSQ', 'XSGAQ', 'XINTQ', 
                       'DVPSXQ', 'CSHOQ', 'FYR', 'FQTR', 'FYEAR', 'NIQ', 'CHEQ', 
                       'DLCQ', 'DLTTQ']
    
    for col in accounting_vars:
        if col in compustat_q.columns:
            compustat_q[col] = pd.to_numeric(compustat_q[col], errors='coerce')
    
    # Sort for lagging
    compustat_q_sorted = compustat_q.sort_values(by=['GVKEY', 'DATADATE']).copy()
    
    # --- 4. Calculate annual dividends ---
    if all(col in compustat_q_sorted.columns for col in ['DVPSXQ', 'CSHOQ']):
        compustat_q_sorted['Total_Common_Dividends_QTR'] = compustat_q_sorted['DVPSXQ'] * compustat_q_sorted['CSHOQ']
        compustat_q_sorted['Annual_Dividends_Common_Total'] = (
            compustat_q_sorted.groupby('GVKEY')['Total_Common_Dividends_QTR']
            .rolling(window=4, min_periods=4).sum().reset_index(level=0, drop=True)
        )
    else:
        compustat_q_sorted['Annual_Dividends_Common_Total'] = np.nan
    
    # --- 5. Apply lagging to accounting variables ---
    vars_to_lag = [v for v in accounting_vars if v not in ['DVPSXQ', 'CSHOQ', 'FYR', 'FQTR', 'FYEAR']]
    
    for var in vars_to_lag:
        if var in compustat_q_sorted.columns:
            compustat_q_sorted[f'{var}_LAG1'] = compustat_q_sorted.groupby('GVKEY')[var].shift(1)
            if var == 'ATQ':
                compustat_q_sorted['ATQ_LAG2'] = compustat_q_sorted.groupby('GVKEY')[var].shift(2)
    
    # --- 6. Create merge keys ---
    compustat_q_sorted['report_quarter_end_merge_key'] = (
        compustat_q_sorted['DATADATE'] + pd.DateOffset(months=3)
    ).dt.to_period('Q').dt.end_time.dt.normalize()
    
    unified_panel['report_quarter_end_merge_key'] = unified_panel['rdate']
    
    # --- 7. Merge data ---
    lagged_compustat_cols = ['GVKEY_formatted', 'report_quarter_end_merge_key', 'Annual_Dividends_Common_Total', 'NAICS']
    
    for var in vars_to_lag:
        if f'{var}_LAG1' in compustat_q_sorted.columns:
            lagged_compustat_cols.append(f'{var}_LAG1')
        if var == 'ATQ' and 'ATQ_LAG2' in compustat_q_sorted.columns:
            lagged_compustat_cols.append('ATQ_LAG2')
    
    merged_panel = pd.merge(
        unified_panel,
        compustat_q_sorted[lagged_compustat_cols],
        on=['GVKEY_formatted', 'report_quarter_end_merge_key'],
        how='left'
    )
    
    # --- 8. Calculate firm characteristics ---
    
    # Book Value of Equity
    if 'CEQQ_LAG1' in merged_panel.columns:
        merged_panel['Book_Value_Equity'] = merged_panel['CEQQ_LAG1']
    elif 'SEQQ_LAG1' in merged_panel.columns:
        merged_panel['Book_Value_Equity'] = merged_panel['SEQQ_LAG1']
    else:
        merged_panel['Book_Value_Equity'] = np.nan
    
    # Ensure positive book value
    merged_panel.loc[merged_panel['Book_Value_Equity'] <= 0, 'Book_Value_Equity'] = np.nan
    
    # x0: Log Market Cap
    if 'Market_Equity' in merged_panel.columns:
        merged_panel['Log_Market_Cap'] = np.log(merged_panel['Market_Equity'].clip(lower=1e-9))
        merged_panel['Log_Market_Cap'] = merged_panel['Log_Market_Cap'].replace([np.inf, -np.inf], np.nan)
    else:
        merged_panel['Log_Market_Cap'] = np.nan
    
    # x5: Log Book Value
    if 'Book_Value_Equity' in merged_panel.columns:
        merged_panel['Log_Book_Value'] = np.log(merged_panel['Book_Value_Equity'].clip(lower=1e-9))
        merged_panel['Log_Book_Value'] = merged_panel['Log_Book_Value'].replace([np.inf, -np.inf], np.nan)
    else:
        merged_panel['Log_Book_Value'] = np.nan
    
    # x2: Dividend-to-Book Ratio
    if all(col in merged_panel.columns for col in ['Annual_Dividends_Common_Total', 'Book_Value_Equity']):
        merged_panel['Dividend_to_Book'] = merged_panel['Annual_Dividends_Common_Total'] / merged_panel['Book_Value_Equity']
        merged_panel['Dividend_to_Book'] = merged_panel['Dividend_to_Book'].replace([np.inf, -np.inf], np.nan)
    else:
        merged_panel['Dividend_to_Book'] = np.nan
    
    # x3: Operating Profitability
    required_profit_cols = ['REVTQ_LAG1', 'Book_Value_Equity']
    if all(col in merged_panel.columns for col in required_profit_cols):
        revt = merged_panel['REVTQ_LAG1']
        book_value = merged_panel['Book_Value_Equity']
        
        # Fill missing expenses with 0
        cogs = merged_panel.get('COGSQ_LAG1', pd.Series(0, index=merged_panel.index)).fillna(0)
        sgna = merged_panel.get('XSGAQ_LAG1', pd.Series(0, index=merged_panel.index)).fillna(0)
        xint = merged_panel.get('XINTQ_LAG1', pd.Series(0, index=merged_panel.index)).fillna(0)
        
        numerator = revt - cogs - sgna - xint
        merged_panel['Operating_Profitability'] = numerator / book_value
        merged_panel['Operating_Profitability'] = merged_panel['Operating_Profitability'].replace([np.inf, -np.inf], np.nan)
        
        # Set to NaN if all expenses were originally NaN
        if all(col in merged_panel.columns for col in ['COGSQ_LAG1', 'XSGAQ_LAG1', 'XINTQ_LAG1']):
            all_expenses_nan = (merged_panel['COGSQ_LAG1'].isna() & 
                               merged_panel['XSGAQ_LAG1'].isna() & 
                               merged_panel['XINTQ_LAG1'].isna())
            merged_panel.loc[all_expenses_nan, 'Operating_Profitability'] = np.nan
    else:
        merged_panel['Operating_Profitability'] = np.nan
    
    # x4: Investment (Asset Growth)
    if all(col in merged_panel.columns for col in ['ATQ_LAG1', 'ATQ_LAG2']):
        merged_panel['Investment'] = ((merged_panel['ATQ_LAG1'] - merged_panel['ATQ_LAG2']) / 
                                     merged_panel['ATQ_LAG2'])
        merged_panel['Investment'] = merged_panel['Investment'].replace([np.inf, -np.inf], np.nan)
    else:
        merged_panel['Investment'] = np.nan
    
    # --- 9. Generate summary statistics ---
    summary = {
        'original_panel_records': len(unified_panel),
        'compustat_records': len(compustat_q),
        'final_records': len(merged_panel),
        'records_with_compustat_data': merged_panel['ATQ_LAG1'].notna().sum() if 'ATQ_LAG1' in merged_panel.columns else 0,
        'records_with_market_cap': merged_panel['Log_Market_Cap'].notna().sum(),
        'records_with_book_value': merged_panel['Log_Book_Value'].notna().sum(),
        'records_with_operating_prof': merged_panel['Operating_Profitability'].notna().sum(),
        'records_with_investment': merged_panel['Investment'].notna().sum()
    }
    try:
        finalpanel = merged_panel[['mgrno', 'rdate', 'LPERMNO', 'gvkey',
                                'shares', 'Monthly_RET', 'CRSP_Monthly_PRC',
                                'CRSP_Monthly_SHROUT', 'CRSP_Monthly_VOL', 
                                'Market_Equity', 'Annual_Dividends_Common_Total', 
                                'NAICS', 'Book_Value_Equity', 'Log_Market_Cap', 
                                'Log_Book_Value', 'Dividend_to_Book', 'Operating_Profitability', 
                                'Investment']]
    except:
        finalpanel = merged_panel[['rdate', 'LPERMNO', 'gvkey',
                                'Monthly_RET', 'CRSP_Monthly_PRC',
                                'CRSP_Monthly_SHROUT', 'CRSP_Monthly_VOL', 
                                'Market_Equity', 'Annual_Dividends_Common_Total', 
                                'NAICS', 'Book_Value_Equity', 'Log_Market_Cap', 
                                'Log_Book_Value', 'Dividend_to_Book', 'Operating_Profitability', 
                                'Investment']]
    # --- 10. Save results ---
    try:
        finalpanel.to_parquet(f'{folder}/Five_Characteristics.parquet', index=False)
        print(f"Final panel with characteristics saved to 'Five_Characteristics.parquet'")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    # --- 11. Print summary ---
    print(f"\n--- Firm Characteristics Summary ---")
    print(f"Original panel records: {summary['original_panel_records']}")
    print(f"Final records: {summary['final_records']}")
    print(f"Records with Compustat data: {summary['records_with_compustat_data']}")
    print(f"Records with Log Market Cap: {summary['records_with_market_cap']}")
    print(f"Records with Log Book Value: {summary['records_with_book_value']}")
    print(f"Records with Operating Profitability: {summary['records_with_operating_prof']}")
    print(f"Records with Investment: {summary['records_with_investment']}")
    
    print(f"\n--- Sample of Final Panel (first 5 rows) ---")
    print(finalpanel.head())
    
    return finalpanel, summary
