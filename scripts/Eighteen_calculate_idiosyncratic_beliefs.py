import pandas as pd
import numpy as np
import os

def compute_idiosyncratic_beliefs(
    chars_with_holdings_df: pd.DataFrame,  # one row per (mgrno, rdate, LPERMNO) with x's and VSD
    FOLDER,                            # folder to save the results
    betas_df: pd.DataFrame,                # one row per (mgrno, rdate) with beta_* cols
) -> pd.DataFrame:
    """
    Compute Overt_Belief and Hidden_Belief for each row in chars_with_holdings_df.
    
    For each row, compute x'beta by multiplying stock characteristics with 
    institution-level beta coefficients. Then:
    
    - Overt_Belief: When Volatility_Scaled_Demand is notna,
      compute min(Volatility_Scaled_Demand, Volatility_Scaled_Demand - x'beta)
    - Hidden_Belief: When Volatility_Scaled_Demand is NaN,
      compute min(0, -x'beta)

    Mapping:
      x0: Log_Market_Cap            -> beta_Log_Market_Cap
      x1: beta                      -> beta_beta
      x2: Investment                -> beta_Investment
      x3: Operating_Profitability   -> beta_Operating_Profitability
      x4: Dividend_to_Book          -> beta_Dividend_to_Book
      x5: Log_Book_Value            -> beta_Log_Book_Value
      x6: control_variable          -> beta_control_variable
      constant: 1                   -> beta_constant (added directly)

    Returns:
      chars_with_holdings_df with appended columns: ['xbeta', 'Overt_Belief', 'Hidden_Belief']
    """

    # --- 0) Column checks ---
    x_to_beta = {
        "Log_Market_Cap": "beta_Log_Market_Cap",
        "beta": "beta_beta",
        "Investment": "beta_Investment",
        "Operating_Profitability": "beta_Operating_Profitability",
        "Dividend_to_Book": "beta_Dividend_to_Book",
        "Log_Book_Value": "beta_Log_Book_Value",
        "control_variable": "beta_control_variable",
    }

    # Check required columns in chars_with_holdings_df
    needed_chars = ['rdate','LPERMNO','mgrno','Volatility_Scaled_Demand'] + list(x_to_beta.keys())
    missing_chars = [c for c in needed_chars if c not in chars_with_holdings_df.columns]
    if missing_chars:
        raise KeyError(f"chars_with_holdings_df missing: {missing_chars}")

    # Check required columns in betas_df
    needed_betas = ['rdate','mgrno'] + list(x_to_beta.values()) + ['beta_constant']
    missing_betas = [c for c in needed_betas if c not in betas_df.columns]
    if missing_betas:
        raise KeyError(f"betas_df missing: {missing_betas}")

    # --- 1) Type hygiene ---
    # Make copies to avoid modifying input dataframes
    df = chars_with_holdings_df.copy()
    bet = betas_df.copy()

    # Convert dates to datetime
    df['rdate'] = pd.to_datetime(df['rdate'])
    bet['rdate'] = pd.to_datetime(bet['rdate'])

    # Convert IDs to integers
    df['LPERMNO'] = df['LPERMNO'].astype(int)
    df['mgrno']   = df['mgrno'].astype(int)
    bet['mgrno']  = bet['mgrno'].astype(int)

    # --- 2) Merge betas_df with chars_with_holdings_df on (mgrno, rdate) ---
    # This attaches the institution-level beta coefficients to each row
    print(f"Merging betas with holdings data...")
    print(f"  Input rows: {len(df):,}")
    print(f"  Unique (mgrno, rdate) in holdings: {df[['mgrno','rdate']].drop_duplicates().shape[0]:,}")
    print(f"  Unique (mgrno, rdate) in betas: {bet[['mgrno','rdate']].drop_duplicates().shape[0]:,}")
    
    # Merge betas onto the holdings dataframe
    df = df.merge(
        bet[['rdate','mgrno'] + list(x_to_beta.values()) + ['beta_constant']],
        on=['mgrno', 'rdate'],
        how='left'
    )
    
    # Check for missing betas after merge
    missing_betas_after_merge = df['beta_constant'].isna().sum()
    if missing_betas_after_merge > 0:
        print(f"  Warning: {missing_betas_after_merge:,} rows could not be matched with betas")
    
    print(f"  Rows after merge: {len(df):,}")

    # --- 3) Compute x'beta for each row ---
    # x'beta = beta_constant + (Log_Market_Cap × beta_Log_Market_Cap) + ... + (control_variable × beta_control_variable)
    print("\nComputing x'beta for each row...")
    
    # Start with the constant term
    df['xbeta'] = df['beta_constant'].astype(float)
    
    # Add each characteristic × beta coefficient
    for xcol, bcol in x_to_beta.items():
        if xcol in df.columns and bcol in df.columns:
            # Convert to float and multiply, handling any NaN values
            df['xbeta'] += df[xcol].astype(float) * df[bcol].astype(float)
        else:
            raise KeyError(f"Missing column for x'beta calculation: {xcol} or {bcol}")
    
    print(f"  Computed x'beta for {df['xbeta'].notna().sum():,} rows")

    # --- 4) Compute Overt_Belief ---
    # Overt_Belief = min(Volatility_Scaled_Demand, Volatility_Scaled_Demand - x'beta)
    # Only computed when Volatility_Scaled_Demand is notna
    print("\nComputing Overt_Belief...")
    
    # Initialize Overt_Belief as NaN
    df['Overt_Belief'] = np.nan
    
    # Find rows where Volatility_Scaled_Demand is notna
    vsd_notna_mask = df['Volatility_Scaled_Demand'].notna()
    n_overt = vsd_notna_mask.sum()
    print(f"  Rows with Volatility_Scaled_Demand notna: {n_overt:,}")
    
    # For these rows, compute min(Volatility_Scaled_Demand, Volatility_Scaled_Demand - x'beta)
    if n_overt > 0:
        vsd_values = df.loc[vsd_notna_mask, 'Volatility_Scaled_Demand'].astype(float)
        xbeta_values = df.loc[vsd_notna_mask, 'xbeta']
        df.loc[vsd_notna_mask, 'Overt_Belief'] = np.minimum(
            vsd_values,
            vsd_values - xbeta_values
        )
        print(f"  Computed Overt_Belief for {df['Overt_Belief'].notna().sum():,} rows")

    # --- 5) Compute Hidden_Belief ---
    # Hidden_Belief = min(0, -x'beta)
    # Only computed when Volatility_Scaled_Demand is NaN
    print("\nComputing Hidden_Belief...")
    
    # Initialize Hidden_Belief as NaN
    df['Hidden_Belief'] = np.nan
    
    # Find rows where Volatility_Scaled_Demand is NaN
    vsd_isna_mask = df['Volatility_Scaled_Demand'].isna()
    n_hidden = vsd_isna_mask.sum()
    print(f"  Rows with Volatility_Scaled_Demand is NaN: {n_hidden:,}")
    
    # For these rows, compute min(0, -x'beta)
    if n_hidden > 0:
        xbeta_values_hidden = df.loc[vsd_isna_mask, 'xbeta']
        df.loc[vsd_isna_mask, 'Hidden_Belief'] = np.minimum(
            0.0,
            -xbeta_values_hidden
        )
        print(f"  Computed Hidden_Belief for {df['Hidden_Belief'].notna().sum():,} rows")

    # --- 6) Summary statistics ---
    print("\n=== Summary ===")
    print(f"Total rows: {len(df):,}")
    print(f"Rows with Overt_Belief: {df['Overt_Belief'].notna().sum():,}")
    print(f"Rows with Hidden_Belief: {df['Hidden_Belief'].notna().sum():,}")
    print(f"Rows with both (should be 0): {(df['Overt_Belief'].notna() & df['Hidden_Belief'].notna()).sum():,}")
    
    if df['Overt_Belief'].notna().sum() > 0:
        overt_values = df.loc[df['Overt_Belief'].notna(), 'Overt_Belief']
        print(f"\nOvert_Belief statistics:")
        print(f"  Mean: {overt_values.mean():.4f}")
        print(f"  Min: {overt_values.min():.4f}")
        print(f"  Max: {overt_values.max():.4f}")
    
    if df['Hidden_Belief'].notna().sum() > 0:
        hidden_values = df.loc[df['Hidden_Belief'].notna(), 'Hidden_Belief']
        print(f"\nHidden_Belief statistics:")
        print(f"  Mean: {hidden_values.mean():.4f}")
        print(f"  Min: {hidden_values.min():.4f}")
        print(f"  Max: {hidden_values.max():.4f}")

    # --- 7) Save updated dataframe to parquet ---
    output_path = f'{FOLDER}/idiosyncratic_beliefs.parquet'
    os.makedirs(FOLDER, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\nSaved updated dataframe to {output_path} ({len(df):,} rows)")

    return df
