import pandas as pd
import numpy as np
import ast
import os

def _safe_parse_list(x):
    """Parse historical_holdings from string representation of list."""
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return []
    return []

def aggregate_hbi_to_stock(
    beliefs_df: pd.DataFrame,
    folder,
    consideration_sets_path: str,
    weight_col: str = "AUM",
    min_managers: int = 5
) -> pd.DataFrame:
    """
    Aggregate manager-level beliefs into stock-level HBI and OBI per quarter.
    
    HBI (Hidden Belief Index): AUM-weighted average of Hidden_Belief, excluding
    stocks that were in the manager's historical_holdings for that (mgrno, rdate).
    
    OBI (Overt Belief Index): AUM-weighted average of Overt_Belief, including
    all stocks (no filtering by historical_holdings).

    Inputs
    ------
    beliefs_df : DataFrame with columns
        ['mgrno','rdate','LPERMNO','Overt_Belief','Hidden_Belief','AUM', ...]
        - Overt_Belief: computed when Volatility_Scaled_Demand is notna
        - Hidden_Belief: computed when Volatility_Scaled_Demand is NaN
        - AUM: may be missing for some rows (will be filled per mgrno-rdate)
    consideration_sets_path : str
        Path to CSV file with columns ['mgrno','rdate','historical_holdings']
        historical_holdings is stored as string representation of list
    weight_col : str
        Column in beliefs_df to use as weight (default: 'AUM')
    min_managers : int
        Minimum number of managers with non-NaN hidden beliefs required to compute HBI
    folder : str
        Output folder for saving Belief_Indexes.csv

    Returns
    -------
    belief_indexes_df : DataFrame
        DataFrame with columns ['rdate', 'LPERMNO', 'HBI', 'OBI', 'Market_Equity']
        (Market_Equity included if present in beliefs_df)
        One row per (rdate, LPERMNO) combination
    """
    print("=== Calculating Hidden Belief Index (HBI) and Overt Belief Index (OBI) ===\n")
    
    # --- 0) Hygiene checks ---
    required_cols = ['mgrno','rdate','LPERMNO','Overt_Belief','Hidden_Belief']
    missing = [c for c in required_cols if c not in beliefs_df.columns]
    if missing:
        raise KeyError(f"beliefs_df missing required columns: {missing}")
    
    if weight_col not in beliefs_df.columns:
        raise KeyError(f"weight_col '{weight_col}' not found in beliefs_df")
    
    # --- 1) Prepare beliefs_df ---
    df = beliefs_df.copy()
    df['rdate'] = pd.to_datetime(df['rdate'])
    df['mgrno'] = df['mgrno'].astype(int)
    df['LPERMNO'] = df['LPERMNO'].astype(int)
    
    print(f"Input beliefs_df: {len(df):,} rows")
    print(f"  Unique (mgrno, rdate, LPERMNO): {df[['mgrno','rdate','LPERMNO']].drop_duplicates().shape[0]:,}")
    print(f"  Unique (mgrno, rdate): {df[['mgrno','rdate']].drop_duplicates().shape[0]:,}")
    
    # --- 2) Fill AUM for each (mgrno, rdate) group ---
    print("\n=== Filling AUM per (mgrno, rdate) group ===")
    aum_before = df[weight_col].notna().sum()
    print(f"  Rows with AUM before filling: {aum_before:,}")
    
    # Fill AUM: take first non-null value per (mgrno, rdate) group
    df[weight_col] = df.groupby(['mgrno', 'rdate'])[weight_col].transform(
        lambda x: x.dropna().iloc[0] if x.dropna().any() else np.nan
    )
    
    aum_after = df[weight_col].notna().sum()
    print(f"  Rows with AUM after filling: {aum_after:,}")
    print(f"  AUM filled for {aum_after - aum_before:,} additional rows")
    
    # --- 3) Load and parse consideration sets file ---
    print("\n=== Loading consideration sets file ===")
    if not os.path.exists(consideration_sets_path):
        raise FileNotFoundError(f"Consideration sets file not found: {consideration_sets_path}")
    
    cons_df = pd.read_csv(consideration_sets_path)
    required_cons_cols = ['mgrno','rdate','historical_holdings']
    missing_cons = [c for c in required_cons_cols if c not in cons_df.columns]
    if missing_cons:
        raise KeyError(f"consideration_sets file missing columns: {missing_cons}")
    
    # Prepare consideration sets
    cons_df['rdate'] = pd.to_datetime(cons_df['rdate'])
    cons_df['mgrno'] = cons_df['mgrno'].astype(int)
    cons_df['historical_holdings'] = cons_df['historical_holdings'].apply(_safe_parse_list)
    
    print(f"Loaded consideration sets: {len(cons_df):,} rows")
    print(f"  Unique (mgrno, rdate) in consideration sets: {cons_df[['mgrno','rdate']].drop_duplicates().shape[0]:,}")
    
    # Create lookup: (mgrno, rdate) -> set of LPERMNOs in historical_holdings
    historical_holdings_lookup = {}
    for _, row in cons_df.iterrows():
        key = (row['mgrno'], row['rdate'])
        historical_holdings_lookup[key] = set(row['historical_holdings'])
    
    print(f"  Created lookup for {len(historical_holdings_lookup):,} (mgrno, rdate) pairs")
    
    # --- 4) Check that all (mgrno, rdate) in beliefs_df exist in consideration sets ---
    print("\n=== Validating (mgrno, rdate) pairs ===")
    beliefs_pairs = set(df[['mgrno','rdate']].drop_duplicates().apply(tuple, axis=1))
    cons_pairs = set(historical_holdings_lookup.keys())
    missing_pairs = beliefs_pairs - cons_pairs
    
    if missing_pairs:
        raise ValueError(
            f"Found {len(missing_pairs):,} (mgrno, rdate) pairs in beliefs_df that are missing "
            f"in consideration sets file. First 10 missing pairs: {list(missing_pairs)[:10]}"
        )
    
    print(f"  All {len(beliefs_pairs):,} (mgrno, rdate) pairs in beliefs_df are present in consideration sets")
    
    # --- 5) Calculate HBI (Hidden Belief Index) ---
    print("\n=== Calculating HBI (Hidden Belief Index) ===")
    
    # Start with rows where Hidden_Belief is not NaN
    hbi_df = df[df['Hidden_Belief'].notna()].copy()
    print(f"  Rows with Hidden_Belief: {len(hbi_df):,}")
    
    # OPTIMIZED: Vectorized filtering using merge instead of apply()
    # Create exploded historical holdings DataFrame for vectorized anti-join
    historical_exploded = []
    for (mgrno, rdate), holdings_set in historical_holdings_lookup.items():
        for lpermno in holdings_set:
            historical_exploded.append({'mgrno': mgrno, 'rdate': rdate, 'LPERMNO': lpermno})
    
    if historical_exploded:
        historical_df = pd.DataFrame(historical_exploded)
        historical_df['rdate'] = pd.to_datetime(historical_df['rdate'])
        historical_df['mgrno'] = historical_df['mgrno'].astype(int)
        historical_df['LPERMNO'] = historical_df['LPERMNO'].astype(int)
        
        # Create indicator column for anti-join
        historical_df['_in_historical'] = True
        
        # Anti-join: keep rows NOT in historical holdings
        hbi_df = hbi_df.merge(
            historical_df[['mgrno', 'rdate', 'LPERMNO', '_in_historical']],
            on=['mgrno', 'rdate', 'LPERMNO'],
            how='left',
            indicator=False
        )
        rows_in_historical = hbi_df['_in_historical'].notna().sum()
        print(f"  Rows where LPERMNO is in historical_holdings: {rows_in_historical:,}")
        
        # Filter out rows in historical holdings
        hbi_df = hbi_df[hbi_df['_in_historical'].isna()].copy()
        hbi_df = hbi_df.drop(columns=['_in_historical'])
    else:
        print(f"  No historical holdings to filter (all rows kept)")
        rows_in_historical = 0
    
    print(f"  Rows remaining after filtering: {len(hbi_df):,}")
    
    # OPTIMIZED: Prepare AUM weights and use transform() instead of groupby + merge
    hbi_df['w_raw'] = pd.to_numeric(hbi_df[weight_col], errors='coerce').fillna(0.0)
    hbi_df.loc[hbi_df['w_raw'] < 0, 'w_raw'] = 0.0
    
    # OPTIMIZED: Use transform() for vectorized sum calculation (no merge needed)
    hbi_df['wsum'] = hbi_df.groupby(['rdate','LPERMNO'])['w_raw'].transform('sum')
    
    # For each manager-stock: calculate weight = manager_AUM / sum_of_all_AUMs_for_that_stock
    hbi_df['w_norm'] = np.where(hbi_df['wsum'] > 0, hbi_df['w_raw'] / hbi_df['wsum'], 0.0)
    
    # For each manager-stock: multiply hidden belief by its weight
    hbi_df['w_hb'] = hbi_df['w_norm'] * hbi_df['Hidden_Belief']
    
    # Aggregate: sum over all managers to get HBI = sum((manager_AUM / total_AUM) * Hidden_Belief)
    hbi_agg = (hbi_df.groupby(['rdate','LPERMNO'])
               .agg(HBI=('w_hb','sum'),
                    managers_used_hbi=('mgrno','nunique'),
                    weight_sum_hbi=('w_raw','sum'))
               .reset_index())
    
    # Apply minimum manager count filter
    hbi_agg.loc[hbi_agg['managers_used_hbi'] < min_managers, 'HBI'] = np.nan
    
    print(f"  HBI computed for {hbi_agg['HBI'].notna().sum():,} stock-quarters")
    print(f"  HBI missing for {hbi_agg['HBI'].isna().sum():,} stock-quarters (insufficient managers)")
    
    # --- 6) Calculate OBI (Overt Belief Index) ---
    print("\n=== Calculating OBI (Overt Belief Index) ===")
    
    # Use rows where Overt_Belief is not NaN (no filtering by historical_holdings)
    obi_df = df[df['Overt_Belief'].notna()].copy()
    print(f"  Rows with Overt_Belief: {len(obi_df):,}")
    
    # OPTIMIZED: Prepare AUM weights and use transform() instead of groupby + merge
    obi_df['w_raw'] = pd.to_numeric(obi_df[weight_col], errors='coerce').fillna(0.0)
    obi_df.loc[obi_df['w_raw'] < 0, 'w_raw'] = 0.0
    
    # OPTIMIZED: Use transform() for vectorized sum calculation (no merge needed)
    obi_df['wsum'] = obi_df.groupby(['rdate','LPERMNO'])['w_raw'].transform('sum')
    
    # For each manager-stock: calculate weight = manager_AUM / sum_of_all_AUMs_for_that_stock
    obi_df['w_norm'] = np.where(obi_df['wsum'] > 0, obi_df['w_raw'] / obi_df['wsum'], 0.0)
    
    # For each manager-stock: multiply overt belief by its weight
    obi_df['w_ob'] = obi_df['w_norm'] * obi_df['Overt_Belief']
    
    # Aggregate: sum over all managers to get OBI = sum((manager_AUM / total_AUM) * Overt_Belief)
    obi_agg = (obi_df.groupby(['rdate','LPERMNO'])
               .agg(OBI=('w_ob','sum'),
                    managers_used_obi=('mgrno','nunique'),
                    weight_sum_obi=('w_raw','sum'))
               .reset_index())
    
    print(f"  OBI computed for {obi_agg['OBI'].notna().sum():,} stock-quarters")
    
    # --- 7) Combine HBI and OBI into Belief_Indexes dataframe ---
    print("\n=== Creating Belief_Indexes dataframe ===")
    
    # Get all unique (rdate, LPERMNO) combinations
    all_stock_quarters = df[['rdate','LPERMNO']].drop_duplicates()
    
    # Get Market_Equity per (rdate, LPERMNO) if it exists
    if 'Market_Equity' in df.columns:
        market_equity_agg = df.groupby(['rdate', 'LPERMNO'])['Market_Equity'].first().reset_index()
        print(f"  Adding Market_Equity column")
    else:
        market_equity_agg = pd.DataFrame(columns=['rdate', 'LPERMNO', 'Market_Equity'])
    
    # Merge HBI, OBI, and Market_Equity
    belief_indexes = all_stock_quarters.merge(
        hbi_agg[['rdate','LPERMNO','HBI']],
        on=['rdate','LPERMNO'],
        how='left'
    ).merge(
        obi_agg[['rdate','LPERMNO','OBI']],
        on=['rdate','LPERMNO'],
        how='left'
    )
    
    # Merge Market_Equity if available
    if 'Market_Equity' in df.columns:
        belief_indexes = belief_indexes.merge(
            market_equity_agg[['rdate','LPERMNO','Market_Equity']],
            on=['rdate','LPERMNO'],
            how='left'
        )
    
    # Sort by rdate, then LPERMNO
    belief_indexes = belief_indexes.sort_values(['rdate','LPERMNO']).reset_index(drop=True)
    
    print(f"Belief_Indexes dataframe: {len(belief_indexes):,} rows")
    print(f"  Unique quarters: {belief_indexes['rdate'].nunique()}")
    print(f"  Unique stocks: {belief_indexes['LPERMNO'].nunique()}")
    print(f"  Rows with HBI: {belief_indexes['HBI'].notna().sum():,}")
    print(f"  Rows with OBI: {belief_indexes['OBI'].notna().sum():,}")
    print(f"  Rows with both: {(belief_indexes['HBI'].notna() & belief_indexes['OBI'].notna()).sum():,}")
    
    # --- 8) Save to CSV ---
    os.makedirs(folder, exist_ok=True)
    output_path = os.path.join(folder, 'Belief_Indexes.csv')
    belief_indexes.to_csv(output_path, index=False)
    print(f"\nSaved Belief_Indexes to {output_path}")
    
    return belief_indexes
