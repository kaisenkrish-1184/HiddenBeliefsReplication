import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

def assign_hbi_quintiles(
    df: pd.DataFrame,
    folder
) -> pd.DataFrame:
    """
    Assign size deciles and HBI quintiles, then calculate four-factor alphas for all portfolios.
    
    Process:
    1. Divide stocks into 10 size deciles based on Market_Equity (within each quarter)
    2. Within each size decile, divide into 5 HBI quintiles (Q1=lowest, Q5=highest)
    3. Discard stocks with missing HBI for that quarter
    4. For each of 50 portfolios (10 deciles × 5 quintiles):
       - Calculate daily returns by averaging equal-weighted returns from 3 prior quarter-end portfolios
       - Calculate excess returns (portfolio_return - rf)
       - Regress excess returns on four factors (mktrf, smb, hml, umd)
       - Annualize alpha: (1 + alpha_daily)^252 - 1
    
    Inputs
    ------
    df : DataFrame with columns
        ['rdate', 'LPERMNO', 'HBI', 'Market_Equity']
        rdate should be quarter-end dates
    folder : str
        Output folder for saving results
    
    Returns
    -------
    results_df : DataFrame
        Regression results with columns:
        ['Size_Decile', 'HBI_Quintile', 'Alpha_Daily', 'Alpha_Annualized',
         'Beta_Mktrf', 'Beta_Smb', 'Beta_Hml', 'Beta_Umd', 'Residual_Std',
         'R_Squared', 'N_Observations']
    """
    print("=== Calculating Four-Factor Alphas for HBI-Quintile Portfolios ===\n")
    
    # --- 0) Hygiene checks ---
    required_cols = ['rdate', 'LPERMNO', 'HBI', 'Market_Equity']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"df missing required columns: {missing}")
    
    # Prepare dataframe
    df = df.copy()
    df['rdate'] = pd.to_datetime(df['rdate'])
    df['LPERMNO'] = df['LPERMNO'].astype(int)
    
    # Filter out stocks with missing HBI for each quarter
    print("Filtering stocks with missing HBI per quarter...")
    df = df[df['HBI'].notna()].copy()
    print(f"  Rows with valid HBI: {len(df):,}")
    
    # --- 1) Load daily returns and factors ---
    print("\n=== Loading daily returns and factors ===")
    daily_returns_path = os.path.join(folder, 'filteredDaily.csv')
    factors_path = os.path.join('fullData', 'fourFactorAlphaCharacteristics.csv')
    
    if not os.path.exists(daily_returns_path):
        raise FileNotFoundError(f"Daily returns file not found: {daily_returns_path}")
    if not os.path.exists(factors_path):
        raise FileNotFoundError(f"Factors file not found: {factors_path}")
    
    # Load daily returns (only need PERMNO, DATE, RET)
    print(f"Loading daily returns from {daily_returns_path}...")
    daily_returns = pd.read_csv(daily_returns_path, usecols=['PERMNO', 'DATE', 'RET'])
    daily_returns['DATE'] = pd.to_datetime(daily_returns['DATE'])
    daily_returns['PERMNO'] = daily_returns['PERMNO'].astype(int)
    daily_returns = daily_returns[daily_returns['RET'].notna()].copy()
    print(f"  Loaded {len(daily_returns):,} daily return observations")
    
    # Load factors
    print(f"Loading factors from {factors_path}...")
    factors = pd.read_csv(factors_path)
    factors['date'] = pd.to_datetime(factors['date'])
    print(f"  Loaded {len(factors):,} factor observations")
    
    # Merge factors with daily returns
    daily_data = daily_returns.merge(
        factors,
        left_on='DATE',
        right_on='date',
        how='inner'
    )
    daily_data = daily_data.drop(columns=['date'])
    print(f"  Merged data: {len(daily_data):,} observations")
    
    # --- 2) Create stock-quarter panel for portfolio assignment ---
    print("\n=== Creating portfolios ===")
    stock_quarter = df.groupby(['rdate', 'LPERMNO']).agg({
        'Market_Equity': 'first',
        'HBI': 'first'
    }).reset_index()
    
    print(f"  Stock-quarters: {len(stock_quarter):,}")
    print(f"  Unique quarters: {stock_quarter['rdate'].nunique()}")
    print(f"  Unique stocks: {stock_quarter['LPERMNO'].nunique()}")
    
    # --- 3) Assign size deciles (within each quarter) ---
    print("\n=== Assigning Size Deciles ===")
    stock_quarter['Size_Decile'] = np.nan
    
    for rdate in stock_quarter['rdate'].unique():
        mask = stock_quarter['rdate'] == rdate
        quarter_data = stock_quarter.loc[mask].copy()
        
        # Get valid Market_Equity values
        valid_size = quarter_data['Market_Equity'].notna()
        if valid_size.sum() >= 10:  # Need at least 10 stocks for deciles
            try:
                deciles = pd.qcut(
                    quarter_data.loc[valid_size, 'Market_Equity'],
                    q=10,
                    labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    duplicates='drop'
                )
                # Convert to numeric and assign using the actual indices
                decile_values = pd.to_numeric(deciles, errors='coerce')
                stock_quarter.loc[valid_size.index, 'Size_Decile'] = decile_values.values
            except Exception as e:
                print(f"  Warning: Could not assign deciles for quarter {rdate}: {e}")
    
    stock_quarter['Size_Decile'] = pd.to_numeric(stock_quarter['Size_Decile'], errors='coerce')
    print(f"  Assigned size deciles: {stock_quarter['Size_Decile'].notna().sum():,} stock-quarters")
    
    # --- 4) Assign HBI quintiles within each size decile (within each quarter) ---
    print("\n=== Assigning HBI Quintiles within Size Deciles ===")
    stock_quarter['HBI_Quintile'] = np.nan
    
    for rdate in stock_quarter['rdate'].unique():
        for size_decile in range(1, 11):
            mask = (stock_quarter['rdate'] == rdate) & (stock_quarter['Size_Decile'] == size_decile)
            decile_data = stock_quarter.loc[mask].copy()
            
            # Get valid HBI values
            valid_hbi = decile_data['HBI'].notna()
            if valid_hbi.sum() >= 5:  # Need at least 5 stocks for quintiles
                try:
                    quintiles = pd.qcut(
                        decile_data.loc[valid_hbi, 'HBI'],
                        q=5,
                        labels=[1, 2, 3, 4, 5],
                        duplicates='drop'
                    )
                    # Convert to numeric and assign using the actual indices
                    quintile_values = pd.to_numeric(quintiles, errors='coerce')
                    stock_quarter.loc[valid_hbi.index, 'HBI_Quintile'] = quintile_values.values
                except Exception as e:
                    # Skip if quintile assignment fails
                    pass
    
    stock_quarter['HBI_Quintile'] = pd.to_numeric(stock_quarter['HBI_Quintile'], errors='coerce')
    print(f"  Assigned HBI quintiles: {stock_quarter['HBI_Quintile'].notna().sum():,} stock-quarters")
    
    # Filter to only stocks with both assignments
    stock_quarter = stock_quarter[
        stock_quarter['Size_Decile'].notna() & stock_quarter['HBI_Quintile'].notna()
    ].copy()
    print(f"  Final portfolio assignments: {len(stock_quarter):,} stock-quarters")
    
    # --- 5) Calculate portfolio daily returns ---
    print("\n=== Calculating Portfolio Daily Returns ===")
    
    # Get all unique dates from daily_data
    all_dates = sorted(daily_data['DATE'].unique())
    all_quarters = sorted(stock_quarter['rdate'].unique())
    
    print(f"  Daily dates: {len(all_dates):,} ({all_dates[0]} to {all_dates[-1]})")
    print(f"  Quarter-ends: {len(all_quarters):,} ({all_quarters[0]} to {all_quarters[-1]})")
    
    # Initialize portfolio returns DataFrame
    portfolio_returns = []
    
    for size_decile in range(1, 11):
        for hbi_quintile in range(1, 6):
            print(f"  Processing portfolio: Size={size_decile}, HBI={hbi_quintile}")
            
            portfolio_daily_returns = []
            
            for date in all_dates:
                # Find the three prior quarter-ends (inclusive)
                prior_quarters = [q for q in all_quarters if q <= date]
                if len(prior_quarters) == 0:
                    continue
                
                # Get the last 3 quarters (or fewer if not available)
                quarters_to_use = prior_quarters[-3:]
                
                # Collect returns from all three quarter portfolios
                quarter_returns = []
                
                for quarter_end in quarters_to_use:
                    # Get stocks in this portfolio at this quarter-end
                    portfolio_stocks = stock_quarter[
                        (stock_quarter['rdate'] == quarter_end) &
                        (stock_quarter['Size_Decile'] == size_decile) &
                        (stock_quarter['HBI_Quintile'] == hbi_quintile)
                    ]['LPERMNO'].unique()
                    
                    if len(portfolio_stocks) == 0:
                        continue
                    
                    # Get returns for these stocks on this date
                    stock_returns = daily_data[
                        (daily_data['DATE'] == date) &
                        (daily_data['PERMNO'].isin(portfolio_stocks))
                    ]['RET'].values
                    
                    # Convert to numeric and calculate equal-weighted return (exclude NaN)
                    stock_returns = pd.to_numeric(stock_returns, errors='coerce')
                    valid_returns = stock_returns[~np.isnan(stock_returns)]
                    if len(valid_returns) > 0:
                        quarter_returns.append(np.mean(valid_returns))
                
                # Average returns from the three quarters
                if len(quarter_returns) > 0:
                    portfolio_return = np.mean(quarter_returns)
                    portfolio_daily_returns.append({
                        'DATE': date,
                        'Portfolio_Return': portfolio_return,
                        'Size_Decile': size_decile,
                        'HBI_Quintile': hbi_quintile
                    })
            
            if len(portfolio_daily_returns) > 0:
                portfolio_returns.extend(portfolio_daily_returns)
    
    portfolio_returns_df = pd.DataFrame(portfolio_returns)
    print(f"\n  Calculated portfolio returns: {len(portfolio_returns_df):,} observations")
    
    # --- 6) Merge with factors and calculate excess returns ---
    print("\n=== Calculating Excess Returns ===")
    portfolio_returns_df = portfolio_returns_df.merge(
        daily_data[['DATE', 'rf', 'mktrf', 'smb', 'hml', 'umd']],
        on='DATE',
        how='inner'
    )
    
    portfolio_returns_df['Excess_Return'] = portfolio_returns_df['Portfolio_Return'] - portfolio_returns_df['rf']
    
    # --- 7) Run four-factor regressions for each portfolio ---
    print("\n=== Running Four-Factor Regressions ===")
    regression_results = []
    
    for size_decile in range(1, 11):
        for hbi_quintile in range(1, 6):
            portfolio_data = portfolio_returns_df[
                (portfolio_returns_df['Size_Decile'] == size_decile) &
                (portfolio_returns_df['HBI_Quintile'] == hbi_quintile)
            ].copy()
            
            if len(portfolio_data) < 10:  # Need at least some observations
                continue
            
            # Prepare regression data
            X = portfolio_data[['mktrf', 'smb', 'hml', 'umd']].values
            y = portfolio_data['Excess_Return'].values
            
            # Remove rows with any NaN
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            
            if len(y_valid) < 10:
                continue
            
            # Run regression
            reg = LinearRegression(fit_intercept=True)
            reg.fit(X_valid, y_valid)
            
            # Calculate predictions and residuals
            y_pred = reg.predict(X_valid)
            residuals = y_valid - y_pred
            residual_std = np.std(residuals)
            
            # Calculate R-squared
            ss_res = np.sum((y_valid - y_pred) ** 2)
            ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Get coefficients
            alpha_daily = reg.intercept_
            beta_mktrf = reg.coef_[0]
            beta_smb = reg.coef_[1]
            beta_hml = reg.coef_[2]
            beta_umd = reg.coef_[3]
            
            # Annualize alpha: (1 + alpha_daily)^252 - 1
            alpha_annualized = (1 + alpha_daily) ** 252 - 1
            
            regression_results.append({
                'Size_Decile': size_decile,
                'HBI_Quintile': hbi_quintile,
                'Alpha_Daily': alpha_daily,
                'Alpha_Annualized': alpha_annualized,
                'Beta_Mktrf': beta_mktrf,
                'Beta_Smb': beta_smb,
                'Beta_Hml': beta_hml,
                'Beta_Umd': beta_umd,
                'Residual_Std': residual_std,
                'R_Squared': r_squared,
                'N_Observations': len(y_valid)
            })
    
    results_df = pd.DataFrame(regression_results)
    print(f"  Completed {len(results_df)} regressions")
    
    # --- 8) Summary statistics ---
    print("\n=== Regression Results Summary ===")
    print(f"Total portfolios with results: {len(results_df)}")
    if len(results_df) > 0:
        print(f"\nAlpha Statistics (Annualized):")
        print(f"  Mean: {results_df['Alpha_Annualized'].mean():.4f}")
        print(f"  Median: {results_df['Alpha_Annualized'].median():.4f}")
        print(f"  Min: {results_df['Alpha_Annualized'].min():.4f}")
        print(f"  Max: {results_df['Alpha_Annualized'].max():.4f}")
        print(f"\nAverage R-squared: {results_df['R_Squared'].mean():.4f}")
        print(f"Average observations per regression: {results_df['N_Observations'].mean():.0f}")
    
    # Sort by Size_Decile and HBI_Quintile
    results_df = results_df.sort_values(['Size_Decile', 'HBI_Quintile']).reset_index(drop=True)
    
    # Save results
    os.makedirs(folder, exist_ok=True)
    output_path = os.path.join(folder, 'four_factor_alpha_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")
    
    return results_df
