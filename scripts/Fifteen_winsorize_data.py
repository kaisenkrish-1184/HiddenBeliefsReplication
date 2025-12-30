import pandas as pd
import numpy as np
from scipy import stats

def winsorize_data(df, folder):
    """
    Winsorize the first stage regression dataframe according to paper specifications:
    - Operating_Profitability, Investment, beta: winsorized at 1% and 99% levels
    - Dividend_to_Book: winsorized at 99% level only (top)
    """
    print("=== Winsorizing First Stage Regression Data ===\n")
    
    # Make a copy to preserve original
    df_winsorized = df.copy()
    
    # Variables to winsorize at 1% and 99% levels
    vars_1_99 = ['Operating_Profitability', 'Investment', 'beta']
    
    # Variables to winsorize at 99% level only (top)
    vars_99_only = ['Dividend_to_Book']
    
    print(f"\n=== Original Statistics ===")
    all_vars = vars_1_99 + vars_99_only
    print(df[all_vars].describe())
    
    # Winsorize at 1% and 99% levels
    print(f"\n=== Winsorizing at 1% and 99% levels ===")
    for var in vars_1_99:
        # Calculate percentiles
        p1 = np.percentile(df[var].dropna(), 1)
        p99 = np.percentile(df[var].dropna(), 99)
        
        # Count observations that will be winsorized
        below_p1 = (df[var] < p1).sum()
        above_p99 = (df[var] > p99).sum()
        
        print(f"{var}:")
        print(f"  1st percentile: {p1:.6f}")
        print(f"  99th percentile: {p99:.6f}")
        print(f"  Observations winsorized at bottom: {below_p1:,}")
        print(f"  Observations winsorized at top: {above_p99:,}")
        
        # Apply winsorization
        df_winsorized[var] = np.where(df[var] < p1, p1, df[var])
        df_winsorized[var] = np.where(df[var] > p99, p99, df_winsorized[var])
        print(f"  ✓ Winsorized")
        print()
    
    # Winsorize at 99% level only (top)
    print(f"=== Winsorizing at 99% level only (top) ===")
    for var in vars_99_only:
        # Calculate 99th percentile
        p99 = np.percentile(df[var].dropna(), 99)
        
        # Count observations that will be winsorized
        above_p99 = (df[var] > p99).sum()
        
        print(f"{var}:")
        print(f"  99th percentile: {p99:.6f}")
        print(f"  Observations winsorized at top: {above_p99:,}")
        
        # Apply winsorization (top only)
        df_winsorized[var] = np.where(df[var] > p99, p99, df[var])
        print(f"  ✓ Winsorized")
        print()
    
    # Show before/after statistics
    print(f"=== Winsorized Statistics ===")
    print(df_winsorized[all_vars].describe())
    
    # Calculate the impact
    print(f"\n=== Winsorization Impact ===")
    total_obs = len(df)
    total_winsorized = 0
    
    for var in vars_1_99:
        original = df[var]
        winsorized = df_winsorized[var]
        changed = (original != winsorized).sum()
        total_winsorized += changed
        print(f"{var}: {changed:,} observations changed ({changed/total_obs*100:.2f}%)")
    
    for var in vars_99_only:
        original = df[var]
        winsorized = df_winsorized[var]
        changed = (original != winsorized).sum()
        total_winsorized += changed
        print(f"{var}: {changed:,} observations changed ({changed/total_obs*100:.2f}%)")
    
    print(f"\nTotal observations affected: {total_winsorized:,}")
    print(f"Percentage of data winsorized: {total_winsorized/total_obs*100:.2f}%")
    
    # Save the winsorized data
    output_file = f'{folder}/first_stage_regression_dataframe_winsorized.parquet'
    df_winsorized.to_parquet(output_file, index=False)
    print(f"\n✓ Winsorized data saved to: {output_file}")
    
    # Verify the winsorization worked
    print(f"\n=== Verification ===")
    for var in vars_1_99:
        min_val = df_winsorized[var].min()
        max_val = df_winsorized[var].max()
        p1 = np.percentile(df[var].dropna(), 1)
        p99 = np.percentile(df[var].dropna(), 99)
        print(f"{var}: min={min_val:.6f} (should be ≥ {p1:.6f}), max={max_val:.6f} (should be ≤ {p99:.6f})")
    
    for var in vars_99_only:
        max_val = df_winsorized[var].max()
        p99 = np.percentile(df[var].dropna(), 99)
        print(f"{var}: max={max_val:.6f} (should be ≤ {p99:.6f})")
    
    return df_winsorized

def compare_regression_coefficients():
    """
    Compare regression coefficients before and after winsorization
    """
    print("\n=== Comparing Regression Impact ===")
    
    from sklearn.linear_model import LinearRegression
    
    # Load original and winsorized data
    df_original = pd.read_parquet('first_stage_regression_dataframe.parquet')
    df_winsorized = pd.read_parquet('first_stage_regression_dataframe_winsorized.parquet')
    
    # Take one quarter for comparison
    quarter = df_original['rdate'].iloc[0]
    df_orig_q = df_original[df_original['rdate'] == quarter].dropna(subset=['Log_Market_Cap', 'Operating_Profitability'])
    df_wins_q = df_winsorized[df_winsorized['rdate'] == quarter].dropna(subset=['Log_Market_Cap', 'Operating_Profitability'])
    
    print(f"Comparing quarter: {quarter}")
    print(f"Original data: {len(df_orig_q):,} observations")
    print(f"Winsorized data: {len(df_wins_q):,} observations")
    
    # Variables for regression
    independent_vars = ['log_price_instrument', 'beta', 'Investment', 
                       'Operating_Profitability', 'Dividend_to_Book', 'Log_Book_Value']
    
    # Run regression on original data
    X_orig = df_orig_q[independent_vars].values
    y_orig = df_orig_q['Log_Market_Cap'].values
    X_orig_const = np.column_stack([np.ones(len(X_orig)), X_orig])
    
    reg_orig = LinearRegression(fit_intercept=False)
    reg_orig.fit(X_orig_const, y_orig)
    
    # Run regression on winsorized data
    X_wins = df_wins_q[independent_vars].values
    y_wins = df_wins_q['Log_Market_Cap'].values
    X_wins_const = np.column_stack([np.ones(len(X_wins)), X_wins])
    
    reg_wins = LinearRegression(fit_intercept=False)
    reg_wins.fit(X_wins_const, y_wins)
    
    # Compare coefficients
    var_names = ['constant'] + independent_vars
    print(f"\nCoefficient Comparison:")
    print(f"{'Variable':<25} {'Original':<15} {'Winsorized':<15} {'Change':<15}")
    print("-" * 70)
    
    for i, var in enumerate(var_names):
        orig_coef = reg_orig.coef_[i]
        wins_coef = reg_wins.coef_[i]
        change = wins_coef - orig_coef
        print(f"{var:<25} {orig_coef:<15.6f} {wins_coef:<15.6f} {change:<15.6f}")
    
    # Calculate R-squared
    r2_orig = reg_orig.score(X_orig_const, y_orig)
    r2_wins = reg_wins.score(X_wins_const, y_wins)
    print(f"\nR-squared:")
    print(f"Original: {r2_orig:.6f}")
    print(f"Winsorized: {r2_wins:.6f}")
    print(f"Change: {r2_wins - r2_orig:.6f}")

if __name__ == "__main__":
    df_winsorized = winsorize_data()
    compare_regression_coefficients() 