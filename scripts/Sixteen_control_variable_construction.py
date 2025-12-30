import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_data(df):
    """
    Load the cleaned data with instruments
    """
    print("Loading cleaned data with instruments...")
    
    # Convert dates
    df['rdate'] = pd.to_datetime(df['rdate'])
    
    # Check required columns
    required_cols = ['Log_Market_Cap', 'log_price_instrument', 'beta', 'Investment', 
                     'Operating_Profitability', 'Dividend_to_Book', 'Log_Book_Value', 
                     'LPERMNO', 'rdate']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"✓ All required columns present")
    print(f"Date range: {df['rdate'].min()} to {df['rdate'].max()}")
    print(f"Unique quarters: {df['rdate'].nunique()}")
    
    return df

def run_first_stage_regression(df_quarter, quarter_date):
    """
    Run first stage regression for a single quarter
    """
    print(f"\nRunning first stage regression for {quarter_date}")
    print(f"Quarter data: {len(df_quarter):,} observations")
    
    # Define variables
    dependent_var = 'Log_Market_Cap'
    independent_vars = ['log_price_instrument', 'beta', 'Investment', 
                       'Operating_Profitability', 'Dividend_to_Book', 'Log_Book_Value']
    
    # Check for missing values
    required_vars = [dependent_var] + independent_vars
    missing_data = df_quarter[required_vars].isnull().any(axis=1)
    
    if missing_data.sum() > 0:
        print(f"  Warning: {missing_data.sum()} observations have missing values - dropping")
        df_clean = df_quarter[~missing_data].copy()
    else:
        df_clean = df_quarter.copy()
    
    print(f"  Clean observations: {len(df_clean):,}")
    
    if len(df_clean) == 0:
        print(f"  ERROR: No clean observations for quarter {quarter_date}")
        return None
    
    # Prepare data
    y = df_clean[dependent_var].values
    X = df_clean[independent_vars].values
    
    # Add constant term
    X_with_constant = np.column_stack([np.ones(len(X)), X])
    
    # Run regression
    reg = LinearRegression(fit_intercept=False)  # We already added constant
    reg.fit(X_with_constant, y)
    
    # Get predictions and residuals
    y_pred = reg.predict(X_with_constant)
    residuals = y - y_pred
    
    # Calculate R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Store results
    results = {
        'quarter_date': quarter_date,
        'n_obs': len(df_clean),
        'r_squared': r_squared,
        'coefficients': reg.coef_,
        'var_names': ['constant'] + independent_vars
    }
    
    # Create results dataframe
    results_df = df_clean[['LPERMNO', 'rdate']].copy()
    results_df['control_variable'] = residuals
    results_df['predicted_log_market_cap'] = y_pred
    results_df['actual_log_market_cap'] = y
    
    print(f"  ✓ Regression completed")
    print(f"  R²: {r_squared:.4f}")
    print(f"  Mean |residual|: {np.mean(np.abs(residuals)):.4f}")
    
    return results_df, results

def process_all_quarters(df):
    """
    Process all quarters to create control variables
    """
    print("\nProcessing all quarters...")
    
    quarters = sorted(df['rdate'].unique())
    print(f"Total quarters to process: {len(quarters)}")
    
    all_control_vars = []
    regression_stats = []
    
    for i, quarter_date in enumerate(quarters):
        print(f"\n--- Quarter {i+1}/{len(quarters)} ---")
        
        # Filter to quarter
        df_quarter = df[df['rdate'] == quarter_date].copy()
        
        # Run regression
        result = run_first_stage_regression(df_quarter, quarter_date)
        
        if result is not None:
            control_vars_df, stats = result
            all_control_vars.append(control_vars_df)
            regression_stats.append(stats)
        else:
            print(f"  Skipping quarter {quarter_date} due to insufficient data")
    
    # Combine all control variables
    if all_control_vars:
        control_variables = pd.concat(all_control_vars, ignore_index=True)
        print(f"\n✓ Created control variables for {len(control_variables):,} observations")
    else:
        raise ValueError("No control variables created - check data quality")
    
    return control_variables, regression_stats

def create_regression_summary(regression_stats):
    """
    Create summary of regression statistics
    """
    print("\nCreating regression summary...")
    
    summary_data = []
    for stats in regression_stats:
        summary_data.append({
            'quarter_date': stats['quarter_date'],
            'n_obs': stats['n_obs'],
            'r_squared': stats['r_squared'],
            'instrument_coef': stats['coefficients'][1],  # log_price_instrument coefficient
            'beta_coef': stats['coefficients'][2],
            'investment_coef': stats['coefficients'][3],
            'op_prof_coef': stats['coefficients'][4],
            'div_to_book_coef': stats['coefficients'][5],
            'log_book_val_coef': stats['coefficients'][6]
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    print(f"✓ Summary created for {len(summary_df)} quarters")
    print(f"\nRegression Summary Statistics:")
    print(f"  Average R²: {summary_df['r_squared'].mean():.4f}")
    print(f"  Average observations per quarter: {summary_df['n_obs'].mean():.0f}")
    print(f"  Average instrument coefficient: {summary_df['instrument_coef'].mean():.4f}")
    
    return summary_df

def merge_with_main_data(df, control_variables):
    """
    Merge control variables back with main dataset with memory safety
    """
    print("\nMerging control variables with main dataset...")
    
    # Check for duplicates in control variables
    print(f"Control vars before dedup: {control_variables.shape}")
    control_vars_clean = control_variables.drop_duplicates(subset=['LPERMNO', 'rdate'])
    print(f"Control vars after dedup: {control_vars_clean.shape}")
    
    # Merge with validation to catch many-to-many issues
    df_with_controls = df.merge(
        control_vars_clean[['LPERMNO', 'rdate', 'control_variable']], 
        on=['LPERMNO', 'rdate'], 
        how='left',
        validate='many_to_one'  # This prevents many-to-many merges
    )
    
    # Check merge success
    missing_controls = df_with_controls['control_variable'].isnull().sum()
    if missing_controls > 0:
        print(f"  Warning: {missing_controls} observations missing control variables")
    else:
        print(f"  ✓ Successfully merged control variables for all observations")
    
    return df_with_controls

def save_results(control_variables, regression_summary, df_with_controls, folder):
    """
    Save all results
    """
    print("\nSaving results...")
    
    # Save control variables
    control_variables.to_csv(f'{folder}/control_variables.csv', index=False)
    print(f"✓ Control variables saved to: control_variables.csv")
    
    # Save regression summary
    regression_summary.to_csv(f'{folder}/first_stage_regression_summary.csv', index=False)
    print(f"✓ Regression summary saved to: first_stage_regression_summary.csv")
    
    # Save final dataset with control variables
    df_with_controls.to_parquet(f'{folder}/regression_ready_dataframe.parquet', index=False)
    print(f"✓ Final dataset saved to: regression_ready_dataframe.parquet")
    
    return ['control_variables.csv', 'first_stage_regression_summary.csv', 'regression_ready_dataframe.parquet']

def main(df, folder):
    """
    Main function for control variable construction
    """
    print("=== Control Variable Construction ===\n")
    
    # Load data
    df = load_data(df)
    
    # Process all quarters
    control_variables, regression_stats = process_all_quarters(df)
    
    # Create regression summary
    regression_summary = create_regression_summary(regression_stats)
    
    # Merge with main data
    df_with_controls = merge_with_main_data(df, control_variables)
    
    # Save results
    output_files = save_results(control_variables, regression_summary, df_with_controls, folder)
    
    # Final summary
    print(f"\n=== Final Summary ===")
    print(f"Total quarters processed: {len(regression_stats)}")
    print(f"Control variables created: {len(control_variables):,}")
    print(f"Final dataset shape: {df_with_controls.shape}")
    print(f"Average R² across quarters: {regression_summary['r_squared'].mean():.4f}")
    print(f"Output files: {output_files}")
    
    return df_with_controls