import pandas as pd
import numpy as np
import statsmodels.api as sm


def calculate_variance_and_beta(
    unified_panel,
    crsp_daily_path,
    folder,
    lookback_days=126,
    min_obs_for_regression=70
):
    """
    Calculate idiosyncratic variance (sigma^2) and market beta for each stock-quarter combination.
    Winsorizes daily returns (RET and VWRETD) at the 1st and 99th percentiles.
    """

    # --- 1. Load Data ---
    try:
        unified_panel = unified_panel
        unified_panel['rdate'] = pd.to_datetime(unified_panel['rdate'])
        unified_panel['LPERMNO'] = unified_panel['LPERMNO'].astype(int)
        print(f"Loaded unified panel: {len(unified_panel)} records")

        # Load CRSP daily data
        crsp_daily = pd.read_csv(crsp_daily_path)
        crsp_daily['DATE'] = pd.to_datetime(crsp_daily['DATE'])
        crsp_daily['PERMNO'] = crsp_daily['PERMNO'].astype(int)
        print(f"Loaded CRSP daily data: {len(crsp_daily)} records")

        # Verify VWRETD column exists
        if 'VWRETD' not in crsp_daily.columns:
            print("ERROR: 'VWRETD' (Value-Weighted Market Return) not found in CRSP daily data.")
            return None, None

    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    # --- 2. Clean and prepare data ---
    crsp_daily['RET'] = pd.to_numeric(crsp_daily['RET'], errors='coerce')
    crsp_daily['VWRETD'] = pd.to_numeric(crsp_daily['VWRETD'], errors='coerce')

    # Drop rows with missing essential data
    initial_daily_count = len(crsp_daily)
    crsp_daily.dropna(subset=['RET', 'VWRETD'], inplace=True)
    print(f"CRSP daily records after cleaning: {len(crsp_daily)} (dropped {initial_daily_count - len(crsp_daily)})")

    # --- 2b. Winsorize RET and VWRETD at 1st and 99th percentiles ---
    for col in ['RET', 'VWRETD']:
        lower, upper = crsp_daily[col].quantile([0.01, 0.99])
        crsp_daily[col] = crsp_daily[col].clip(lower, upper)
    print("Winsorized RET and VWRETD at 1st and 99th percentiles.")

    # --- 3. Prepare unique stock-quarter combinations ---
    unique_permno_quarters = (
        unified_panel[['LPERMNO', 'rdate']]
        .drop_duplicates()
        .sort_values(by=['LPERMNO', 'rdate'])
    )
    print(f"Unique PERMNO-quarter combinations: {len(unique_permno_quarters)}")

    relevant_permnos = unique_permno_quarters['LPERMNO'].unique()
    crsp_daily_relevant = crsp_daily[crsp_daily['PERMNO'].isin(relevant_permnos)].copy()
    crsp_daily_relevant = crsp_daily_relevant.sort_values(by=['PERMNO', 'DATE'])
    print(f"Filtered daily data to relevant PERMNOs: {len(crsp_daily_relevant)} records")

    # --- 4. Calculate variance and beta ---
    results = []
    grouped_daily = crsp_daily_relevant.groupby('PERMNO')

    successful_calcs = 0
    insufficient_data_count = 0
    regression_failures = 0

    for index, row in unique_permno_quarters.iterrows():
        permno = row['LPERMNO']
        quarter_end_date = row['rdate']

        try:
            stock_data = grouped_daily.get_group(permno)
        except KeyError:
            results.append({'LPERMNO': permno, 'rdate': quarter_end_date,
                            'sigma2': np.nan, 'beta': np.nan})
            continue

        # Lookback window
        window_data = stock_data[stock_data['DATE'] <= quarter_end_date].copy()
        window_data = window_data.tail(lookback_days)

        if len(window_data) < min_obs_for_regression:
            results.append({'LPERMNO': permno, 'rdate': quarter_end_date,
                            'sigma2': np.nan, 'beta': np.nan})
            insufficient_data_count += 1
            continue

        # Regression
        Y = window_data['RET']
        X = sm.add_constant(window_data['VWRETD'])

        regression_df = pd.concat([Y, X], axis=1).dropna()
        if len(regression_df) < min_obs_for_regression:
            results.append({'LPERMNO': permno, 'rdate': quarter_end_date,
                            'sigma2': np.nan, 'beta': np.nan})
            insufficient_data_count += 1
            continue

        Y_reg = regression_df.iloc[:, 0]
        X_reg = regression_df.iloc[:, 1:]

        try:
            model = sm.OLS(Y_reg, X_reg)
            results_ols = model.fit()

            sigma2 = np.var(results_ols.resid, ddof=1)
            beta = results_ols.params.get('VWRETD', np.nan)

            results.append({'LPERMNO': permno, 'rdate': quarter_end_date,
                            'sigma2': sigma2, 'beta': beta})
            successful_calcs += 1

        except Exception:
            results.append({'LPERMNO': permno, 'rdate': quarter_end_date,
                            'sigma2': np.nan, 'beta': np.nan})
            regression_failures += 1

    variance_beta_df = pd.DataFrame(results)

    # --- 6. Summary ---
    summary = {
        'total_permno_quarters': len(unique_permno_quarters),
        'successful_calculations': successful_calcs,
        'insufficient_data': insufficient_data_count,
        'regression_failures': regression_failures,
        'sigma2_nan_count': variance_beta_df['sigma2'].isna().sum(),
        'beta_nan_count': variance_beta_df['beta'].isna().sum(),
        'success_rate': successful_calcs / len(unique_permno_quarters) if len(unique_permno_quarters) > 0 else 0
    }

    # --- 7. Save ---
    try:
        variance_beta_df.to_csv(f'{folder}/variance_and_beta.csv', index=False)
        print(f"Variance and beta results saved to {folder}/variance_and_beta.csv")
    except Exception as e:
        print(f"Error saving results: {e}")

    # --- 8. Print summary ---
    print(f"\n--- Variance and Beta Calculation Summary ---")
    for k,v in summary.items():
        print(f"{k}: {v}")

    print(f"\n--- Sample Results (first 5 rows) ---")
    print(variance_beta_df.head())

    return variance_beta_df, summary