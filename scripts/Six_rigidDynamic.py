import pandas as pd
import numpy as np
import time


def classify_manager_rigidity(
    dataframe,
    folder,
    window_quarters=13,
    criterion1_threshold=0.95,
    criterion2a_overlap_threshold=0.90,
    criterion2a_weight_threshold=0.1
):
    """
    Classify managers as Rigid or Dynamic based on Graves Definition 1 criteria.
    
    Args:
        dataframe (DataFrame): Panel with manager holdings and characteristics
        output_path (str): Path for output CSV file
        window_quarters (int): Rolling window size in quarters (default: 13)
        criterion1_threshold (float): Overlap threshold for Criterion 1 (default: 0.95)
        criterion2a_overlap_threshold (float): Overlap threshold for Criterion 2a (default: 0.90)
        criterion2a_weight_threshold (float): Weight difference threshold for Criterion 2a (default: 0.1)
        
    Returns:
        tuple: (classified_df, summary_dict) with manager classifications and statistics
    """
    
    # Validate required columns
    required_cols = ['mgrno', 'rdate', 'LPERMNO', 'Value_Share', 'Market_Equity']
    missing_cols = [col for col in required_cols if col not in dataframe.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return None, None
    
    # Prepare data
    df = dataframe.copy()
    df['rdate'] = pd.to_datetime(df['rdate'])
    df['LPERMNO'] = df['LPERMNO'].astype(int)
    df['mgrno'] = df['mgrno'].astype(int)
    df = df.sort_values(['mgrno', 'rdate']).reset_index(drop=True)
    
    print(f"Processing {len(df)} records for {df['mgrno'].nunique()} managers")
    
    # Pre-compute holdings and value shares by manager-quarter
    holdings_by_mgr_quarter = df.groupby(['mgrno', 'rdate'])['LPERMNO'].apply(set)
    value_shares_by_mgr_quarter = df.groupby(['mgrno', 'rdate']).apply(
        lambda x: x.set_index('LPERMNO')['Value_Share'].to_dict(), include_groups=False
    )
    
    # Pre-compute market equity lookup (ultra-fast for large datasets)
    print("Pre-computing market equity lookup...")
    # Create flat dictionary with tuple keys for maximum speed
    market_equity_lookup = df.set_index(['mgrno', 'rdate', 'LPERMNO'])['Market_Equity'].to_dict()
    print("Market equity lookup completed.")
    
    # Initialize rigidity status storage
    rigidity_status = {}
    num_overlap_pairs = window_quarters - 1
    
    # Initialize timing variables
    start_time = time.time()
    total_managers = df['mgrno'].nunique()
    manager_ids = df['mgrno'].unique()
    
    print(f"Starting processing of {total_managers} managers...")
    
    # Process each manager
    for manager_idx, mgr_id in enumerate(manager_ids):
        mgr_quarters = holdings_by_mgr_quarter.loc[mgr_id].sort_index()
        mgr_value_shares = value_shares_by_mgr_quarter.loc[mgr_id].sort_index()
        
        # Initialize all quarters as None (insufficient history)
        for rdate in mgr_quarters.index:
            rigidity_status[(mgr_id, rdate)] = None
        
        # Evaluate quarters with sufficient history
        for i in range(window_quarters - 1, len(mgr_quarters)):
            rdate = mgr_quarters.index[i]
            
            # Get window data
            window_start = i - window_quarters + 1
            window_end = i + 1
            holdings_window = list(mgr_quarters.iloc[window_start:window_end])
            value_share_window = list(mgr_value_shares.iloc[window_start:window_end])
            
            # Initialize criteria flags
            is_rigid_c1 = is_rigid_c2a = is_rigid_c2b = False
            
            # Criterion 1: Average overlap > 95%
            overlap_ratios = []
            for j in range(1, window_quarters):
                h_prev, h_curr = holdings_window[j-1], holdings_window[j]
                if len(h_prev) > 0:
                    overlap = len(h_prev.intersection(h_curr)) / len(h_prev)
                    overlap_ratios.append(overlap)
                else:
                    overlap_ratios.append(0)
            
            if len(overlap_ratios) == num_overlap_pairs and np.mean(overlap_ratios) >= criterion1_threshold:
                is_rigid_c1 = True
                # Early exit: If Criterion 1 passes, immediately set rigidity and skip Criteria 2a and 2b
                rigidity_status[(mgr_id, rdate)] = True
                continue
            
            # Criterion 2a: All quarters ≥90% overlap AND weight stability <0.1
            # Check all overlaps ≥90%
            all_overlaps_90 = True
            for j in range(1, window_quarters):
                h_prev, h_curr = holdings_window[j-1], holdings_window[j]
                if len(h_prev) == 0 or len(h_prev.intersection(h_curr)) / len(h_prev) < criterion2a_overlap_threshold:
                    all_overlaps_90 = False
                    break
            
            # Check weight stability if overlap criterion met
            all_weights_stable = True
            if all_overlaps_90:
                for j in range(1, window_quarters):
                    h_prev = holdings_window[j-1]
                    ws_prev = value_share_window[j-1]
                    ws_curr = value_share_window[j]
                    
                    weight_diff_sum = sum(abs(ws_curr.get(stock, 0) - ws_prev.get(stock, 0)) 
                                        for stock in h_prev)
                    
                    if weight_diff_sum >= criterion2a_weight_threshold:
                        all_weights_stable = False
                        break
            
            if all_overlaps_90 and all_weights_stable:
                is_rigid_c2a = True
                # Early exit: If Criterion 2a passes, immediately set rigidity and skip Criterion 2b
                rigidity_status[(mgr_id, rdate)] = True
                continue
                
            # Criterion 2b: Fixed-share schemes with market cap adjustments
            if all_overlaps_90:  # Only check 2b if overlap criterion is met
                all_adjusted_weights_stable = True
                
                for j in range(1, window_quarters):
                    h_prev, h_curr = holdings_window[j-1], holdings_window[j]
                    common_stocks = h_prev.intersection(h_curr)
                    
                    if len(common_stocks) == 0:
                        all_adjusted_weights_stable = False
                        break
                    
                    # Get market cap data for adjustment using optimized lookup
                    prev_rdate = mgr_quarters.index[window_start + j - 1]
                    curr_rdate = mgr_quarters.index[window_start + j]
                    
                    total_adjusted_diff = 0
                    for stock in common_stocks:
                        # Get market caps using pre-computed flat lookup
                        prev_mc = market_equity_lookup.get((mgr_id, prev_rdate, stock), np.nan)
                        curr_mc = market_equity_lookup.get((mgr_id, curr_rdate, stock), np.nan)
                        
                        # Skip if market cap data is missing
                        if not (pd.isna(prev_mc) or pd.isna(curr_mc)) and prev_mc > 0:
                            # Get weights
                            w_prev = value_share_window[j-1].get(stock, 0)
                            w_curr = value_share_window[j].get(stock, 0)
                            
                            # Adjust current weight for market cap change
                            mc_ratio = curr_mc / prev_mc
                            w_curr_adjusted = w_curr / mc_ratio
                            
                            total_adjusted_diff += abs(w_curr_adjusted - w_prev)
                    
                    if total_adjusted_diff >= criterion2a_weight_threshold:
                        all_adjusted_weights_stable = False
                        break
                    
                if all_adjusted_weights_stable:
                    is_rigid_c2b = True
            
            # Store rigidity status (only reached if no early exits occurred)
            rigidity_status[(mgr_id, rdate)] = is_rigid_c2b
        
        # Print progress every 1 manager
        if (manager_idx + 1) % 1 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_manager = elapsed_time / (manager_idx + 1)
            remaining_managers = total_managers - (manager_idx + 1)
            estimated_remaining_time = avg_time_per_manager * remaining_managers
            
            print(f"Progress: {manager_idx + 1:,}/{total_managers:,} managers processed "
                  f"({(manager_idx + 1)/total_managers*100:.1f}%) | "
                  f"Elapsed: {elapsed_time:.1f}s | "
                  f"Avg: {avg_time_per_manager:.3f}s/mgr | "
                  f"ETA: {estimated_remaining_time:.1f}s", flush=True)
    
    # Apply classification to dataframe
    def get_manager_type(rigidity):
        if rigidity is None:
            return ""  # Insufficient history
        return "Rigid" if rigidity else "Dynamic"
    
    df['Manager_Type'] = df.apply(
        lambda row: get_manager_type(rigidity_status.get((row['mgrno'], row['rdate']), None)), 
        axis=1
    )
    
    # Generate summary statistics
    type_counts = df['Manager_Type'].value_counts(dropna=False)
    total_records = len(df)
    
    summary = {
        'total_records': total_records,
        'total_managers': df['mgrno'].nunique(),
        'rigid_records': type_counts.get('Rigid', 0),
        'dynamic_records': type_counts.get('Dynamic', 0),
        'insufficient_history': type_counts.get('', 0),
        'rigid_percentage': (type_counts.get('Rigid', 0) / total_records * 100) if total_records > 0 else 0,
        'dynamic_percentage': (type_counts.get('Dynamic', 0) / total_records * 100) if total_records > 0 else 0,
        'window_quarters': window_quarters,
        'criterion1_threshold': criterion1_threshold,
        'criterion2a_overlap_threshold': criterion2a_overlap_threshold,
        'criterion2a_weight_threshold': criterion2a_weight_threshold
    }
    
    # Save results
    try:
        df.to_parquet(f'{folder}/ManagersSortedByRigidity.parquet', index=False)
        print(f"Results saved to 'ManagersSortedByRigidity'")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    # Print summary
    print(f"\n--- Manager Classification Summary ---")
    print(f"Total records: {summary['total_records']:,}")
    print(f"Total managers: {summary['total_managers']:,}")
    print(f"Rigid records: {summary['rigid_records']:,} ({summary['rigid_percentage']:.1f}%)")
    print(f"Dynamic records: {summary['dynamic_records']:,} ({summary['dynamic_percentage']:.1f}%)")
    print(f"Insufficient history: {summary['insufficient_history']:,}")
    
    print(f"\n--- Sample Classifications ---")
    print(df[['mgrno', 'rdate', 'LPERMNO', 'Manager_Type']].head(10))
    
    return df, summary
