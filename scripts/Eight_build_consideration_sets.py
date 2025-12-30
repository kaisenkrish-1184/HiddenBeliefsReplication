import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import time
import os
from collections import Counter

# Try to import psutil for memory monitoring, fallback if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Memory monitoring disabled.")

def load_and_prepare_data(dataframe):
    """
    Load the main data file and prepare for consideration set construction
    Keep ALL managers for historical holdings, but identify which ones are dynamic at each time t
    """
    print("Loading data...")
    df = dataframe
    
    # Convert dates to datetime
    df['rdate'] = pd.to_datetime(df['rdate'])
    
    # Ensure NAICS codes are strings and extract first 4 digits
    df['NAICS'] = df['NAICS'].astype(str)
    df['NAICS_4DIGIT'] = df['NAICS'].str[:4]
    
    print(f"Original data shape: {df.shape}")
    print(f"Number of unique managers: {df['mgrno'].nunique()}")
    print(f"Date range: {df['rdate'].min()} to {df['rdate'].max()}")
    print(f"Unique 4-digit NAICS codes: {df['NAICS_4DIGIT'].nunique()}")
    
    # Don't filter out rigid managers yet - we need their historical holdings
    # We'll filter for dynamic managers only when building consideration sets at time t
    return df

def load_stock_universe(universe):
    """
    Load stockUniverse.csv which contains all stocks that fit the original criteria
    """
    print("Loading stock universe...")
    stock_universe = universe
    
    # Convert dates to datetime
    stock_universe['rdate'] = pd.to_datetime(stock_universe['rdate'])
    
    # Ensure NAICS codes are strings and extract first 4 digits
    stock_universe['NAICS'] = stock_universe['NAICS'].astype(str)
    stock_universe['NAICS_4DIGIT'] = stock_universe['NAICS'].str[:4]
    
    # Filter out stocks with incomplete NAICS codes (< 4 digits)
    stock_universe = stock_universe[stock_universe['NAICS_4DIGIT'].str.len() >= 4]
    
    print(f"Stock universe shape: {stock_universe.shape}")
    print(f"Unique stocks in universe: {stock_universe['LPERMNO'].nunique()}")
    print(f"Unique 4-digit NAICS codes in universe: {stock_universe['NAICS_4DIGIT'].nunique()}")
    
    return stock_universe

def get_historical_holdings(df, quarters_back=12):
    """
    Get historical holdings for each manager going back specified number of quarters
    Keep ALL managers (both rigid and dynamic) for historical data
    """
    print(f"Getting historical holdings going back {quarters_back} quarters...")
    
    # Group by manager and date to get holdings
    holdings = df.groupby(['mgrno', 'rdate', 'LPERMNO']).agg({
        'shares': 'sum',
        'AUM': 'first',
        'Manager_Type': 'first',
        'NAICS_4DIGIT': 'first'
    }).reset_index()
    
    # Add quarter information
    holdings['year'] = holdings['rdate'].dt.year
    holdings['quarter'] = holdings['rdate'].dt.quarter
    
    print(f"Historical holdings shape: {holdings.shape}")
    print(f"Unique managers in historical data: {holdings['mgrno'].nunique()}")
    
    return holdings

def get_memory_usage():
    """Get current memory usage in MB"""
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    else:
        return 0.0  # Return 0 if psutil not available

def precompute_lookups(historical_holdings, stock_universe):
    """
    Pre-compute all lookups for maximum speed optimization
    """
    print("=== Phase 1 Optimization: Pre-computing lookups ===")
    start_memory = get_memory_usage()
    start_time = time.time()
    
    # 1. Pre-compute current holdings lookup: {(mgrno, rdate): [holdings_list]}
    print("Pre-computing current holdings lookup...")
    current_holdings_lookup = {}
    for (mgrno, rdate), group in historical_holdings.groupby(['mgrno', 'rdate']):
        current_holdings_lookup[(mgrno, rdate)] = group['LPERMNO'].tolist()
    print(f"  Current holdings lookup: {len(current_holdings_lookup):,} entries")
    
    # 2. Pre-compute historical date ranges: {(mgrno, rdate): [list_of_13_historical_dates]}
    print("Pre-computing historical date ranges...")
    historical_dates_lookup = {}
    manager_dates = historical_holdings.groupby('mgrno')['rdate'].apply(lambda x: sorted(x.unique())).to_dict()
    
    for mgrno, all_dates in manager_dates.items():
        for i, current_date in enumerate(all_dates):
            # Get up to 13 most recent dates (including current)
            start_idx = max(0, i + 1 - 13)
            historical_dates = all_dates[start_idx:i+1]
            historical_dates_lookup[(mgrno, current_date)] = historical_dates
    print(f"  Historical dates lookup: {len(historical_dates_lookup):,} entries")
    
    # 3. Pre-compute historical holdings using vectorized approach
    print("Pre-computing historical holdings (vectorized)...")
    historical_holdings_lookup = {}
    
    # Group holdings by manager for efficient processing
    holdings_by_manager = historical_holdings.groupby('mgrno')
    
    for mgrno, manager_data in holdings_by_manager:
        # Get all dates for this manager
        manager_dates = sorted(manager_data['rdate'].unique())
        
        # For each date, get historical holdings efficiently
        for current_date in manager_dates:
            if (mgrno, current_date) in historical_dates_lookup:
                historical_dates = historical_dates_lookup[(mgrno, current_date)]
                
                # Filter manager's data for historical dates (much faster on smaller subset)
                historical_data = manager_data[manager_data['rdate'].isin(historical_dates)]
                historical_stocks = historical_data['LPERMNO'].unique().tolist()
                historical_holdings_lookup[(mgrno, current_date)] = historical_stocks
    
    print(f"  Historical holdings lookup: {len(historical_holdings_lookup):,} entries")
    
    # 4. Pre-compute NAICS codes using same vectorized approach
    print("Pre-computing NAICS codes (vectorized)...")
    naics_lookup = {}
    
    for mgrno, manager_data in holdings_by_manager:
        manager_dates = sorted(manager_data['rdate'].unique())
        
        for current_date in manager_dates:
            if (mgrno, current_date) in historical_dates_lookup:
                historical_dates = historical_dates_lookup[(mgrno, current_date)]
                
                # Filter manager's data for historical dates
                historical_data = manager_data[manager_data['rdate'].isin(historical_dates)]
                naics_codes = historical_data['NAICS_4DIGIT'].dropna().tolist()  # Keep duplicates for proper counting
                naics_lookup[(mgrno, current_date)] = naics_codes
    
    print(f"  NAICS lookup: {len(naics_lookup):,} entries")
    
    # 5. Pre-compute stock universe by date and NAICS: {(rdate, naics_4digit): [lpermno_list]}
    print("Pre-computing stock universe lookup...")
    stock_universe_lookup = {}
    for (rdate, naics), group in stock_universe.groupby(['rdate', 'NAICS_4DIGIT']):
        stock_universe_lookup[(rdate, naics)] = group['LPERMNO'].unique().tolist()
    print(f"  Stock universe lookup: {len(stock_universe_lookup):,} entries")
    
    end_memory = get_memory_usage()
    precompute_time = time.time() - start_time
    
    print(f"=== Pre-computation Complete ===")
    print(f"Time: {precompute_time:.2f} seconds")
    if PSUTIL_AVAILABLE:
        print(f"Memory usage: {start_memory:.1f} MB -> {end_memory:.1f} MB (+{end_memory-start_memory:.1f} MB)")
    else:
        print("Memory monitoring: Not available (psutil not installed)")
    
    return {
        'current_holdings': current_holdings_lookup,
        'historical_dates': historical_dates_lookup,
        'historical_holdings': historical_holdings_lookup,
        'naics_codes': naics_lookup,
        'stock_universe': stock_universe_lookup
    }

def build_consideration_sets_optimized(df, historical_holdings, stock_universe, validate=True):
    """
    OPTIMIZED: Build consideration sets for each dynamic manager at each date
    Based on Graves' methodology with Phase 1 optimizations
    """
    print("Building consideration sets for dynamic managers (OPTIMIZED)...")
    
    # Get unique manager-date combinations where the manager is dynamic
    dynamic_manager_dates = df[df['Manager_Type'] == 'Dynamic'][['mgrno', 'rdate']].drop_duplicates()
    print(f"Number of dynamic manager-date combinations: {len(dynamic_manager_dates)}")
    
    # Pre-compute all lookups
    lookups = precompute_lookups(historical_holdings, stock_universe)
    
    consideration_sets = []
    start_time = time.time()
    
    # Validation: store results for comparison if requested
    validation_results = [] if validate else None
    
    for i, (_, row) in enumerate(dynamic_manager_dates.iterrows()):
        if i % 100 == 0:  # Progress indicator every 100 iterations
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(dynamic_manager_dates) - i - 1) / rate if rate > 0 else 0
            if PSUTIL_AVAILABLE:
                current_memory = get_memory_usage()
                print(f"Progress: {i+1}/{len(dynamic_manager_dates)} ({elapsed:.1f}s elapsed, {rate:.2f} iter/s, {eta:.1f}s remaining, {current_memory:.1f} MB)")
            else:
                print(f"Progress: {i+1}/{len(dynamic_manager_dates)} ({elapsed:.1f}s elapsed, {rate:.2f} iter/s, {eta:.1f}s remaining)")
        
        mgrno = row['mgrno']
        rdate = row['rdate']
        
        # OPTIMIZED: Use pre-computed lookup instead of DataFrame filtering
        current_holdings = lookups['current_holdings'].get((mgrno, rdate), [])
        
        # OPTIMIZED: Use pre-computed historical holdings
        historical_stocks = lookups['historical_holdings'].get((mgrno, rdate), [])
        
        # OPTIMIZED: Use pre-computed NAICS codes
        naics_codes = lookups['naics_codes'].get((mgrno, rdate), [])
        
        # Find 4-digit industries with 2+ holdings from RECENT holdings
        # Count NAICS codes using fast Python collections.Counter (preserves duplicates)
        naics_counts = Counter(naics_codes)
        industries_to_expand = [naics for naics, count in naics_counts.items() if count >= 2]
        
        # OPTIMIZED: Use pre-computed stock universe lookup
        expanded_stocks = []
        for naics in industries_to_expand:
            stocks_in_industry = lookups['stock_universe'].get((rdate, naics), [])
            expanded_stocks.extend(stocks_in_industry)
        
        # Final consideration set = historical holdings + industry expansion
        consideration_set = list(set(historical_stocks + expanded_stocks))
        
        result = {
            'mgrno': mgrno,
            'rdate': rdate,
            'current_holdings': current_holdings,
            'historical_holdings': historical_stocks,
            'consideration_set': consideration_set,
            'consideration_set_size': len(consideration_set),
            'current_holdings_count': len(current_holdings),
            'historical_holdings_count': len(historical_stocks),
            'industries_expanded': industries_to_expand,
            'naics_4digit_codes': list(set(naics_codes))  # Store unique NAICS for reference
        }
        
        consideration_sets.append(result)
        
        # Store for validation if requested
        if validate:
            validation_results.append(result)
    
    total_time = time.time() - start_time
    print(f"Optimized processing completed in {total_time:.1f} seconds")
    print(f"Average time per iteration: {total_time/len(dynamic_manager_dates):.3f} seconds")
    
    result_df = pd.DataFrame(consideration_sets)
    
    # Validation against original logic (sample-based)
    if validate and len(validation_results) > 0:
        print("\n=== Validation: Comparing optimized vs original logic ===")
        validate_optimization(df, historical_holdings, stock_universe, validation_results[:min(10, len(validation_results))])
    
    return result_df

def validate_optimization(df, historical_holdings, stock_universe, optimized_sample):
    """
    Validate that optimized results match original logic for a sample
    """
    print("Running validation on sample of optimized results...")
    
    mismatches = 0
    for i, opt_result in enumerate(optimized_sample):
        mgrno = opt_result['mgrno']
        rdate = opt_result['rdate']
        
        # Run original logic for this manager-date
        current_holdings_orig = historical_holdings[
            (historical_holdings['mgrno'] == mgrno) & 
            (historical_holdings['rdate'] == rdate)
        ]['LPERMNO'].tolist()
        
        historical_dates_orig = historical_holdings[
            (historical_holdings['mgrno'] == mgrno) & 
            (historical_holdings['rdate'] <= rdate)
        ]['rdate'].unique()
        historical_dates_orig = sorted(historical_dates_orig)[-13:]
        
        historical_stocks_orig = historical_holdings[
            (historical_holdings['mgrno'] == mgrno) & 
            (historical_holdings['rdate'].isin(historical_dates_orig))
        ]['LPERMNO'].unique().tolist()
        
        # Compare results
        current_match = set(opt_result['current_holdings']) == set(current_holdings_orig)
        historical_match = set(opt_result['historical_holdings']) == set(historical_stocks_orig)
        
        if not current_match or not historical_match:
            mismatches += 1
            print(f"  Mismatch {mismatches}: Manager {mgrno}, Date {rdate}")
            if not current_match:
                print(f"    Current holdings differ: {len(opt_result['current_holdings'])} vs {len(current_holdings_orig)}")
            if not historical_match:
                print(f"    Historical holdings differ: {len(opt_result['historical_holdings'])} vs {len(historical_stocks_orig)}")
    
    if mismatches == 0:
        print(f"✓ Validation PASSED: All {len(optimized_sample)} samples match original logic")
    else:
        print(f"✗ Validation FAILED: {mismatches}/{len(optimized_sample)} samples had mismatches")
    
    return mismatches == 0

def build_consideration_sets(df, historical_holdings, stock_universe):
    """
    Main wrapper function - uses optimized version by default
    """
    return build_consideration_sets_optimized(df, historical_holdings, stock_universe, validate=True)

def main(dataframe, universe, folder):
    """
    Main function to build consideration sets
    """
    print("=== Building Consideration Sets for Dynamic Managers ===\n")
    
    # Load data (keep ALL managers for historical holdings)
    df = load_and_prepare_data(dataframe)
    
    # Load stock universe for industry expansion
    stock_universe = load_stock_universe(universe)
    
    # Get historical holdings from ALL managers
    historical_holdings = get_historical_holdings(df)
    
    # Build consideration sets only for dynamic managers at time t
    consideration_sets = build_consideration_sets(df, historical_holdings, stock_universe)
    
    # Save results
    consideration_sets.to_csv(f'{folder}/dynamic_manager_consideration_sets.csv', index=False)
    print(f"\nConsideration sets saved to: dynamic_manager_consideration_sets.csv")
    
    # Summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"Total consideration sets: {len(consideration_sets)}")
    print(f"Unique managers: {consideration_sets['mgrno'].nunique()}")
    print(f"Average consideration set size: {consideration_sets['consideration_set_size'].mean():.0f}")
    print(f"Average current holdings: {consideration_sets['current_holdings_count'].mean():.0f}")
    print(f"Average historical holdings: {consideration_sets['historical_holdings_count'].mean():.0f}")
    print(f"Average expansion ratio: {consideration_sets['consideration_set_size'].mean() / consideration_sets['current_holdings_count'].mean():.1f}x")
    
    # Show sample of results
    print(f"\n=== Sample Results ===")
    print(consideration_sets.head(5))
    
    # Show distribution of consideration set sizes
    print(f"\n=== Consideration Set Size Distribution ===")
    size_dist = consideration_sets['consideration_set_size'].describe()
    print(size_dist)

    return consideration_sets