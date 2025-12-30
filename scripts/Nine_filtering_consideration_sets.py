import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_and_filter_consideration_sets(consideration_sets):
    """
    Load the consideration sets and filter out those with less than 25 current holdings
    """
    print("Loading consideration sets...")
    
    print(f"Original consideration sets: {len(consideration_sets)}")
    print(f"Original unique managers: {consideration_sets['mgrno'].nunique()}")
    
    # Filter out rows with less than 25 current holdings
    filtered_consideration_sets = consideration_sets[
        consideration_sets['current_holdings_count'] >= 25
    ].copy()
    
    print(f"Filtered consideration sets: {len(filtered_consideration_sets)}")
    print(f"Filtered unique managers: {filtered_consideration_sets['mgrno'].nunique()}")
    
    # Show how many were filtered out
    filtered_out = len(consideration_sets) - len(filtered_consideration_sets)
    print(f"Filtered out: {filtered_out} manager-date combinations")
    
    return filtered_consideration_sets

def display_summary_statistics(consideration_sets):
    """
    Display comprehensive summary statistics for the filtered consideration sets
    """
    print(f"\n=== Summary Statistics ===")
    print(f"Total consideration sets: {len(consideration_sets)}")
    print(f"Unique managers: {consideration_sets['mgrno'].nunique()}")
    print(f"Average consideration set size: {consideration_sets['consideration_set_size'].mean():.0f}")
    print(f"Average current holdings: {consideration_sets['current_holdings_count'].mean():.0f}")
    print(f"Average historical holdings: {consideration_sets['historical_holdings_count'].mean():.0f}")
    
    if consideration_sets['current_holdings_count'].mean() > 0:
        print(f"Average expansion ratio: {consideration_sets['consideration_set_size'].mean() / consideration_sets['current_holdings_count'].mean():.1f}x")
    
    # Show distribution of consideration set sizes
    print(f"\n=== Consideration Set Size Distribution ===")
    size_dist = consideration_sets['consideration_set_size'].describe()
    print(size_dist)
    
    # Show distribution of current holdings counts
    print(f"\n=== Current Holdings Count Distribution ===")
    current_dist = consideration_sets['current_holdings_count'].describe()
    print(current_dist)
    
    # Show distribution of historical holdings counts
    print(f"\n=== Historical Holdings Count Distribution ===")
    historical_dist = consideration_sets['historical_holdings_count'].describe()
    print(historical_dist)
    
    # Show manager frequency
    print(f"\n=== Manager Frequency ===")
    manager_freq = consideration_sets['mgrno'].value_counts()
    print(f"Most frequent manager appears: {manager_freq.max()} times")
    print(f"Least frequent manager appears: {manager_freq.min()} times")
    print(f"Median manager frequency: {manager_freq.median():.0f} times")
    
    # Show date range
    print(f"\n=== Date Range ===")
    consideration_sets['rdate'] = pd.to_datetime(consideration_sets['rdate'])
    print(f"Date range: {consideration_sets['rdate'].min()} to {consideration_sets['rdate'].max()}")
    
    # Show industries expanded
    print(f"\n=== Industry Expansion ===")
    # Count how many industries were expanded for each manager-date
    consideration_sets['industries_expanded_count'] = consideration_sets['industries_expanded'].apply(
        lambda x: len(eval(x)) if isinstance(x, str) else 0
    )
    print(f"Average industries expanded: {consideration_sets['industries_expanded_count'].mean():.1f}")
    print(f"Max industries expanded: {consideration_sets['industries_expanded_count'].max()}")
    print(f"Min industries expanded: {consideration_sets['industries_expanded_count'].min()}")

def save_filtered_data(consideration_sets, folder):
    """
    Save the filtered consideration sets to a new file
    """
    consideration_sets.to_csv(f'{folder}/filtered_consideration_sets_25plus_holdings.csv', index=False)
    print(f"\nFiltered consideration sets saved to: filtered_consideration_sets_25plus_holdings")

def main(consideration_sets, folder):
    """
    Main function to filter and analyze consideration sets
    """
    print("=== Filtering Consideration Sets (25+ Current Holdings) ===\n")
    
    # Load and filter consideration sets
    filtered_consideration_sets = load_and_filter_consideration_sets(consideration_sets)
    
    if len(filtered_consideration_sets) == 0:
        print("No consideration sets meet the criteria (25+ current holdings)")
        return
    
    # Display summary statistics
    display_summary_statistics(filtered_consideration_sets)
    
    # Save filtered data
    save_filtered_data(filtered_consideration_sets, folder)
    
    # Show sample of results
    print(f"\n=== Sample Results ===")
    print(filtered_consideration_sets.head(5))

    return filtered_consideration_sets
