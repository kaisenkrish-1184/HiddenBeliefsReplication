import pandas as pd
import numpy as np
from datetime import datetime

def load_data(df):
    """
    Load the full dataset for instrument construction
    """
    print("Loading parquet file...")
    
    # Convert dates
    df['rdate'] = pd.to_datetime(df['rdate'])
    
    print(f"Full dataset shape: {df.shape}")
    print(f"Date range: {df['rdate'].min()} to {df['rdate'].max()}")
    print(f"Unique quarters: {df['rdate'].nunique()}")
    
    return df

def get_institution_data(df_quarter):
    """
    Get institution-level data efficiently using the fact that AUM is only in holdings
    """
    print("\nGetting institution-level data...")
    
    # Get AUM from holdings (where AUM is not null)
    institution_aum = df_quarter[df_quarter['AUM'].notna()].groupby('mgrno').agg({
        'AUM': 'first',  # Should be the same for all holdings of same manager
        'Manager_Type': 'first'
    }).reset_index()
    
    print(f"Institutions with AUM data: {len(institution_aum)}")
    
    # Get consideration sets for ALL institutions (including those without AUM)
    consideration_sets = df_quarter.groupby('mgrno')['LPERMNO'].apply(list).reset_index()
    consideration_sets.columns = ['mgrno', 'consideration_set']
    
    # Merge AUM data with consideration sets
    institution_data = consideration_sets.merge(institution_aum, on='mgrno', how='left')
    
    print(f"Total institutions: {len(institution_data)}")
    print(f"Institutions with AUM: {institution_data['AUM'].notna().sum()}")
    print(f"Institutions without AUM: {institution_data['AUM'].isna().sum()}")
    
    return institution_data

def get_stock_characteristics(df_quarter):
    """
    Get stock characteristics for the quarter
    """
    print("\nGetting stock characteristics...")
    
    # Get unique stocks with their characteristics
    stock_chars = df_quarter.groupby('LPERMNO').agg({
        'Book_Value_Equity': 'first',
        'Market_Equity': 'first',
        'Log_Market_Cap': 'first',
        'Log_Book_Value': 'first'
    }).reset_index()
    
    print(f"Number of unique stocks: {len(stock_chars)}")
    
    return stock_chars

def compute_equal_weighted_demand(institution_data, stock_chars):
    """
    Compute equal-weighted counterfactual demand for each stock
    Only for institutions with AUM data
    """
    print("\nComputing equal-weighted counterfactual demand...")
    
    # Initialize demand dictionary
    equal_weighted_demand = {}
    
    # Only process institutions with AUM data
    institutions_with_aum = institution_data[institution_data['AUM'].notna()]
    
    for _, row in institutions_with_aum.iterrows():
        mgrno = row['mgrno']
        aum = row['AUM']
        consideration_set = row['consideration_set']
        
        # Equal weight per stock in consideration set
        if len(consideration_set) > 0:
            weight_per_stock = aum / len(consideration_set)
            
            # Add demand for each stock in consideration set
            for stock in consideration_set:
                if stock not in equal_weighted_demand:
                    equal_weighted_demand[stock] = 0
                equal_weighted_demand[stock] += weight_per_stock
    
    print(f"Stocks with equal-weighted demand: {len(equal_weighted_demand)}")
    print(f"Institutions contributing to equal-weighted demand: {len(institutions_with_aum)}")
    
    return equal_weighted_demand

def compute_book_weighted_demand(institution_data, stock_chars):
    """
    Compute book-value-weighted counterfactual demand for each stock
    Only for institutions with AUM data
    """
    print("\nComputing book-value-weighted counterfactual demand...")
    
    # Initialize demand dictionary
    book_weighted_demand = {}
    
    # Only process institutions with AUM data
    institutions_with_aum = institution_data[institution_data['AUM'].notna()]
    
    for _, row in institutions_with_aum.iterrows():
        mgrno = row['mgrno']
        aum = row['AUM']
        consideration_set = row['consideration_set']
        
        if len(consideration_set) > 0:
            # Get book values for stocks in consideration set
            consideration_stocks = stock_chars[stock_chars['LPERMNO'].isin(consideration_set)]
            
            # Skip if no book value data
            if len(consideration_stocks) == 0:
                continue
                
            # Calculate total book value in consideration set
            total_book_value = consideration_stocks['Book_Value_Equity'].sum()
            
            # Skip if total book value is zero or missing
            if total_book_value == 0 or pd.isna(total_book_value):
                continue
            
            # Book-value weight for each stock
            for _, stock_row in consideration_stocks.iterrows():
                stock = stock_row['LPERMNO']
                book_value = stock_row['Book_Value_Equity']
                
                if pd.isna(book_value) or book_value <= 0:
                    continue
                    
                weight = aum * (book_value / total_book_value)
                
                if stock not in book_weighted_demand:
                    book_weighted_demand[stock] = 0
                book_weighted_demand[stock] += weight
    
    print(f"Stocks with book-weighted demand: {len(book_weighted_demand)}")
    print(f"Institutions contributing to book-weighted demand: {len(institutions_with_aum)}")
    
    return book_weighted_demand

def create_instrument(equal_weighted_demand, book_weighted_demand, stock_chars, quarter_date):
    """
    Create the log price instrument
    """
    print("\nCreating log price instrument...")
    
    # Get all stocks that have either type of demand
    all_stocks = set(equal_weighted_demand.keys()) | set(book_weighted_demand.keys())
    
    instrument_data = []
    
    for stock in all_stocks:
        eq_demand = equal_weighted_demand.get(stock, 0)
        book_demand = book_weighted_demand.get(stock, 0)
        
        # Total counterfactual demand
        total_demand = eq_demand + book_demand
        
        # Log price instrument
        log_instrument = np.log(1 + total_demand)
        
        instrument_data.append({
            'LPERMNO': stock,
            'rdate': quarter_date,
            'equal_weighted_demand': eq_demand,
            'book_weighted_demand': book_demand,
            'total_counterfactual_demand': total_demand,
            'log_price_instrument': log_instrument
        })
    
    instrument_df = pd.DataFrame(instrument_data)
    
    print(f"Instrument created for {len(instrument_df)} stocks")
    print(f"Average log instrument: {instrument_df['log_price_instrument'].mean():.4f}")
    print(f"Log instrument range: {instrument_df['log_price_instrument'].min():.4f} to {instrument_df['log_price_instrument'].max():.4f}")
    
    return instrument_df

def process_quarter(df_quarter, quarter_date):
    """
    Process a single quarter to create instruments
    """
    print(f"\n{'='*60}")
    print(f"Processing quarter: {quarter_date}")
    print(f"{'='*60}")
    
    # Get institution data (AUM + consideration sets)
    institution_data = get_institution_data(df_quarter)
    
    # Get stock characteristics
    stock_chars = get_stock_characteristics(df_quarter)
    
    # Compute counterfactual demands
    equal_weighted_demand = compute_equal_weighted_demand(institution_data, stock_chars)
    book_weighted_demand = compute_book_weighted_demand(institution_data, stock_chars)
    
    # Create instrument
    instrument_df = create_instrument(equal_weighted_demand, book_weighted_demand, stock_chars, quarter_date)
    
    return instrument_df

def main(df, FOLDER):
    """
    Main function to create instruments for all quarters
    """
    print("=== Instrument Construction for All Quarters ===\n")
    
    # Load full dataset
    df = load_data(df)
    
    # Get unique quarters
    quarters = sorted(df['rdate'].unique())
    print(f"\nProcessing {len(quarters)} quarters...")
    
    # Process each quarter
    all_instruments = []
    
    for i, quarter_date in enumerate(quarters):
        print(f"\nProgress: {i+1}/{len(quarters)}")
        
        # Filter to current quarter
        df_quarter = df[df['rdate'] == quarter_date].copy()
        
        # Process quarter
        instrument_df = process_quarter(df_quarter, quarter_date)
        
        # Add to results
        all_instruments.append(instrument_df)
        
        # Clear memory
        del df_quarter, instrument_df
    
    # Combine all results
    print(f"\n{'='*60}")
    print("Combining results from all quarters...")
    print(f"{'='*60}")
    
    final_instruments = pd.concat(all_instruments, ignore_index=True)
    
    # Show final results
    print(f"\n=== Final Results Summary ===")
    print(f"Total quarters processed: {len(quarters)}")
    print(f"Total instrument observations: {len(final_instruments)}")
    print(f"Unique stocks: {final_instruments['LPERMNO'].nunique()}")
    print(f"Date range: {final_instruments['rdate'].min()} to {final_instruments['rdate'].max()}")
    
    # Show sample of results
    print(f"\nSample of final instrument values:")
    print(final_instruments.head(10))
    
    # Save results
    output_filename = f'{FOLDER}/instruments_all_quarters.csv'
    final_instruments.to_csv(output_filename, index=False)
    print(f"\nAll instruments saved to: {output_filename}")
    
    return final_instruments

if __name__ == "__main__":
    instruments_df = main()