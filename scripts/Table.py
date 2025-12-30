import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_alpha_table(csv_path='test/four_factor_alpha_results.csv'):
    """
    Create a matplotlib table showing annualized alpha by size decile and HBI quintile.
    
    Parameters
    ----------
    csv_path : str
        Path to the four_factor_alpha_results.csv file
    """
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Convert annualized alpha to percentage
    df['Alpha_Pct'] = df['Alpha_Annualized'] * 100
    
    # Pivot table: rows = Size_Decile (D10 to D1), columns = HBI_Quintile (Q1 to Q5)
    # Note: Size_Decile 1 = D1 (smallest), Size_Decile 10 = D10 (largest)
    # So D10 at top, D1 at bottom
    pivot_table = df.pivot_table(
        index='Size_Decile',
        columns='HBI_Quintile',
        values='Alpha_Pct',
        aggfunc='first'
    )
    
    # Sort by Size_Decile in descending order (D10 at top, D1 at bottom)
    pivot_table = pivot_table.sort_index(ascending=False)
    
    # Sort columns by HBI_Quintile (Q1 to Q5)
    pivot_table = pivot_table.sort_index(axis=1)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Format values as percentages with 2 decimal places
    formatted_table = pivot_table.applymap(lambda x: f'{x:.2f}%' if not pd.isna(x) else 'N/A')
    
    # Create row labels (D1 to D10)
    row_labels = [f'D{int(i)}' for i in pivot_table.index]
    
    # Create column labels (HBI Q1 to HBI Q5)
    col_labels = [f'HBI Q{int(i)}' for i in pivot_table.columns]
    
    # Create the table
    table = ax.table(
        cellText=formatted_table.values,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Get all cell positions to understand the table structure
    cell_keys = sorted(table._cells.keys())
    
    # Debug: Print table structure to understand indexing
    if len(cell_keys) > 0:
        max_row = max(k[0] for k in cell_keys)
        max_col = max(k[1] for k in cell_keys)
        print(f"Table structure: Max row={max_row}, Max col={max_col}")
        print(f"Expected: {len(row_labels)} data rows, {len(col_labels)} data columns")
        print(f"Sample cell keys: {cell_keys[:10]}")
    
    # In matplotlib tables with rowLabels and colLabels:
    # - Row 0: header row (colLabels), columns 0 to len(col_labels)-1 (or 1 to len(col_labels)?)
    # - Column 0: row labels column, rows 1 to len(row_labels)
    # - Data cells: rows 1 to len(row_labels), columns 1 to len(col_labels)
    
    # Style header row cells (row 0, columns 1 to len(col_labels))
    for j in range(len(col_labels)):
        col_idx = j + 1  # Column 0 is for row labels, so data starts at 1
        if (0, col_idx) in table._cells:
            cell = table[(0, col_idx)]
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
    
    # Style row label cells (column 0, rows 1 to len(row_labels))
    for i in range(len(row_labels)):
        row_idx = i + 1  # Row 0 is for headers, so data starts at 1
        if (row_idx, 0) in table._cells:
            cell = table[(row_idx, 0)]
            cell.set_facecolor('#2196F3')
            cell.set_text_props(weight='bold', color='white')
    
    # Data cells - no color coding (keep default white background)
    
    # Set title
    plt.title('Table 3: The HBI and Portfolio Returns: Size × HBI Sorts\n(Annualized Alpha %)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('test/four_factor_alpha_table.png', dpi=300, bbox_inches='tight')
    plt.savefig('test/four_factor_alpha_table.pdf', bbox_inches='tight')
    print("Table saved to test/four_factor_alpha_table.png and test/four_factor_alpha_table.pdf")
    
    # Also display it
    plt.show()
    
    return fig, ax, table

if __name__ == "__main__":
    create_alpha_table()

