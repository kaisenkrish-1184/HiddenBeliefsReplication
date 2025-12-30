import pandas as pd

# -------- CONFIG --------
# IN_FILE  = "consideration_sets_dataframe.parquet"
# OUT_FILE = "consideration_sets_dataframe_clean.parquet"
# MIN_COLS = 10
# ------------------------

def main(df, FOLDER, MIN_COLS = 10):

    print(f"Original shape: {df.shape}")

    # Count non-null values per row
    mask = df.notna().sum(axis=1) >= MIN_COLS
    cleaned = df[mask].copy()

    print(f"After filtering (<{MIN_COLS} non-nulls removed): {cleaned.shape}")
    print(f"Dropped rows: {len(df) - len(cleaned)}")

    cleaned.to_parquet(f'{FOLDER}/consideration_sets_dataframe_clean.parquet', index=False)
    print(f"Saved cleaned data to consideration_sets_dataframe_clean.parquet")

    return cleaned