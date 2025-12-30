import pandas as pd
import numpy as np
import ast
import json
import gc
import time
from pathlib import Path

# # ----------------- CONFIG -----------------
# CONS_CSV  = "filtered_consideration_sets_25plus_holdings.csv"
# FINAL_CSV = "finalFileBeforeZeros.csv"     # ~10GB
# UNI_CSV   = "stockUniverse.csv"            # ~0.5GB
# OUT_FILE  = "consideration_sets_dataframe.parquet"  # change to .csv if desired
# # ------------------------------------------

def _safe_parse_list(x):
    """Fast parsing of consideration_set - optimized for common cases"""
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        # Fast path for empty strings
        x_stripped = x.strip()
        if not x_stripped or x_stripped == '[]':
            return []
        # Try json.loads first (faster for large lists), fallback to ast.literal_eval
        try:
            return json.loads(x_stripped)
        except (ValueError, json.JSONDecodeError):
            try:
                return ast.literal_eval(x_stripped)
            except (ValueError, SyntaxError):
                return []
    return []

def load_inputs(consideration_sets, df, universe):
    print("Loading input CSVs...")
    # Return fields can have letter codes (e.g., 'B','C'); accept them as NA
    na_codes = ["B", "C"]

    cons = consideration_sets

    final = df

    uni = universe

    print(f"consideration_sets: {cons.shape}, finalFile: {final.shape}, stockUniverse: {uni.shape}")

    # Key dtypes for fast joins
    for df, has_mgr in [(final, True), (uni, False)]: 
        df["LPERMNO"] = df["LPERMNO"].astype("int64", copy=False)
        df["rdate"] = pd.to_datetime(df["rdate"])  # Ensure datetime type
        if has_mgr and "mgrno" in df.columns:
            df["mgrno"] = df["mgrno"].astype("int64", copy=False)

    cons["mgrno"] = cons["mgrno"].astype("int64", copy=False)
    cons["rdate"] = pd.to_datetime(cons["rdate"])  # Ensure datetime type for consideration sets

    return cons, final, uni

def build_panel(cons, final, uni, chunk_size=1000, FOLDER="test"):
    """
    Build panel by exploding consideration sets in chunks to reduce memory usage.
    Optimized for memory and speed with incremental disk writes.
    
    Args:
        cons: Consideration sets DataFrame
        final: Final DataFrame for merging
        uni: Universe DataFrame for fallback merging
        chunk_size: Number of consideration set rows to process at once (default: 1000)
        FOLDER: Folder path for temporary files
    """
    print("Parsing consideration sets...")
    # Check if already parsed (if first value is already a list, skip parsing)
    sample_val = cons["consideration_set"].iloc[0] if len(cons) > 0 else None
    print(f"  Sample value type: {type(sample_val)}")
    if isinstance(sample_val, list):
        print("  Consideration sets already parsed (lists), skipping parsing step")
    else:
        # Check sample string length to estimate parsing time
        if isinstance(sample_val, str):
            sample_len = len(sample_val)
            print(f"  Sample string length: {sample_len:,} characters")
            if sample_len > 10000:
                print(f"  Warning: Large strings detected - parsing may take a while...")
        
        print(f"  Parsing {len(cons):,} consideration sets (this may take a moment)...")
        # Use smaller chunks for very large strings
        chunk_parse_size = 1000 if isinstance(sample_val, str) and len(str(sample_val)) > 5000 else 5000
        total_chunks = (len(cons) + chunk_parse_size - 1) // chunk_parse_size
        print(f"  Using chunk size: {chunk_parse_size} ({total_chunks} total chunks)")
        parsed_lists = []
        
        start_time = time.time()
        
        for i in range(0, len(cons), chunk_parse_size):
            chunk_num = i // chunk_parse_size + 1
            chunk_start = time.time()
            
            # More frequent progress updates
            if chunk_num % 2 == 0 or chunk_num == 1:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(cons) - i) / rate if rate > 0 else 0
                print(f"    Parsed {i:,}/{len(cons):,} rows ({chunk_num}/{total_chunks} chunks) - "
                      f"Rate: {rate:.0f} rows/sec, ETA: {eta/60:.1f} min...", flush=True)
            
            chunk = cons["consideration_set"].iloc[i:i+chunk_parse_size]
            # Use map instead of apply for slightly better performance
            parsed_chunk = chunk.map(_safe_parse_list)
            parsed_lists.append(parsed_chunk)
            
            # Periodic garbage collection every 10 chunks
            if chunk_num % 10 == 0:
                gc.collect()
        
        print(f"    Concatenating {len(parsed_lists)} parsed chunks...")
        cons["consideration_set"] = pd.concat(parsed_lists, ignore_index=True)
        del parsed_lists
        gc.collect()
        total_time = time.time() - start_time
        print(f"  Parsing completed in {total_time/60:.1f} minutes")
    
    # Preserve final's column order for output
    final_cols = list(final.columns)
    key_final = ["LPERMNO", "mgrno", "rdate"]
    key_uni = ["LPERMNO", "rdate"]
    
    # Create indexed DataFrames for efficient merging
    print("Indexing DataFrames for efficient merging...")
    final_indexed = final.set_index(key_final).sort_index()
    uni_indexed = uni.set_index(key_uni).sort_index()
    
    # Pre-compute which columns need coalescing (only those that exist in both)
    uni_cols_set = set(uni.columns)
    cols_to_coalesce = [c for c in final_cols if c in uni_cols_set]
    
    # Free original DataFrames to save memory
    del final, uni
    gc.collect()
    
    # Setup temporary directory for incremental writes
    temp_dir = Path(FOLDER) / 'temp_panel_chunks'
    temp_dir.mkdir(parents=True, exist_ok=True)
    chunk_files = []
    
    # Process in chunks
    print(f"Processing {len(cons):,} consideration sets in chunks of {chunk_size}...")
    total_exploded = 0
    num_chunks = (len(cons) + chunk_size - 1) // chunk_size
    
    overall_start_time = time.time()
    
    for i in range(0, len(cons), chunk_size):
        chunk_num = i // chunk_size + 1
        chunk_iter_start = time.time()
        
        # Progress reporting - show every chunk
        elapsed = time.time() - overall_start_time
        rate = i / elapsed if elapsed > 0 else 0
        eta = (len(cons) - i) / rate if rate > 0 else 0
        print(f"  Processing chunk {chunk_num}/{num_chunks} ({i:,}/{len(cons):,} rows) - "
              f"Rate: {rate:.0f} rows/sec, ETA: {eta/60:.1f} min...", flush=True)
        
        # Get chunk of consideration sets (use iloc without copy for speed)
        cons_chunk = cons.iloc[i:i+chunk_size]
        
        # Explode this chunk
        explode_start = time.time()
        exploded = cons_chunk.explode("consideration_set", ignore_index=True)
        if len(exploded) == 0:
            del cons_chunk
            continue
        
        # Rename and convert in one step
        exploded = exploded.rename(columns={"consideration_set": "LPERMNO"})
        exploded["LPERMNO"] = pd.to_numeric(exploded["LPERMNO"], errors='coerce', downcast='integer')
        exploded = exploded[exploded["LPERMNO"].notna()]  # Remove invalid LPERMNOs
        
        if len(exploded) == 0:
            del cons_chunk, exploded
            continue
        
        total_exploded += len(exploded)
        explode_time = time.time() - explode_start
        
        # Progress for every chunk
        print(f"    Chunk {chunk_num}: Exploded to {len(exploded):,} rows in {explode_time:.2f}s", flush=True)
        
        # Set index on exploded for merge (no copy needed)
        merge_start = time.time()
        exploded_indexed = exploded.set_index(key_final)
        del exploded
        
        # 1) Preferred merge to final on (LPERMNO, mgrno, rdate)
        # Use join instead of merge for slightly better performance on indexed DataFrames
        m = exploded_indexed.join(final_indexed, how='left', rsuffix='_drop')
        del exploded_indexed
        
        # Drop duplicate columns efficiently
        drop_cols = [c for c in m.columns if c.endswith('_drop')]
        if drop_cols:
            m = m.drop(columns=drop_cols)
        
        # Reset index to get keys back as columns
        m = m.reset_index()
        
        # 2) Fallback merge to stockUniverse on (LPERMNO, rdate)
        m_uni_key = m.set_index(key_uni)
        m_uni = m_uni_key.join(uni_indexed, how='left', rsuffix='_su')
        del m_uni_key, m
        
        # Reset index
        m_uni = m_uni.reset_index()
        merge_time = time.time() - merge_start
        
        # Vectorized coalescing: fill NAs in final columns from _su (stockUniverse) where available
        coalesce_start = time.time()
        for c in cols_to_coalesce:
            c_su = c + "_su"
            if c_su in m_uni.columns:
                # Vectorized fillna is faster than where()
                mask = m_uni[c].isna()
                if mask.any():
                    m_uni.loc[mask, c] = m_uni.loc[mask, c_su]
                # Drop the _su column to save memory
                m_uni = m_uni.drop(columns=[c_su])
        
        # Ensure all final columns exist (vectorized)
        missing_cols = [c for c in final_cols if c not in m_uni.columns]
        if missing_cols:
            for c in missing_cols:
                m_uni[c] = np.nan
        
        # Keep only final's columns in original order
        chunk_result = m_uni[final_cols]
        del m_uni
        coalesce_time = time.time() - coalesce_start
        
        # Write chunk to disk immediately instead of keeping in memory
        write_start = time.time()
        chunk_file = temp_dir / f'chunk_{chunk_num:06d}.parquet'
        chunk_result.to_parquet(chunk_file, index=False, engine='pyarrow', compression='snappy')
        chunk_files.append(chunk_file)
        del chunk_result, cons_chunk
        write_time = time.time() - write_start
        
        chunk_iter_time = time.time() - chunk_iter_start
        
        # Detailed timing for every chunk
        print(f"    Chunk {chunk_num} timing - Explode: {explode_time:.2f}s, "
              f"Merge: {merge_time:.2f}s, Coalesce: {coalesce_time:.2f}s, "
              f"Write: {write_time:.2f}s, Total: {chunk_iter_time:.2f}s", flush=True)
        
        # Periodic garbage collection (every 20 chunks)
        if chunk_num % 20 == 0:
            gc.collect()
            print(f"    Garbage collection at chunk {chunk_num}...", flush=True)
        
        # Progress summary every 10 chunks
        if chunk_num % 10 == 0:
            avg_time = elapsed / chunk_num if chunk_num > 0 else 0
            remaining_chunks = num_chunks - chunk_num
            est_remaining = avg_time * remaining_chunks
            print(f"    Progress: {chunk_num}/{num_chunks} chunks ({chunk_num/num_chunks*100:.1f}%) - "
                  f"Avg: {avg_time:.2f}s/chunk, Est. remaining: {est_remaining/60:.1f} min", flush=True)
    
    total_chunk_time = time.time() - overall_start_time
    print(f"\nTotal exploded rows: {total_exploded:,}")
    print(f"Chunk processing completed in {total_chunk_time/60:.1f} minutes")
    print(f"Combining {len(chunk_files)} chunks from disk...")
    
    # Combine chunks in batches to avoid memory spikes
    batch_size = 50
    combined_batches = []
    combine_start_time = time.time()
    num_batches = (len(chunk_files) + batch_size - 1) // batch_size
    
    for batch_num, batch_start in enumerate(range(0, len(chunk_files), batch_size), 1):
        if batch_num % 5 == 0 or batch_num == 1:
            print(f"  Combining batch {batch_num}/{num_batches} ({batch_start:,}/{len(chunk_files)} chunks)...", flush=True)
        
        batch_files = chunk_files[batch_start:batch_start + batch_size]
        batch_dfs = [pd.read_parquet(f) for f in batch_files]
        batch_combined = pd.concat(batch_dfs, ignore_index=True)
        combined_batches.append(batch_combined)
        del batch_dfs, batch_combined
        gc.collect()
    
    combine_time = time.time() - combine_start_time
    print(f"  Batch combination completed in {combine_time:.1f} seconds")
    
    # Final combination
    if len(combined_batches) > 1:
        out = pd.concat(combined_batches, ignore_index=True)
        del combined_batches
    else:
        out = combined_batches[0]
        del combined_batches
    
    # Clean up temporary files
    for chunk_file in chunk_files:
        chunk_file.unlink()
    temp_dir.rmdir()
    
    gc.collect()
    
    return out

def main(consideration_sets, df, universe, FOLDER):
    print("=== Fast consideration panel build ===")
    cons, final, uni = load_inputs(consideration_sets, df, universe)
    out = build_panel(cons, final, uni, FOLDER=FOLDER)

    print(f"Final shape: {out.shape}")

    out.to_parquet(f'{FOLDER}/consideration_sets_dataframe.parquet', index=False)

    # Summary
    print("\n=== Summary ===")
    print("Unique managers:", out["mgrno"].nunique())
    print("Unique dates:   ", out["rdate"].nunique())
    print("Unique stocks:  ", out["LPERMNO"].nunique())

    return out
