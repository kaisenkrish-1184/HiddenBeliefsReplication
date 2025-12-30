# Constructs dataframe for SCQR

# Imports
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import time
import os
from sklearn.linear_model import LinearRegression
# Files
import Zero_Merge as zm
import One_MergeKeys as mk
import Two_MergeCRSPAndCleanDaily as mc
import Three_firmCharacteristicsFundQ as fc
import Four_VarianceAndBeta as vb
import Five_VSDandBeta as vsd
import Six_rigidDynamic as rd
import Seven_universeCreator as uc
import Eight_build_consideration_sets as cs
import Nine_filtering_consideration_sets as fs
import Ten_build_consideration_panel as bp
import Eleven_removeAllDelisted as rm
import Twelve_append_rigid_managers as am
import Thirteen_instrument_construction as ic
import Fourteen_cleaningMissingChars as cmc
import Fifteen_winsorize_data as wd
import Sixteen_control_variable_construction as cvc

FOLDER = 'replication'

# Step flags - Set to True to run, False to skip
RUN_FIRST_MERGEKEYS = True
RUN_SECOND_MERGEKEYS = True
RUN_CRSP_MERGE = True
RUN_FIRM_CHARACTERISTICS = True
RUN_VARIANCE_AND_BETA = True
RUN_VSD_AND_BETA = True
RUN_RIGID_DYNAMIC = True
UNIVERSE_CREATOR = True
BUILD_CONSIDERATION_SETS = True
FILTER_CONSIDERATION_SETS = True
BUILD_CONSIDERATION_PANEL = True
REMOVE_ALL_DELISTED = True
APPEND_RIGID_MANAGERS = True
INSTRUMENT_CONSTRUCTION = True
CLEAN_MISSING_CHARS = True
WINSORIZE_DATA = True
CONTROL_VARIABLE = True

# Step 0: First Merge Keys (PERMNO)
if RUN_FIRST_MERGEKEYS:
    result_df, summary = zm.merge_keys(f'{FOLDER}/13f_holdings.parquet', f'{FOLDER}/names.csv', FOLDER)
    print("Step 0 completed - PERMNO merged")

# Step 1: Second Merge Keys (GVKEY)
if RUN_SECOND_MERGEKEYS:
    # Ensure result_df is loaded
    if 'result_df' not in locals():
        result_df = pd.read_parquet(f'{FOLDER}/13F_1stMerge.parquet')
    result_df, summary = mk.merge_keys(result_df, f'{FOLDER}/GVKEY_link.csv', FOLDER)
    print("Step 1 completed - GVKEY merged")

# Step 2: CRSP Merge
if RUN_CRSP_MERGE:
    # Ensure result_df is loaded
    if 'result_df' not in locals():
        result_df = pd.read_parquet(f'{FOLDER}/13F_MergeKeys.parquet')
    result_df, dailyReturns, summary = mc.merge_crsp_and_clean_daily(result_df, f'{FOLDER}/crsp_monthly.csv', f'{FOLDER}/crsp_daily.csv', FOLDER)
    print("Step 2 completed - CRSP merge completed")

# Step 3: Firm Characteristics
if RUN_FIRM_CHARACTERISTICS:
    # Ensure result_df is loaded
    if 'result_df' not in locals():
        result_df = pd.read_parquet(f'{FOLDER}/13F_and_CRSPM.parquet')
    result_df, summary = fc.add_firm_characteristics(result_df, f'{FOLDER}/fundamentals_quarterly.parquet', FOLDER)
    print("Step 3 completed - Firm characteristics added")

# Step 4: Variance and Beta
if RUN_VARIANCE_AND_BETA:
    # Ensure result_df is loaded
    if 'result_df' not in locals():
        result_df = pd.read_parquet(f'{FOLDER}/Five_Characteristics.parquet')
    variance_beta_df, summary = vb.calculate_variance_and_beta(result_df, f'{FOLDER}/filteredDaily.csv', FOLDER)
    print("Step 4 completed - Variance and beta added")

# Step 5: VSD and Beta
if RUN_VSD_AND_BETA:
    # Ensure result_df is loaded
    if 'result_df' not in locals():
        result_df = pd.read_parquet(f'{FOLDER}/Five_Characteristics.parquet')
    result_df, summary = vsd.merge_and_calculate_vsd(result_df, f'{FOLDER}/variance_and_beta.csv', FOLDER)
    print("Step 5 completed - VSD and beta added")

# Step 6: Rigid Dynamic Classification
if RUN_RIGID_DYNAMIC:
    # Ensure result_df is loaded
    if 'result_df' not in locals():
        result_df = pd.read_parquet(f'{FOLDER}/LHSandSixCharacteristics.parquet')
    result_df, summary = rd.classify_manager_rigidity(result_df, FOLDER)
    print("Step 6 completed - Rigid dynamic added")

# Step Universe Creator
if UNIVERSE_CREATOR:
    universe = uc.merge_keys(f'{FOLDER}/crsp_monthly.csv', f'{FOLDER}/GVKEY_link.csv', f'{FOLDER}/Universe')
    universe, dailyReturns = uc.filter_crsp_and_clean_daily(universe, f'{FOLDER}/crsp_daily.csv', f'{FOLDER}/Universe')
    universe, summary = fc.add_firm_characteristics(universe, f'{FOLDER}/fundamentals_quarterly.parquet', f'{FOLDER}/Universe')
    variance_beta_df, summary = vb.calculate_variance_and_beta(universe, f'{FOLDER}/Universe/CRSP_daily_cleaned.csv', f'{FOLDER}/Universe')
    universe = uc.merge_and_calculate_vsd(universe, f'{FOLDER}/Universe/variance_and_beta.csv', f'{FOLDER}/Universe')
    print("Universe Creator completed - Universe created")

# Step 7: Build Consideration Sets
if BUILD_CONSIDERATION_SETS:
    # Ensure result_df and universe are loaded
    if 'result_df' not in locals():
        result_df = pd.read_parquet(f'{FOLDER}/ManagersSortedByRigidity.parquet')
    if 'universe' not in locals():
        universe = pd.read_parquet(f'{FOLDER}/Universe/Universe.parquet')
    consideration_sets = cs.main(result_df, universe, FOLDER)
    print("Step 7 completed - Consideration sets built")

# Step 8: Filter Consideration Sets
if FILTER_CONSIDERATION_SETS:
    # Ensure consideration_sets is loaded
    if 'consideration_sets' not in locals():
        consideration_sets = pd.read_csv(f'{FOLDER}/dynamic_manager_consideration_sets.csv')
    consideration_sets = fs.main(consideration_sets, FOLDER)
    print("Step 8 completed - Consideration sets filtered")

# Step 9: Build Consideration Panel
if BUILD_CONSIDERATION_PANEL:
    # Ensure all required variables are loaded
    if 'consideration_sets' not in locals():
        consideration_sets = pd.read_csv(f'{FOLDER}/filtered_consideration_sets_25plus_holdings.csv')
    if 'result_df' not in locals():
        result_df = pd.read_parquet(f'{FOLDER}/ManagersSortedByRigidity.parquet')
    if 'universe' not in locals():
        universe = pd.read_parquet(f'{FOLDER}/Universe/Universe.parquet')
    result_df = bp.main(consideration_sets, result_df, universe, FOLDER)
    print("Step 9 completed - Consideration panel built")

# Step 10: Remove All Delisted
if REMOVE_ALL_DELISTED:
    # Ensure result_df is loaded
    if 'result_df' not in locals():
        result_df = pd.read_parquet(f'{FOLDER}/consideration_sets_dataframe.parquet')
    result_df = rm.main(result_df, FOLDER)
    print("Step 10 completed - Delisted stocks removed")

# Step 11: Append Rigid Managers
if APPEND_RIGID_MANAGERS:
    # Ensure result_df is loaded
    if 'result_df' not in locals():
        result_df = pd.read_parquet(f'{FOLDER}/consideration_sets_dataframe_clean.parquet')
    # Load classification_df
    classification_df = pd.read_parquet(f'{FOLDER}/ManagersSortedByRigidity.parquet')
    instrument_construction_df = am.main(classification_df, result_df, FOLDER)
    print("Step 11 completed - Rigid managers appended")

# Step 12: Instrument Construction
if INSTRUMENT_CONSTRUCTION:
    # Ensure instrument_construction_df is loaded
    if 'instrument_construction_df' not in locals():
        instrument_construction_df = pd.read_parquet(f'{FOLDER}/instrument_construction_DF.parquet')
    instruments = ic.main(instrument_construction_df, FOLDER)
    print("Step 12 completed - Instruments constructed")

# Step 13: Clean Missing Characteristics
if CLEAN_MISSING_CHARS:
    # Ensure result_df and instruments are loaded
    if 'result_df' not in locals():
        result_df = pd.read_parquet(f'{FOLDER}/consideration_sets_dataframe_clean.parquet')
    if 'instruments' not in locals():
        instruments = pd.read_csv(f'{FOLDER}/instruments_all_quarters.csv')
    result_df = cmc.main(result_df, instruments, FOLDER)
    print("Step 13 completed - Missing characteristics cleaned")

# Step 14: Winsorize Data
if WINSORIZE_DATA:
    # Ensure result_df is loaded
    if 'result_df' not in locals():
        result_df = pd.read_parquet(f'{FOLDER}/first_stage_regression_dataframe.parquet')
    result_df = wd.winsorize_data(result_df, FOLDER)
    wd.compare_regression_coefficients()
    print("Step 14 completed - Data winsorized")

# Step 15: Control Variable Construction
if CONTROL_VARIABLE:
    # Ensure result_df is loaded
    if 'result_df' not in locals():
        result_df = pd.read_parquet(f'{FOLDER}/first_stage_regression_dataframe_winsorized.parquet')
    result_df = cvc.main(result_df, FOLDER)
    print("Step 15 completed - Control variables constructed")

print("\n=== Pipeline completed ===")
