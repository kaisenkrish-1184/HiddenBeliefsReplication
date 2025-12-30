# Replication of Daniel Graves JMP: "What Lies Beneath Zero: Censoring, Demand Estimation, and Hidden Beliefs"
This respository contains code for replicating Graves' paper, creating a trading strategy based around Hidden Beliefs. The paper can be accessed [here]([url](https://yale.app.box.com/s/vlmkafxglm89f4cwra9ecav9nniyukyw)).
## Data

This replication uses CRSP, Compustat, and 13F data accessed via WRDS.
Due to licensing restrictions, data cannot be redistributed.

Users must download data directly from WRDS and place it in:

data/raw/

UPDATE: This code is based around legacy files from WRDS. The code will have to be adjusted accordingly to acommadate the new format if needed. 

These are the necessary data files
- Thomson-Reuters 13F Files
- CRSP Monthly Stock File
- CRSP Daily Stock File
- Fundamentals Quarterly File
- Fama-French 4 Factors + Momentum File
- Names File
- GVKEY-PERMNO-CUSIP Linking File


## Estimating Hidden Beliefs

1. This code was compiled via Google Cloud's compute engine.
2. Once the necessary data has been downloaded, running mainframe.py from the replication folder will produce the necessary dataframe to run sequential-censored quantile regression (SCQR). This code will take a considerable amount of time based on the amount of data ran on. 
3. Running scqr.py from the replication folder will produce the demand estimation results.
4. Running fourFactorAlpha.py from the replication folder will produce esimates of four-factor alpha for 5 HBI quintiles by 10 size deciles. Finally, run table.py to produce a matplotlib table to neatly display the final results. 

## Results for 2022-24 Data (obtained by using data from 2019-24, due to historical holdings condition)

1. The example folder contains necessary files from a test compilation that estimated HBI for 2022Q1-2024Q4. 
