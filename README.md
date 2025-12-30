# Replication of Daniel Graves JMP  
**“What Lies Beneath Zero: Censoring, Demand Estimation, and Hidden Beliefs”**

This repository contains code to replicate empirical results of Daniel Graves’ JMP, including the estimation of institutional hidden beliefs and the construction of a trading strategy based on the Hidden Beliefs Index (HBI).

The paper can be accessed [here](https://yale.app.box.com/s/vlmkafxglm89f4cwra9ecav9nniyukyw).

### Note on WRDS formats
This code is written using **legacy WRDS file formats**.  
Users working with **post-2024 WRDS extracts** may need to make minor adjustments to variable names and file structure.

---

## Required Data Files

- **Thomson Reuters 13F Holdings**
  - `mgrno`, `cusip`, `shares`, `prc`, `fdate`, `rdate`

- **CRSP Monthly Stock File**
  - `PERMNO`, `SHRCD`, `EXCHCD`, `CUSIP`, `PRC`, `VOL`, `RET`, `SHROUT`, `RETX`

- **CRSP Daily Stock File**
  - `PERMNO`, `PRC`, `VOL`, `RET`, `SHROUT`, `RETX`, `vwretd`, `ewretd`

- **Compustat Fundamentals Quarterly**
  - `gvkey`, `datadate`, `conm`, `naics`, `atq`, `seqq`, `ceqq`, `cheq`,
    `cogsq`, `cshoq`, `dlcq`, `dlttq`, `niq`, `revtq`, `xintq`, `xsgaq`,
    `dvpspq`, `dvpsxq`, `mkvaltq`, `prclq`

- **Fama–French 5 Factors + Momentum**

- **CRSP Stock Names**

- **GVKEY–PERMNO–CUSIP Linking File**

---

## Estimating Hidden Beliefs

1. The code was tested using **Google Cloud Compute Engine** due to the scale of the data.
2. After downloading the required data, running `mainframe.py` from the replication root builds the full panel needed for **Sequential Censored Quantile Regression (SCQR)**.
   - Runtime depends heavily on sample size and compute resources.
3. Running `scqr.py` estimates institution-level demand parameters under short-sale constraints.
4. Running `fourFactorAlpha.py`:
   - Constructs **HBI-sorted portfolios**
   - Forms **5 HBI quintiles × 10 size deciles (50 equal-weighted portfolios)**
   - Computes **annualized four-factor alphas**
5. `Table.py` generates a formatted matplotlib table summarizing the final results.

Hidden-belief portfolios are constructed following Graves (2023), using **returns averaged over the previous three quarters**.

---

## Results (2022–2024 Sample)

The `example/` folder contains intermediate outputs from a test replication estimating HBI for **2022Q1–2024Q4**, using data from **2019–2024** to satisfy the historical holdings requirement.

---

## Notes

- This repository contains **code only**; no proprietary data are included.
- Results may differ slightly depending on WRDS vintage, sample filters, and computing environment.
