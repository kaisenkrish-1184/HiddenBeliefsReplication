#!/usr/bin/env python3
"""
SCQR Demand Estimation — Section 3.3 compliant (Grids, J0/J1 selection, control function)
-----------------------------------------------------------------------------------------
Implements the three-step Sequential Censored Quantile Regression (SCQR) exactly as laid out:
  1) Build a descending grid of quantiles from τ_start to τ_end.
  2) Initialize at the highest τ using standard quantile regression on the full sample.
  3) For each next τ:
       J0 = {i : z_i' β_prev > Q_{0.01}( {z_r' β_prev | z_r' β_prev > 0} ) }
       β0(τ_next) = QR on J0 at τ_next
       J1 = {i : z_i' β0(τ_next) > Q_{0.005} ( {z_r' β0(τ_next) | z_r' β0(τ_next) > 0} ) }
       β(τ_next)  = QR on J1 at τ_next
  4) Stop at τ = 0.50 (median). If J0 or J1 empty at any point => fails to converge for that manager-quarter.

Model:
    y = max(0, z'β + u),  z = [ 1, Log_Market_Cap, beta, Investment, Operating_Profitability,
                                 Dividend_to_Book, Log_Book_Value, control_variable ]

Inputs:
  - regression_ready_dataframe.parquet
    Required cols:
      ['mgrno','LPERMNO','rdate','Volatility_Scaled_Demand',
       'Log_Market_Cap','beta','Investment','Operating_Profitability',
       'Dividend_to_Book','Log_Book_Value','control_variable','AUM']

Outputs:
  - demand_estimation_results.csv
  - demand_estimation_summary.csv
  - institution_demand_parameters.parquet
"""

from __future__ import annotations
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, as_completed

FOLDER = 'gravesReplication'
DEFAULT_INPUT = f"{FOLDER}/regression_ready_dataframe.parquet"
WORKERS = 8
TAU_START = 0.99
TAU_END   = 0.50
GRID_MIN_STEPS = 40
EPS_SMOOTH = 1e-3
MAXITER_QR = 200
FTOL = 1e-9

# Manager-level guards
MIN_HOLDINGS_POS = 25       # at least this many positive VSD holdings
MIN_PANEL_ROWS    = 1       # total rows for that manager-quarter (consideration set)
MIN_UNCENSORED    = 1       # lower bound (we still rely on J0/J1 emptiness to fail)

# Regressor order (constant first)
VAR_ORDER = [
    'constant',
    'Log_Market_Cap',
    'beta',
    'Investment',
    'Operating_Profitability',
    'Dividend_to_Book',
    'Log_Book_Value',
    'control_variable'
]

REQ_COLS = [
    'mgrno','LPERMNO','rdate','Volatility_Scaled_Demand','AUM',
    'Log_Market_Cap','beta','Investment','Operating_Profitability',
    'Dividend_to_Book','Log_Book_Value','control_variable'
]

# ---------------------------
# Smoothed check loss (ε-Huberized)
# ---------------------------

def _check_loss(u: np.ndarray, tau: float, eps: float = EPS_SMOOTH) -> np.ndarray:
    absu = np.abs(u)
    # Linear parts (standard check loss)
    lin = u * (tau - (u < 0).astype(float))
    # Quadratic smoothing around zero
    quad = 0.5/eps * (u**2) + (tau - 0.5)*u + 0.5*eps*(0.5 - np.abs(tau - 0.5))
    return np.where(absu > eps, lin, quad)

def _check_grad(u: np.ndarray, tau: float, eps: float = EPS_SMOOTH) -> np.ndarray:
    g_lin  = tau - (u < 0).astype(float)
    g_quad = (u / eps) + (tau - 0.5)
    return np.where(np.abs(u) > eps, g_lin, g_quad)

def quantile_regression_smoothed(y: np.ndarray, Z: np.ndarray, tau: float,
                                 beta0: np.ndarray | None = None,
                                 maxiter: int = MAXITER_QR, ftol: float = FTOL):
    n, k = Z.shape
    if beta0 is None:
        try:
            beta0 = np.linalg.lstsq(Z, y, rcond=None)[0]
        except Exception:
            beta0 = np.zeros(k)

    def obj(b):
        r = y - Z @ b
        return _check_loss(r, tau).sum()

    def grad(b):
        r = y - Z @ b
        return -(_check_grad(r, tau) @ Z)

    res = minimize(obj, beta0, jac=grad, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": ftol})
    return res.x, res.success

# ---------------------------
# Section 3.3: SCQR core (J0/J1 selection)
# ---------------------------

def _positive_pred_quantile(Z: np.ndarray, b: np.ndarray, q: float) -> float:
    pred = Z @ b
    pos  = pred[pred > 0]
    if pos.size == 0:
        return np.inf  # forces empty set upstream
    return float(np.quantile(pos, q))

def scqr_section33(y: np.ndarray, Z: np.ndarray,
                   tau_start: float = TAU_START,
                   tau_end: float   = TAU_END,
                   grid_min_steps: int = GRID_MIN_STEPS,
                   maxiter_qr: int = MAXITER_QR):
    """
    Strict Section 3.3 implementation:
      - Descending τ grid
      - J0 at 1% positive-prediction quantile under β(τ_prev)
      - J1 at 0.5% positive-prediction quantile under β0(τ_next)
      - Fail if J0 or J1 empty at any step
    Returns (beta_hat, "Success") or (None, reason)
    """
    n = len(y)
    # Build τ grid
    L = max(grid_min_steps, int(np.sqrt(max(n, 1))))
    taus = np.linspace(tau_start, tau_end, L)

    # Initialize at highest τ on full sample
    beta, ok = quantile_regression_smoothed(y, Z, taus[0])
    if not ok:
        return None, "Initial QR failed"

    for t in taus[1:]:
        # ---- Step 1: J0 using β_prev and 1% positive-prediction quantile
        q01 = _positive_pred_quantile(Z, beta, 0.01)
        J0  = (Z @ beta) > q01
        if J0.sum() == 0:
            return None, f"Empty J0 at tau={t:.3f}"

        # Initial estimator on J0
        b0, ok = quantile_regression_smoothed(y[J0], Z[J0], t, beta0=beta, maxiter=maxiter_qr)
        if not ok:
            return None, f"QR(J0) failed at tau={t:.3f}"

        # ---- Step 2: J1 using b0 and 0.5% positive-prediction quantile
        q005 = _positive_pred_quantile(Z, b0, 0.005)
        J1   = (Z @ b0) > q005
        if J1.sum() == 0:
            return None, f"Empty J1 at tau={t:.3f}"

        # Final estimator on J1
        beta, ok = quantile_regression_smoothed(y[J1], Z[J1], t, beta0=b0, maxiter=maxiter_qr)
        if not ok:
            return None, f"QR(J1) failed at tau={t:.3f}"

    return beta, "Success"

# ---------------------------
# Per manager-quarter prep
# ---------------------------

def build_y_Z(df_mq: pd.DataFrame):
    """
    y = VSD with zeros for non-holdings
    Z = [1, regressors in VAR_ORDER[1:]]
    """
    # guard on positive holdings count (uncensored candidates)
    pos_hold = (df_mq['Volatility_Scaled_Demand'].fillna(0.0) > 0).sum()
    if pos_hold < MIN_HOLDINGS_POS:
        return None, None, f"Only {pos_hold} positive holdings (<{MIN_HOLDINGS_POS})"

    y = df_mq['Volatility_Scaled_Demand'].fillna(0.0).to_numpy()
    X = df_mq[VAR_ORDER[1:]].to_numpy()
    Z = np.column_stack([np.ones(len(X)), X])

    if not np.isfinite(Z).all():
        return None, None, "Non-finite regressors"

    return y, Z, "OK"

def estimate_one_manager(args):
    mgrno, rdate, df_mq, scqr_kwargs = args

    if len(df_mq) < MIN_PANEL_ROWS:
        return dict(mgrno=mgrno, rdate=rdate, status="failed",
                    reason=f"Panel too small ({len(df_mq)})",
                    n_obs=len(df_mq), n_holdings=int((df_mq['Volatility_Scaled_Demand'].fillna(0.0) > 0).sum()),
                    estimation_time=0.0)

    y, Z, msg = build_y_Z(df_mq)
    if y is None:
        return dict(mgrno=mgrno, rdate=rdate, status="failed", reason=msg,
                    n_obs=len(df_mq), n_holdings=int((df_mq['Volatility_Scaled_Demand'].fillna(0.0) > 0).sum()),
                    estimation_time=0.0)

    t0 = time.time()
    beta, status = scqr_section33(y, Z, **scqr_kwargs)
    dt = time.time() - t0

    if beta is None:
        return dict(mgrno=mgrno, rdate=rdate, status="failed", reason=status,
                    n_obs=len(y), n_holdings=int((y > 0).sum()), estimation_time=dt)

    out = dict(mgrno=mgrno, rdate=rdate, status="success", reason=status,
               n_obs=len(y), n_holdings=int((y > 0).sum()), estimation_time=dt)
    for i, nm in enumerate(VAR_ORDER):
        out[f"beta_{nm}"] = float(beta[i])
    return out

# ---------------------------
# Quarter orchestration
# ---------------------------

def process_quarter(df_q: pd.DataFrame, qdate: pd.Timestamp, workers: int, scqr_kwargs: dict):
    print(f"\nProcessing quarter {qdate.date()}  (rows: {len(df_q):,})")

    mgr_sizes = df_q.groupby('mgrno').size()
    mgrs = mgr_sizes.index.to_numpy()

    tasks = []
    for m in mgrs:
        tasks.append((int(m), qdate, df_q[df_q['mgrno'] == m], scqr_kwargs))

    results = []
    if not tasks:
        print("  No managers found.")
        return results

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(estimate_one_manager, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            res = f.result()
            results.append(res)
            if i % 50 == 0:
                ok = sum(1 for r in results if r['status'] == 'success')
                print(f"    Progress: {i}/{len(tasks)}  (success so far: {ok})")

    ok = sum(1 for r in results if r['status'] == 'success')
    print(f"  Quarter done. Success: {ok}/{len(results)}")
    return results

def process_all(df: pd.DataFrame, workers: int, scqr_kwargs: dict):
    print("\nProcessing all quarters…")
    quarters = sorted(df['rdate'].unique())
    print(f"Total quarters: {len(quarters)}")

    all_res = []
    for i, q in enumerate(quarters, 1):
        print(f"\n--- Quarter {i}/{len(quarters)}: {pd.Timestamp(q).date()} ---")
        res_q = process_quarter(df[df['rdate'] == q], pd.Timestamp(q), workers, scqr_kwargs)
        all_res.extend(res_q)
    print("✓ Finished all quarters.")
    return all_res

# ---------------------------
# Summaries + Save
# ---------------------------

def summarize_and_save(results: list[dict]):
    if not results:
        raise ValueError("No results to summarize.")

    df = pd.DataFrame(results)
    total = len(df)
    succ = int((df['status'] == 'success').sum())
    rate = succ / total if total else 0.0

    print("\nOverall:")
    print(f"  Attempts:  {total:,}")
    print(f"  Successes: {succ:,}  ({rate:.1%})")

    byq = df.groupby('rdate').agg(
        successful=('status', lambda s: int((s == 'success').sum())),
        total=('status', 'count'),
        avg_obs=('n_obs', 'mean'),
        avg_time=('estimation_time', 'mean')
    )
    byq['success_rate'] = byq['successful'] / byq['total']

    df.to_csv(f'{FOLDER}/demand_estimation_results.csv', index=False)
    byq.to_csv(f'{FOLDER}/demand_estimation_summary.csv', index=False)

    succ_df = df[df['status'] == 'success'].copy()
    if not succ_df.empty:
        succ_df.to_parquet(f'{FOLDER}/institution_demand_parameters.parquet', index=False)
        print(f"  ✓ Saved {FOLDER}/institution_demand_parameters.parquet")
    print(f"  ✓ Saved {FOLDER}/demand_estimation_results.csv and {FOLDER}/demand_estimation_summary.csv")

# ---------------------------
# Data load
# ---------------------------

def load_data(path: str) -> pd.DataFrame:
    print("Loading regression-ready data…")
    df = pd.read_parquet(path)

    miss = [c for c in REQ_COLS if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    df['rdate'] = pd.to_datetime(df['rdate'])

    # Clean regressors (y can be NaN; we turn it to 0 later)
    colsX = VAR_ORDER[1:]
    df[colsX] = df[colsX].replace([np.inf, -np.inf], np.nan)
    bad = df[colsX].isna().any(axis=1)
    if bad.any():
        nbad = int(bad.sum())
        print(f"  ⚠ Dropping {nbad:,} rows with missing regressors")
        df = df.loc[~bad].copy()

    print(f"✓ Loaded: {df.shape}")
    print(f"Date range: {df['rdate'].min().date()} → {df['rdate'].max().date()}")
    print(f"Quarters:   {df['rdate'].nunique()}")
    print(f"Institutions: {df['mgrno'].nunique():,}")
    return df

# ---------------------------
# CLI / Main
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="SCQR (Section 3.3 compliant)")
    ap.add_argument('--input', default=DEFAULT_INPUT)
    ap.add_argument('--workers', type=int, default=WORKERS)
    ap.add_argument('--tau-start', type=float, default=TAU_START)
    ap.add_argument('--tau-end', type=float, default=TAU_END)
    ap.add_argument('--grid-min-steps', type=int, default=GRID_MIN_STEPS)
    ap.add_argument('--maxiter', type=int, default=MAXITER_QR)
    return ap.parse_args()

def main():
    args = parse_args()
    t0 = time.time()

    df = load_data(args.input)

    scqr_kwargs = dict(
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        grid_min_steps=args.grid_min_steps,
        maxiter_qr=args.maxiter
    )

    results = process_all(df, workers=args.workers, scqr_kwargs=scqr_kwargs)
    summarize_and_save(results)

    dt = time.time() - t0
    print("\n=== Final Summary ===")
    print(f"Total time: {dt/60:.1f} min")
    print(f"Total institution-quarters: {len(results):,}")
    print(f"Successful estimations: {sum(r['status']=='success' for r in results):,}")

if __name__ == '__main__':
    main()
