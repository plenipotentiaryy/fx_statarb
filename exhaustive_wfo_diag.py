import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings
warnings.filterwarnings("ignore")

from config import DATA_DIR, TICKERS

# ── Exhaustive WFO Parameters ────────────────────────────────────────────────
WFO_TRAIN_MONTHS = 12
WFO_TEST_MONTHS  = 12
WFO_STEP_MONTHS  = 12
WFO_EXPANDING    = False # Rolling

_JOH_CRIT_IDX = {0.90: 0, 0.95: 1, 0.99: 2}

def test_johansen(df, t1, t2, crit_level=0.95):
    pc = df[[t1, t2]].dropna()
    if len(pc) < 200: return False, 0, 0, 0
    try:
        res = coint_johansen(pc, det_order=0, k_ar_diff=1)
        trace = float(res.lr1[0])
        crit = float(res.cvt[0, _JOH_CRIT_IDX[crit_level]])
        evec = res.evec[:, 0]
        beta = -evec[1] / evec[0]
        return trace > crit, round(trace, 2), round(crit, 2), round(beta, 4)
    except:
        return False, 0, 0, 0

def main():
    daily_path = DATA_DIR / "closes_daily.csv"
    if not daily_path.exists():
        print("Error: data/closes_daily.csv not found.")
        return

    daily = pd.read_csv(daily_path, index_col=0, parse_dates=True)
    if daily.index.tz is None:
        daily.index = pd.to_datetime(daily.index, utc=True)
    
    symbols = daily.columns.tolist()
    all_pairs = list(combinations(sorted(symbols), 2))
    
    print(f"Starting exhaustive WFO diagnostic for {len(all_pairs)} pairs...")
    print(f"Logic: Train={WFO_TRAIN_MONTHS}m, Test={WFO_TEST_MONTHS}m, Step={WFO_STEP_MONTHS}m")
    
    # Generate windows
    anchor = daily.index[0].date()
    end    = daily.index[-1].date()
    windows = []
    cur = anchor
    while True:
        tr_s = cur
        tr_e = tr_s + relativedelta(months=WFO_TRAIN_MONTHS)
        te_s = tr_e
        te_e = te_s + relativedelta(months=WFO_TEST_MONTHS)
        if te_e > end: break
        windows.append((tr_s, tr_e, te_s, te_e))
        cur = cur + relativedelta(months=WFO_STEP_MONTHS)

    print(f"Total windows: {len(windows)}")
    
    results = []
    
    # To speed up, we'll process window by window and keep track of pairs
    for w_idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        print(f"  Processing window {w_idx+1}/{len(windows)}: {tr_s} -> {te_e} ...")
        
        tr_s_ts = pd.Timestamp(tr_s, tz="UTC")
        tr_e_ts = pd.Timestamp(tr_e, tz="UTC")
        te_s_ts = pd.Timestamp(te_s, tz="UTC")
        te_e_ts = pd.Timestamp(te_e, tz="UTC")
        
        df_train = daily[(daily.index >= tr_s_ts) & (daily.index < tr_e_ts)]
        df_test  = daily[(daily.index >= te_s_ts) & (daily.index < te_e_ts)]
        
        # Log returns for correlation
        log_ret_tr = np.log(df_train / df_train.shift(1)).dropna(axis=1, how='all')
        corr_matrix_tr = log_ret_tr.corr()
        
        # For each pair
        for t1, t2 in all_pairs:
            if t1 not in df_train.columns or t2 not in df_train.columns: continue
            if t1 not in df_test.columns or t2 not in df_test.columns: continue
            
            # 1. Train Stats
            corr_tr = corr_matrix_tr.loc[t1, t2] if t1 in corr_matrix_tr.columns and t2 in corr_matrix_tr.columns else np.nan
            if np.isnan(corr_tr): continue
            
            is_coint_tr, trace_tr, crit_tr, beta_tr = test_johansen(df_train, t1, t2)
            
            # Only record if it was interesting in train (e.g. cointegrated or high corr)
            # But the user wants "every pair with every pair", so we should at least record cointegration status
            if is_coint_tr:
                # 2. Test Stats (Persistence)
                is_coint_te, trace_te, crit_te, beta_te = test_johansen(df_test, t1, t2)
                
                results.append({
                    "window": w_idx + 1,
                    "pair": f"{t1}-{t2}",
                    "corr_train": round(corr_tr, 3),
                    "is_coint_train": is_coint_tr,
                    "is_coint_test": is_coint_te,
                    "beta_train": beta_tr,
                    "trace_diff_train": round(trace_tr - crit_tr, 2),
                    "trace_diff_test": round(trace_te - crit_te, 2)
                })

    if not results:
        print("No cointegrated pairs found in any window.")
        return

    res_df = pd.DataFrame(results)
    
    # Analysis
    print("\n" + "="*60)
    print("EXHAUSTIVE WFO DIAGNOSTIC REPORT")
    print("="*60)
    
    # 1. Best pairs by cointegration persistence
    print("\nTOP PAIRS BY COINTEGRATION PERSISTENCE (Train -> Test):")
    pair_stats = res_df.groupby("pair").agg({
        "window": "count",
        "is_coint_test": "sum",
        "corr_train": "mean"
    })
    pair_stats.columns = ["Windows_Coint_Train", "Windows_Stayed_Coint_Test", "Avg_Corr_Train"]
    pair_stats["Persistence_%"] = (pair_stats["Windows_Stayed_Coint_Test"] / pair_stats["Windows_Coint_Train"] * 100).round(1)
    
    # Filter for pairs that were cointegrated in at least 3 windows
    top_persistent = pair_stats[pair_stats["Windows_Coint_Train"] >= 3].sort_values("Persistence_%", ascending=False).head(20)
    print(top_persistent.to_string())
    
    # 2. Summary stats
    total_tr = len(res_df)
    total_te = res_df["is_coint_test"].sum()
    print(f"\nOverall Persistence: {total_te}/{total_tr} ({total_te/total_tr*100:.1f}%)")
    
    # Save results
    res_df.to_csv(DATA_DIR / "exhaustive_wfo_diag.csv", index=False)
    print(f"\nDetailed results saved to data/exhaustive_wfo_diag.csv")

if __name__ == "__main__":
    main()
