import pandas as pd
import numpy as np
from pathlib import Path
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_DIR, WFO_TRAIN_MONTHS, WFO_TEST_MONTHS, WFO_STEP_MONTHS, 
    WFO_EXPANDING, BARS_PER_DAY
)

def hurst_exponent(series: np.ndarray, max_lag: int = 100) -> float:
    lags = range(2, min(max_lag, len(series) // 4))
    tau  = [np.std(series[lag:] - series[:-lag]) for lag in lags]
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])

_JOH_CRIT_IDX = {0.90: 0, 0.95: 1, 0.99: 2}

def test_johansen(df, t1, t2, crit_level=0.95):
    pc = df[[t1, t2]].dropna()
    if len(pc) < 100: return False, 0, 0, 0
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
    # Load data
    daily_path = DATA_DIR / "closes_daily.csv"
    pairs_path = DATA_DIR / "pairs_selected.csv"
    
    if not daily_path.exists() or not pairs_path.exists():
        print("Missing data/closes_daily.csv or data/pairs_selected.csv")
        return

    daily = pd.read_csv(daily_path, index_col=0, parse_dates=True)
    if daily.index.tz is None:
        daily.index = pd.to_datetime(daily.index, utc=True)
    
    pairs_df = pd.read_csv(pairs_path)
    pairs_list = pairs_df["pair"].tolist()

    # Define windows
    anchor = daily.index[0].date()
    end    = daily.index[-1].date()
    windows = []
    cur = anchor
    while True:
        tr_s = anchor if WFO_EXPANDING else cur
        tr_e = cur + relativedelta(months=WFO_TRAIN_MONTHS)
        te_s = tr_e
        te_e = te_s + relativedelta(months=WFO_TEST_MONTHS)
        if te_e > end: break
        windows.append((tr_s, tr_e))
        cur = cur + relativedelta(months=WFO_STEP_MONTHS)

    print(f"Checking {len(pairs_list)} pairs across {len(windows)} WFO windows...")
    print(f"Window Type: {'EXPANDING' if WFO_EXPANDING else 'ROLLING'}")
    print("-" * 100)

    results = []
    for w_idx, (tr_s, tr_e) in enumerate(windows):
        tr_s_ts = pd.Timestamp(tr_s, tz="UTC")
        tr_e_ts = pd.Timestamp(tr_e, tz="UTC")
        slice_df = daily[(daily.index >= tr_s_ts) & (daily.index < tr_e_ts)]
        
        if len(slice_df) < 100: continue
        
        log_ret = np.log(slice_df / slice_df.shift(1)).dropna(axis=1, how='all')
        corr_matrix = log_ret.corr()

        for pair in pairs_list:
            t1, t2 = pair.split("-")
            if t1 not in slice_df.columns or t2 not in slice_df.columns:
                continue
            
            # Correlation
            corr = corr_matrix.loc[t1, t2] if t1 in corr_matrix.columns and t2 in corr_matrix.columns else np.nan
            
            # Cointegration
            is_coint, trace, crit, beta = test_johansen(slice_df, t1, t2)
            
            # Hurst
            pc = slice_df[[t1, t2]].dropna()
            if len(pc) > 50:
                spread = pc[t1] - beta * pc[t2] if is_coint else pc[t1] - (pc[t1].iloc[0]/pc[t2].iloc[0]) * pc[t2]
                hurst = hurst_exponent(spread.dropna().values)
            else:
                hurst = np.nan
            
            results.append({
                "window": w_idx + 1,
                "start": tr_s,
                "end": tr_e,
                "pair": pair,
                "corr": round(corr, 3),
                "is_coint": is_coint,
                "trace": trace,
                "crit": crit,
                "beta": beta,
                "hurst": round(hurst, 3)
            })

    report_df = pd.DataFrame(results)
    
    # Summary per pair
    print("\nSUMMARY ACROSS ALL WINDOWS:")
    summary = report_df.groupby("pair").agg({
        "is_coint": ["sum", "count"],
        "corr": "mean",
        "hurst": "mean"
    })
    summary.columns = ["Coint_Windows", "Total_Windows", "Avg_Corr", "Avg_Hurst"]
    summary["Coint_Ratio"] = (summary["Coint_Windows"] / summary["Total_Windows"] * 100).round(1)
    
    print(summary[["Total_Windows", "Coint_Windows", "Coint_Ratio", "Avg_Corr", "Avg_Hurst"]].to_string())
    
    # Check for specific pairs stability
    print("\nRECENT 5 WINDOWS CHECK (Current state):")
    recent = report_df[report_df["window"] > (len(windows) - 5)]
    print(recent[["window", "pair", "corr", "is_coint", "beta", "hurst"]].sort_values(["window", "pair"]).to_string(index=False))

if __name__ == "__main__":
    main()
