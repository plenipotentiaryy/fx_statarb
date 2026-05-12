import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm

from config import DATA_DIR, COINT_WINDOW_DAYS, TICKERS, SSD_PERCENTILE, NON_FX_TICKERS, BARS_PER_DAY

# ── Tunable parameters ────────────────────────────────────────────────────────
CORR_MIN      = 0.40
JOH_CRIT      = 0.95
BETA_MIN      = 0.10
BETA_MAX      = 15.0
HURST_MAX     = 0.45   # tightened: H must be clearly mean-reverting
MIN_OBS       = 252
HALF_LIFE_MAX = 1440   # bars on intraday tf — max 1 trading day at 1-min (= 1 day at 15-min: 96)

# Pairs that failed systematic backtesting (near-zero WR regardless of parameters)
BLACKLIST = {
    "nzdchf-nzdusd",
    "nzdcad-nzdchf",
    "eurusd-nzdcad",
}

# ── File paths ────────────────────────────────────────────────────────────────
DAILY_UNIV   = DATA_DIR / "closes_daily.csv"
OUT_FILE     = DATA_DIR / "pairs_selected.csv"

def hurst_exponent(series: np.ndarray, max_lag: int = 100) -> float:
    lags = range(2, min(max_lag, len(series) // 4))
    tau  = [np.std(series[lag:] - series[:-lag]) for lag in lags]
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])

def compute_half_life(spread: pd.Series) -> float:
    aligned = pd.concat([spread.diff(), spread.shift(1)], axis=1).dropna()
    aligned.columns = ["diff", "lag"]
    theta = sm.OLS(aligned["diff"], sm.add_constant(aligned["lag"])).fit().params["lag"]
    return -np.log(2) / theta if theta < 0 else float("inf")

_JOH_CRIT_IDX = {0.90: 0, 0.95: 1, 0.99: 2}

def _johansen_beta(pc: pd.DataFrame) -> tuple[float, float, float]:
    res   = coint_johansen(pc.dropna(), det_order=0, k_ar_diff=1)
    trace = float(res.lr1[0])
    crit  = float(res.cvt[0, _JOH_CRIT_IDX[JOH_CRIT]])
    evec  = res.evec[:, 0]
    beta  = -evec[1] / evec[0]
    return trace, crit, float(beta)

def main():
    if not DAILY_UNIV.exists():
        print(f"Error: {DAILY_UNIV} not found. Run step0b_generate_daily.py first.")
        return

    closes = pd.read_csv(DAILY_UNIV, index_col=0, parse_dates=True)
    # Keep only pure FX tickers — filter out equity indices and commodities
    symbols = [s for s in closes.columns if s.lower() not in NON_FX_TICKERS]
    closes = closes[symbols]

    print(f"Loaded {len(symbols)} symbols and {len(closes)} days.")

    # Layer 1: All combinations
    all_pairs = list(combinations(sorted(symbols), 2))
    print(f"Total potential pairs: {len(all_pairs)}")
    
    # Layer 2: Correlation
    window = COINT_WINDOW_DAYS
    recent = closes.tail(window)
    # Drop columns that are entirely NaN in the window
    recent = recent.dropna(axis=1, how="all")
    # Also drop columns that have too many NaNs
    recent = recent.loc[:, recent.notna().sum() > window * 0.8]
    
    log_ret = np.log(recent / recent.shift(1))
    corr_matrix = log_ret.corr()
    
    passed_corr = []
    for t1, t2 in all_pairs:
        if t1 in corr_matrix.columns and t2 in corr_matrix.columns:
            c = corr_matrix.loc[t1, t2]
            if c >= CORR_MIN:
                passed_corr.append((t1, t2, c))
    
    print(f"Layer 2 (corr >= {CORR_MIN}): {len(passed_corr)} pairs remain")
    
    # Layer 2b: SSD
    scored_ssd = []
    for t1, t2, c in passed_corr:
        s1, s2 = recent[t1].dropna(), recent[t2].dropna()
        common = s1.index.intersection(s2.index)
        if len(common) < 30: continue
        s1, s2 = s1.loc[common], s2.loc[common]
        p1_norm, p2_norm = s1 / s1.iloc[0], s2 / s2.iloc[0]
        ssd = float(((p1_norm - p2_norm) ** 2).sum())
        scored_ssd.append((ssd, t1, t2, c))
    
    if scored_ssd:
        ssd_vals = [s for s, _, _, _ in scored_ssd]
        cutoff = float(np.percentile(ssd_vals, SSD_PERCENTILE))
        passed_ssd = [(t1, t2, c) for ssd, t1, t2, c in scored_ssd if ssd <= cutoff]
        print(f"Layer 2b (SSD <= p{SSD_PERCENTILE}): {len(passed_ssd)} pairs remain")
    else:
        passed_ssd = passed_corr
        
    # Layer 4+5: Johansen
    print(f"Testing {len(passed_ssd)} pairs with Johansen...")
    results = []
    for t1, t2, corr in passed_ssd:
        pc = recent[[t1, t2]].dropna()
        try:
            trace, crit, beta = _johansen_beta(pc)
            if trace > crit and BETA_MIN <= abs(beta) <= BETA_MAX and beta > 0:
                spread = pc[t1] - beta * pc[t2]
                hurst = hurst_exponent(spread.values)
                if hurst < HURST_MAX:
                    hl = compute_half_life(spread)
                    hl_bars = round(hl * BARS_PER_DAY, 0)   # correct for any bar size
                    if hl_bars > HALF_LIFE_MAX or hl_bars <= 0:
                        continue
                    results.append({
                        "pair": f"{t1}-{t2}",
                        "t1": t1, "t2": t2,
                        "corr": round(corr, 4),
                        "joh_trace": round(trace, 3),
                        "joh_crit_95": round(crit, 3),
                        "joh_margin": round(trace - crit, 3),
                        "beta": round(beta, 4),
                        "hurst": round(hurst, 4),
                        "half_life_bars": hl_bars
                    })
        except: continue
        
    if not results:
        print("No cointegrated pairs found.")
        return
        
    df = pd.DataFrame(results).sort_values("joh_margin", ascending=False)
    before = len(df)
    df = df[~df["pair"].isin(BLACKLIST)]
    if before > len(df):
        print(f"Blacklist removed {before - len(df)} pairs: {BLACKLIST & set(df['pair'].tolist()) or BLACKLIST}")
    df.to_csv(OUT_FILE, index=False)
    print(f"Saved {len(df)} pairs to {OUT_FILE}")
    print(df.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
