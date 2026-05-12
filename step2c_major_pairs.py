"""
step2c_major_pairs.py — FX major pairs using return-spread (not price cointegration).

Why return-spread instead of price cointegration on 1-min:
  FX prices are I(1) processes → price-level spread has half-life of days/weeks.
  Return spread = cum_r1 - beta × cum_r2 over a short rolling window IS stationary.
  Half-life: 5–30 minutes → many signals per day → suitable for prop firm.

Signal logic (implemented in build_signals when USE_RETURN_SPREAD=True):
  cum_r1[t] = sum of pct_change for t1 over last RETURN_WINDOW bars
  cum_r2[t] = sum of pct_change for t2 over last RETURN_WINDOW bars
  spread[t]  = cum_r1[t] - beta × cum_r2[t]   ← stationary, ~zero mean
  zscore[t]  = spread[t] / rolling_std(spread, window)

Entry: |zscore| > ENTRY_Z → one leg outperformed → expect mean reversion.
Exit:  zscore crosses EXIT_Z in the opposite direction.

Output: data/pairs_selected.csv (same schema as step2b / step4a)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from config import DATA_DIR, BARS_PER_DAY, BAR_MINUTES, RETURN_WINDOW

# ── Pre-defined FX major pairs ────────────────────────────────────────────────
MAJOR_PAIRS = [
    ("eurusd", "gbpusd"),   # core EUR+GBP vs USD — correlation ~0.90
    ("eurusd", "audusd"),   # EUR+AUD vs USD — risk-on ~0.80
    ("eurusd", "nzdusd"),   # EUR+NZD vs USD — risk ~0.75
    ("gbpusd", "audusd"),   # GBP+AUD vs USD — risk bloc ~0.78
    ("audusd", "nzdusd"),   # Pacific pair — tightest ~0.95
    # eurjpy-gbpjpy removed: consistently 0% WR in backtest (JPY intervention spikes)
    ("eurusd", "usdchf"),   # EUR / CHF inverse mirror ~0.90
    ("gbpusd", "nzdusd"),   # GBP + NZD commodity FX ~0.70
]

# Analysis window: last N months of 1-min data for return correlation
ANALYSIS_MONTHS = 3

# Minimum return correlation to include a pair
MIN_CORR = 0.55

# Expected half-life on return spread (bars) — must be short for 1-min stat arb
HL_MIN_BARS = 2    # at least 2 bars
HL_MAX_BARS = 90   # at most 90 minutes


def load_closes(tickers: list[str]) -> pd.DataFrame:
    pq = DATA_DIR / f"closes_{BAR_MINUTES}min.parquet"
    if pq.exists():
        return pd.read_parquet(pq, columns=tickers)
    csv = DATA_DIR / f"closes_{BAR_MINUTES}min.csv"
    return pd.read_csv(csv, index_col=0, parse_dates=True, usecols=["datetime"] + tickers)


def compute_return_spread_halflife(r1: pd.Series, r2: pd.Series,
                                    beta: float, window: int) -> float:
    """Half-life of the cumulative return spread."""
    cum_r1 = r1.rolling(window).sum().dropna()
    cum_r2 = r2.rolling(window).sum().dropna()
    spread  = (cum_r1 - beta * cum_r2).dropna()

    delta = spread.diff().dropna()
    lag   = spread.shift(1).dropna()
    aligned = pd.concat([delta, lag], axis=1).dropna()
    aligned.columns = ["delta", "lag"]
    if len(aligned) < 50:
        return float("inf")

    try:
        theta = sm.OLS(aligned["delta"],
                       sm.add_constant(aligned["lag"])).fit().params["lag"]
        return -np.log(2) / theta if theta < 0 else float("inf")
    except Exception:
        return float("inf")


def main():
    all_tickers = sorted({t for pair in MAJOR_PAIRS for t in pair})
    print(f"Loading 1-min closes for {len(all_tickers)} tickers …")

    try:
        closes = load_closes(all_tickers)
    except Exception as e:
        print(f"Error: {e}"); return

    if closes.index.tz is None:
        closes.index = closes.index.tz_localize("UTC")

    cutoff = closes.index[-1] - pd.DateOffset(months=ANALYSIS_MONTHS)
    recent = closes[closes.index >= cutoff].dropna()
    print(f"Using {len(recent):,} bars  ({cutoff.date()} → {recent.index[-1].date()})")
    print(f"Return window: {RETURN_WINDOW} bars = {RETURN_WINDOW} minutes\n")

    results = []
    for t1, t2 in MAJOR_PAIRS:
        if t1 not in recent.columns or t2 not in recent.columns:
            print(f"  SKIP {t1}-{t2}: missing data"); continue

        sub = recent[[t1, t2]].dropna()
        r1  = sub[t1].pct_change().fillna(0)
        r2  = sub[t2].pct_change().fillna(0)

        # Return correlation
        corr     = r1.corr(r2)
        corr_abs = abs(corr)
        if corr_abs < MIN_CORR:
            print(f"  SKIP {t1}-{t2}: low_corr={corr_abs:.2f}"); continue

        # OLS beta on cumulative returns
        cum_r1 = r1.rolling(RETURN_WINDOW).sum().dropna()
        cum_r2 = r2.rolling(RETURN_WINDOW).sum().dropna()
        idx    = cum_r1.index.intersection(cum_r2.index)
        y, x   = cum_r1.loc[idx], cum_r2.loc[idx]
        try:
            beta = sm.OLS(y, sm.add_constant(x)).fit().params[t2]
        except Exception:
            print(f"  SKIP {t1}-{t2}: OLS failed"); continue

        # Half-life on return spread
        hl = compute_return_spread_halflife(r1, r2, beta, RETURN_WINDOW)
        if not (HL_MIN_BARS <= hl <= HL_MAX_BARS):
            print(f"  SKIP {t1}-{t2}: hl={hl:.1f}bars"); continue

        # Hurst on return spread
        spread     = (cum_r1 - beta * cum_r2).dropna()
        diff1      = spread.diff(1).dropna()
        diff4      = spread.diff(4).dropna()
        v1, v4     = diff1.var(), diff4.var()
        hurst      = 0.5 * np.log(v4 / max(v1, 1e-15)) / np.log(4) if v1 > 0 else 0.5

        print(f"  OK   {t1}-{t2}  corr={corr_abs:.2f}  beta={beta:.4f}  "
              f"H={hurst:.3f}  hl={hl:.1f}bars ({hl:.0f}min)")
        results.append({
            "pair":           f"{t1}-{t2}",
            "t1":             t1,
            "t2":             t2,
            "corr":           round(corr_abs, 4),
            "beta":           round(beta, 4),
            "hurst":          round(hurst, 4),
            "half_life_bars": round(hl, 1),
            "joh_trace":      0.0,
            "joh_crit_95":    0.0,
            "joh_margin":     round(corr_abs, 4),
        })

    if not results:
        print("\nNo pairs passed. Relax MIN_CORR or HL_MAX_BARS."); return

    df = pd.DataFrame(results).sort_values("joh_margin", ascending=False)
    out = DATA_DIR / "pairs_selected.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} pairs → {out}")
    print(df[["pair","corr","beta","hurst","half_life_bars"]].to_string(index=False))


if __name__ == "__main__":
    main()
