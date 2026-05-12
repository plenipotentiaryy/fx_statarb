"""
step2_pairs.py — Pair selection with rolling-window cointegration as primary filter.

Primary filter : Rolling 90-day daily Engle-Granger (Yahoo Finance, p < 0.15)
                 Uses only the most recent COINT_WINDOW_DAYS trading days so the
                 test reflects current macro regime instead of 10-year averages.
Secondary      : Hurst < 0.50, log-return correlation > 0.50,
                 recent 120-day correlation > 0.50, beta_daily in [0.1, 15.0],
                 cost viability (cost_fraction < 10%)
Info columns   : intraday EG p-value, Johansen trace/eigen stats (not filters)

beta_daily (OLS on rolling window) is the primary hedge ratio saved to CSV.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm
import yfinance as yf
from config import (
    PAIRS, TICKERS, RECENT_BARS, TRAIN_RATIO, RTH_START, RTH_END, DATA_DIR,
    HURST_MAX, CORR_MIN, RECENT_CORR_DAYS, RECENT_CORR_MIN,
    BARS_PER_DAY, CLOSES_FILE, DAILY_START, COST_TAKER,
    COINT_WINDOW_DAYS, HALF_LIFE_MAX_BARS,
)

BETA_MIN = 0.1
BETA_MAX = 15.0
MIN_PAIR_OVERLAP = 500
DAILY_CACHE = DATA_DIR / "closes_daily.csv"
DAILY_COINT_P  = 0.35      # EG p-value pre-screen (permissive — Johansen is primary)
TRAIN_COINT_WINDOW = 500   # days of training-period daily data for initial pair selection
JOH_OR_EG_PASS = True      # pair passes if EITHER Johansen OR EG confirms (not both required)
DAILY_CACHE_MAX_AGE = 7    # re-download daily cache if older than this many days

# Max fraction of one spread-sigma that transaction costs may consume per trade.
MAX_COST_FRACTION = 0.10


# ── Daily closes via yfinance (cached) ───────────────────────────────────────

def load_daily_closes() -> pd.DataFrame:
    if DAILY_CACHE.exists():
        age_days = (pd.Timestamp.now() -
                    pd.Timestamp(DAILY_CACHE.stat().st_mtime, unit="s")).days
        if age_days < DAILY_CACHE_MAX_AGE:
            df = pd.read_csv(DAILY_CACHE, index_col=0, parse_dates=True)
            print(f"Daily cache loaded  ({len(df)} rows × {len(df.columns)} tickers, age {age_days}d)")
            return df

    print("Downloading daily closes via yfinance …", flush=True)
    raw = yf.download(TICKERS, start=DAILY_START, auto_adjust=True,
                      progress=False, threads=True)
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    closes = closes.dropna(how="all")
    closes.to_csv(DAILY_CACHE)
    print(f"Daily cache saved  ({len(closes)} rows × {len(closes.columns)} tickers)")
    return closes


# ── Intraday (Alpha Vantage) ──────────────────────────────────────────────────

def _load_raw() -> pd.DataFrame:
    path = DATA_DIR / CLOSES_FILE
    if not path.exists():
        fallback = DATA_DIR / "closes_15min.csv"
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(f"No data file: {CLOSES_FILE}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        df.index = df.index.tz_convert("UTC")
    return df.between_time(RTH_START, RTH_END)


def load_closes() -> pd.DataFrame:
    raw  = _load_raw()
    good = [c for c in raw.columns if raw[c].notna().sum() > 1000]
    print(f"Tickers available: {len(good)} / {len(raw.columns)}")
    return raw[good]


def load_closes_dense() -> pd.DataFrame:
    raw   = _load_raw()
    n     = len(raw)
    dense = [c for c in raw.columns if raw[c].notna().sum() >= int(n * 0.99)]
    return raw[dense].dropna()


# ── Stats helpers ─────────────────────────────────────────────────────────────

def johansen_info(df2: pd.DataFrame) -> tuple[float, float, bool]:
    """
    Run Johansen cointegration test on a 2-column DataFrame.
    Returns (trace_stat, eigen_stat, cointegrated_at_95pct).
    Critical values: 95% CI for r=0 test with 2 series, det_order=0.
    Trace critical value: 15.41 | Eigen critical value: 14.07
    """
    try:
        result = coint_johansen(df2.dropna(), det_order=0, k_ar_diff=1)
        trace_stat = float(result.lr1[0])   # trace stat for H0: r=0
        eigen_stat = float(result.lr2[0])   # max-eigen stat for H0: r=0
        # critical values: index 0=90%, 1=95%, 2=99%
        trace_crit = float(result.cvt[0, 1])
        eigen_crit = float(result.cvm[0, 1])
        cointegrated = (trace_stat > trace_crit) or (eigen_stat > eigen_crit)
        return round(trace_stat, 3), round(eigen_stat, 3), cointegrated
    except Exception:
        return float("nan"), float("nan"), False


def hurst_exponent(series: np.ndarray, max_lag: int = 200) -> float:
    lags = range(2, min(max_lag, len(series) // 4))
    # Use RMS instead of std to preserve structural drift
    tau  = [np.sqrt(np.mean((series[lag:] - series[:-lag])**2)) for lag in lags]
    return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])


def compute_half_life(spread: pd.Series) -> float:
    aligned = pd.concat([spread.diff(), spread.shift(1)], axis=1).dropna()
    aligned.columns = ["diff", "lag"]
    theta = sm.OLS(aligned["diff"], sm.add_constant(aligned["lag"])).fit().params["lag"]
    return -np.log(2) / theta if theta < 0 else float("inf")


# ── Load data ─────────────────────────────────────────────────────────────────

DATA_DIR.mkdir(exist_ok=True)
daily = load_daily_closes()

closes = load_closes()
_dense = load_closes_dense()

split_date   = _dense.index[int(len(_dense) * TRAIN_RATIO)]
closes_train = closes[closes.index <= split_date]
closes_test  = closes[closes.index  > split_date]

n_train = len(_dense[_dense.index <= split_date])
n_test  = len(_dense[_dense.index  > split_date])
print(f"\nTotal bars (dense): {len(_dense)}")
print(f"TRAIN ({TRAIN_RATIO:.0%}): {n_train} bars  "
      f"{_dense.index[0].date()} → {split_date.date()}")
print(f"TEST  ({1-TRAIN_RATIO:.0%}): {n_test} bars  "
      f"{closes_test.index[0].date()} → {closes_test.index[-1].date()}")

# Intraday window for spread metrics (last RECENT_BARS of training period)
_coint_ref   = _dense[_dense.index <= split_date].tail(RECENT_BARS)
coint_start  = _coint_ref.index[0]
closes_coint = closes_train[closes_train.index >= coint_start]

print(f"\nIntraday spread window: {coint_start.date()} → {split_date.date()}"
      f"  ({len(closes.columns)} tickers available)")
print(f"Daily data: {daily.index[0].date()} → {daily.index[-1].date()}"
      f"  ({len(daily)} trading days)")
print(f"\nTesting {len(PAIRS)} predefined pairs\n")


# ── Pair loop ─────────────────────────────────────────────────────────────────

results = []

for t1, t2 in PAIRS:
    label = f"{t1}-{t2}"

    if t1 not in closes.columns or t2 not in closes.columns:
        print(f"  SKIP {label}: missing intraday data")
        continue

    # ── PRIMARY FILTER: rolling-window daily cointegration ───────────────────
    if t1 not in daily.columns or t2 not in daily.columns:
        print(f"  SKIP {label}: missing daily data")
        continue

    pc_d = daily[[t1, t2]].dropna()
    if len(pc_d) < COINT_WINDOW_DAYS:
        print(f"  SKIP {label}: only {len(pc_d)} daily bars (need {COINT_WINDOW_DAYS})")
        continue

    # Use TRAINING portion of daily data for initial selection.
    # split_date is timezone-aware; daily index may be naive — normalise.
    split_naive = split_date.tz_localize(None) if split_date.tzinfo else split_date
    if pc_d.index.tz is not None:
        pc_d_train = pc_d[pc_d.index <= split_date]
    else:
        pc_d_train = pc_d[pc_d.index <= split_naive]

    # Use up to TRAIN_COINT_WINDOW days from the training period end
    pc_window = pc_d_train.tail(TRAIN_COINT_WINDOW) if len(pc_d_train) >= COINT_WINDOW_DAYS \
                else pc_d.tail(COINT_WINDOW_DAYS)

    _, pvalue_daily, _ = coint(pc_window[t1], pc_window[t2])
    eg_pass = pvalue_daily < DAILY_COINT_P

    # Johansen — primary criterion for 20-year datasets
    joh_trace, joh_eigen, joh_coint = johansen_info(pc_window)

    # Require EITHER Johansen OR EG to confirm cointegration
    if not (eg_pass or joh_coint):
        print(f"  FAIL {label}: EG p={pvalue_daily:.4f} AND Johansen t={joh_trace:.1f} — both fail")
        continue

    # Daily beta — OLS on training window
    beta_daily = float(sm.OLS(pc_window[t1], sm.add_constant(pc_window[t2])).fit().params.iloc[1])
    if beta_daily < 0 or not (BETA_MIN <= abs(beta_daily) <= BETA_MAX):
        print(f"  SKIP {label}: beta_daily={beta_daily:.4f} outside valid range")
        continue

    # ── Intraday spread metrics (computed with beta_daily for consistency) ────
    pc = closes_coint[[t1, t2]].dropna()
    if len(pc) < MIN_PAIR_OVERLAP:
        print(f"  SKIP {label}: only {len(pc)} intraday bars")
        continue

    spread    = pc[t1] - beta_daily * pc[t2]
    half_life = compute_half_life(spread)
    hurst     = hurst_exponent(spread.values)
    log_ret   = np.log(pc / pc.shift(1)).dropna()
    corr      = round(float(log_ret[t1].corr(log_ret[t2])), 4)

    # ── Cost viability ────────────────────────────────────────────────────────
    avg_notional  = float(pc[t1].mean() + beta_daily * pc[t2].mean())
    sigma_spread  = float(spread.std())
    cost_fraction = (2 * COST_TAKER * avg_notional) / max(sigma_spread, 1e-8)
    if cost_fraction > MAX_COST_FRACTION:
        print(f"  SKIP {label}: cost_fraction={cost_fraction:.1%} > {MAX_COST_FRACTION:.0%} "
              f"(notional=${avg_notional:.0f}, σ_spread=${sigma_spread:.2f})")
        continue

    # ── Hurst filter ──────────────────────────────────────────────────────────
    if hurst >= HURST_MAX:
        print(f"  FAIL {label}: Hurst={hurst:.3f} >= {HURST_MAX}")
        continue

    # ── Half-life filter: too slow = spread won't revert before stop is hit ──
    if half_life > HALF_LIFE_MAX_BARS:
        print(f"  FAIL {label}: HL={half_life:.0f}b > {HALF_LIFE_MAX_BARS} (too slow)")
        continue

    # ── Correlation filter ────────────────────────────────────────────────────
    if corr < CORR_MIN:
        print(f"  FAIL {label}: corr={corr:.3f} < {CORR_MIN}")
        continue

    # ── Recent 120-day correlation ────────────────────────────────────────────
    pc_recent   = closes[[t1, t2]].dropna().tail(RECENT_CORR_DAYS * BARS_PER_DAY)
    log_ret_rec = np.log(pc_recent / pc_recent.shift(1)).dropna()
    recent_corr = round(float(log_ret_rec[t1].corr(log_ret_rec[t2])), 4)
    # recent_corr kept as info — not a blocking filter for historical backtest
    # (backtest's CointegrationFilter handles dynamic breaks at runtime)

    # ── INFO: intraday cointegration (not a filter) ──────────────────────────
    _, pvalue_intraday, _ = coint(pc[t1], pc[t2])
    adf_stat, adf_pvalue, *_ = adfuller(spread)
    beta_intraday = float(sm.OLS(pc[t1], sm.add_constant(pc[t2])).fit().params.iloc[1])

    results.append({
        "pair":               label,
        "correlation":        corr,
        "recent_corr_120d":   recent_corr,
        "coint_pvalue":       round(pvalue_intraday, 6),   # intraday — info only
        "coint_pvalue_daily": round(pvalue_daily, 6),      # rolling 90d EG — primary filter
        "beta":               round(beta_intraday, 4),     # intraday beta (info)
        "beta_daily":         round(beta_daily, 4),        # rolling-window daily beta (used by backtest)
        "adf_pvalue":         round(adf_pvalue, 6),
        "adf_stat":           round(adf_stat, 4),
        "half_life_bars":     round(half_life, 1),
        "hurst":              round(hurst, 4),
        "cost_fraction":      round(cost_fraction, 4),
        "johansen_trace":     joh_trace,                   # Johansen trace stat — info
        "johansen_eigen":     joh_eigen,                   # Johansen eigen stat — info
        "johansen_coint":     joh_coint,                   # True if Johansen confirms coint at 95%
        "coint_window_days":  COINT_WINDOW_DAYS,
        "test_start_date":    str(closes_test.index[0].date()),
    })

    joh_flag = "✓" if joh_coint else "✗"
    print(f"  PASS {label}: rolling{COINT_WINDOW_DAYS}d_p={pvalue_daily:.4f}  "
          f"beta_d={beta_daily:.4f}  HL={half_life:.0f}b  H={hurst:.3f}"
          f"  corr={corr}  rc120={recent_corr}"
          f"  Johansen={joh_flag}(t={joh_trace:.1f},e={joh_eigen:.1f})"
          f"  [intraday_p={pvalue_intraday:.4f}]")


# ── Results ───────────────────────────────────────────────────────────────────

if not results:
    print("\nNo pairs passed all filters.")
    pd.DataFrame(columns=["pair", "correlation", "recent_corr_120d",
                           "coint_pvalue", "coint_pvalue_daily",
                           "beta", "beta_daily", "adf_pvalue", "adf_stat",
                           "half_life_bars", "hurst",
                           "johansen_trace", "johansen_eigen", "johansen_coint",
                           "coint_window_days", "test_start_date"]
                 ).to_csv(DATA_DIR / "pairs_selected.csv", index=False)
    raise SystemExit(0)

df = pd.DataFrame(results).sort_values("coint_pvalue_daily")
print("\n" + "=" * 120)
print(df[["pair", "correlation", "recent_corr_120d", "coint_pvalue_daily",
          "johansen_coint", "johansen_trace", "beta_daily", "half_life_bars", "hurst"]
        ].to_string(index=False))
print("=" * 120)
joh_both = df["johansen_coint"].sum()
print(f"\nPairs confirmed by both EG + Johansen: {joh_both}/{len(df)}")

df.to_csv(DATA_DIR / "pairs_selected.csv", index=False)
print(f"\nSaved {len(df)} pairs  (test starts {results[0]['test_start_date']})")

good = df[df["half_life_bars"].between(5, 500)]
print(f"\nRecommended (HL 5–500 bars): {len(good)}")
for _, row in good.iterrows():
    print(f"  {row['pair']}: beta_daily={row['beta_daily']}  "
          f"HL={row['half_life_bars']} bars  "
          f"daily_coint={row['coint_pvalue_daily']:.4f}  "
          f"[intraday_coint={row['coint_pvalue']:.4f}]")
