import warnings
warnings.filterwarnings("ignore")

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm

from config import DATA_DIR, COINT_WINDOW_DAYS, TICKERS, SSD_PERCENTILE, NON_FX_TICKERS, BARS_PER_DAY

# ── Tunable parameters ────────────────────────────────────────────────────────
CORR_MIN      = 0.25
JOH_CRIT      = 0.95
BETA_MIN      = 0.10
BETA_MAX      = 15.0
HURST_MAX     = 0.48
MIN_OBS       = 252
HALF_LIFE_MAX_DAYS = 30

BLACKLIST = {
    "nzdchf-nzdusd",
    "nzdcad-nzdchf",
    "eurusd-nzdcad",
}

DAILY_UNIV   = DATA_DIR / "closes_daily.csv"
OUT_FILE     = DATA_DIR / "pairs_selected.csv"

_OUTPUT_COLUMNS = [
    "pair", "t1", "t2", "corr", "joh_trace", "joh_crit_95",
    "joh_margin", "beta", "hurst", "half_life_bars",
    "train_end_date", "test_start_date", "test_end_date", "snapshot_id",
]


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


def _load_daily_universe() -> pd.DataFrame:
    if not DAILY_UNIV.exists():
        raise FileNotFoundError(f"{DAILY_UNIV} not found. Run step0b_generate_daily.py first.")
    closes = pd.read_csv(DAILY_UNIV, index_col=0, parse_dates=True)
    symbols = [s for s in closes.columns if s.lower() not in NON_FX_TICKERS]
    return closes[symbols]


def build_pairs_universe(
    train_end: str | pd.Timestamp | None = None,
    test_start: str | pd.Timestamp | None = None,
    test_end: str | pd.Timestamp | None = None,
    out_path: str | Path | None = None,
    closes: pd.DataFrame | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build a point-in-time FX pairs universe.

    train_end : ISO date or Timestamp. Pair selection uses ONLY closes <= train_end.
                If None, the full available history is used (legacy mode).
    test_start, test_end : optional metadata persisted into the output CSV.
    out_path : optional CSV path. If given, the result is written there.

    Returns the DataFrame of selected pairs with metadata columns attached.
    """
    if closes is None:
        closes = _load_daily_universe()
    else:
        closes = closes.copy()

    train_end_ts = pd.Timestamp(train_end) if train_end is not None else None
    if train_end_ts is not None:
        closes = closes[closes.index <= train_end_ts]
        # Hard safeguard against look-ahead.
        max_date = closes.index.max() if not closes.empty else None
        if max_date is not None and max_date > train_end_ts:
            raise AssertionError(
                f"Look-ahead detected: closes.index.max()={max_date} > train_end={train_end_ts}"
            )
        if verbose:
            print(f"Date filter: closes clipped to <= {train_end_ts.date()} ({len(closes)} days)")

    symbols = list(closes.columns)
    if verbose:
        print(f"Loaded {len(symbols)} symbols and {len(closes)} days.")

    window = COINT_WINDOW_DAYS
    recent = closes.tail(window)
    recent = recent.dropna(axis=1, how="all")
    recent = recent.loc[:, recent.notna().sum() > window * 0.8]

    all_pairs = list(combinations(sorted(recent.columns), 2))
    if verbose:
        print(f"Total potential pairs: {len(all_pairs)}")

    log_ret = np.log(recent / recent.shift(1))
    corr_matrix = log_ret.corr()

    passed_corr = []
    for t1, t2 in all_pairs:
        c = corr_matrix.loc[t1, t2] if (t1 in corr_matrix.columns and t2 in corr_matrix.columns) else np.nan
        if pd.notna(c) and c >= CORR_MIN:
            passed_corr.append((t1, t2, float(c)))
    if verbose:
        print(f"Layer 2 (corr >= {CORR_MIN}): {len(passed_corr)} pairs remain")

    scored_ssd = []
    for t1, t2, c in passed_corr:
        s1, s2 = recent[t1].dropna(), recent[t2].dropna()
        common = s1.index.intersection(s2.index)
        if len(common) < 30:
            continue
        s1, s2 = s1.loc[common], s2.loc[common]
        p1_norm, p2_norm = s1 / s1.iloc[0], s2 / s2.iloc[0]
        ssd = float(((p1_norm - p2_norm) ** 2).sum())
        scored_ssd.append((ssd, t1, t2, c))

    if scored_ssd:
        ssd_vals = [s for s, _, _, _ in scored_ssd]
        cutoff = float(np.percentile(ssd_vals, SSD_PERCENTILE))
        passed_ssd = [(t1, t2, c) for ssd, t1, t2, c in scored_ssd if ssd <= cutoff]
        if verbose:
            print(f"Layer 2b (SSD <= p{SSD_PERCENTILE}): {len(passed_ssd)} pairs remain")
    else:
        passed_ssd = passed_corr

    if verbose:
        print(f"Testing {len(passed_ssd)} pairs with Johansen...")
    results = []
    for t1, t2, corr in passed_ssd:
        pc = recent[[t1, t2]].dropna()
        try:
            trace, crit, beta = _johansen_beta(pc)
            if trace > crit and beta > 0 and BETA_MIN <= beta <= BETA_MAX:
                spread = pc[t1] - beta * pc[t2]
                hurst = hurst_exponent(spread.values)
                if hurst < HURST_MAX:
                    hl = compute_half_life(spread)
                    if hl <= 0 or hl > HALF_LIFE_MAX_DAYS:
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
                        "half_life_bars": round(hl * BARS_PER_DAY, 1),
                    })
        except Exception:
            continue

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("joh_margin", ascending=False)
        df = df[~df["pair"].isin(BLACKLIST)].reset_index(drop=True)

    # Attach window metadata
    test_start_str = pd.Timestamp(test_start).date().isoformat() if test_start is not None else None
    test_end_str   = pd.Timestamp(test_end).date().isoformat()   if test_end   is not None else None
    train_end_str  = train_end_ts.date().isoformat()             if train_end_ts is not None else None
    snapshot_id    = pd.Timestamp(test_start).strftime("%Y%m%d") if test_start is not None else None

    if df.empty:
        df = pd.DataFrame(columns=_OUTPUT_COLUMNS)
    else:
        df["train_end_date"]  = train_end_str
        df["test_start_date"] = test_start_str
        df["test_end_date"]   = test_end_str
        df["snapshot_id"]     = snapshot_id

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        if verbose:
            print(f"Saved {len(df)} pairs to {out_path}")
            if not df.empty:
                print(df.head(20).to_string(index=False))

    return df


def main():
    parser = argparse.ArgumentParser(description="FX pair discovery — point-in-time or full-history mode")
    parser.add_argument("--train-end",  type=str, default=None,
                        help="ISO date: use only closes <= this date for pair selection (no look-ahead)")
    parser.add_argument("--test-start", type=str, default=None,
                        help="ISO date: written into output as test_start_date metadata column")
    parser.add_argument("--test-end",   type=str, default=None,
                        help="ISO date: written into output as test_end_date metadata column")
    parser.add_argument("--out",        type=str, default=str(OUT_FILE),
                        help="Output CSV path (default: data/pairs_selected.csv)")
    args = parser.parse_args()

    build_pairs_universe(
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
