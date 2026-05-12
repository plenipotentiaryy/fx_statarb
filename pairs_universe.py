"""
pairs_universe.py — Auto-discovery of cointegrated pairs from the S&P 500.

Funnel (cheap → expensive):
  Layer 1: Same GICS sub-industry (Wikipedia)    ~125k → ~3,000 pairs
  Layer 2: Correlation matrix > 0.30             ~3k   →   ~500 pairs  (garbage filter only)
  Layer 2b: SSD pre-filter (Gatev et al. 2006)  ~500  →   ~350 pairs  (discard most divergent)
  Layer 3: Individual ADF pre-screen (I(1))      ~350  →   ~200 pairs
  Layer 4: Johansen cointegration test           ~200  →    ~40 pairs
  Layer 5: Beta range, Hurst < 0.50              ~40   →  final list

Layer 4 uses Johansen (not Engle-Granger) because:
  - Symmetric by construction: no Y/X assignment bias
  - One eigendecomposition returns BOTH the test stat AND the optimal hedge ratio (β)
  - For a 2-asset pair this is faster than symmetric EG (2x coint + 2x OLS)
  - The eigenvector β accounts for noise in both series (equivalent to TLS regression)

Runs on daily closes only (yfinance).
Output: data/pairs_universe.csv  — paste the 'pair' column into config.py PAIRS.
Also prints which new pairs overlap with your existing intraday data.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

import ssl
import urllib.request
from io import StringIO
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm

from config import DATA_DIR, DAILY_START, COINT_WINDOW_DAYS, TICKERS, SSD_PERCENTILE

# ── Tunable parameters ────────────────────────────────────────────────────────
CORR_MIN     = 0.30   # Layer 2: garbage filter only — correlation ≠ cointegration.
                      # Cuts obvious noise; does NOT rank quality. Good cointegrated
                      # pairs can have corr ~0.4. Real quality selection happens in L4+5.
JOH_CRIT     = 0.95   # Layer 4: Johansen confidence level (0.90 / 0.95 / 0.99)
BETA_MIN     = 0.10   # Layer 5: min OLS beta
BETA_MAX     = 15.0   # Layer 5: max OLS beta
HURST_MAX    = 0.50   # Layer 5: spread must be mean-reverting
MIN_OBS      = 252    # minimum daily bars per ticker (1 year)

# ── File paths ────────────────────────────────────────────────────────────────
SP500_CACHE  = DATA_DIR / "sp500_sectors.csv"
DAILY_UNIV   = DATA_DIR / "closes_daily_universe.csv"
OUT_FILE     = DATA_DIR / "pairs_universe.csv"
CACHE_AGE    = 7   # days before re-downloading


# ── Layer 0: S&P 500 sector map ───────────────────────────────────────────────

def get_sp500_sectors() -> pd.DataFrame:
    """Scrape Wikipedia S&P 500 list → DataFrame[Symbol, Sector, SubIndustry]."""
    if SP500_CACHE.exists():
        age = (pd.Timestamp.now() -
               pd.Timestamp(SP500_CACHE.stat().st_mtime, unit="s")).days
        if age < CACHE_AGE:
            df = pd.read_csv(SP500_CACHE)
            print(f"Sector cache loaded  ({len(df)} tickers, age {age}d)")
            return df

    print("Fetching S&P 500 sector map from Wikipedia …", flush=True)
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"})
    with urllib.request.urlopen(req, context=ctx) as resp:
        html = resp.read().decode("utf-8")
    tables = pd.read_html(StringIO(html))
    sp = tables[0].copy()
    # Column names vary slightly across Wikipedia edits — normalise
    sp.columns = [c.strip() for c in sp.columns]
    col_map = {}
    for c in sp.columns:
        cl = c.lower()
        if "symbol" in cl:         col_map[c] = "symbol"
        elif "sector" in cl and "sub" not in cl: col_map[c] = "sector"
        elif "sub" in cl and "industry" in cl:   col_map[c] = "sub_industry"
    sp = sp.rename(columns=col_map)[["symbol", "sector", "sub_industry"]]
    # Wikipedia uses dots; yfinance uses dashes (BRK.B → BRK-B)
    sp["symbol"] = sp["symbol"].str.replace(".", "-", regex=False)
    sp.to_csv(SP500_CACHE, index=False)
    print(f"Sector map saved  ({len(sp)} tickers)")
    return sp


# ── Layer 0: bulk daily closes ────────────────────────────────────────────────

def download_universe(symbols: list[str]) -> pd.DataFrame:
    """Download / load cached daily closes for all universe tickers."""
    if DAILY_UNIV.exists():
        age = (pd.Timestamp.now() -
               pd.Timestamp(DAILY_UNIV.stat().st_mtime, unit="s")).days
        if age < CACHE_AGE:
            df = pd.read_csv(DAILY_UNIV, index_col=0, parse_dates=True)
            print(f"Universe daily cache loaded  "
                  f"({len(df)} rows × {len(df.columns)} tickers, age {age}d)")
            return df

    print(f"Downloading daily closes for {len(symbols)} tickers …", flush=True)
    raw = yf.download(symbols, start=DAILY_START, auto_adjust=True,
                      progress=True, threads=True)
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    closes = closes.dropna(how="all")
    closes.to_csv(DAILY_UNIV)
    print(f"Universe cache saved  ({len(closes)} rows × {len(closes.columns)} tickers)")
    return closes


# ── Layer 1: same sub-industry filter ────────────────────────────────────────

def layer1_sector(symbols: list[str], sector_df: pd.DataFrame) -> list[tuple]:
    """Keep only pairs from the same GICS sub-industry."""
    sym_set = set(symbols)
    sector_df = sector_df[sector_df["symbol"].isin(sym_set)]
    groups = sector_df.groupby("sub_industry")["symbol"].apply(list)

    pairs = []
    for sub, members in groups.items():
        if len(members) < 2:
            continue
        for t1, t2 in combinations(sorted(members), 2):
            pairs.append((t1, t2, sub))

    print(f"\nLayer 1 (same sub-industry): {len(pairs):,} pairs "
          f"across {len(groups)} sub-industries")
    return pairs


# ── Layer 2: vectorized correlation matrix ────────────────────────────────────

def layer2_correlation(pairs: list[tuple], closes: pd.DataFrame,
                       window: int = COINT_WINDOW_DAYS) -> list[tuple]:
    """Filter pairs by pairwise log-return correlation using the rolling window."""
    recent = closes.tail(window)
    log_ret = np.log(recent / recent.shift(1)).dropna()

    # Build correlation matrix once — O(n²) but vectorized
    corr = log_ret.corr()

    passed = []
    for t1, t2, sub in pairs:
        if t1 not in corr.columns or t2 not in corr.columns:
            continue
        c = corr.loc[t1, t2]
        if c >= CORR_MIN:
            passed.append((t1, t2, sub, round(c, 4)))

    print(f"Layer 2 (corr ≥ {CORR_MIN}):    {len(passed):,} pairs remain")
    return passed


# ── Layer 2b: SSD pre-filter (Gatev et al. 2006) ────────────────────────────────

def layer2b_ssd(pairs: list[tuple], closes: pd.DataFrame,
               window: int = COINT_WINDOW_DAYS,
               percentile: float = SSD_PERCENTILE) -> list[tuple]:
    """
    Sum of Squared Deviations pre-filter (Gatev, Goetzmann, Rouwenhorst 2006).

    Normalises each price series to 1.0 at the start of the formation window,
    then keeps only pairs whose normalised paths stayed closest together
    (SSD below the `percentile` threshold).

    This is a cheap O(N*T) filter that runs *before* Johansen (O(N*T^2)),
    cutting the most obviously divergent pairs and speeding up Layer 4 by ~5x.

    Note: low SSD is a necessary but NOT sufficient condition for cointegration.
    The full Johansen test in Layer 4 remains the authoritative gate.
    """
    recent = closes.tail(window)

    # Compute SSD for every pair
    scored = []
    for item in pairs:
        t1, t2 = item[0], item[1]
        if t1 not in recent.columns or t2 not in recent.columns:
            continue
        s1 = recent[t1].dropna()
        s2 = recent[t2].dropna()
        # Align on common index
        common = s1.index.intersection(s2.index)
        if len(common) < 30:
            continue
        s1, s2 = s1.loc[common], s2.loc[common]
        # Normalise to 1.0 at start (cumulative total return index)
        p1_norm = s1 / s1.iloc[0]
        p2_norm = s2 / s2.iloc[0]
        ssd = float(((p1_norm - p2_norm) ** 2).sum())
        scored.append((ssd, item))

    if not scored:
        return pairs  # fallback: if something went wrong, don't filter

    # Keep pairs with SSD below the percentile threshold
    ssd_values = [s for s, _ in scored]
    cutoff = float(np.percentile(ssd_values, percentile))
    passed = [item for ssd, item in scored if ssd <= cutoff]

    pct_removed = (1 - len(passed) / max(len(scored), 1)) * 100
    print(f"Layer 2b (SSD ≤ p{percentile:.0f}):   {len(passed):,} pairs remain "
          f"(removed {pct_removed:.0f}% most divergent,  cutoff SSD={cutoff:.2f})")
    return passed


# ── Layer 3: ADF pre-screen (both series must be I(1)) ───────────────────────

def layer3_adf(pairs: list[tuple], closes: pd.DataFrame,
               window: int = COINT_WINDOW_DAYS) -> list[tuple]:
    """
    Keep pairs where BOTH tickers are individually I(1).
    I(1) ↔ ADF p > 0.05 on levels (unit root present).
    Compute once per ticker, not per pair.
    """
    recent = closes.tail(window)
    needed = set(t for t2 in pairs for t in (t2[0], t2[1]))
    needed = needed & set(recent.columns)

    i1 = set()
    for t in needed:
        s = recent[t].dropna()
        if len(s) < 30:
            continue
        p = adfuller(s)[1]
        if p > 0.05:   # fail to reject unit root → I(1)
            i1.add(t)

    passed = [(t1, t2, sub, c) for t1, t2, sub, c in pairs
              if t1 in i1 and t2 in i1]

    print(f"Layer 3 (both I(1)):         {len(passed):,} pairs remain"
          f"  ({len(i1)} I(1) tickers out of {len(needed)} tested)")
    return passed


# ── Helpers ───────────────────────────────────────────────────────────────────

def hurst_exponent(series: np.ndarray, max_lag: int = 100) -> float:
    lags = range(2, min(max_lag, len(series) // 4))
    tau  = [np.std(series[lag:] - series[:-lag]) for lag in lags]
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])


def compute_half_life(spread: pd.Series) -> float:
    aligned = pd.concat([spread.diff(), spread.shift(1)], axis=1).dropna()
    aligned.columns = ["diff", "lag"]
    theta = sm.OLS(aligned["diff"],
                   sm.add_constant(aligned["lag"])).fit().params["lag"]
    return -np.log(2) / theta if theta < 0 else float("inf")


# ── Layer 4+5: Johansen cointegration + spread quality ───────────────────────

# Johansen 95% critical values for k=2 series, det_order=0:
# trace H0(r=0): 15.41  |  max-eigen H0(r=0): 14.07
_JOH_CRIT_IDX = {0.90: 0, 0.95: 1, 0.99: 2}


def _johansen_beta(pc: pd.DataFrame) -> tuple[float, float, float]:
    """
    Run Johansen on a 2-column DataFrame.
    Returns (trace_stat, crit_95, beta) where beta = optimal hedge ratio.

    Eigenvector [e0, e1] defines spread = e0*P1 + e1*P2.
    Normalised to P1 convention: beta = -e1/e0  (long 1 share P1, short beta shares P2).
    """
    res   = coint_johansen(pc.dropna(), det_order=0, k_ar_diff=1)
    trace = float(res.lr1[0])
    crit  = float(res.cvt[0, _JOH_CRIT_IDX[JOH_CRIT]])
    evec  = res.evec[:, 0]          # first (most significant) cointegrating vector
    beta  = -evec[1] / evec[0]      # normalise: spread = P1 - beta * P2
    return trace, crit, float(beta)


def layer45_johansen(pairs: list[tuple], closes: pd.DataFrame,
                     window: int = COINT_WINDOW_DAYS) -> list[dict]:
    """
    Johansen test replaces Engle-Granger here.
    - No Y/X assignment: symmetric by construction.
    - One call returns both test stat AND optimal beta (eigenvector).
    - For 2 assets: cheaper than symmetric EG (2x coint + 2x OLS).
    """
    recent = closes.tail(window)
    results = []
    n = len(pairs)

    for i, (t1, t2, sub, corr) in enumerate(pairs, 1):
        pc = recent[[t1, t2]].dropna()
        if len(pc) < window // 2:
            continue

        try:
            trace, crit, beta = _johansen_beta(pc)
        except Exception:
            continue

        if trace <= crit:          # H0: no cointegration not rejected
            continue

        if beta < 0 or not (BETA_MIN <= abs(beta) <= BETA_MAX):
            continue

        spread    = pc[t1] - beta * pc[t2]
        hurst     = hurst_exponent(spread.values)
        if hurst >= HURST_MAX:
            continue

        half_life = compute_half_life(spread)

        results.append({
            "pair":          f"{t1}-{t2}",
            "t1":            t1,
            "t2":            t2,
            "sub_industry":  sub,
            "corr":          corr,
            "joh_trace":     round(trace, 3),
            "joh_crit_95":   round(crit, 3),
            "joh_margin":    round(trace - crit, 3),   # higher = more confident
            "beta":          round(beta, 4),
            "hurst":         round(hurst, 4),
            "half_life_d":   round(half_life, 1),
            "window_days":   window,
        })

        if i % 20 == 0 or i == n:
            print(f"  Johansen progress: {i}/{n}  ({len(results)} passed)", end="\r")

    print()
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    DATA_DIR.mkdir(exist_ok=True)

    # Layer 0a: sector map
    sector_df = get_sp500_sectors()
    all_symbols = sector_df["symbol"].tolist()

    # Layer 0b: daily closes
    closes = download_universe(all_symbols)

    # Drop tickers with insufficient history
    good = [c for c in closes.columns if closes[c].notna().sum() >= MIN_OBS]
    closes = closes[good].dropna(how="all")
    print(f"Tickers with ≥{MIN_OBS} days history: {len(good)}")

    # Layer 1: same sub-industry
    pairs1 = layer1_sector(good, sector_df)

    # Layer 2: correlation
    pairs2 = layer2_correlation(pairs1, closes)

    # Layer 2b: SSD pre-filter (Gatev et al.) — cheap pass before ADF + Johansen
    pairs2 = layer2b_ssd(pairs2, closes)

    # Layer 3: ADF I(1) pre-screen
    pairs3 = layer3_adf(pairs2, closes)

    # Layers 4+5: Johansen + spread quality
    print(f"\nLayer 4+5 (Johansen @{JOH_CRIT:.0%}, beta, Hurst): "
          f"testing {len(pairs3)} pairs …")
    results = layer45_johansen(pairs3, closes)

    if not results:
        print("\nNo pairs passed all filters.")
        return

    df = (pd.DataFrame(results)
            .sort_values("joh_margin", ascending=False)
            .reset_index(drop=True))

    # ── Sector cap: Top-2 per sub-industry (by joh_margin) ───────────────────
    # Prevents over-concentration: 7 Application Software pairs all blow up
    # together on a NASDAQ shock. Keep only the 2 strongest per sub-industry.
    TOP_PER_SECTOR = 2
    df = df.groupby("sub_industry").head(TOP_PER_SECTOR).reset_index(drop=True)
    print(f"\nAfter sector cap (top {TOP_PER_SECTOR}/sub-industry): {len(df)} pairs")

    df.to_csv(OUT_FILE, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*110}")
    print(df[["pair", "sub_industry", "corr", "joh_trace", "joh_crit_95",
              "joh_margin", "beta", "hurst", "half_life_d"]].to_string(index=False))
    print(f"{'='*110}")
    print(f"\nTotal new pairs found: {len(df)}")
    print(f"Saved → {OUT_FILE}")

    # ── Overlap with existing intraday data ───────────────────────────────────
    existing = set(TICKERS)
    overlap = df[(df["t1"].isin(existing)) & (df["t2"].isin(existing))]
    if len(overlap):
        print(f"\nPairs ALREADY in your intraday data ({len(overlap)}) "
              "→ run pairs.py + backtest immediately:")
        for _, row in overlap.iterrows():
            print(f"  (\"{row['t1']}\", \"{row['t2']}\"),  "
                  f"# {row['sub_industry']}  "
                  f"joh_margin={row['joh_margin']:.2f}  H={row['hurst']:.3f}")
    else:
        print("\nNo overlap with current intraday tickers — "
              "add new pairs to config.py PAIRS then run download.py.")

    # ── New pairs needing download ─────────────────────────────────────────────
    new_pairs = df[~((df["t1"].isin(existing)) & (df["t2"].isin(existing)))]
    if len(new_pairs):
        print(f"\nNew pairs needing intraday download ({len(new_pairs)}):")
        for _, row in new_pairs.head(20).iterrows():
            print(f"  (\"{row['t1']}\", \"{row['t2']}\"),  "
                  f"# {row['sub_industry']}  "
                  f"joh_margin={row['joh_margin']:.2f}  H={row['hurst']:.3f}")
        if len(new_pairs) > 20:
            print(f"  … and {len(new_pairs)-20} more in {OUT_FILE}")

    # ── Half-life filter ──────────────────────────────────────────────────────
    good_hl = df[df["half_life_d"].between(3, 60)]
    print(f"\nPairs with tradeable half-life (3–60 days): {len(good_hl)}")
    for _, row in good_hl.iterrows():
        print(f"  {row['pair']}: joh_margin={row['joh_margin']:.2f}  "
              f"HL={row['half_life_d']}d  H={row['hurst']:.3f}  "
              f"beta={row['beta']}  [{row['sub_industry']}]")


if __name__ == "__main__":
    main()
