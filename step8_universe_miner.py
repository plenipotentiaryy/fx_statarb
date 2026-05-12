"""
step8_universe_miner.py — Institutional-Grade Pair Mining Engine.

Mining Strategy:
1. Universe: S&P 500 Tickers + Sector ETFs.
2. Layer 0: Correlation Filter (Matrix-based, fast).
3. Layer 1: SSD (Sum of Squared Deviations) Filter.
4. Layer 2: Johansen Cointegration (Parallelized).
5. Layer 3: Quality Gates (Beta, Hurst, Half-life).

This script finds cointegrated pairs across DIFFERENT sectors to maximize diversification.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import ssl
import urllib.request
from io import StringIO
import yfinance as yf
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm

# Import from config if available, else use defaults
try:
    from config import DATA_DIR, DAILY_START, COINT_WINDOW_DAYS
except ImportError:
    DATA_DIR = Path("data")
    DAILY_START = "2015-01-01"
    COINT_WINDOW_DAYS = 252 * 2

# --- Mining Parameters ---
CORR_MIN = 0.50        # Higher threshold for cross-sector search to reduce noise
SSD_PERCENTILE = 20    # Keep only the top 20% closest pairs (Gatev et al.)
JOH_CRIT = 0.95        # 95% Confidence
BETA_MIN = 0.20
BETA_MAX = 5.0
HURST_MAX = 0.65       # Tier-1 relaxed threshold: allow borderline mean-reverting pairs
HL_MAX_DAYS = 40       # Fast mean-reversion preferred
MAX_PAIRS = 200        # Final target size

ETFS = ["XLF", "XLK", "XLE", "XLV", "XLY", "XLP", "XLU", "XLB", "XLI", "XLC"]
SP500_CACHE = DATA_DIR / "sp500_sectors.csv"
DAILY_UNIV = DATA_DIR / "closes_daily_universe.csv"
OUT_FILE = DATA_DIR / "pairs_mined.csv"

def get_sp500_tickers():
    if SP500_CACHE.exists():
        return pd.read_csv(SP500_CACHE)["symbol"].tolist()
    
    print("Fetching S&P 500 tickers...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, context=ctx) as resp:
            html = resp.read().decode("utf-8")
        tables = pd.read_html(StringIO(html))
        sp = tables[0].copy()
        sp.columns = [c.strip() for c in sp.columns]
        symbol_col = [c for c in sp.columns if "Symbol" in c][0]
        tickers = sp[symbol_col].str.replace(".", "-", regex=False).tolist()
        # Cache it
        pd.DataFrame({"symbol": tickers}).to_csv(SP500_CACHE, index=False)
        return tickers
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "JPM", "V", "MA"]

def download_data(tickers):
    if DAILY_UNIV.exists():
        age = (pd.Timestamp.now() - pd.Timestamp(DAILY_UNIV.stat().st_mtime, unit="s")).days
        if age < 7:
            df = pd.read_csv(DAILY_UNIV, index_col=0, parse_dates=True)
            # Filter if tickers changed
            existing = set(df.columns)
            if all(t in existing for t in tickers):
                return df
            
    print(f"Downloading daily data for {len(tickers)} tickers...")
    df = yf.download(tickers, start=DAILY_START, auto_adjust=True, progress=True)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    df = df.dropna(how="all").ffill()
    df.to_csv(DAILY_UNIV)
    return df

def compute_half_life(spread: pd.Series) -> float:
    try:
        aligned = pd.concat([spread.diff(), spread.shift(1)], axis=1).dropna()
        aligned.columns = ["diff", "lag"]
        theta = sm.OLS(aligned["diff"], sm.add_constant(aligned["lag"])).fit().params["lag"]
        return -np.log(2) / theta if theta < 0 else 999.0
    except:
        return 999.0

def hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    try:
        lags = range(2, max_lag)
        tau = [np.std(series[lag:] - series[:-lag]) for lag in lags]
        return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])
    except:
        return 0.5

def check_pair(t1, t2, closes_subset):
    """Worker function for parallel Johansen test."""
    pc = closes_subset[[t1, t2]].dropna()
    if len(pc) < 252: return None
    
    try:
        res = coint_johansen(pc, det_order=0, k_ar_diff=1)
        trace = float(res.lr1[0])
        crit = float(res.cvt[0, 1]) # 95%
        if trace <= crit: return None
        
        evec = res.evec[:, 0]
        beta = -evec[1] / evec[0]
        if beta <= 0 or not (BETA_MIN <= beta <= BETA_MAX): return None
        
        spread = pc[t1] - beta * pc[t2]
        hl = compute_half_life(spread)
        if hl > HL_MAX_DAYS: return None
        
        h = hurst_exponent(spread.values)
        if h > HURST_MAX: return None
        
        return {
            "pair": f"{t1}-{t2}",
            "t1": t1, "t2": t2,
            "beta": round(beta, 4),
            "joh_trace": round(trace, 2),
            "half_life": round(hl, 1),
            "hurst": round(h, 3)
        }
    except:
        return None

def main():
    DATA_DIR.mkdir(exist_ok=True)
    tickers = sorted(list(set(get_sp500_tickers() + ETFS)))
    closes = download_data(tickers)
    
    # Pre-filter tickers with low data
    valid_tickers = [c for c in closes.columns if closes[c].notna().sum() > 504]
    mining_window = 504 # 2 years for robust mining
    closes_subset = closes[valid_tickers].tail(mining_window)
    print(f"Mining across {len(valid_tickers)} tickers using {mining_window}-day window...")
    
    # 1. Correlation Matrix (Fast)
    print("Calculating correlation matrix...")
    corr = closes_subset.pct_change().corr()
    pairs = []
    for i in range(len(valid_tickers)):
        for j in range(i + 1, len(valid_tickers)):
            t1, t2 = valid_tickers[i], valid_tickers[j]
            c = corr.loc[t1, t2]
            if c >= CORR_MIN:
                pairs.append((t1, t2, c))
    
    print(f"Layer 1: {len(pairs):,} pairs passed Correlation gate (Corr >= {CORR_MIN})")
    
    # 2. SSD Filter (Top 30% for broader search)
    print("Applying SSD filter...")
    ssd_percentile = 30
    ssd_scores = []
    for t1, t2, c in pairs:
        s1 = closes_subset[t1] / closes_subset[t1].iloc[0]
        s2 = closes_subset[t2] / closes_subset[t2].iloc[0]
        ssd = float(((s1 - s2)**2).sum())
        ssd_scores.append((ssd, t1, t2))
    
    cutoff = np.percentile([s for s, t1, t2 in ssd_scores], ssd_percentile)
    pairs_to_test = [(t1, t2) for s, t1, t2 in ssd_scores if s <= cutoff]
    print(f"Layer 2: {len(pairs_to_test):,} pairs passed SSD gate (Top {ssd_percentile}%)")
    
    # 3. Johansen Parallel Test
    print(f"Layer 3: Testing {len(pairs_to_test)} pairs via Johansen (Parallel)...")
    results = []
    
    # Limit number of workers to avoid memory issues
    num_workers = min(mp.cpu_count(), 8)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use subset of closes for workers to reduce serialization overhead
        futures = [executor.submit(check_pair, t1, t2, closes_subset[[t1, t2]]) for t1, t2 in pairs_to_test]
        for i, f in enumerate(futures):
            res = f.result()
            if res: 
                results.append(res)
            if i % 100 == 0: 
                print(f"  Progress: {i}/{len(pairs_to_test)} (Found {len(results)})", end="\r")
            
    print(f"\nFound {len(results)} cointegrated pairs.")
    
    if not results:
        print("No pairs found. Try relaxing filters.")
        return

    # 4. Save and Summary
    df = pd.DataFrame(results).sort_values("joh_trace", ascending=False).head(MAX_PAIRS)
    df.to_csv(OUT_FILE, index=False)
    print(f"Saved Top {len(df)} pairs to {OUT_FILE}")
    print("\nSample Mined Pairs:")
    print(df.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
