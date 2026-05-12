"""
step0_download.py — Two-stage data pipeline.

Stage 1  Daily closes via yfinance (10 years, FREE)
         → data/closes_daily.csv
         Source : Yahoo Finance
         Purpose: cointegration pair selection in step2

Stage 2  Intraday 15-min via Polygon (~2 years, paid Starter plan)
         → data/closes_15min.csv
         Source : Polygon.io REST API
         Purpose: z-score signals, backtest, regime detection

Run order:  step0  →  step2  →  step4  →  step5
"""

import os
import time
import threading
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from config import (
    TICKERS, DAILY_START, END_DATE, START_DATE,
    DATA_DIR, REQUEST_SLEEP, BAR_MINUTES,
    CLOSES_FILE, VOLUMES_FILE, VWAPS_FILE,
    RTH_START, RTH_END,
)

load_dotenv()
DATA_DIR.mkdir(exist_ok=True)

DAILY_CACHE = DATA_DIR / "closes_daily.csv"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Daily closes via yfinance (10 years)
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("STAGE 1  Daily closes — yfinance (10 years)")
print(f"         {DAILY_START} → today")
print("=" * 60)

refresh_daily = True
if DAILY_CACHE.exists():
    existing_daily = pd.read_csv(DAILY_CACHE, index_col=0, parse_dates=True)
    first_bar = pd.Timestamp(existing_daily.index[0].date())
    age_hours = (pd.Timestamp.now() -
                 pd.Timestamp(DAILY_CACHE.stat().st_mtime, unit="s")).total_seconds() / 3600
    target_start = pd.Timestamp(DAILY_START)

    if first_bar <= target_start + pd.Timedelta(days=30) and age_hours < 12:
        print(f"Cache OK  ({len(existing_daily)} rows × {existing_daily.shape[1]} tickers  "
              f"| {existing_daily.index[0].date()} → {existing_daily.index[-1].date()}  "
              f"| {age_hours:.0f}h old)")
        refresh_daily = False
    elif first_bar > target_start + pd.Timedelta(days=30):
        print(f"Cache starts {first_bar.date()} — extending back to {DAILY_START} …")
    else:
        print(f"Cache {age_hours:.0f}h old — refreshing …")

if refresh_daily:
    print(f"Downloading {len(TICKERS)} tickers from Yahoo Finance …", flush=True)
    try:
        raw = yf.download(
            TICKERS,
            start=DAILY_START,
            auto_adjust=True,
            progress=True,
            threads=True,
        )
        closes_daily = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        closes_daily = closes_daily.dropna(how="all")

        # Report failed tickers
        missing = [t for t in TICKERS if t not in closes_daily.columns]
        if missing:
            print(f"\nNot found on Yahoo ({len(missing)}): {missing}")

        closes_daily.to_csv(DAILY_CACHE)
        print(f"\nSaved {DAILY_CACHE.name}: "
              f"{len(closes_daily)} rows × {closes_daily.shape[1]} tickers  "
              f"| {closes_daily.index[0].date()} → {closes_daily.index[-1].date()}")
    except Exception as e:
        print(f"yfinance error: {e}")
        if DAILY_CACHE.exists():
            print("Using existing cache.")
        else:
            raise SystemExit("No daily data available.")
else:
    closes_daily = existing_daily

# Coverage summary
n_tickers      = closes_daily.shape[1]
n_years        = (closes_daily.index[-1] - closes_daily.index[0]).days / 365.25
nan_pct        = closes_daily.isna().mean()
good_tickers   = (nan_pct < 0.05).sum()
short_tickers  = ((closes_daily.notna().apply(lambda s: s.idxmax()) >
                   pd.Timestamp(DAILY_START) + pd.Timedelta(days=365))
                  .sum())

print(f"\nDaily coverage:")
print(f"  Tickers total  : {n_tickers}")
print(f"  Full history   : {good_tickers} tickers (< 5% NaN)")
print(f"  Years          : {n_years:.1f}")
print(f"  Short history  : {short_tickers} tickers (listed after {DAILY_START}+1y)")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Intraday 15-min via Polygon (audit existing data)
# ══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 60)
print(f"STAGE 2  Intraday {BAR_MINUTES}-min — Polygon (~2 years)")
print(f"         {START_DATE} → {END_DATE}")
print("=" * 60)

intraday_path = DATA_DIR / CLOSES_FILE
if not intraday_path.exists():
    print(f"\n{CLOSES_FILE} not found.")
    print("Run  step1_download.py  to download intraday data from Polygon.")
else:
    closes_id = pd.read_csv(intraday_path, index_col=0, parse_dates=True)
    if closes_id.index.tz is None:
        closes_id.index = closes_id.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        closes_id.index = closes_id.index.tz_convert("US/Eastern")
    rth = closes_id.between_time(RTH_START, RTH_END)

    bars_per_day  = 6.5 * 60 / BAR_MINUTES
    n_rows        = len(rth)
    n_cols        = rth.shape[1]
    date_range    = f"{rth.index[0].date()} → {rth.index[-1].date()}"
    cal_days      = (rth.index[-1] - rth.index[0]).days
    trading_days  = round(n_rows / bars_per_day)

    print(f"\nFile    : {CLOSES_FILE}")
    print(f"Shape   : {n_rows} RTH bars × {n_cols} tickers")
    print(f"Period  : {date_range}  ({cal_days} calendar days)")
    print(f"Trading : ~{trading_days} trading days  (~{cal_days/365:.1f} years)")

    # Per-ticker gap audit
    problems = []
    ok_count = 0
    for ticker in TICKERS:
        if ticker not in rth.columns:
            problems.append(("MISSING", ticker, "-", "-", 0))
            continue
        s = rth[ticker]
        valid = s.dropna()
        if valid.empty:
            problems.append(("EMPTY", ticker, "-", "-", 0))
            continue
        window   = s.loc[valid.index[0]:valid.index[-1]]
        nan_mask = window.isna()
        max_run = run = 0
        for v in nan_mask:
            run = run + 1 if v else 0
            max_run = max(max_run, run)
        max_gap_d = max_run / bars_per_day
        daily_c   = valid.resample("B").count()
        sparse    = int((daily_c < bars_per_day * 0.2).sum())

        if max_gap_d > 3:
            problems.append(("GAP", ticker, str(valid.index[0].date()),
                             str(valid.index[-1].date()), round(max_gap_d, 1)))
        elif sparse / max(len(daily_c), 1) > 0.05:
            problems.append(("SPARSE", ticker, str(valid.index[0].date()),
                             str(valid.index[-1].date()), sparse))
        else:
            ok_count += 1

    print(f"\nIntraday quality:")
    print(f"  OK      : {ok_count} tickers")
    if problems:
        print(f"  Issues  : {len(problems)}")
        for status, t, f, l, v in problems:
            label = f"gap={v}d" if status == "GAP" else (f"sparse={v}d" if status == "SPARSE" else "")
            print(f"    {status:<8} {t:<6}  {f} → {l}  {label}")
    else:
        print(f"  No issues detected.")

    # Tickers in daily but not intraday
    only_daily = [t for t in TICKERS
                  if t in closes_daily.columns and t not in rth.columns]
    if only_daily:
        print(f"\n  Tickers in daily but missing intraday ({len(only_daily)}): {only_daily}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)

daily_ok   = DAILY_CACHE.exists()
intraday_ok = intraday_path.exists()

print(f"\n  Daily   (Yahoo, {BAR_MINUTES=}d)  : "
      f"{'✓' if daily_ok else '✗'}  "
      f"{n_years:.1f} years  →  pair SELECTION in step2")
print(f"  Intraday (Polygon, {BAR_MINUTES}min): "
      f"{'✓' if intraday_ok else '✗'}  "
      f"~{cal_days/365:.1f} years  →  SIGNALS & BACKTEST in step4/5")

print()
if daily_ok and intraday_ok:
    print("  Both data sources ready.  Next:  python RUN_ALL.py 2a 4a 4b 4c 4d 4e 5a 6e 6d")
elif daily_ok:
    print("  Daily ready.  Run step1_download.py to get intraday from Polygon.")
else:
    print("  Something missing — check errors above.")
