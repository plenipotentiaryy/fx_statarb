"""
step1_download.py — Download / backfill OHLCV data from Polygon.

Handles three cases in one pass:
  1. New tickers (not yet in any CSV)      → full download START_DATE → END_DATE
  2. Tickers with insufficient history     → backfill from START_DATE to first_existing_bar
  3. Tickers with a gap in the middle      → flagged in the gap audit at the end

After downloading, writes a per-ticker gap report to data/download_audit.csv.
"""
import os
import time
import threading
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from polygon import RESTClient
from dotenv import load_dotenv
from config import (
    TICKERS, START_DATE, END_DATE, DATA_DIR,
    REQUEST_SLEEP, BAR_MINUTES,
    CLOSES_FILE, VOLUMES_FILE, VWAPS_FILE,
    RTH_START, RTH_END,
)

load_dotenv()

keys_raw = os.getenv("API_KEYS", "")
API_KEYS = [k.strip() for k in keys_raw.split(",") if k.strip()]
if not API_KEYS:
    raise ValueError("No API_KEYS in .env")

clients = [RESTClient(api_key=k) for k in API_KEYS]
DATA_DIR.mkdir(exist_ok=True)

_START_TS  = pd.Timestamp(START_DATE)
_END_TS    = pd.Timestamp(END_DATE)
# Tickers whose history starts later than this threshold need a backfill pass
_BACKFILL_THRESHOLD_DAYS = 90


def parse_bar(bar) -> dict:
    if isinstance(bar, dict):
        return {
            "timestamp": pd.to_datetime(bar["t"], unit="ms"),
            "open": bar["o"], "high": bar["h"], "low": bar["l"],
            "close": bar["c"], "volume": bar["v"],
            "vwap": bar.get("vwap") or bar.get("vwp"),
        }
    ts = bar.timestamp
    return {
        "timestamp": pd.to_datetime(ts, unit="ms") if isinstance(ts, (int, float)) else pd.to_datetime(ts),
        "open": bar.open, "high": bar.high, "low": bar.low,
        "close": bar.close, "volume": bar.volume,
        "vwap": getattr(bar, "vwap", None),
    }


def fetch_range(client: RESTClient, ticker: str,
                from_date: str, to_date: str) -> pd.DataFrame | None:
    """Download all bars for ticker between from_date and to_date (paginates)."""
    bars = []
    page_start = from_date
    end_ts_ms  = int(pd.Timestamp(to_date).timestamp() * 1000)
    while True:
        page = client.get_aggs(
            ticker=ticker,
            multiplier=BAR_MINUTES,
            timespan="minute",
            from_=page_start,
            to=to_date,
            limit=50000,
            sort="asc",
        )
        if not page:
            break
        bars.extend(page)
        last_ts_ms = page[-1].timestamp
        if last_ts_ms >= end_ts_ms or len(page) < 50000:
            break
        # More data available — sleep to respect rate limit, then advance cursor
        time.sleep(REQUEST_SLEEP)
        page_start = last_ts_ms + BAR_MINUTES * 60 * 1000

    if not bars:
        return None
    df = pd.DataFrame([parse_bar(b) for b in bars]).set_index("timestamp")
    return df[~df.index.duplicated(keep="first")]


# ── Build download work list ──────────────────────────────────────────────────
closes_path = DATA_DIR / CLOSES_FILE
existing_closes: pd.DataFrame | None = None

if closes_path.exists():
    existing_closes = pd.read_csv(closes_path, index_col=0, parse_dates=True)
    if existing_closes.index.tz is None:
        existing_closes.index = existing_closes.index.tz_localize("UTC")

existing_cols = set(existing_closes.columns) if existing_closes is not None else set()

_MAX_GAP_BARS = int(3 * 6.5 * 60 / BAR_MINUTES)   # 3 trading days in bars


def _max_rth_gap(col: pd.Series) -> int:
    """Return the length (in bars) of the longest consecutive NaN run inside the
    ticker's valid range (first_valid … last_valid), in RTH hours only."""
    if existing_closes is None:
        return 0
    # Use the raw index — RTH filtering is applied per-col via between_time-like logic
    valid_idx = col.dropna()
    if valid_idx.empty:
        return 0
    window  = col.loc[valid_idx.index[0]:valid_idx.index[-1]]
    max_run = run = 0
    for v in window.isna():
        run     = run + 1 if v else 0
        max_run = max(max_run, run)
    return max_run


# (ticker, from_date_str, to_date_str, mode)
work_list: list[tuple[str, str, str, str]] = []

for ticker in TICKERS:
    if ticker not in existing_cols:
        # Case 1: completely new ticker
        work_list.append((ticker, START_DATE, END_DATE, "new"))
        continue

    col = existing_closes[ticker].dropna()
    if col.empty:
        work_list.append((ticker, START_DATE, END_DATE, "new"))
        continue

    # Case 2: gap in the middle of existing data — full re-download to fill it
    gap_bars = _max_rth_gap(existing_closes[ticker])
    if gap_bars > _MAX_GAP_BARS:
        work_list.append((ticker, START_DATE, END_DATE, "redownload"))
        continue

    # Case 3: history starts too late — backfill from START_DATE
    first_bar_date = pd.Timestamp(col.index[0].date())
    if first_bar_date > _START_TS + pd.Timedelta(days=_BACKFILL_THRESHOLD_DAYS):
        backfill_end = (first_bar_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        work_list.append((ticker, START_DATE, backfill_end, "backfill"))

# Sort: new/redownload first, then backfills
work_list.sort(key=lambda x: (x[3] not in ("new", "redownload"), x[0]))

if not work_list:
    print("All tickers have sufficient history. Nothing to download.")
else:
    new_count        = sum(1 for w in work_list if w[3] == "new")
    backfill_count   = sum(1 for w in work_list if w[3] == "backfill")
    redownload_count = sum(1 for w in work_list if w[3] == "redownload")
    n_keys = len(API_KEYS)
    print(f"Already have: {len(existing_cols)} tickers")
    print(f"New:  {new_count}  |  Backfills: {backfill_count}  |  Re-downloads (gap fix): {redownload_count}")
    print(f"API keys: {n_keys}  →  parallel workers: {n_keys}\n")
    for ticker, f, t, mode in work_list[:10]:
        print(f"  [{mode:10s}] {ticker:6s}  {f} → {t}")
    if len(work_list) > 10:
        print(f"  ... and {len(work_list) - 10} more")
    print()

# ── Parallel download ─────────────────────────────────────────────────────────
all_data: dict[str, tuple[pd.DataFrame, str]] = {}   # ticker → (df, mode)
_lock    = threading.Lock()
_counter = [0]


def download_worker(key_idx: int, items: list) -> None:
    client  = clients[key_idx]
    key_str = API_KEYS[key_idx][:4]
    if key_idx > 0:
        time.sleep(key_idx * (REQUEST_SLEEP / len(clients)))
    for j, (ticker, from_date, to_date, mode) in enumerate(items):
        if j > 0:
            time.sleep(REQUEST_SLEEP)
        with _lock:
            _counter[0] += 1
            pos = _counter[0]
        print(f"  [{pos}/{len(work_list)}] {ticker:6s} [{mode:8s}] "
              f"{from_date} → {to_date}  (key {key_str}...)")
        try:
            df = fetch_range(client, ticker, from_date, to_date)
            if df is None or df.empty:
                print(f"    -> no data")
                continue
            with _lock:
                all_data[ticker] = (df, mode)
            print(f"    -> {len(df)} bars  ({df.index[0].date()} → {df.index[-1].date()})")
        except Exception as e:
            print(f"    -> error: {e}")


if work_list:
    n_keys = len(API_KEYS)
    splits = [work_list[i::n_keys] for i in range(n_keys)]
    with ThreadPoolExecutor(max_workers=n_keys) as pool:
        futures = [pool.submit(download_worker, i, splits[i]) for i in range(n_keys)]
        for f in as_completed(futures):
            f.result()

# ── Merge into existing CSVs ──────────────────────────────────────────────────
if all_data:
    for filename, col_key in [
        (CLOSES_FILE,  "close"),
        (VOLUMES_FILE, "volume"),
        (VWAPS_FILE,   "vwap"),
    ]:
        path = DATA_DIR / filename
        base = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else pd.DataFrame()

        for ticker, (df, mode) in all_data.items():
            if col_key not in df.columns:
                continue
            new_series = df[col_key]
            if ticker in base.columns:
                if mode == "redownload":
                    # Full fresh data — merge new into existing; new wins for overlapping rows
                    base[ticker] = new_series.combine_first(base[ticker])
                elif mode == "backfill":
                    # New data is historical; existing (recent) data wins for any overlap
                    base[ticker] = base[ticker].combine_first(new_series)
                else:
                    base[ticker] = base[ticker].combine_first(new_series)
            else:
                base[ticker] = new_series

        base.sort_index(inplace=True)
        base.to_csv(path)
        print(f"Saved {filename}: {base.shape[1]} tickers, {base.shape[0]} rows")
else:
    print("No new data downloaded — CSVs unchanged.")


# ── Gap audit ─────────────────────────────────────────────────────────────────
print("\n── Gap audit ─────────────────────────────────────────────────────────")
closes = pd.read_csv(closes_path, index_col=0, parse_dates=True)
if closes.index.tz is None:
    closes.index = closes.index.tz_localize("UTC").tz_convert("US/Eastern")
else:
    closes.index = closes.index.tz_convert("US/Eastern")
rth = closes.between_time(RTH_START, RTH_END)

# Build reference calendar from tickers with the most data
ref_ticker = rth.notna().sum().idxmax()
ref_calendar = rth[ref_ticker].dropna().index

audit_rows = []
problems = []
for ticker in TICKERS:
    if ticker not in rth.columns:
        audit_rows.append({"ticker": ticker, "status": "MISSING", "first_bar": None,
                           "last_bar": None, "bars": 0, "nan_pct": 1.0,
                           "gap_days": None})
        problems.append(f"  MISSING  {ticker}")
        continue

    s     = rth[ticker]
    valid = s.dropna()
    if valid.empty:
        audit_rows.append({"ticker": ticker, "status": "EMPTY", "first_bar": None,
                           "last_bar": None, "bars": 0, "nan_pct": 1.0,
                           "gap_days": None})
        problems.append(f"  EMPTY    {ticker}")
        continue

    first_bar = valid.index[0]
    last_bar  = valid.index[-1]

    # Find the longest consecutive NaN gap within [first_bar, last_bar]
    window   = s.loc[first_bar:last_bar]
    nan_mask = window.isna()
    max_gap_bars = 0
    run = 0
    for v in nan_mask:
        run = run + 1 if v else 0
        max_gap_bars = max(max_gap_bars, run)
    bars_per_day = 6.5 * 60 / BAR_MINUTES
    max_gap_days = round(max_gap_bars / bars_per_day, 1)

    # Sparse days: days where the ticker has fewer than 20% of expected bars
    # (catches GOLD-style partial data where NaN gaps are scattered, not consecutive)
    daily_counts  = valid.resample("B").count()
    sparse_days   = int((daily_counts < bars_per_day * 0.2).sum())
    sparse_days_pct = round(sparse_days / max(len(daily_counts), 1), 3)

    nan_pct      = round(nan_mask.mean(), 4)
    history_days = (last_bar - first_bar).days

    status = "OK"
    if max_gap_days > 3:
        status = "GAP"
        problems.append(f"  GAP      {ticker:6s}  longest gap: {max_gap_days:.0f} trading days")
    if sparse_days_pct > 0.05:
        if status == "OK":
            status = "SPARSE"
        problems.append(f"  SPARSE   {ticker:6s}  {sparse_days} sparse days "
                        f"({sparse_days_pct*100:.0f}% of trading days)")
    # SHORT check: only flag if ticker is missing >50% of the expected history
    # AND the gap is not just the plan limit (Polygon Starter covers ~2 years)
    plan_limit_days = 365 * 2 + 90   # ~2.25 years — Starter plan coverage
    expected_days   = (_END_TS - _START_TS).days
    if history_days < min(expected_days * 0.5, plan_limit_days * 0.7):
        if status == "OK":
            status = "SHORT"
        problems.append(f"  SHORT    {ticker:6s}  history: {history_days} calendar days "
                        f"(first bar: {first_bar.date()})")

    audit_rows.append({
        "ticker":           ticker,
        "status":           status,
        "first_bar":        str(first_bar.date()),
        "last_bar":         str(last_bar.date()),
        "bars":             len(valid),
        "nan_pct":          nan_pct,
        "max_gap_days":     max_gap_days,
        "sparse_days_pct":  sparse_days_pct,
    })

audit_df = pd.DataFrame(audit_rows).set_index("ticker")
audit_df.to_csv(DATA_DIR / "download_audit.csv")

ok_count  = (audit_df["status"] == "OK").sum()
print(f"Tickers OK:      {ok_count} / {len(TICKERS)}")
if problems:
    print(f"Issues ({len(problems)}):")
    for p in problems:
        print(p)
else:
    print("No issues detected.")
print(f"\nFull report saved to {DATA_DIR / 'download_audit.csv'}")
