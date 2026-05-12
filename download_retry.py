"""Re-download tickers whose 5-min data ends before a cutoff date.

Fetches only the missing tail (from last_bar + 5min to END_DATE) for each
incomplete ticker, then merges the new rows into the existing CSV files.
"""
import os
import time
import threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from polygon import RESTClient
from dotenv import load_dotenv
from config import END_DATE, DATA_DIR, REQUEST_SLEEP, BAR_MINUTES, CLOSES_FILE, VOLUMES_FILE, VWAPS_FILE

load_dotenv()

keys_raw = os.getenv("API_KEYS", "")
API_KEYS = [k.strip() for k in keys_raw.split(",") if k.strip()]
if not API_KEYS:
    raise ValueError("No API_KEYS in .env")

clients = [RESTClient(api_key=k) for k in API_KEYS]

CUTOFF = pd.Timestamp("2026-04-01")   # tickers ending before this date need re-download


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


# ── Find tickers needing re-download ─────────────────────────────────────────
closes_path = DATA_DIR / CLOSES_FILE
if not closes_path.exists():
    raise FileNotFoundError(f"No existing data: {closes_path}")

existing_closes = pd.read_csv(closes_path, index_col=0, parse_dates=True)

to_retry: list[tuple[str, str]] = []   # (ticker, fetch_from_date_str)
for col in existing_closes.columns:
    valid = existing_closes[col].dropna()
    if valid.empty:
        continue
    last_ts = valid.index[-1]
    if last_ts < CUTOFF:
        # Start just after the last known bar
        fetch_from = last_ts + pd.Timedelta(minutes=BAR_MINUTES)
        to_retry.append((col, fetch_from.strftime("%Y-%m-%d")))

if not to_retry:
    print("All tickers are up to date.")
    raise SystemExit(0)

n_keys = len(API_KEYS)
print(f"Tickers needing retry: {len(to_retry)}")
print(f"API keys: {n_keys}  →  parallel workers: {n_keys}")
print(f"Sleep per key: {REQUEST_SLEEP}s\n")
for t, d in to_retry[:10]:
    print(f"  {t} from {d}")
if len(to_retry) > 10:
    print(f"  ... and {len(to_retry) - 10} more")
print()

# ── Parallel download ─────────────────────────────────────────────────────────
all_data: dict = {}
_lock = threading.Lock()
_counter = [0]


def download_worker(key_idx: int, ticker_list: list[tuple[str, str]]) -> None:
    client  = clients[key_idx]
    key_str = API_KEYS[key_idx][:4]
    # Stagger workers so they never fire at the same moment
    if key_idx > 0:
        time.sleep(key_idx * (REQUEST_SLEEP / len(clients)))
    for j, (ticker, fetch_from) in enumerate(ticker_list):
        if j > 0:
            time.sleep(REQUEST_SLEEP)
        with _lock:
            _counter[0] += 1
            pos = _counter[0]
        print(f"  [{pos}/{len(to_retry)}] {ticker} from {fetch_from}  (key {key_str}...)")
        try:
            bars = []
            page_start  = fetch_from
            end_ts_ms   = int(pd.Timestamp(END_DATE).timestamp() * 1000)
            while True:
                page = client.get_aggs(
                    ticker=ticker,
                    multiplier=BAR_MINUTES,
                    timespan="minute",
                    from_=page_start,
                    to=END_DATE,
                    limit=50000,
                    sort="asc",
                )
                if not page:
                    break
                bars.extend(page)
                last_ts_ms = page[-1].timestamp
                if last_ts_ms >= end_ts_ms:
                    break
                time.sleep(REQUEST_SLEEP)
                page_start = last_ts_ms + BAR_MINUTES * 60 * 1000

            if not bars:
                print(f"    -> no new data")
                continue
            df = pd.DataFrame([parse_bar(b) for b in bars]).set_index("timestamp")
            df = df[~df.index.duplicated(keep="first")]
            with _lock:
                all_data[ticker] = df
            print(f"    -> {len(df)} new bars  (up to {df.index[-1].date()})")
        except Exception as e:
            print(f"    -> error: {e}")


splits = [to_retry[i::n_keys] for i in range(n_keys)]

with ThreadPoolExecutor(max_workers=n_keys) as pool:
    futures = [pool.submit(download_worker, i, splits[i]) for i in range(n_keys)]
    for f in as_completed(futures):
        f.result()

if not all_data:
    print("No new data downloaded.")
    raise SystemExit(0)

# ── Merge new tail with existing data ────────────────────────────────────────
print(f"\nMerging {len(all_data)} tickers into existing files…")

for filename, col in [(CLOSES_FILE, "close"), (VOLUMES_FILE, "volume"), (VWAPS_FILE, "vwap")]:
    path = DATA_DIR / filename
    if not path.exists():
        print(f"  SKIP {filename}: file not found")
        continue
    existing = pd.read_csv(path, index_col=0, parse_dates=True)
    for ticker, df in all_data.items():
        if col not in df.columns:
            continue
        new_series = df[col]
        if ticker in existing.columns:
            existing[ticker] = existing[ticker].combine_first(new_series)
        else:
            existing[ticker] = new_series
    existing.sort_index(inplace=True)
    existing.to_csv(path)
    print(f"  Saved {filename}: {existing.shape}")

print(f"\nDone. Re-downloaded {len(all_data)} tickers.")

# ── Quick verification ────────────────────────────────────────────────────────
updated = pd.read_csv(closes_path, index_col=0, parse_dates=True)
still_short = []
for t, _ in to_retry:
    if t in updated.columns:
        v = updated[t].dropna()
        if not v.empty and v.index[-1] < CUTOFF:
            still_short.append((t, v.index[-1].date(), len(v)))

if still_short:
    print(f"\nWARNING: {len(still_short)} tickers still have short data:")
    for t, end, n in still_short:
        print(f"  {t}: ends {end} ({n} bars)")
else:
    print(f"\nAll {len(to_retry)} tickers now have data through at least {CUTOFF.date()}.")
