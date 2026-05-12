"""
step1_redownload.py — Докачка тикеров, у которых данные оборвались раньше 2026-03-01.
Определяет автоматически: смотрит closes_5min.csv, находит последний бар каждого тикера,
и скачивает только недостающий кусок (от last_date+1 до END_DATE).
Потом сливает новые данные в существующие CSV.
"""

import os
import time
import threading
import pandas as pd
from polygon import RESTClient
from dotenv import load_dotenv
from config import END_DATE, DATA_DIR, REQUEST_SLEEP, BAR_MINUTES, CLOSES_FILE, VOLUMES_FILE, VWAPS_FILE

load_dotenv()

CUTOFF = "2026-03-01"  # тикеры с данными до этой даты будут докачаны

keys_raw = os.getenv("API_KEYS", "")
API_KEYS = [k.strip() for k in keys_raw.split(",") if k.strip()]
if not API_KEYS:
    raise ValueError("No API_KEYS in .env")

clients = [RESTClient(api_key=k) for k in API_KEYS]


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


# ── Определяем что нужно докачать ────────────────────────────────────────────
closes_path = DATA_DIR / CLOSES_FILE
if not closes_path.exists():
    print(f"Файл {CLOSES_FILE} не найден — сначала запусти step1_download.py")
    raise SystemExit(1)

closes = pd.read_csv(closes_path, index_col=0, parse_dates=True)

to_download = []  # (ticker, start_date_str)
for col in closes.columns:
    s = closes[col].dropna()
    if len(s) == 0:
        continue
    last = s.index[-1]
    if last < pd.Timestamp(CUTOFF):
        # Начинаем с дня ПОСЛЕ последнего бара
        start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        to_download.append((col, start))

if not to_download:
    print("Все тикеры актуальны — докачка не нужна!")
    raise SystemExit(0)

to_download.sort(key=lambda x: x[1])

n_keys = len(API_KEYS)
est_min = len(to_download) * REQUEST_SLEEP // 60 // max(n_keys, 1)
print(f"Тикеров к докачке: {len(to_download)}")
print(f"API keys: {n_keys}  →  parallel workers: {n_keys}")
print(f"Estimated: ~{est_min} min\n")

for t, start in to_download[:5]:
    print(f"  {t:6s}: данные до обрыва, докачка с {start}")
print(f"  ... и ещё {len(to_download)-5}\n")


# ── Параллельная докачка ─────────────────────────────────────────────────────
all_data: dict = {}
_lock = threading.Lock()
_counter = [0]


def download_worker(key_idx: int, ticker_list: list) -> None:
    client = clients[key_idx]
    key_str = API_KEYS[key_idx][:4]
    for j, (ticker, start_date) in enumerate(ticker_list):
        if j > 0:
            time.sleep(REQUEST_SLEEP)
        with _lock:
            _counter[0] += 1
            pos = _counter[0]
        print(f"  [{pos}/{len(to_download)}] {ticker}  {start_date} → {END_DATE}  (key {key_str}...)")
        try:
            bars = []
            page_start = start_date
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
                if len(page) < 50000:
                    break
                time.sleep(REQUEST_SLEEP)
                last_ts = page[-1].timestamp
                page_start = last_ts + BAR_MINUTES * 60 * 1000

            if not bars:
                print(f"    -> no new data")
                continue
            df = pd.DataFrame([parse_bar(b) for b in bars]).set_index("timestamp")
            df = df[~df.index.duplicated(keep="first")]
            with _lock:
                all_data[ticker] = df
            print(f"    -> {len(df)} new bars")
        except Exception as e:
            print(f"    -> error: {e}")


# Разбиваем по ключам
from concurrent.futures import ThreadPoolExecutor, as_completed

splits = [to_download[i::n_keys] for i in range(n_keys)]

with ThreadPoolExecutor(max_workers=n_keys) as pool:
    futures = [pool.submit(download_worker, i, splits[i]) for i in range(n_keys)]
    for f in as_completed(futures):
        f.result()

if not all_data:
    print("\nНичего нового не скачалось.")
    raise SystemExit(0)


# ── Сливаем с существующими данными ──────────────────────────────────────────
for filename, field in [
    (CLOSES_FILE,  "close"),
    (VOLUMES_FILE, "volume"),
    (VWAPS_FILE,   "vwap"),
]:
    path = DATA_DIR / filename
    existing = pd.read_csv(path, index_col=0, parse_dates=True)

    new_cols = {}
    for ticker, df in all_data.items():
        if field in df.columns:
            new_cols[ticker] = df[field]

    if not new_cols:
        continue

    new_df = pd.DataFrame(new_cols)

    # Для каждого докачанного тикера: склеиваем старые + новые данные
    for ticker in new_df.columns:
        if ticker in existing.columns:
            combined = pd.concat([existing[ticker], new_df[ticker]])
            combined = combined[~combined.index.duplicated(keep="first")]
            existing[ticker] = combined.sort_index()
        else:
            existing = existing.join(new_df[[ticker]], how="outer")

    existing = existing.sort_index()
    existing.to_csv(path)
    print(f"Updated {filename}: {existing.shape[1]} tickers, {existing.shape[0]} rows")

print(f"\nГотово! Докачано {len(all_data)} тикеров.")
print("Теперь запусти step2_pairs.py заново.")
