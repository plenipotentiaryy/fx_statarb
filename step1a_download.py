"""
download_av.py — Alpha Vantage intraday downloader (15-min, resumable).

Architecture:
  - Each (ticker, month) slice saved as data/av_cache/TICKER_YYYY-MM.csv
  - File existence = success marker → restart resumes from where it stopped
  - Rate limiter: 75 req/min (premium) = 0.82s between calls
  - Exponential backoff: 3 retries with 2s / 4s / 8s delays on failure
  - After all downloads: merges into closes_15min.csv + volumes_15min.csv
"""

import os
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from config import TICKERS, DATA_DIR, RTH_START, RTH_END, BAR_MINUTES

load_dotenv()

API_KEY    = os.getenv("ALPHAVANTAGE_API_KEY", "")
if not API_KEY:
    raise ValueError("ALPHAVANTAGE_API_KEY not set in .env")

INTERVAL   = f"{BAR_MINUTES}min"
CACHE_DIR  = DATA_DIR / "av_cache"
FAILED_LOG = DATA_DIR / "av_failed.log"
CLOSES_OUT = DATA_DIR / f"closes_{BAR_MINUTES}min.csv"
VOLUMES_OUT = DATA_DIR / f"volumes_{BAR_MINUTES}min.csv"

HISTORY_YEARS = 20  # full 20-year history for robust research
REQ_PER_MIN   = 75
SLEEP_S       = 60.0 / REQ_PER_MIN   # 0.80s between requests
MAX_RETRIES   = 3
MAX_THREADS   = 5   # parallel downloads to saturate the network/API bandwidth

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Month slice list ──────────────────────────────────────────────────────────

def month_slices(years: int = HISTORY_YEARS) -> list[str]:
    """Return list of 'YYYY-MM' strings from (now - years) to last month."""
    end   = datetime.today().replace(day=1) - timedelta(days=1)   # last day of prev month
    start = end - relativedelta(years=years)
    start = start.replace(day=1)
    slices = []
    cur = start
    while cur <= end:
        slices.append(cur.strftime("%Y-%m"))
        cur += relativedelta(months=1)
    return slices


# ── Single-slice download ─────────────────────────────────────────────────────

def fetch_slice_v2(ticker: str, month: str) -> pd.DataFrame | str | None:
    """
    Download one month of 15-min bars for a ticker.
    Returns DataFrame, "INVALID_API_CALL" if pre-IPO, or None on failure.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_INTRADAY"
        f"&symbol={ticker}"
        f"&interval={INTERVAL}"
        f"&month={month}"
        f"&outputsize=full"
        f"&datatype=csv"
        f"&apikey={API_KEY}"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()

            if resp.text.startswith("{"):
                msg = resp.json()
                info = (msg.get("Information") or msg.get("Note") or msg.get("Error Message") or str(msg)).lower()
                
                if "invalid api call" in info:
                    return "INVALID_API_CALL"
                
                if "premium" in info or "call frequency" in info:
                    log.warning(f"  Rate limit hit — sleeping 60s")
                    time.sleep(60)
                    continue
                
                log.warning(f"  {ticker} {month}: API message: {info[:120]}")
                return None

            from io import StringIO
            df = pd.read_csv(StringIO(resp.text), parse_dates=["timestamp"])
            if df.empty:
                return None
            return df

        except Exception as e:
            wait = 2 ** attempt
            log.warning(f"  {ticker} {month} attempt {attempt}/{MAX_RETRIES}: {e} — retry in {wait}s")
            time.sleep(wait)

    return None


def download_all(tickers: list[str], slices: list[str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    total    = len(tickers) * len(slices)
    done     = 0
    skipped  = 0
    failed   = []

    log.info(f"Tickers: {len(tickers)}  Months: {len(slices)}  "
             f"Total requests: {total}  ETA: ~{total / REQ_PER_MIN:.0f} min")

    for ticker in tickers:
        skip_until_year = 0
        for i, month in enumerate(slices):
            year = int(month.split("-")[0])
            if year < skip_until_year:
                done += 1
                skipped += 1
                continue

            cache_file = CACHE_DIR / f"{ticker}_{month}.csv"
            if cache_file.exists():
                skipped += 1
                done += 1
                continue

            df_or_err = fetch_slice_v2(ticker, month)
            time.sleep(SLEEP_S)
            done += 1
            pct = done / total * 100

            if isinstance(df_or_err, str) and df_or_err == "INVALID_API_CALL":
                # Ticker likely didn't exist yet. Skip 12 months ahead to find IPO faster.
                log.info(f"  EMPTY {ticker} {month} (Pre-IPO?) -> Skipping 12 months forward...")
                skip_until_year = year + 1
                continue
            
            if df_or_err is None:
                failed.append(f"{ticker}_{month}")
                log.warning(f"  FAIL  {ticker} {month}  [{done}/{total} {pct:.1f}%]")
                continue

            df_or_err.to_csv(cache_file, index=False)
            log.info(f"  OK    {ticker} {month}  {len(df_or_err):4d} bars  [{done}/{total} {pct:.1f}%]  skip={skipped}")

    log.info(f"\nDownload complete. Failed: {len(failed)}")
    if failed:
        log.info(f"  Failed slices logged to {FAILED_LOG}")


# ── Merge slices → CSV + Parquet ─────────────────────────────────────────────

def merge_to_csv(tickers: list[str], rebuild: bool = False) -> None:
    """
    Merge av_cache slices into columnar closes/volumes files.

    rebuild=True  → ignore existing files, rebuild everything from scratch.
                    Use this after a full download to get the complete 20-year matrix.
    rebuild=False → merge new tickers into existing files (incremental update).
    """
    log.info(f"\nMerging slices into CSV + Parquet  (rebuild={rebuild}) …")

    all_closes:  dict[str, pd.Series] = {}
    all_volumes: dict[str, pd.Series] = {}

    for ticker in tickers:
        files = sorted(CACHE_DIR.glob(f"{ticker}_*.csv"))
        if not files:
            log.warning(f"  No slices for {ticker} — skipping")
            continue

        frames = []
        for f in files:
            try:
                frames.append(pd.read_csv(f, parse_dates=["timestamp"]))
            except Exception as e:
                log.warning(f"  Bad slice {f.name}: {e}")

        if not frames:
            continue

        df = (pd.concat(frames, ignore_index=True)
                .drop_duplicates("timestamp")
                .sort_values("timestamp")
                .set_index("timestamp"))

        if df.index.tz is None:
            df.index = df.index.tz_localize("US/Eastern")
        else:
            df.index = df.index.tz_convert("US/Eastern")

        df = df.between_time(RTH_START, RTH_END)

        all_closes[ticker]  = df["close"]
        all_volumes[ticker] = df["volume"]
        log.info(f"  {ticker}: {len(df):,} bars")

    if not all_closes:
        log.error("No data to merge.")
        return

    closes_df  = pd.DataFrame(all_closes)
    volumes_df = pd.DataFrame(all_volumes)

    for out_csv, new_df in [(CLOSES_OUT, closes_df), (VOLUMES_OUT, volumes_df)]:
        if rebuild or not out_csv.exists():
            final_df = new_df
        else:
            existing = pd.read_csv(out_csv, index_col=0, parse_dates=True)
            if existing.index.tz is None:
                existing.index = existing.index.tz_localize("US/Eastern")
            for col in new_df.columns:
                existing[col] = new_df[col]
            final_df = existing

        # ── CSV (human-readable backup) ───────────────────────────────────
        final_df.to_csv(out_csv)
        log.info(f"  CSV  → {out_csv}  "
                 f"({len(final_df):,} rows × {len(final_df.columns)} tickers)")

        # ── Parquet (fast columnar format for data_loader.py) ─────────────
        out_parquet = out_csv.with_suffix(".parquet")
        final_df.to_parquet(out_parquet, engine="pyarrow",
                            compression="snappy", index=True)
        size_mb = out_parquet.stat().st_size / 1e6
        log.info(f"  Parquet → {out_parquet}  ({size_mb:.1f} MB)")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    rebuild = "--rebuild" in sys.argv

    slices = month_slices(HISTORY_YEARS)
    log.info(f"Date range: {slices[0]} → {slices[-1]}  ({len(slices)} months)")

    download_all(TICKERS, slices)
    merge_to_csv(TICKERS, rebuild=rebuild)

    log.info("\nDone. Run: python pairs.py → backtest.py")
    log.info("Tip: python download_av.py --rebuild  — full 20-year rebuild from av_cache")