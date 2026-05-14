"""
data_loader.py — Unified Parquet/CSV loader for configured intraday OHLCV data.

Priority: Parquet (column-pruning, ~10x faster) → CSV fallback.

Usage:
    from data_loader import load_closes, load_volumes

    closes = load_closes()                            # all configured tickers
    closes = load_closes(["JPM", "BAC"])              # 2 columns only
    closes = load_closes(start="2015-01-01")          # date slice
    closes = load_closes(["SPY"], freq="daily")       # resample to daily
"""

import gc
import pandas as pd
import numpy as np
from pathlib import Path

from config import (
    DATA_DIR, RTH_START, RTH_END, BAR_MINUTES,
    CLOSES_FILE, VOLUMES_FILE, VWAPS_FILE, WFO_SKIP_VOLUMES,
)
from utils import fast_read

CLOSES_PARQUET  = DATA_DIR / f"closes_{BAR_MINUTES}min.parquet"
VOLUMES_PARQUET = DATA_DIR / f"volumes_{BAR_MINUTES}min.parquet"
VWAPS_PARQUET   = DATA_DIR / f"vwaps_{BAR_MINUTES}min.parquet"
CLOSES_CSV      = DATA_DIR / CLOSES_FILE
VOLUMES_CSV     = DATA_DIR / VOLUMES_FILE
VWAPS_CSV       = DATA_DIR / VWAPS_FILE

_TZ = "UTC"


# ── Core loader ───────────────────────────────────────────────────────────────

def _normalise_tz(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(_TZ)
    else:
        df.index = df.index.tz_convert(_TZ)
    return df


def _apply_filters(df: pd.DataFrame,
                   start: str | None,
                   end:   str | None,
                   rth:   bool,
                   freq:  str | None) -> pd.DataFrame:
    if rth:
        df = df.between_time(RTH_START, RTH_END)
    if start:
        df = df[df.index >= pd.Timestamp(start).tz_localize(_TZ)]
    if end:
        df = df[df.index <= pd.Timestamp(end).tz_localize(_TZ)]
    if freq and freq != f"{BAR_MINUTES}min":
        df = df.resample(freq).last().dropna(how="all")
    return df


def _load(parquet_path: Path, csv_path: Path,
          tickers: list[str] | None,
          start:   str | None,
          end:     str | None,
          rth:     bool,
          freq:    str | None) -> pd.DataFrame:
    if not parquet_path.exists() and not csv_path.exists():
        raise FileNotFoundError(
            f"Neither {parquet_path} nor {csv_path} found. "
            "Run download_av.py to build the dataset."
        )
    df = fast_read(
        csv_path,
        columns=tickers,
        log_label=csv_path.name,
    )
    df = _normalise_tz(df)
    df = _apply_filters(df, start, end, rth, freq)
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def load_closes(tickers: list[str] | None = None,
                start:   str | None = None,
                end:     str | None = None,
                rth:     bool = True,
                freq:    str | None = None) -> pd.DataFrame:
    """
    Load configured-bar close prices.

    Parameters
    ----------
    tickers : list of str, optional
        Column subset — uses Parquet column pruning when available.
        None = all configured tickers.
    start / end : "YYYY-MM-DD" strings, optional
    rth   : filter to configured trading hours
    freq  : resample rule, e.g. "1D", "1W". None = keep configured bars.
    """
    return _load(CLOSES_PARQUET, CLOSES_CSV, tickers, start, end, rth, freq)


def load_volumes(tickers: list[str] | None = None,
                 start:   str | None = None,
                 end:     str | None = None,
                 rth:     bool = True,
                 freq:    str | None = None) -> pd.DataFrame:
    """Load configured-bar volume data. Falls back to dummy volume (ones) if unavailable."""
    if WFO_SKIP_VOLUMES:
        print("  [Loader] WFO_SKIP_VOLUMES=True — skipping volume load.")
        return None
    try:
        return _load(VOLUMES_PARQUET, VOLUMES_CSV, tickers, start, end, rth, freq)
    except FileNotFoundError:
        # Fallback: return ones with the same index/columns as closes
        closes = load_closes(tickers, start=start, end=end, rth=rth, freq=freq)
        return pd.DataFrame(1.0, index=closes.index, columns=closes.columns)


def load_vwaps(tickers: list[str] | None = None,
               start:   str | None = None,
               end:     str | None = None,
               rth:     bool = True,
               freq:    str | None = None) -> pd.DataFrame:
    """Load configured-bar VWAP data. Falls back to closes if VWAP is unavailable."""
    try:
        return _load(VWAPS_PARQUET, VWAPS_CSV, tickers, start, end, rth, freq)
    except FileNotFoundError:
        return load_closes(tickers, start=start, end=end, rth=rth, freq=freq)


def load_pair(t1: str, t2: str,
              start: str | None = None,
              end:   str | None = None) -> pd.DataFrame:
    """
    Load closes for exactly two tickers and drop rows where either is NaN.
    Convenience wrapper for the most common backtest use-case.
    """
    df = load_closes([t1, t2], start=start, end=end)
    return df.dropna()


# ── Daily returns helper (for pair screening / clustering) ────────────────────

def load_daily_returns(tickers: list[str] | None = None,
                       start: str | None = None) -> pd.DataFrame:
    """
    Resample intraday closes to daily and compute log-returns.
    Used by pairs_screener.py for the K-Means clustering step.
    """
    closes = load_closes(tickers, start=start, rth=True, freq="1D")
    return np.log(closes / closes.shift(1)).dropna(how="all")


# ── Pair-loop memory helper ───────────────────────────────────────────────────

def release(*dfs) -> None:
    """Delete DataFrames and trigger GC. Call at end of each pair iteration."""
    for df in dfs:
        del df
    gc.collect()
