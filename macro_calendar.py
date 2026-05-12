"""
macro_calendar.py — High-impact macro event calendar for FX entry blocking.

Auto-generated events (rule-based, no external data needed):
  NFP  — Non-Farm Payrolls: first Friday of each month, 13:30 UTC
  CPI  — US Consumer Price Index: ~second Wednesday, 13:30 UTC (approximate)

Custom events loaded from data/macro_events.csv:
  Add FOMC, ECB, BOE, PMI, GDP releases here.
  Format: datetime (UTC), event, impact
  Example row: 2024-03-20 18:00:00+00:00,FOMC,HIGH

Usage:
    from macro_calendar import build_event_blackout
    blackout = build_event_blackout(signal_index, DATA_DIR, bars_before=2, bars_after=4)
    # blackout: boolean array aligned to signal_index — True = no entry
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ── Rule-based event generators ───────────────────────────────────────────────

def _first_friday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year, month, 1)
    offset = (4 - d.dayofweek) % 7       # days until Friday (weekday 4)
    return d + pd.Timedelta(days=offset)


def _second_wednesday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year, month, 1)
    offset = (2 - d.dayofweek) % 7       # days until first Wednesday
    first_wed = d + pd.Timedelta(days=offset)
    return first_wed + pd.Timedelta(weeks=1)   # second Wednesday


def _nfp_events(start_year: int, end_year: int) -> list[pd.Timestamp]:
    """First Friday of each month at 13:30 UTC (NFP release time)."""
    out = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            dt = _first_friday(year, month).replace(hour=13, minute=30)
            out.append(dt.tz_localize("UTC"))
    return out


def _cpi_events(start_year: int, end_year: int) -> list[pd.Timestamp]:
    """~Second Wednesday of each month at 13:30 UTC (US CPI, approximate)."""
    out = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            dt = _second_wednesday(year, month).replace(hour=13, minute=30)
            out.append(dt.tz_localize("UTC"))
    return out


# ── CSV loader ────────────────────────────────────────────────────────────────

def _load_custom_events(data_dir: Path) -> list[pd.Timestamp]:
    """
    Load high-impact events from data/macro_events.csv.

    Required columns: datetime, impact
    Optional column:  event (label only, not used for filtering)

    Only rows with impact == 'HIGH' are loaded.
    datetime must be timezone-aware or will be assumed UTC.
    """
    path = data_dir / "macro_events.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"])

    if "impact" in df.columns:
        df = df[df["impact"].str.upper() == "HIGH"]

    return df["datetime"].tolist()


# ── Main public API ───────────────────────────────────────────────────────────

def load_event_timestamps(data_dir: Path,
                          start: str,
                          end: str) -> pd.DatetimeIndex:
    """
    Return a DatetimeIndex of all high-impact event timestamps (UTC).

    Combines:
      - Auto-generated NFP  (first Friday of each month, 13:30 UTC)
      - Auto-generated CPI  (second Wednesday, 13:30 UTC, approximate)
      - Custom events from  data/macro_events.csv (FOMC, ECB, BOE, …)
    """
    ts_start = pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=1)
    ts_end   = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1)

    y0, y1 = ts_start.year, ts_end.year

    all_ts = (
        _nfp_events(y0, y1)
        + _cpi_events(y0, y1)
        + _load_custom_events(data_dir)
    )

    dti = pd.DatetimeIndex(sorted(set(all_ts)), tz="UTC")
    return dti[(dti >= ts_start) & (dti <= ts_end)]


def build_event_blackout(signal_index: pd.DatetimeIndex,
                         data_dir: Path,
                         bars_before: int = 2,
                         bars_after: int  = 4,
                         bar_minutes: int = 15) -> np.ndarray:
    """
    Build a boolean blackout array aligned to signal_index.

    True  → this bar is within the event blackout window → block entry.
    False → safe to enter.

    bars_before × 15min before the event (avoid pre-release positioning risk).
    bars_after  × 15min after  the event (avoid post-release whipsaw).

    Default: 2 bars before (30 min) and 4 bars after (60 min).
    Tune up: ECB/Fed press conferences → 8 bars after (2 hours).
    """
    if len(signal_index) == 0:
        return np.zeros(0, dtype=bool)

    idx_utc = signal_index.tz_convert("UTC")
    start   = str(idx_utc[0].date())
    end     = str(idx_utc[-1].date())

    events = load_event_timestamps(data_dir, start, end)
    if len(events) == 0:
        return np.zeros(len(signal_index), dtype=bool)

    delta_before = pd.Timedelta(minutes=bars_before * bar_minutes)
    delta_after  = pd.Timedelta(minutes=bars_after  * bar_minutes)

    blackout = np.zeros(len(signal_index), dtype=bool)
    for ev in events:
        mask = (idx_utc >= ev - delta_before) & (idx_utc <= ev + delta_after)
        blackout |= np.asarray(mask)

    return blackout
