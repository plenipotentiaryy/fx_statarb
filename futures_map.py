"""
futures_map.py — Spot-FX ↔ CME FX-futures mapping and volume loader.

WHY FUTURES
-----------
Spot FX is OTC and has no consolidated traded volume. CME FX futures, by contrast,
are exchange-traded with a real, price-stamped volume tape. Mapping each spot pair
to its CME contract gives us genuine "traded volume" to build a volume profile from.

QUOTING DIRECTION
-----------------
CME FX futures are quoted as USD-per-foreign-currency, i.e. they track the
*foreign/USD* rate. So:

  * EURUSD, GBPUSD, AUDUSD, NZDUSD  → 6E, 6B, 6A, 6N  — SAME direction as spot.
  * USDJPY, USDCAD, USDCHF          → 6J, 6C, 6S       — INVERSE of spot
    (the future tracks JPY/USD = 1/USDJPY). Price levels are reciprocal, but
    VOLUME is direction-agnostic — a trade is a trade — so for a *volume* profile
    the inverse quoting does not matter. We only flag it (``inverse``) for any
    future price-level overlay work; volume is mapped straight across.

DATA FILE
---------
download_futures_volume.py writes data/futures_volumes_{bar}min.parquet with one
column per CME root symbol (continuous front-month volume). load_futures_volume_as_spot()
relabels those columns back to spot ticker names so the rest of the codebase keeps
using "eurusd", "gbpusd", ... transparently.
"""

from __future__ import annotations

import pandas as pd

from config import BAR_MINUTES, DATA_DIR, RTH_START, RTH_END

# spot ticker → (CME root, inverse-quoted?)
SPOT_TO_FUTURE: dict[str, tuple[str, bool]] = {
    "eurusd": ("6E", False),
    "gbpusd": ("6B", False),
    "audusd": ("6A", False),
    "nzdusd": ("6N", False),
    "usdjpy": ("6J", True),
    "usdcad": ("6C", True),
    "usdchf": ("6S", True),
    "usdmxn": ("6M", True),
    "usdzar": ("6Z", True),   # CME ZAR/USD (thin — use with care)
}

# Reverse lookup: CME root → spot ticker
FUTURE_TO_SPOT: dict[str, str] = {root: spot for spot, (root, _) in SPOT_TO_FUTURE.items()}

FUTURES_VOLUMES_PARQUET = DATA_DIR / f"futures_volumes_{BAR_MINUTES}min.parquet"
FUTURES_VOLUMES_CSV     = DATA_DIR / f"futures_volumes_{BAR_MINUTES}min.csv"


def mappable_spot(tickers: list[str] | None) -> list[str]:
    """Subset of ``tickers`` that have a CME futures equivalent."""
    if tickers is None:
        return list(SPOT_TO_FUTURE.keys())
    return [t for t in tickers if t.lower() in SPOT_TO_FUTURE]


def load_futures_volume_as_spot(tickers: list[str] | None = None,
                                start: str | None = None,
                                end: str | None = None,
                                rth: bool = True,
                                freq: str | None = None) -> pd.DataFrame | None:
    """Load CME futures volume, relabelled to spot ticker names.

    Only spot pairs with a direct CME contract are returned (others fall back to
    tick volume upstream). Returns None when the futures file is absent.
    """
    path = FUTURES_VOLUMES_PARQUET if FUTURES_VOLUMES_PARQUET.exists() else FUTURES_VOLUMES_CSV
    if not path.exists():
        return None

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, index_col=0, parse_dates=True)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Relabel CME roots → spot tickers; keep only requested/mappable pairs.
    wanted = mappable_spot(tickers)
    cols: dict[str, pd.Series] = {}
    for spot in wanted:
        root, _ = SPOT_TO_FUTURE[spot]
        if root in df.columns:
            cols[spot] = df[root]
    if not cols:
        return None

    out = pd.DataFrame(cols).sort_index()
    if rth:
        out = out.between_time(RTH_START, RTH_END)
    if start:
        out = out[out.index >= pd.Timestamp(start).tz_localize("UTC")]
    if end:
        out = out[out.index <= pd.Timestamp(end).tz_localize("UTC")]
    if freq and freq != f"{BAR_MINUTES}min":
        out = out.resample(freq).sum().dropna(how="all")
    return out
