"""
download_futures_volume.py — Pull REAL CME FX-futures volume for the zone engine.

Spot FX has no consolidated volume, so we source genuine traded volume from the
CME FX futures complex (6E, 6B, 6J, 6A, 6C, 6S, 6N, ...) and map it back to the
spot pairs in futures_map.py.

PRIMARY SOURCE — Databento (GLBX.MDP3, ohlcv-1m, continuous front-month)
    pip install databento
    export DATABENTO_API_KEY=db-...
    python download_futures_volume.py                 # all mapped contracts
    python download_futures_volume.py 6E 6B 6J        # subset

Databento's continuous symbology "ROOT.c.0" gives a volume-rolled front-month
series, which is exactly what a continuous volume profile wants (no manual roll).

FALLBACK SOURCE — Polygon futures (if POLYGON_FUTURES=1 and no Databento key).
    Polygon's futures feed is per-contract; we sum active contracts per bar.

Output: data/futures_volumes_{BAR_MINUTES}min.parquet (+ .csv)
        one column per CME root, UTC index, volume only.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

from config import BAR_MINUTES, DATA_DIR, START_DATE, END_DATE
from futures_map import (
    SPOT_TO_FUTURE, FUTURES_VOLUMES_PARQUET, FUTURES_VOLUMES_CSV,
)

ALL_ROOTS = sorted({root for root, _ in SPOT_TO_FUTURE.values()})
_RESAMPLE = f"{BAR_MINUTES}min"


# ── Databento path ────────────────────────────────────────────────────────────

def _fetch_databento(roots: list[str]) -> pd.DataFrame:
    import databento as db

    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError("DATABENTO_API_KEY not set")
    client = db.Historical(key)

    series: dict[str, pd.Series] = {}
    for root in roots:
        symbol = f"{root}.c.0"   # continuous front-month
        print(f"  [databento] {symbol}  {START_DATE} → {END_DATE}")
        data = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            schema="ohlcv-1m",
            symbols=[symbol],
            stype_in="continuous",
            start=START_DATE,
            end=END_DATE,
        )
        df = data.to_df()
        if df.empty:
            print(f"    (no data for {symbol})")
            continue
        # Databento df is indexed by ts_event (UTC); 'volume' is the bar volume.
        vol = df["volume"].astype("float64")
        vol.index = pd.to_datetime(df.index, utc=True)
        if BAR_MINUTES != 1:
            vol = vol.resample(_RESAMPLE).sum()
        series[root] = vol.rename(root)

    return pd.DataFrame(series).sort_index()


# ── Polygon futures fallback ──────────────────────────────────────────────────

def _fetch_polygon(roots: list[str]) -> pd.DataFrame:
    """Best-effort Polygon futures volume (per-contract, summed per bar).

    Polygon's futures product keys contracts individually; without an explicit
    roll calendar we aggregate volume across the listed contracts per root, which
    is adequate for a *volume* profile (we only need where volume concentrates).
    """
    from polygon import RESTClient

    key = os.environ.get("POLYGON_API_KEY")
    if not key:
        raise RuntimeError("POLYGON_API_KEY not set")
    client = RESTClient(key)

    series: dict[str, pd.Series] = {}
    for root in roots:
        print(f"  [polygon] futures root {root}")
        frames = []
        try:
            contracts = client.list_futures_contracts(product_code=root, limit=100)
        except Exception as e:  # noqa: BLE001
            print(f"    contract listing failed for {root}: {e}")
            continue
        for c in contracts:
            ticker = getattr(c, "ticker", None)
            if not ticker:
                continue
            try:
                bars = client.list_aggs(ticker, BAR_MINUTES, "minute",
                                        START_DATE, END_DATE, limit=50000)
                rows = [(pd.Timestamp(b.timestamp, unit="ms", tz="UTC"), b.volume)
                        for b in bars]
                if rows:
                    s = pd.Series(dict(rows))
                    frames.append(s)
            except Exception:  # noqa: BLE001
                continue
        if frames:
            agg = pd.concat(frames, axis=1).sum(axis=1)
            if BAR_MINUTES != 1:
                agg = agg.resample(_RESAMPLE).sum()
            series[root] = agg.rename(root)

    return pd.DataFrame(series).sort_index()


# ── Driver ────────────────────────────────────────────────────────────────────

def main() -> None:
    roots = [r.upper() for r in sys.argv[1:]] or ALL_ROOTS
    print(f"Downloading CME futures volume for: {', '.join(roots)}  "
          f"(BAR_MINUTES={BAR_MINUTES})")

    use_polygon = os.environ.get("POLYGON_FUTURES") == "1" or not os.environ.get("DATABENTO_API_KEY")
    if use_polygon and not os.environ.get("DATABENTO_API_KEY"):
        print("  DATABENTO_API_KEY absent → trying Polygon futures fallback.")
        df = _fetch_polygon(roots)
    else:
        df = _fetch_databento(roots)

    if df.empty:
        print("No futures volume downloaded — check API key / entitlements.")
        return

    df = df[~df.index.duplicated(keep="last")]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FUTURES_VOLUMES_CSV)
    try:
        df.to_parquet(FUTURES_VOLUMES_PARQUET, engine="pyarrow")
        print(f"Saved {FUTURES_VOLUMES_PARQUET}  ({len(df):,} rows × {df.shape[1]} roots)")
    except Exception as e:  # noqa: BLE001
        print(f"Parquet skipped: {e}")
    print(f"Saved {FUTURES_VOLUMES_CSV}")


if __name__ == "__main__":
    main()
