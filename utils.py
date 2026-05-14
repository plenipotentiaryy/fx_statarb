from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from config import FAST_RUN_BARS


def fast_read_csv(path, fast_bars: int | None = None) -> pd.DataFrame:
    """Read CSV, optionally only the last N rows via skiprows."""
    return fast_read(
        path,
        fast_bars=FAST_RUN_BARS if fast_bars is None else fast_bars,
        prefer_parquet=False,
    )


def fast_read(
    path,
    columns: Iterable[str] | None = None,
    fast_bars: int | None = None,
    *,
    index_col: int | str | None = 0,
    parse_dates: bool = True,
    prefer_parquet: bool = True,
    log_label: str | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Read market data quickly with Parquet priority and optional tail-only CSV load.

    If a sibling `.parquet` exists it is preferred, with optional column pruning.
    Otherwise CSV is loaded; when `fast_bars > 0` only the last N data rows are read.
    """
    csv_path = Path(path)
    parquet_path = csv_path.with_suffix(".parquet")
    fast_bars = FAST_RUN_BARS if fast_bars is None else fast_bars
    label = log_label or csv_path.name
    start = time.perf_counter()

    if prefer_parquet and parquet_path.exists():
        try:
            df = pd.read_parquet(
                parquet_path,
                columns=list(columns) if columns is not None else None,
                engine="pyarrow",
            )
            if fast_bars and fast_bars > 0:
                df = df.tail(int(fast_bars))
            elapsed = time.perf_counter() - start
            print(f"Loaded {label} (parquet): {len(df)} rows in {elapsed:.1f}s")
            return df
        except Exception as exc:
            print(f"Parquet fallback for {label}: {exc}")

    if columns is not None:
        keep = set(columns)

        def _usecols(col_name: str) -> bool:
            if index_col is None:
                return col_name in keep
            if isinstance(index_col, str):
                return col_name == index_col or col_name in keep
            return True
    else:
        _usecols = None

    if fast_bars and fast_bars > 0:
        with csv_path.open("r", encoding=read_csv_kwargs.get("encoding", "utf-8"), newline="") as fh:
            total_rows = sum(1 for _ in fh) - 1
        skip = max(0, total_rows - int(fast_bars))
        df = pd.read_csv(
            csv_path,
            index_col=index_col,
            parse_dates=parse_dates,
            skiprows=range(1, skip + 1),
            usecols=_usecols,
            **read_csv_kwargs,
        )
    else:
        df = pd.read_csv(
            csv_path,
            index_col=index_col,
            parse_dates=parse_dates,
            usecols=_usecols,
            **read_csv_kwargs,
        )

    if columns is not None and _usecols is None:
        cols = [c for c in columns if c in df.columns]
        df = df[cols]
    elapsed = time.perf_counter() - start
    print(f"Loaded {label}: {len(df)} rows in {elapsed:.1f}s")
    return df


def save_with_parquet(df: pd.DataFrame, path, *, index: bool = True) -> None:
    """Save CSV and best-effort Parquet sibling without deleting CSV fallback."""
    csv_path = Path(path)
    df.to_csv(csv_path, index=index)
    parquet_path = csv_path.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, engine="pyarrow", index=index)
    except Exception as exc:
        print(f"Parquet save skipped for {csv_path.name}: {exc}")
