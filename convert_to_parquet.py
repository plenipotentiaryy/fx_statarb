from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import DATA_DIR


def human_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def convert_csv(path: Path) -> None:
    parquet_path = path.with_suffix(".parquet")
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        df = pd.read_csv(path)
    df.to_parquet(parquet_path, engine="pyarrow")

    csv_size = path.stat().st_size
    pq_size = parquet_path.stat().st_size
    compression = (1.0 - pq_size / max(csv_size, 1)) * 100.0
    print(
        f"{path.name:<28} -> {parquet_path.name:<28}  "
        f"{human_mb(csv_size):>10} -> {human_mb(pq_size):>10}  "
        f"({compression:5.1f}% smaller)"
    )


def main() -> None:
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {DATA_DIR}")

    print(f"Converting {len(csv_files)} CSV files in {DATA_DIR} to Parquet...\n")
    for path in csv_files:
        convert_csv(path)


if __name__ == "__main__":
    main()
