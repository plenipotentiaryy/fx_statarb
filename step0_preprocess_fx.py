import pandas as pd
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import gc
import sys
from config import BAR_MINUTES, DATA_DIR

OUTPUT_DIR = DATA_DIR

# When BAR_MINUTES == 1 skip the resample step — raw M1 data is already 1-min.
RESAMPLE = BAR_MINUTES > 1


def process_pair(pair_dir: Path, output_kind: str):
    pair_name = pair_dir.name
    csv_files = sorted(pair_dir.glob("*.csv"))
    if not csv_files:
        return None

    chunks = []
    for csv_file in csv_files:
        try:
            # Format: YYYYMMDD HHMMSS;O;H;L;C;V
            df = pd.read_csv(
                csv_file,
                sep=";",
                header=None,
                usecols=[0, 1, 2, 3, 4, 5],
                names=["datetime", "open", "high", "low", "close", "volume"],
                engine="pyarrow",
            )
            df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d %H%M%S")
            df.set_index("datetime", inplace=True)
            df["typical"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
            activity = ((df["high"] - df["low"]) / df["close"].replace(0, float("nan")) * 1_000_000)
            activity = activity.replace([float("inf"), float("-inf")], float("nan")).fillna(0.0)
            weight = df["volume"] if df["volume"].sum() > 0 else activity

            if output_kind == "close":
                close = df["close"].resample(f"{BAR_MINUTES}min").last().ffill()
                series = close if RESAMPLE else df["close"]
            elif output_kind == "volume":
                if RESAMPLE:
                    series = weight.resample(f"{BAR_MINUTES}min").sum().fillna(0.0)
                else:
                    series = weight.fillna(0.0)
            elif output_kind == "vwap":
                if RESAMPLE:
                    volume = weight.resample(f"{BAR_MINUTES}min").sum().fillna(0.0)
                    pv = (df["typical"] * weight).resample(f"{BAR_MINUTES}min").sum()
                    series = (pv / volume.replace(0, float("nan"))).astype("float64").ffill()
                else:
                    series = df["typical"]
            else:
                raise ValueError(f"Unknown output kind: {output_kind}")

            chunks.append(series)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    if not chunks:
        return None

    full = pd.concat(chunks)
    full = full[~full.index.duplicated(keep="last")]
    full.name = pair_name
    return full


def write_output(pair_dirs: list[Path], stem: str, output_kind: str) -> None:
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(process_pair, d, output_kind): d.name for d in pair_dirs}
        for fut in tqdm(concurrent.futures.as_completed(futures),
                        total=len(pair_dirs), desc=f"Processing {stem}"):
            name = futures[fut]
            try:
                s = fut.result()
                if s is not None:
                    results[name] = s
            except Exception as e:
                print(f"  {name} failed: {e}")

    if not results:
        print(f"No {stem} data processed.")
        return

    print(f"Consolidating {stem}: {len(results)} tickers …")
    df = pd.DataFrame(results).sort_index()
    csv_out = OUTPUT_DIR / f"{stem}_{BAR_MINUTES}min.csv"
    df.to_csv(csv_out)
    print(f"Saved  {csv_out}  ({len(df):,} rows × {df.shape[1]} tickers)")

    try:
        pq_out = OUTPUT_DIR / f"{stem}_{BAR_MINUTES}min.parquet"
        df.to_parquet(pq_out, engine="pyarrow")
        print(f"Saved  {pq_out}")
    except Exception as e:
        print(f"Parquet skipped for {stem}: {e}")

    del df
    results.clear()
    gc.collect()


def main():
    wanted = {x.lower() for x in sys.argv[1:]}
    pair_dirs = sorted(
        d for d in DATA_DIR.iterdir()
        if d.is_dir() and any(d.glob("*.csv")) and (not wanted or d.name.lower() in wanted)
    )
    print(f"Found {len(pair_dirs)} pair directories  |  BAR_MINUTES={BAR_MINUTES}")

    outputs = [
        ("closes", "close"),
        ("volumes", "volume"),
        ("vwaps", "vwap"),
    ]

    for stem, key in outputs:
        write_output(pair_dirs, stem, key)

    print("Done!")


if __name__ == "__main__":
    main()
