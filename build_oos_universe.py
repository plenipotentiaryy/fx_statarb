"""
build_oos_universe.py — rolling, point-in-time OOS universe builder.

Scheme per snapshot:
    train     window = 24 months
    validate  window =  3 months
    test      window =  1 month   (the OOS block)
    step              =  1 month  (next snapshot's test_start shifts forward 1m)

For each rolling test block, pair selection uses ONLY closes
in [train_start, validate_end] (validate_end = test_start - 1 day).
Selection is delegated to step2b_pairs_fx.build_pairs_universe() — no
duplication of cointegration logic.

Outputs:
    data/oos_universe/universe_YYYYMMDD.csv   (per-snapshot universe, YYYYMMDD = test_start)
    data/oos_universe/manifest.csv            (index of all snapshots)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta

from config import DATA_DIR
from step2b_pairs_fx import _load_daily_universe, build_pairs_universe


DEFAULT_OUT_DIR = DATA_DIR / "oos_universe"
TRAIN_MONTHS    = 24
VALIDATE_MONTHS = 3
TEST_MONTHS     = 1
STEP_MONTHS     = 1


def _ensure_ts(date_val: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(date_val)


def build(
    start: str | pd.Timestamp,
    end:   str | pd.Timestamp,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    *,
    train_months: int = TRAIN_MONTHS,
    validate_months: int = VALIDATE_MONTHS,
    test_months: int = TEST_MONTHS,
    step_months: int = STEP_MONTHS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build a rolling set of point-in-time universe snapshots."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    closes = _load_daily_universe()
    if closes.index.tz is not None:
        closes.index = closes.index.tz_localize(None)

    start_ts = _ensure_ts(start)
    end_ts   = _ensure_ts(end)

    # First snapshot's test_start can't be earlier than data_first + train+validate.
    data_first = closes.index.min()
    earliest_test_start = data_first + relativedelta(months=train_months + validate_months)
    test_start = max(start_ts, earliest_test_start)

    manifest_rows: list[dict] = []

    while test_start <= end_ts:
        train_start  = test_start - relativedelta(months=train_months + validate_months)
        validate_end = test_start - pd.Timedelta(days=1)  # selection cutoff
        test_end     = test_start + relativedelta(months=test_months) - pd.Timedelta(days=1)
        snapshot_id  = test_start.strftime("%Y%m%d")

        out_path = out_dir / f"universe_{snapshot_id}.csv"

        if verbose:
            print(f"\n[{snapshot_id}] train={train_start.date()}..{validate_end.date()} | "
                  f"test={test_start.date()}..{test_end.date()}")

        # Slice closes to the [train_start, validate_end] window before calling the
        # selection function. The function applies its own train_end safeguard.
        snap_closes = closes[(closes.index >= train_start) & (closes.index <= validate_end)]
        df = build_pairs_universe(
            train_end=validate_end,
            test_start=test_start,
            test_end=test_end,
            out_path=out_path,
            closes=snap_closes,
            verbose=False,
        )

        manifest_rows.append({
            "snapshot_id":      snapshot_id,
            "train_start_date": train_start.date().isoformat(),
            "train_end_date":   validate_end.date().isoformat(),
            "test_start_date":  test_start.date().isoformat(),
            "test_end_date":    test_end.date().isoformat(),
            "path":             str(out_path),
            "n_pairs":          int(len(df)),
        })
        if verbose:
            print(f"  → {len(df)} pairs → {out_path.name}")

        test_start = test_start + relativedelta(months=step_months)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    if verbose:
        print(f"\nManifest: {len(manifest)} snapshots → {manifest_path}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build rolling point-in-time OOS universe snapshots.")
    parser.add_argument("--start",   type=str, required=True, help="First test_start (ISO date).")
    parser.add_argument("--end",     type=str, required=True, help="Last test_start (ISO date).")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--train-months",    type=int, default=TRAIN_MONTHS)
    parser.add_argument("--validate-months", type=int, default=VALIDATE_MONTHS)
    parser.add_argument("--test-months",     type=int, default=TEST_MONTHS)
    parser.add_argument("--step-months",     type=int, default=STEP_MONTHS)
    args = parser.parse_args()

    build(
        start=args.start,
        end=args.end,
        out_dir=args.out_dir,
        train_months=args.train_months,
        validate_months=args.validate_months,
        test_months=args.test_months,
        step_months=args.step_months,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
