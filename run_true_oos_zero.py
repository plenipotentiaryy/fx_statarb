from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_oos_universe import TRAIN_MONTHS as DEFAULT_TRAIN_MONTHS, TEST_MONTHS as DEFAULT_TEST_MONTHS, STEP_MONTHS as DEFAULT_STEP_MONTHS, build as build_oos_universe
from step2b_pairs_fx import _load_daily_universe as load_daily_universe
from config import DATA_DIR, OUTPUT_DIR, INITIAL_CAPITAL
from regime_block_bootstrap import RegimeBlockBootstrap, attach_regime_to_trades
from utils import save_with_parquet


OOS_UNIVERSE_DIR = DATA_DIR / "oos_universe"
OOS_PARAMS_DIR = DATA_DIR / "oos_params"
OOS_BLOCK_DIR = DATA_DIR / "oos_blocks"
FINAL_TRADES_PATH = DATA_DIR / "oos_stitched_trades.csv"
FINAL_CURATED_PATH = DATA_DIR / "pairs_true_oos_zero_universe.csv"
FINAL_RESEARCH_PATH = DATA_DIR / "true_oos_research_report.csv"
FINAL_ZERO_MTM_PATH = DATA_DIR / "true_oos_zero_mtm_report.csv"
FINAL_MANIFEST_PATH = DATA_DIR / "true_oos_manifest.csv"
FINAL_RUN_CONFIG = DATA_DIR / "true_oos_run_config.json"
FINAL_RUN_LOG = DATA_DIR / "true_oos_run_log.txt"


def _log(msg: str, log_lines: list[str]) -> None:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    line = f"[{stamp}] {msg}"
    print(line)
    log_lines.append(line)


def _validate_trades_window(trades: pd.DataFrame, test_start: str, test_end: str, snapshot_id: str) -> None:
    """Hard guard: trades must fall strictly inside [test_start, test_end]."""
    if trades.empty:
        return
    ts_start = pd.Timestamp(test_start)
    ts_end = pd.Timestamp(test_end)
    et = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    xt = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce")
    if ts_start.tzinfo is None:
        ts_start = ts_start.tz_localize("UTC")
    else:
        ts_start = ts_start.tz_convert("UTC")
    if ts_end.tzinfo is None:
        ts_end = ts_end.tz_localize("UTC")
    else:
        ts_end = ts_end.tz_convert("UTC")
    # test_end is end-of-day exclusive — add a day to be permissive on intraday bars
    ts_end_excl = ts_end + pd.Timedelta(days=1)
    bad_entry = et < ts_start
    bad_exit = xt > ts_end_excl
    if bad_entry.any() or bad_exit.any():
        n_bad = int(bad_entry.sum() + bad_exit.sum())
        raise ValueError(
            f"Snapshot {snapshot_id}: {n_bad} trades fall outside OOS window "
            f"[{ts_start.date()}, {ts_end.date()}]. Data leakage — refusing to stitch."
        )


class BlockSubprocessError(RuntimeError):
    pass


def _run(cmd: list[str], *, env: dict[str, str] | None = None, label: str = "") -> subprocess.CompletedProcess[str]:
    pretty = " ".join(cmd)
    if label:
        print(f"\n[{label}] {pretty}")
    else:
        print(f"\n{pretty}")
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise BlockSubprocessError(f"Command failed with exit code {proc.returncode}: {pretty}")

    lines = (proc.stdout or "").splitlines()
    tail = lines[-30:] if len(lines) > 30 else lines
    if tail:
        print("\n".join(tail))
    if proc.stderr:
        err_tail = (proc.stderr or "").splitlines()[-10:]
        if err_tail:
            print("\n".join(err_tail))
    return proc


def _load_global_hmm() -> pd.Series | None:
    path = DATA_DIR / "global_hmm_regime.csv"
    if not path.exists():
        return None
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    s.index = pd.to_datetime(s.index, utc=True).tz_localize(None)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def _compute_research_report(trades: pd.DataFrame, regime_series: pd.Series | None = None) -> dict[str, Any]:
    if trades.empty:
        return {
            "trades": 0,
            "trades_per_year": 0.0,
            "trade_hit_rate": float("nan"),
            "gross_pnl": 0.0,
            "costs": 0.0,
            "total_pnl": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe_annualized": float("nan"),
            "trade_sharpe_annualized": float("nan"),
            "daily_sharpe_annualized": float("nan"),
            "bootstrap_total_pnl_lo": 0.0,
            "bootstrap_total_pnl_hi": 0.0,
            "bootstrap_sharpe_lo": 0.0,
            "bootstrap_sharpe_hi": 0.0,
            "bootstrap_max_drawdown_lo": 0.0,
            "bootstrap_max_drawdown_hi": 0.0,
        }

    pnl = pd.to_numeric(trades["net_pnl"], errors="coerce").fillna(0.0)
    total_pnl = float(pnl.sum())
    gross_pnl = float(trades["gross_pnl"].sum()) if "gross_pnl" in trades.columns else 0.0
    costs = float((trades["tx_cost"] + trades["borrow_cost"]).sum()) if {"tx_cost", "borrow_cost"}.issubset(trades.columns) else 0.0
    hit_rate = float((pnl > 0).mean() * 100)

    if len(trades) > 1:
        exit_times = pd.to_datetime(trades["exit_time"])
        span_days = max((exit_times.iloc[-1] - exit_times.iloc[0]).days, 1)
        trades_per_year = len(trades) / max(span_days / 365.25, 1 / 365.25)
    else:
        trades_per_year = 0.0

    trade_sharpe = float(pnl.mean() / pnl.std() * np.sqrt(trades_per_year)) if pnl.std() > 0 else 0.0
    pnl_for_daily = pd.to_numeric(
        trades["dollar_pnl"] if "dollar_pnl" in trades.columns else trades["net_pnl"],
        errors="coerce",
    ).fillna(0.0)
    exit_times_daily = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce")
    daily = (
        pd.DataFrame({"exit_time": exit_times_daily, "pnl": pnl_for_daily})
        .dropna(subset=["exit_time"])
        .assign(trade_day=lambda x: x["exit_time"].dt.floor("D"))
        .groupby("trade_day")["pnl"].sum()
        .sort_index()
    )
    if not daily.empty:
        full_days = pd.date_range(daily.index.min(), daily.index.max(), freq="D", tz=daily.index.tz)
        daily = daily.reindex(full_days, fill_value=0.0)
    daily_returns = daily / float(INITIAL_CAPITAL) if not daily.empty else pd.Series(dtype=float)
    daily_sharpe = (
        float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
        if len(daily_returns) > 1 and daily_returns.std() > 0
        else 0.0
    )
    cumulative = pnl.cumsum()
    max_drawdown = float((cumulative - cumulative.cummax()).min())
    winning = pnl[pnl > 0]
    losing = pnl[pnl <= 0]
    profit_factor = float(winning.sum() / abs(losing.sum())) if len(losing) > 0 and losing.sum() != 0 else float("inf")

    boot = RegimeBlockBootstrap(block_size=20, n_bootstrap=10_000, pnl_col="net_pnl", regime_col="hmm_regime", random_seed=42)
    boot_input = trades.copy()
    if regime_series is not None:
        boot_input = attach_regime_to_trades(boot_input, regime_series)
    else:
        boot_input["hmm_regime"] = 0
    boot_result = boot.bootstrap_metrics(boot_input)

    return {
        "trades": int(len(trades)),
        "trades_per_year": float(trades_per_year),
        "trade_hit_rate": hit_rate,
        "gross_pnl": gross_pnl,
        "costs": costs,
        "total_pnl": total_pnl,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "sharpe_annualized": daily_sharpe,
        "daily_sharpe_annualized": daily_sharpe,
        "trade_sharpe_annualized": trade_sharpe,
        "bootstrap_total_pnl_lo": float(boot_result["confidence_intervals"]["total_pnl"][0]),
        "bootstrap_total_pnl_hi": float(boot_result["confidence_intervals"]["total_pnl"][1]),
        "bootstrap_sharpe_lo": float(boot_result["confidence_intervals"]["sharpe"][0]),
        "bootstrap_sharpe_hi": float(boot_result["confidence_intervals"]["sharpe"][1]),
        "bootstrap_max_drawdown_lo": float(boot_result["confidence_intervals"]["max_drawdown"][0]),
        "bootstrap_max_drawdown_hi": float(boot_result["confidence_intervals"]["max_drawdown"][1]),
    }


def _compute_zero_mtm_report(trades: pd.DataFrame, zero_block_reports: list[pd.DataFrame]) -> dict[str, Any]:
    if trades.empty:
        return {
            "daily_breach": False,
            "trailing_breach": False,
            "forced_liquidation_count": 0,
            "max_liquidation_drawdown": 0.0,
            "inactivity_days": None,
            "max_trade_days_30d": 0,
            "max_profit_days_30d": 0,
            "max_trade_days_60d": 0,
            "max_profit_days_60d": 0,
        }

    trades = trades.copy()
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True).dt.tz_convert("US/Eastern")
    trades = trades.sort_values("exit_time")
    trades["trade_day"] = trades["exit_time"].dt.normalize()
    daily_pnl = trades.groupby("trade_day")["dollar_pnl"].sum().sort_index() if "dollar_pnl" in trades.columns else trades.groupby("trade_day")["net_pnl"].sum().sort_index()
    full_days = pd.date_range(daily_pnl.index.min(), daily_pnl.index.max(), freq="D", tz=daily_pnl.index.tz)
    daily_pnl = daily_pnl.reindex(full_days, fill_value=0.0)

    trade_days = trades["trade_day"].drop_duplicates().sort_values()
    inactivity_days = None
    if len(trade_days) >= 2:
        gaps = trade_days.diff().dt.days.dropna()
        inactivity_days = int(gaps.max() - 1) if not gaps.empty else 0
    elif len(trade_days) == 1:
        inactivity_days = 0

    def _rolling_window_activity(daily_pnl: pd.Series, window_days: int) -> tuple[int, int, float]:
        trade_days = (daily_pnl != 0).rolling(window_days, min_periods=window_days).sum()
        profit_days = (daily_pnl > 0).rolling(window_days, min_periods=window_days).sum()
        pnl_window = daily_pnl.rolling(window_days, min_periods=window_days).sum()
        trade_max = int(trade_days.max()) if pd.notna(trade_days.max()) else 0
        profit_max = int(profit_days.max()) if pd.notna(profit_days.max()) else 0
        pnl_max = float(pnl_window.max()) if pd.notna(pnl_window.max()) else 0.0
        return trade_max, profit_max, pnl_max

    max_trade_days_30d, max_profit_days_30d, max_30d_pnl = _rolling_window_activity(daily_pnl, 30)
    max_trade_days_60d, max_profit_days_60d, max_60d_pnl = _rolling_window_activity(daily_pnl, 60)

    def _as_bool(series_name: str) -> bool:
        if not zero_block_reports:
            return False
        return any(bool(df.get(series_name, pd.Series(dtype=bool)).fillna(False).astype(bool).any()) for df in zero_block_reports if not df.empty)

    forced_liquidation_count = 0
    max_liquidation_drawdown = 0.0
    for df in zero_block_reports:
        if df.empty:
            continue
        if "forced_liquidations" in df.columns:
            forced_liquidation_count += int(pd.to_numeric(df["forced_liquidations"], errors="coerce").fillna(0).sum())
        if "max_liquidation_drawdown" in df.columns:
            max_liquidation_drawdown = min(
                max_liquidation_drawdown,
                float(pd.to_numeric(df["max_liquidation_drawdown"], errors="coerce").fillna(0.0).min()),
            )

    return {
        "daily_breach": _as_bool("daily_breach"),
        "trailing_breach": _as_bool("trailing_breach"),
        "forced_liquidation_count": int(forced_liquidation_count),
        "max_liquidation_drawdown": float(max_liquidation_drawdown),
        "inactivity_days": inactivity_days,
        "max_trade_days_30d": int(max_trade_days_30d),
        "max_profit_days_30d": int(max_profit_days_30d),
        "max_trade_days_60d": int(max_trade_days_60d),
        "max_profit_days_60d": int(max_profit_days_60d),
    }


def _curate_universe(block_universe_rows: list[pd.DataFrame]) -> pd.DataFrame:
    if not block_universe_rows:
        return pd.DataFrame()
    combined = pd.concat(block_universe_rows, ignore_index=True)
    if combined.empty:
        return combined
    if "pair" not in combined.columns:
        return pd.DataFrame()

    agg = combined.groupby("pair").agg(
        present_blocks=("pair", "size"),
        passed_blocks=("zero_universe_ok", "sum") if "zero_universe_ok" in combined.columns else ("pair", "size"),
        mean_net_pnl=("net_pnl", "mean") if "net_pnl" in combined.columns else ("pair", "size"),
        sum_net_pnl=("net_pnl", "sum") if "net_pnl" in combined.columns else ("pair", "size"),
        mean_trades=("trades", "mean") if "trades" in combined.columns else ("pair", "size"),
        mean_win_rate=("win_rate", "mean") if "win_rate" in combined.columns else ("pair", "size"),
        max_60d_pnl=("max_60d_pnl", "max") if "max_60d_pnl" in combined.columns else ("pair", "size"),
        max_inactive_days=("max_inactive_days", "max") if "max_inactive_days" in combined.columns else ("pair", "size"),
    ).reset_index()
    agg["pass_rate"] = agg["passed_blocks"] / agg["present_blocks"].replace(0, np.nan)
    curated = agg[
        (agg["present_blocks"] >= 2)
        & (agg["pass_rate"] >= 0.8)
        & (agg["sum_net_pnl"] > 0)
    ].copy()
    if curated.empty:
        return curated
    return curated.sort_values(["max_60d_pnl", "sum_net_pnl"], ascending=False).reset_index(drop=True)


def _load_manifest() -> pd.DataFrame:
    manifest = OOS_UNIVERSE_DIR / "manifest.csv"
    if manifest.exists():
        df = pd.read_csv(manifest)
        if not df.empty:
            return df
    return pd.DataFrame()


def _ensure_universe(args: argparse.Namespace) -> pd.DataFrame:
    if args.rebuild_universe or not (OOS_UNIVERSE_DIR / "manifest.csv").exists():
        manifest = build_oos_universe(
            start=args.start_date or "2000-01-01",
            end=args.end_date or pd.Timestamp.today().strftime("%Y-%m-%d"),
            out_dir=OOS_UNIVERSE_DIR,
            train_months=args.train_months,
            test_months=args.test_months,
            step_months=args.step_months,
        )
        if manifest.empty:
            raise SystemExit("No universe snapshots built.")
        return manifest
    manifest = _load_manifest()
    if manifest.empty:
        raise SystemExit("Universe manifest missing or empty.")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a strict true OOS / MTM Zero validation stack.")
    parser.add_argument("--rebuild-universe", action="store_true", help="Rebuild OOS universe snapshots first.")
    parser.add_argument("--start-date", type=str, default=None, help="Optional start date for universe builder.")
    parser.add_argument("--end-date", type=str, default=None, help="Optional end date for universe builder.")
    parser.add_argument("--train-months", type=int, default=DEFAULT_TRAIN_MONTHS, help="Universe train window in months.")
    parser.add_argument("--test-months", type=int, default=DEFAULT_TEST_MONTHS, help="Universe test window in months.")
    parser.add_argument("--step-months", type=int, default=DEFAULT_STEP_MONTHS, help="Universe step in months.")
    parser.add_argument("--max-snapshots", type=int, default=None, help="Cap number of OOS snapshots.")
    parser.add_argument("--skip-step3j", action="store_true", help="Assume WFO params already exist for each snapshot.")
    parser.add_argument("--limit-blocks", type=int, default=None, help="Limit number of snapshots to process.")
    args = parser.parse_args()

    OOS_UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    OOS_PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    OOS_BLOCK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = _ensure_universe(args)
    manifest = manifest.sort_values(["test_start_date", "snapshot_id"]).reset_index(drop=True)
    if "pairs" in manifest.columns:
        manifest = manifest[pd.to_numeric(manifest["pairs"], errors="coerce").fillna(0).astype(int) > 0].copy()
    if args.limit_blocks is not None:
        manifest = manifest.head(args.limit_blocks).copy()
    if manifest.empty:
        raise SystemExit("No snapshots available for OOS run.")

    log_lines: list[str] = []
    _log(f"pipeline started: {len(manifest)} candidate snapshots", log_lines)

    # No-fallback hard guard
    for forbidden in ("AFES_PAIRS_PATH",):
        if os.environ.get(forbidden):
            _log(f"WARNING: {forbidden} present in environment — will be overridden per-block", log_lines)

    stitched_trades: list[pd.DataFrame] = []
    zero_block_reports: list[pd.DataFrame] = []
    universe_rows: list[pd.DataFrame] = []
    block_manifest: list[dict[str, Any]] = []

    for _, row in manifest.iterrows():
        snapshot_id = str(row["snapshot_id"])
        pairs_path = Path(row["path"])
        test_start = str(row["test_start_date"])
        test_end = str(row["test_end_date"])
        # Snapshot id used by step3j / step4a is YYYYMMDD of test_start
        snap_id_ymd = pd.Timestamp(test_start).strftime("%Y%m%d")

        block_params_path = OOS_PARAMS_DIR / f"params_{snap_id_ymd}.csv"
        block_trades_path = OOS_BLOCK_DIR / f"trades_{snapshot_id}.csv"
        block_research_path = OOS_BLOCK_DIR / f"research_{snapshot_id}.csv"
        block_zero_path = OOS_BLOCK_DIR / f"zero_mtm_{snapshot_id}.csv"
        block_universe_path = OOS_BLOCK_DIR / f"pairs_zero_{snapshot_id}.csv"

        if not args.skip_step3j:
            # End data at test_start-1day so WFO optimization never sees the test block.
            opt_end = (pd.Timestamp(test_start) - pd.Timedelta(days=1)).date().isoformat()
            try:
                _run(
                    [
                        "python3",
                        "step3j_wfo.py",
                        "--pairs",       str(pairs_path),
                        "--end",         opt_end,
                        "--test-start",  test_start,
                        "--test-end",    test_end,
                        "--snapshot-id", snap_id_ymd,
                    ],
                env={
                    **os.environ,
                    "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", "/private/tmp/afes-mpl"),
                    "BACKTEST_SKIP_PLOTS": "1",
                    "AFES_ALLOW_EMPTY_RUN": "1",
                },
                label=f"WFO {snapshot_id}",
            )
            except BlockSubprocessError as e:
                _log(f"block {snapshot_id} SKIPPED: WFO failed ({e})", log_lines)
                block_manifest.append({
                    "snapshot_id": snapshot_id, "test_start_date": test_start, "test_end_date": test_end,
                    "universe_path": str(pairs_path), "params_path": "",
                    "status": "skipped", "skip_reason": "wfo_subprocess_failed",
                    "n_trades": 0, "trades_path": "", "research_report_path": "", "zero_mtm_report_path": "",
                })
                continue
            src = DATA_DIR / "wfo_params.csv"
            if src.exists():
                shutil.copy2(src, block_params_path)
            else:
                _log(f"block {snapshot_id} SKIPPED: missing wfo_params.csv", log_lines)
                continue
        elif not block_params_path.exists():
            _log(f"block {snapshot_id} SKIPPED: missing per-block params", log_lines)
            continue

        try:
            _run(
            [
                "python3",
                "step4a_backtest.py",
            ],
            env={
                **os.environ,
                "AFES_PAIRS_PATH": str(pairs_path),
                "AFES_OOS_PARAMS_DIR": str(OOS_PARAMS_DIR),
                "AFES_DISABLE_OPT_OVERRIDE": "1",
                "AFES_ALLOW_EMPTY_RUN": "1",
                "AFES_TEST_END_DATE": test_end,
                "AFES_TRUE_OOS": os.environ.get("AFES_TRUE_OOS", "0"),
                "BACKTEST_SKIP_PLOTS": "1",
                "BACKTEST_PROGRESS": "0",
                "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", "/private/tmp/afes-mpl"),
            },
            label=f"BACKTEST {snapshot_id}",
            )
        except BlockSubprocessError as e:
            _log(f"block {snapshot_id} SKIPPED: backtest failed ({e})", log_lines)
            block_manifest.append({
                "snapshot_id": snapshot_id, "test_start_date": test_start, "test_end_date": test_end,
                "universe_path": str(pairs_path), "params_path": str(block_params_path),
                "status": "skipped", "skip_reason": "backtest_subprocess_failed",
                "n_trades": 0, "trades_path": "", "research_report_path": "", "zero_mtm_report_path": "",
            })
            continue

        trades_path = DATA_DIR / "trades.csv"
        research_path = DATA_DIR / "research_report.csv"
        zero_path = DATA_DIR / "zero_mtm_report.csv"
        universe_path = DATA_DIR / "pairs_zero_universe.csv"

        if not trades_path.exists():
            raise SystemExit(f"Missing trades output after step4a run: {trades_path}")

        block_trades = pd.read_csv(trades_path)
        _validate_trades_window(block_trades, test_start, test_end, snapshot_id)
        block_trades["snapshot_id"] = snapshot_id
        block_trades["block_test_start"] = str(row["test_start_date"])
        block_trades["block_test_end"] = str(row["test_end_date"])
        block_trades["universe_path"] = str(pairs_path)
        block_trades["params_path"] = str(block_params_path)
        stitched_trades.append(block_trades)
        shutil.copy2(trades_path, block_trades_path)
        _log(
            f"block {snapshot_id} [{test_start}..{test_end}] backtest OK n_trades={len(block_trades)}",
            log_lines,
        )
        block_manifest.append({
            "snapshot_id": snapshot_id,
            "test_start_date": test_start,
            "test_end_date": test_end,
            "universe_path": str(pairs_path),
            "params_path": str(block_params_path),
            "status": "ok",
            "skip_reason": "",
            "n_trades": int(len(block_trades)),
            "trades_path": str(block_trades_path),
            "research_report_path": str(block_research_path) if research_path.exists() else "",
            "zero_mtm_report_path": str(block_zero_path) if zero_path.exists() else "",
        })

        if research_path.exists():
            shutil.copy2(research_path, block_research_path)
        if zero_path.exists() and zero_path.stat().st_size > 0:
            try:
                zero_df = pd.read_csv(zero_path)
                zero_block_reports.append(zero_df)
            except Exception:
                pass
            shutil.copy2(zero_path, block_zero_path)
        if universe_path.exists() and universe_path.stat().st_size > 0:
            try:
                uni_df = pd.read_csv(universe_path)
            except Exception:
                uni_df = pd.DataFrame()
            if not uni_df.empty:
                uni_df["snapshot_id"] = snapshot_id
                universe_rows.append(uni_df)
            shutil.copy2(universe_path, block_universe_path)

    stitched = pd.concat(stitched_trades, ignore_index=True) if stitched_trades else pd.DataFrame()
    stitched = stitched.sort_values("exit_time").reset_index(drop=True) if not stitched.empty else stitched
    save_with_parquet(stitched, FINAL_TRADES_PATH, index=False)

    global_hmm = _load_global_hmm()
    research = _compute_research_report(stitched, global_hmm)
    zero_mtm = _compute_zero_mtm_report(stitched, zero_block_reports)
    curated = _curate_universe(universe_rows)

    if not curated.empty:
        save_with_parquet(curated, FINAL_CURATED_PATH, index=False)
    else:
        pd.DataFrame().to_csv(FINAL_CURATED_PATH, index=False)

    # Separation of concerns: research vs. Zero MTM reports are NEVER merged.
    research_row = {**research, "blocks": int(len(manifest)), "symbols_curated": int(len(curated))}
    zero_row = {**zero_mtm, "blocks": int(len(manifest))}
    _RESEARCH_FORBIDDEN = {"daily_breach", "trailing_breach", "forced_liquidation_count",
                           "max_liquidation_drawdown", "inactivity_days"}
    _ZERO_FORBIDDEN = {"sharpe", "sharpe_annualized", "hit_rate", "trade_hit_rate",
                       "bootstrap_sharpe_lo", "bootstrap_sharpe_hi",
                       "bootstrap_total_pnl_lo", "bootstrap_total_pnl_hi",
                       "total_pnl", "gross_pnl"}
    leak_r = _RESEARCH_FORBIDDEN & set(research_row.keys())
    leak_z = _ZERO_FORBIDDEN & set(zero_row.keys())
    assert not leak_r, f"research report leaks Zero fields: {leak_r}"
    assert not leak_z, f"zero report leaks research fields: {leak_z}"

    save_with_parquet(pd.DataFrame([research_row]), FINAL_RESEARCH_PATH, index=False)
    save_with_parquet(pd.DataFrame([zero_row]), FINAL_ZERO_MTM_PATH, index=False)

    # Per-block manifest
    if block_manifest:
        save_with_parquet(pd.DataFrame(block_manifest), FINAL_MANIFEST_PATH, index=False)
    else:
        pd.DataFrame().to_csv(FINAL_MANIFEST_PATH, index=False)

    # Run config for reproducibility
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False,
        ).stdout.strip() or None
    except Exception:
        git_commit = None
    run_config = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "train_months": args.train_months,
        "test_months": args.test_months,
        "step_months": args.step_months,
        "max_snapshots": args.max_snapshots,
        "limit_blocks": args.limit_blocks,
        "universe_dir": str(OOS_UNIVERSE_DIR),
        "params_dir": str(OOS_PARAMS_DIR),
        "out_dir": str(DATA_DIR),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "python_version": sys.version.split()[0],
        "command_line_args": sys.argv,
        "n_blocks_processed": int(len(block_manifest)),
    }
    FINAL_RUN_CONFIG.write_text(json.dumps(run_config, indent=2, default=str))

    _log(
        f"pipeline completed: n_trades={research['trades']} "
        f"daily_sharpe={research['daily_sharpe_annualized']:.2f} "
        f"trade_sharpe={research['trade_sharpe_annualized']:.2f} "
        f"daily_breach={zero_mtm['daily_breach']} trailing_breach={zero_mtm['trailing_breach']} "
        f"forced_liquidations={zero_mtm['forced_liquidation_count']}",
        log_lines,
    )
    FINAL_RUN_LOG.write_text("\n".join(log_lines) + "\n")

    print(f"\n{'='*72}")
    print("TRUE OOS ZERO RESULT")
    print(f"{'='*72}")
    print(f"Blocks processed:      {len(manifest)}")
    print(f"Stitched OOS trades:   {research['trades']}")
    print(f"Daily Sharpe:          {research['daily_sharpe_annualized']:.2f}")
    print(f"Trade Sharpe:          {research['trade_sharpe_annualized']:.2f}  (diagnostic; high-frequency inflated)")
    print(f"Research hit rate:     {research['trade_hit_rate']:.1f}%")
    print(f"Bootstrap trade Sharpe CI: [{research['bootstrap_sharpe_lo']:.2f}, {research['bootstrap_sharpe_hi']:.2f}]")
    print(f"Bootstrap P&L CI:      [{research['bootstrap_total_pnl_lo']:+.2f}, {research['bootstrap_total_pnl_hi']:+.2f}]")
    print(f"Zero daily breach:     {'BREACH' if zero_mtm['daily_breach'] else 'OK'}")
    print(f"Zero trailing breach:  {'BREACH' if zero_mtm['trailing_breach'] else 'OK'}")
    print(f"Forced liquidations:   {zero_mtm['forced_liquidation_count']}")
    print(f"Max liquidation DD:    {zero_mtm['max_liquidation_drawdown']:+.2f}")
    print(f"Curated universe:      {len(curated)} pairs")
    if not curated.empty:
        print(curated.head(20).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
