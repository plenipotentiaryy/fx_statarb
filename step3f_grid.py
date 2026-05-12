"""
step4d_pair_grid.py — Per-pair parameter optimisation (train) + validation (test).

Grid (18 combos per pair):
  entry_z : 1.65 / 1.7 / 1.8
  exit_z  : -0.1 / 0.0
  stop_z  : 3.0 / 3.2 / 3.5

Stop ceiling 4.0: beyond Z=4 cointegration is likely broken (0.007% prob of random fluctuation).

Workflow:
  1. Run grid on TRAINING data only  → find best params per pair (by Sharpe)
  2. Validate those params on TEST data (OOS check)
  3. Save per-pair optimal params → data/optimal_params.csv
     (step4_backtest.py reads this and uses per-pair settings)
  4. Heatmaps + equity curves

Why train/test split here:
  We optimise on the first ~55 % of the data and validate on the last 45 %.
  This avoids in-sample overfitting while still customising per pair.
"""

import itertools
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import (
    CLOSES_FILE, COST_TAKER, BORROW_RATE_ANNUAL,
    RTH_START, RTH_END, SIGNAL_START, TRAIN_RATIO,
    BARS_PER_DAY, DATA_DIR, OUTPUT_DIR, PAIRS,
    USE_VWZ, VWZ_MIN_VOLUME, USE_VWAP,
    USE_RETURN_SPREAD, RETURN_WINDOW,
    USE_VELOCITY_GATE, VELOCITY_WINDOW,
    USE_RVOL_GATE, RVOL_THRESHOLD, RVOL_WINDOW,
    USE_VWAP_MTF, VWAP_MTF_TF,
    KALMAN_DELTA, TARGET_RISK_USD
)
from kalman import kalman_hedge
from data_loader import load_volumes, load_vwaps

ENTRY_Z_GRID = [1.9, 2.0, 2.2, 2.5, 3.0]   # raised floor; filters mean higher Z is viable
STOP_Z_GRID  = [3.2, 3.5, 4.0]
EXIT_Z_GRID  = [-0.1, 0.0]
COMBOS       = list(itertools.product(ENTRY_Z_GRID, EXIT_Z_GRID, STOP_Z_GRID))
N_COMBOS     = len(COMBOS)
MIN_TRADES   = 8


# ── Numba JIT grid kernel ─────────────────────────────────────────────────────
# Runs ALL combos in a single pass over the bars.
# Zero-Pandas rule: only float64 / int64 NumPy arrays inside.
#
# results[c] = [total_pnl, n_trades, n_wins, sum_pnl_sq, max_dd, n_stops]

def _grid_kernel(zscore:    np.ndarray,   # float64[n_bars]
                 spread:    np.ndarray,   # float64[n_bars]
                 velocity:  np.ndarray,   # float64[n_bars]
                 rvol:      np.ndarray,   # float64[n_bars]
                 sv_mtf:    np.ndarray,   # float64[n_bars]
                 t1_price:  np.ndarray,   # float64[n_bars]
                 t2_price:  np.ndarray,   # float64[n_bars]
                 entry_arr: np.ndarray,   # float64[n_combos]
                 exit_arr:  np.ndarray,   # float64[n_combos]
                 stop_arr:  np.ndarray,   # float64[n_combos]
                 beta:           float,
                 cost_taker:     float,
                 borrow_rate:    float,
                 bars_per_day:   float,
                 use_vel_gate:   bool,
                 use_rvol_gate:  bool,
                 rvol_thresh:    float,
                 use_mtf_vwap:   bool) -> np.ndarray:

    n_bars   = len(zscore)
    n_combos = len(entry_arr)

    # Per-combo mutable state (pre-allocated, no Python lists)
    pos    = np.zeros(n_combos, dtype=np.int64)
    e_sp   = np.zeros(n_combos)
    e_t1   = np.zeros(n_combos)
    e_t2   = np.zeros(n_combos)
    e_bar  = np.zeros(n_combos, dtype=np.int64)

    # Results matrix — 6 stats per combo
    results  = np.zeros((n_combos, 6))
    cum_pnl  = np.zeros(n_combos)
    peak_pnl = np.zeros(n_combos)

    for i in range(n_bars):
        z  = zscore[i]
        s  = spread[i]
        p1 = t1_price[i]
        p2 = t2_price[i]

        for c in range(n_combos):
            ez = entry_arr[c]
            xz = exit_arr[c]
            sz = stop_arr[c]
            pc = pos[c]

            # ── Exit / stop ───────────────────────────────────────────────
            if pc != 0:
                is_exit = (pc == 1 and z >= xz) or (pc == -1 and z <= -xz)
                is_stop = (pc == 1 and z <= -sz) or (pc == -1 and z >= sz)

                if is_exit or is_stop:
                    # gross is in 'return units' (e.g. 0.0010 = 10 bps)
                    gross    = pc * (s - e_sp[c])
                    
                    # tx cost in return units: cost_taker per leg, scaled by leg weight
                    # (1.0 for leg 1, beta for leg 2)
                    tx       = (1.0 + beta) * cost_taker * 2.0  # open + close
                    
                    hold_d   = (i - e_bar[c]) / bars_per_day
                    # borrow cost (simplified percentage)
                    borrow   = (1.0 + beta) * (borrow_rate / 252.0) * hold_d
                    
                    net      = gross - tx - borrow
                    
                    if np.isnan(net):
                        pos[c] = 0
                        continue

                    results[c, 0] += net          # total_pnl
                    results[c, 1] += 1.0          # n_trades
                    if net > 0.0:
                        results[c, 2] += 1.0      # n_wins
                    results[c, 3] += net * net    # sum_pnl_sq  (for Sharpe std)
                    if is_stop:
                        results[c, 5] += 1.0      # n_stops

                    cum_pnl[c] += net
                    if cum_pnl[c] > peak_pnl[c]:
                        peak_pnl[c] = cum_pnl[c]
                    dd = cum_pnl[c] - peak_pnl[c]
                    if dd < results[c, 4]:
                        results[c, 4] = dd        # max_dd (negative number)

                    pos[c] = 0

            # ── Entry ─────────────────────────────────────────────────────
            if pos[c] == 0:
                # 1. RVOL check
                if use_rvol_gate and (np.isnan(rvol[i]) or rvol[i] < rvol_thresh):
                    continue

                if z < -ez:
                    # 2. Velocity gate
                    if use_vel_gate and (np.isnan(velocity[i]) or velocity[i] < 0):
                        continue
                    # 3. MTF VWAP Anchor
                    if use_mtf_vwap and not np.isnan(sv_mtf[i]) and s > sv_mtf[i]:
                        continue
                    pos[c]  = 1
                    e_sp[c] = s; e_t1[c] = p1; e_t2[c] = p2; e_bar[c] = i
                elif z > ez:
                    # 2. Velocity gate
                    if use_vel_gate and (np.isnan(velocity[i]) or velocity[i] > 0):
                        continue
                    # 3. MTF VWAP Anchor
                    if use_mtf_vwap and not np.isnan(sv_mtf[i]) and s < sv_mtf[i]:
                        continue
                    pos[c]  = -1
                    e_sp[c] = s; e_t1[c] = p1; e_t2[c] = p2; e_bar[c] = i

    return results


# ── Python wrapper ────────────────────────────────────────────────────────────

def run_grid_numba(df: pd.DataFrame, t1: str, t2: str,
                   beta: float, combos: list, days: float,
                   pair: str = "") -> pd.DataFrame:
    """
    Run full grid search via the Numba kernel in one pass.

    Parameters
    ----------
    df     : DataFrame with columns zscore, spread, {t1}_close, {t2}_close
    combos : list of (entry_z, exit_z, stop_z) — invalid ones are filtered
    days   : number of calendar days in the period (for annualised Sharpe)

    Returns
    -------
    DataFrame with one row per valid combo, sorted by Sharpe descending.
    """
    # Filter invalid combos (stop must be > entry, exit must be < entry)
    valid = [(ez, xz, sz) for ez, xz, sz in combos if sz > ez and xz < ez]
    if not valid:
        return pd.DataFrame()

    # Pandas → contiguous float64 NumPy arrays (zero-copy where possible)
    zscore   = np.ascontiguousarray(df["zscore"].to_numpy(np.float64))
    spread   = np.ascontiguousarray(df["spread"].to_numpy(np.float64))
    velocity = np.ascontiguousarray(df["velocity"].to_numpy(np.float64)) if "velocity" in df.columns else np.zeros_like(zscore)
    rvol     = np.ascontiguousarray(df["rvol"].to_numpy(np.float64)) if "rvol" in df.columns else np.ones_like(zscore)
    sv_mtf   = np.ascontiguousarray(df["spread_vwap_mtf"].to_numpy(np.float64)) if "spread_vwap_mtf" in df.columns else np.full_like(zscore, np.nan)
    t1_price = np.ascontiguousarray(df[f"{t1}_close"].to_numpy(np.float64))
    t2_price = np.ascontiguousarray(df[f"{t2}_close"].to_numpy(np.float64))

    entry_arr = np.array([c[0] for c in valid], dtype=np.float64)
    exit_arr  = np.array([c[1] for c in valid], dtype=np.float64)
    stop_arr  = np.array([c[2] for c in valid], dtype=np.float64)

    results = _grid_kernel(zscore, spread, velocity, rvol, sv_mtf, t1_price, t2_price,
                           entry_arr, exit_arr, stop_arr,
                           float(beta),
                           float(COST_TAKER),
                           float(BORROW_RATE_ANNUAL),
                           float(BARS_PER_DAY),
                           bool(USE_VELOCITY_GATE),
                           bool(USE_RVOL_GATE),
                           float(RVOL_THRESHOLD),
                           bool(USE_VWAP_MTF))

    # Convert results matrix → DataFrame
    years = max(days / 365.25, 1e-9)
    rows  = []
    for i, (ez, xz, sz) in enumerate(valid):
        n_trades = int(results[i, 1])
        if n_trades < MIN_TRADES:
            continue

        total_pnl = results[i, 0]
        n_wins    = int(results[i, 2])
        sum_pnl2  = results[i, 3]
        max_dd    = results[i, 4]
        n_stops   = int(results[i, 5])

        mean_pnl  = total_pnl / n_trades
        var_pnl   = max(sum_pnl2 / n_trades - mean_pnl ** 2, 0.0)
        std_pnl   = np.sqrt(var_pnl)
        tpy       = n_trades / years
        sharpe    = mean_pnl / std_pnl * np.sqrt(tpy) if std_pnl > 0.0 else 0.0

        rows.append({
            "pair":      pair,
            "entry_z":   ez,
            "exit_z":    xz,
            "stop_z":    sz,
            "trades":    n_trades,
            "win_rate":  n_wins / n_trades * 100.0,
            "sharpe":    sharpe,
            "total_pnl": total_pnl,
            "avg_pnl":   mean_pnl,
            "max_dd":    max_dd,
            "stop_rate": n_stops / n_trades * 100.0,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_closes_split() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Returns (train_closes, test_closes, days_train)."""
    path = DATA_DIR / CLOSES_FILE
    if not path.exists():
        path = DATA_DIR / "closes_15min.csv"
    closes = pd.read_csv(path, index_col=0, parse_dates=True)
    closes.index = pd.to_datetime(closes.index, utc=True).tz_convert("US/Eastern")
    closes = closes.between_time(RTH_START, RTH_END)

    pairs_path = DATA_DIR / "pairs_selected.csv"
    if pairs_path.exists():
        meta = pd.read_csv(pairs_path)
        if not meta.empty:
            needed    = {t for p in meta["pair"] for t in p.split("-")}
            available = [t for t in needed if t in closes.columns]
            # Keep ALL rows — each pair drops its own NaN in build_signals
            closes = closes[available]

            if "test_start_date" in meta.columns:
                test_start = pd.Timestamp(meta["test_start_date"].iloc[0]).tz_localize("US/Eastern")
                train = closes[closes.index < test_start]
                test  = closes[closes.index >= test_start]
                if len(train) > 200 and len(test) > 200:
                    # days_train = full calendar span of training data
                    days_train = (train.index[-1] - train.index[0]).days
                    return train, test, float(days_train)

    # Fallback: split by TRAIN_RATIO, no NaN-dropping across all pairs
    n     = len(closes)
    split = int(n * TRAIN_RATIO)
    train = closes.iloc[:split]
    test  = closes.iloc[split:]
    return train, test, float((train.index[-1] - train.index[0]).days)


def _combined_volume(volumes: pd.DataFrame | None,
                     index: pd.DatetimeIndex,
                     t1: str,
                     t2: str) -> pd.Series | None:
    if volumes is None or t1 not in volumes.columns or t2 not in volumes.columns:
        return None
    vol = (
        volumes[t1].reindex(index).fillna(0.0)
        + volumes[t2].reindex(index).fillna(0.0)
    )
    return vol.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=VWZ_MIN_VOLUME)


def _volume_weighted_zscore(spread: pd.Series,
                            vol: pd.Series,
                            window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    w_sum = vol.rolling(window=window, min_periods=window).sum()
    mean = (spread * vol).rolling(window=window, min_periods=window).sum() / w_sum
    var = (((spread - mean) ** 2) * vol).rolling(window=window, min_periods=window).sum() / w_sum
    std = var.pow(0.5)
    zscore = (spread - mean) / std.replace(0, np.nan)
    return zscore, mean, std


def build_signals(closes: pd.DataFrame, t1: str, t2: str,
                  beta: float, half_life: float,
                  volumes: pd.DataFrame | None = None,
                  vwaps: pd.DataFrame | None = None) -> pd.DataFrame:
    # 1. Price input
    if USE_VWAP and vwaps is not None and t1 in vwaps.columns and t2 in vwaps.columns:
        p1 = vwaps[t1].reindex(closes.index).fillna(closes[t1])
        p2 = vwaps[t2].reindex(closes.index).fillna(closes[t2])
    else:
        p1, p2 = closes[t1], closes[t2]

    vol = _combined_volume(volumes, closes.index, t1, t2)

    # 2. Return-spread mode
    if USE_RETURN_SPREAD:
        r1 = p1.pct_change().fillna(0)
        r2 = p2.pct_change().fillna(0)
        cum_r1 = r1.rolling(RETURN_WINDOW).sum()
        cum_r2 = r2.rolling(RETURN_WINDOW).sum()
        
        # Use Kalman for smoothed beta
        _, beta_arr, _, _ = kalman_hedge(
            cum_r1.fillna(0).values, cum_r2.fillna(0).values,
            delta=KALMAN_DELTA, beta_init=float(beta),
        )
        beta_s = pd.Series(beta_arr, index=closes.index)
        spread = (cum_r1 - beta_s * cum_r2).rename("spread")
        window = RETURN_WINDOW * 3
        if USE_VWZ and vol is not None:
            zscore, spread_mean, spread_std = _volume_weighted_zscore(spread, vol, window)
        else:
            spread_mean = spread.rolling(window).mean()
            spread_std  = (spread - spread_mean).rolling(window).std()
            zscore      = (spread - spread_mean) / spread_std.replace(0, np.nan)
    else:
        spread = p1 - beta * p2
        window = max(20, min(int(half_life), 200))
        
        if USE_VWZ and vol is not None:
            zscore, spread_mean, spread_std = _volume_weighted_zscore(spread, vol, window)
        else:
            spread_mean = spread.rolling(window).mean()
            spread_std = spread.rolling(window).std()
            zscore = (spread - spread_mean) / spread_std

    velocity = zscore.diff(VELOCITY_WINDOW)

    # 3. RVOL: Relative Volume
    if vol is not None:
        rvol = vol / vol.rolling(RVOL_WINDOW, min_periods=RVOL_WINDOW).mean().replace(0, np.nan)
    else:
        rvol = pd.Series(1.0, index=closes.index)

    # 4. MTF VWAP Anchor
    if USE_VWAP_MTF and vwaps is not None and t1 in vwaps.columns and t2 in vwaps.columns:
        p1_v = vwaps[t1].reindex(closes.index).fillna(closes[t1])
        p2_v = vwaps[t2].reindex(closes.index).fillna(closes[t2])
        spread_v = p1_v - beta_s * p2_v if USE_RETURN_SPREAD else p1_v - beta * p2_v
        spread_vwap_mtf = spread_v.resample(VWAP_MTF_TF).mean().reindex(closes.index, method="ffill")
    else:
        spread_vwap_mtf = pd.Series(np.nan, index=closes.index)

    return pd.DataFrame({
        "zscore":          zscore,
        "spread":          spread,
        "spread_mean":     spread_mean,
        "spread_std":      spread_std,
        "velocity":        velocity,
        "rvol":            rvol,
        "spread_vwap_mtf": spread_vwap_mtf,
        f"{t1}_close":     closes[t1],
        f"{t2}_close":     closes[t2],
    }).dropna().between_time(SIGNAL_START, RTH_END)


# ── Backtest engine ───────────────────────────────────────────────────────────

def backtest(df: pd.DataFrame, t1: str, t2: str, beta: float,
             entry_z: float, exit_z: float, stop_z: float) -> list[dict]:
    t1c, t2c = f"{t1}_close", f"{t2}_close"
    pos = es = et1 = et2 = 0.0
    ebar = 0
    trades = []

    for i in range(len(df)):
        z  = df["zscore"].iloc[i]
        s  = df["spread"].iloc[i]
        p1 = df[t1c].iloc[i]
        p2 = df[t2c].iloc[i]

        if pos != 0:
            ex = (pos == 1 and z >= exit_z) or (pos == -1 and z <= -exit_z)
            st = (pos == 1 and z <= -stop_z) or (pos == -1 and z >= stop_z)
            if ex or st:
                gross    = pos * (s - es)
                notional = et1 + beta * et2
                tx       = 2 * notional * COST_TAKER
                hold_d   = (i - ebar) / BARS_PER_DAY
                borrow   = (beta * et2 if pos == 1 else et1) * BORROW_RATE_ANNUAL * hold_d / 252
                trades.append({
                    "net_pnl":      gross - tx - borrow,
                    "gross_pnl":    gross,
                    "exit_reason":  "STOP" if st else "SIGNAL",
                    "holding_bars": i - ebar,
                })
                pos = 0

        if pos == 0:
            if z < -entry_z:   pos = 1
            elif z > entry_z:  pos = -1
            if pos != 0:
                es = s; et1 = p1; et2 = p2; ebar = i

    return trades


def calc_metrics(trades: list[dict], days: float) -> dict | None:
    if len(trades) < MIN_TRADES:
        return None
    pnl   = np.array([t["net_pnl"] for t in trades])
    curve = np.cumsum(pnl)
    wr    = float((pnl > 0).mean() * 100)
    tpy   = len(pnl) / max(days / 365.25, 0.01)
    sh    = float(pnl.mean() / pnl.std() * np.sqrt(tpy)) if pnl.std() > 0 else 0.0
    dd    = float((curve - np.maximum.accumulate(curve)).min())
    wins  = pnl[pnl > 0].sum()
    loss  = abs(pnl[pnl <= 0].sum())
    pf    = float(wins / loss) if loss > 0 else float("inf")
    stops = sum(1 for t in trades if t["exit_reason"] == "STOP")
    return {
        "trades":    len(pnl),
        "win_rate":  wr,
        "sharpe":    sh,
        "total_pnl": float(pnl.sum()),
        "avg_pnl":   float(pnl.mean()),
        "max_dd":    dd,
        "pf":        pf,
        "stops":     stops,
        "stop_rate": stops / len(pnl) * 100,
        "curve":     curve,
    }


# ── Load data ─────────────────────────────────────────────────────────────────

closes_train, closes_test, days_train = load_closes_split()
days_test = (closes_test.index[-1] - closes_test.index[0]).days
pairs = pd.read_csv(DATA_DIR / "pairs_selected.csv")

if pairs.empty:
    raise SystemExit("pairs_selected.csv is empty — run step2_pairs.py first")

print(f"Per-pair grid  |  {N_COMBOS} combos  |  optimise on TRAIN → validate on TEST")
print(f"TRAIN: {closes_train.index[0].date()} → {closes_train.index[-1].date()}  "
      f"({days_train:.0f} days)")
print(f"TEST:  {closes_test.index[0].date()}  → {closes_test.index[-1].date()}   "
      f"({days_test} days)")
print(f"Entry : {ENTRY_Z_GRID}")
print(f"Exit  : {EXIT_Z_GRID}")
print(f"Stop  : {STOP_Z_GRID}\n")

OUTPUT_DIR.mkdir(exist_ok=True)

# ── Per-pair grid search ──────────────────────────────────────────────────────

all_best: list[dict] = []
all_pair_train_results: dict = {}

for _, row in pairs.iterrows():
    pair_name = row["pair"]
    t1, t2    = pair_name.split("-")
    
    # ── Only process pairs defined in config.py ──────────────────────────────
    if (t1, t2) not in PAIRS and (t2, t1) not in PAIRS:
        continue
        
    beta      = float(row.get("beta_daily", row["beta"]) or row["beta"])
    half_life = float(row["half_life_bars"])

    if t1 not in closes_train.columns or t2 not in closes_train.columns:
        print(f"  SKIP {pair_name}: missing data")
        continue

    # ── Load volumes/vwaps for VW-Z/VWAP ─────────────────────────────────────
    volumes = load_volumes() if USE_VWZ else None
    vwaps   = load_vwaps()   if USE_VWAP else None

    # ── Build signals for both splits ────────────────────────────────────────
    sig_train = build_signals(closes_train, t1, t2, beta, half_life, volumes, vwaps)
    sig_test  = build_signals(closes_test,  t1, t2, beta, half_life, volumes, vwaps)

    if len(sig_train) < 200:
        print(f"  SKIP {pair_name}: too few train bars ({len(sig_train)})")
        continue

    print(f"  {pair_name}  train={len(sig_train)} bars  test={len(sig_test)} bars ...",
          end="", flush=True)

    # ── Grid search on TRAIN — single Numba pass over all combos ─────────────
    df_train = run_grid_numba(sig_train, t1, t2, beta, COMBOS, days_train, pair_name)

    if df_train.empty:
        print("  no valid train combos")
        continue

    all_pair_train_results[pair_name] = df_train
    best_train = df_train.iloc[0]
    
    if best_train["sharpe"] < 0.05:
        print(f"  FAILED: best train Sharpe ({best_train['sharpe']:.2f}) too low")
        continue

    # ── Validate best params on TEST — single-combo Numba pass ───────────────
    best_combo = [(float(best_train["entry_z"]),
                   float(best_train["exit_z"]),
                   float(best_train["stop_z"]))]
    df_test_best = run_grid_numba(sig_test, t1, t2, beta, best_combo, days_test)
    if df_test_best.empty:
        test_m = {"trades": 0, "win_rate": 0.0, "sharpe": 0.0,
                  "total_pnl": 0.0, "max_dd": 0.0, "stop_rate": 0.0}
    else:
        test_m = df_test_best.iloc[0].to_dict()

    print(f"  TRAIN best → entry={best_train['entry_z']}  "
          f"exit={best_train['exit_z']:+.1f}  stop={best_train['stop_z']}  "
          f"Sh={best_train['sharpe']:.2f}  "
          f"│  TEST Sh={test_m['sharpe']:.2f}  WR={test_m['win_rate']:.0f}%")

    all_best.append({
        "pair":            pair_name,
        "entry_z":         float(best_train["entry_z"]),
        "exit_z":          float(best_train["exit_z"]),
        "stop_z":          float(best_train["stop_z"]),
        # train metrics
        "train_sharpe":    round(float(best_train["sharpe"]), 2),
        "train_wr":        round(float(best_train["win_rate"]), 1),
        "train_trades":    int(best_train["trades"]),
        "train_pnl":       round(float(best_train["total_pnl"]), 4),
        # test (OOS) metrics
        "test_sharpe":     round(float(test_m["sharpe"]), 2),
        "test_wr":         round(float(test_m["win_rate"]), 1),
        "test_trades":     int(test_m["trades"]),
        "test_pnl":        round(float(test_m["total_pnl"]), 4),
    })

print()
if not all_best:
    raise SystemExit("No valid results found.")

df_best = pd.DataFrame(all_best)

# ── Summary table ─────────────────────────────────────────────────────────────

print("=" * 110)
print(f"{'OPTIMAL PARAMS PER PAIR':^110}")
print("=" * 110)
print(f"{'Pair':<10} {'Entry':>6} {'Exit':>6} {'Stop':>6}  "
      f"{'── TRAIN ──':^30}  {'── TEST (OOS) ──':^30}")
print(f"{'':>30}  {'Sh':>8} {'WR':>6} {'Tr':>5} {'P&L':>10}  "
      f"{'Sh':>8} {'WR':>6} {'Tr':>5} {'P&L':>10}")
print("-" * 110)
for _, r in df_best.iterrows():
    oos_flag = " ✓" if r["test_sharpe"] > 0 else " ✗"
    print(f"{r['pair']:<10} {r['entry_z']:>6.1f} {r['exit_z']:>+6.1f} {r['stop_z']:>6.2f}  "
          f"{r['train_sharpe']:>8.2f} {r['train_wr']:>5.1f}% {r['train_trades']:>5} "
          f"{r['train_pnl']:>+10.4f}  "
          f"{r['test_sharpe']:>8.2f} {r['test_wr']:>5.1f}% {r['test_trades']:>5} "
          f"{r['test_pnl']:>+10.4f}{oos_flag}")
print("=" * 110)

save_cols = ["pair", "entry_z", "exit_z", "stop_z",
             "train_sharpe", "train_wr", "train_trades", "train_pnl",
             "test_sharpe", "test_wr", "test_trades", "test_pnl"]
df_best[save_cols].to_csv(DATA_DIR / "optimal_params.csv", index=False)
print(f"\nSaved optimal params → data/optimal_params.csv")
print("step4_backtest.py will use these per-pair parameters automatically.\n")

# ── Per-pair ranked tables (train) ────────────────────────────────────────────
print()
for pair_name, df_t in all_pair_train_results.items():
    top = df_t.head(10)
    print(f"\n{'─'*75}  {pair_name}  top-10 TRAIN combos")
    print(f"{'#':>3} {'entry':>6} {'exit':>6} {'stop':>6} "
          f"{'trades':>7} {'WR':>6} {'Sharpe':>8} {'P&L':>10} {'Stop%':>6}")
    for rank, (_, r) in enumerate(top.iterrows()):
        m = " ◄" if rank == 0 else ""
        print(f"{rank+1:>3} {r['entry_z']:>6.1f} {r['exit_z']:>+6.1f} {r['stop_z']:>6.2f} "
              f"{r['trades']:>7.0f} {r['win_rate']:>5.1f}% {r['sharpe']:>8.2f} "
              f"{r['total_pnl']:>+10.4f} {r['stop_rate']:>5.1f}%{m}")

# ── Per-pair heatmaps (train) ─────────────────────────────────────────────────
for pair_name, df_t in all_pair_train_results.items():
    n_stop = len(STOP_Z_GRID)
    fig, axes = plt.subplots(1, n_stop, figsize=(5 * n_stop, 5))
    if n_stop == 1:
        axes = [axes]

    vmin, vmax = df_t["sharpe"].min(), df_t["sharpe"].max()

    for ax, stop_z in zip(axes, STOP_Z_GRID):
        sub = df_t[df_t["stop_z"] == stop_z]
        if sub.empty:
            ax.set_visible(False)
            continue
        pivot = sub.pivot(index="entry_z", columns="exit_z", values="sharpe")
        pivot = pivot.reindex(index=sorted(pivot.index, reverse=True),
                               columns=sorted(pivot.columns, reverse=True))

        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:+.1f}" for v in pivot.columns], fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.1f}" for v in pivot.index], fontsize=9)
        ax.set_xlabel("exit_z")
        ax.set_ylabel("entry_z")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    c = "white" if abs(val) > 0.5 * max(abs(vmin), abs(vmax)) else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, fontweight="bold", color=c)
        ax.set_title(f"stop={stop_z}", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(f"{pair_name}  TRAIN Sharpe grid (entry × exit per stop)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = OUTPUT_DIR / f"pair_grid_{pair_name.replace('-', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Heatmap → {out}")

# ── Top-5 test equity curves per pair ────────────────────────────────────────
for _, row in df_best.iterrows():
    pair_name = row["pair"]
    t1, t2    = pair_name.split("-")
    _pair_row = pairs.loc[pairs["pair"] == pair_name]
    beta      = float((_pair_row["beta_daily"] if "beta_daily" in pairs.columns
                       else _pair_row["beta"]).fillna(_pair_row["beta"]).iloc[0])
    hl        = float(pairs.loc[pairs["pair"] == pair_name, "half_life_bars"].iloc[0])

    sig_test  = build_signals(closes_test, t1, t2, beta, hl)
    if sig_test.empty or t1 not in closes_test.columns:
        continue

    df_t = all_pair_train_results.get(pair_name)
    if df_t is None:
        continue

    colors_5 = plt.cm.RdYlGn(np.linspace(0.15, 0.85, 5))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: top-5 OOS equity curves — use old Python backtest() for curve shape
    ax = axes[0]
    for rank, (_, r) in enumerate(df_t.head(5).iterrows()):
        tr = backtest(sig_test, t1, t2, beta,
                      float(r["entry_z"]), float(r["exit_z"]), float(r["stop_z"]))
        if not tr:
            continue
        curve = np.cumsum([t["net_pnl"] for t in tr])
        lbl   = (f"#{rank+1} e={r['entry_z']} x={r['exit_z']:+.1f} s={r['stop_z']}  "
                 f"[train Sh={r['sharpe']:.2f}]")
        ax.plot(curve, label=lbl, color=colors_5[rank],
                lw=2.5 if rank == 0 else 1.2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"{pair_name}  TEST equity curves (train-optimal params)")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative net P&L")
    ax.legend(fontsize=7)

    # Right: train vs test Sharpe scatter — Numba batch for all combos at once
    ax2 = axes[1]
    merged = df_t.copy()
    all_test_combos = list(zip(merged["entry_z"], merged["exit_z"], merged["stop_z"]))
    df_test_all = run_grid_numba(sig_test, t1, t2, beta,
                                  all_test_combos, days_test)
    key_cols = ["entry_z", "exit_z", "stop_z", "sharpe"]
    if not df_test_all.empty:
        df_test_all = df_test_all[key_cols].rename(columns={"sharpe": "test_sharpe"})
        merged = merged.merge(df_test_all, on=["entry_z", "exit_z", "stop_z"], how="left")
        merged["test_sharpe"] = merged["test_sharpe"].fillna(0.0)
    else:
        merged["test_sharpe"] = 0.0

    ax2.scatter(merged["sharpe"], merged["test_sharpe"], alpha=0.5, s=20, c="steelblue")
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.axvline(0, color="gray", lw=0.8, ls="--")
    ax2.set_xlabel("TRAIN Sharpe")
    ax2.set_ylabel("TEST Sharpe")
    ax2.set_title(f"{pair_name}  Train vs Test Sharpe (all {len(merged)} combos)")

    corr = merged[["sharpe", "test_sharpe"]].corr().iloc[0, 1]
    ax2.text(0.05, 0.95, f"r = {corr:.2f}", transform=ax2.transAxes,
             fontsize=10, va="top")

    plt.tight_layout()
    out = OUTPUT_DIR / f"pair_traintest_{pair_name.replace('-', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Train/Test chart → {out}")
