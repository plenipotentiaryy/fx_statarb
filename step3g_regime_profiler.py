"""
regime_profiler.py — Regime-Conditioned Dynamic Z-Score Threshold Profiling (RCDP).

Splits historical data by HMM regime (Normal=0, Volatile=1) and runs
independent grid searches on each slice to find optimal entry/exit/stop
thresholds per regime per pair.

Output: data/regime_thresholds.csv
    pair, regime, entry_z, exit_z, stop_z, sharpe, trades, win_rate

The backtest (backtest.py) loads this file and dynamically switches
thresholds based on the HMM regime label at each bar.

Pipeline position: 3g (after hmm.py, before grid.py and backtest.py)
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import (
    CLOSES_FILE, COST_PER_SIDE, BORROW_RATE_ANNUAL,
    RTH_START, RTH_END, SIGNAL_START, TRAIN_RATIO,
    BARS_PER_DAY, DATA_DIR, OUTPUT_DIR,
    ENTRY_Z, EXIT_Z, STOP_Z, ENTRY_Z_VOLATILE,
    RCDP_ENTRY_GRID, RCDP_EXIT_GRID, RCDP_STOP_GRID,
    RCDP_MIN_TRADES_VOLATILE,
)

# Minimum trades for normal-regime slice (same as grid.py)
MIN_TRADES_NORMAL = 8

# Build combo list
COMBOS = list(itertools.product(RCDP_ENTRY_GRID, RCDP_EXIT_GRID, RCDP_STOP_GRID))
N_COMBOS = len(COMBOS)

REGIME_NAMES = {0: "Normal", 1: "Volatile"}


# ── Grid kernel (pure Python — same logic as grid.py) ────────────────────────

def _grid_kernel(zscore:    np.ndarray,
                 spread:    np.ndarray,
                 t1_price:  np.ndarray,
                 t2_price:  np.ndarray,
                 entry_arr: np.ndarray,
                 exit_arr:  np.ndarray,
                 stop_arr:  np.ndarray,
                 beta:           float,
                 cost_per_side:  float,
                 borrow_rate:    float,
                 bars_per_day:   float) -> np.ndarray:

    n_bars   = len(zscore)
    n_combos = len(entry_arr)

    pos    = np.zeros(n_combos, dtype=np.int64)
    e_sp   = np.zeros(n_combos)
    e_t1   = np.zeros(n_combos)
    e_t2   = np.zeros(n_combos)
    e_bar  = np.zeros(n_combos, dtype=np.int64)

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

            if pc != 0:
                is_exit = (pc == 1 and z >= xz) or (pc == -1 and z <= -xz)
                is_stop = (pc == 1 and z <= -sz) or (pc == -1 and z >= sz)

                if is_exit or is_stop:
                    gross    = pc * (s - e_sp[c])
                    notional = e_t1[c] + beta * e_t2[c]
                    tx       = 2.0 * notional * cost_per_side
                    hold_d   = (i - e_bar[c]) / bars_per_day
                    short_n  = (beta * e_t2[c]) if pc == 1 else e_t1[c]
                    borrow   = short_n * borrow_rate * hold_d / 252.0
                    net      = gross - tx - borrow

                    results[c, 0] += net
                    results[c, 1] += 1.0
                    if net > 0.0:
                        results[c, 2] += 1.0
                    results[c, 3] += net * net
                    if is_stop:
                        results[c, 5] += 1.0

                    cum_pnl[c] += net
                    if cum_pnl[c] > peak_pnl[c]:
                        peak_pnl[c] = cum_pnl[c]
                    dd = cum_pnl[c] - peak_pnl[c]
                    if dd < results[c, 4]:
                        results[c, 4] = dd

                    pos[c] = 0

            if pos[c] == 0:
                if z < -ez:
                    pos[c]  = 1
                    e_sp[c] = s; e_t1[c] = p1; e_t2[c] = p2; e_bar[c] = i
                elif z > ez:
                    pos[c]  = -1
                    e_sp[c] = s; e_t1[c] = p1; e_t2[c] = p2; e_bar[c] = i

    return results


def run_grid(df: pd.DataFrame, t1: str, t2: str,
             beta: float, combos: list, days: float,
             min_trades: int = MIN_TRADES_NORMAL) -> pd.DataFrame:
    """Run grid search. Returns DataFrame sorted by Sharpe descending."""
    valid = [(ez, xz, sz) for ez, xz, sz in combos if sz > ez and xz < ez]
    if not valid:
        return pd.DataFrame()

    zscore   = np.ascontiguousarray(df["zscore"].to_numpy(np.float64))
    spread   = np.ascontiguousarray(df["spread"].to_numpy(np.float64))
    t1_price = np.ascontiguousarray(df[f"{t1}_close"].to_numpy(np.float64))
    t2_price = np.ascontiguousarray(df[f"{t2}_close"].to_numpy(np.float64))

    entry_arr = np.array([c[0] for c in valid], dtype=np.float64)
    exit_arr  = np.array([c[1] for c in valid], dtype=np.float64)
    stop_arr  = np.array([c[2] for c in valid], dtype=np.float64)

    results = _grid_kernel(zscore, spread, t1_price, t2_price,
                           entry_arr, exit_arr, stop_arr,
                           float(beta),
                           float(COST_PER_SIDE),
                           float(BORROW_RATE_ANNUAL),
                           float(BARS_PER_DAY))

    years = max(days / 365.25, 1e-9)
    rows  = []
    for i, (ez, xz, sz) in enumerate(valid):
        n_trades = int(results[i, 1])
        if n_trades < min_trades:
            continue

        total_pnl = results[i, 0]
        n_wins    = int(results[i, 2])
        sum_pnl2  = results[i, 3]
        n_stops   = int(results[i, 5])

        mean_pnl  = total_pnl / n_trades
        var_pnl   = max(sum_pnl2 / n_trades - mean_pnl ** 2, 0.0)
        std_pnl   = np.sqrt(var_pnl)
        tpy       = n_trades / years
        sharpe    = mean_pnl / std_pnl * np.sqrt(tpy) if std_pnl > 0.0 else 0.0

        rows.append({
            "entry_z":   ez,
            "exit_z":    xz,
            "stop_z":    sz,
            "trades":    n_trades,
            "win_rate":  round(n_wins / n_trades * 100.0, 1),
            "sharpe":    round(sharpe, 2),
            "total_pnl": round(total_pnl, 4),
            "stop_rate": round(n_stops / n_trades * 100.0, 1),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


# ── Signal builder (same as grid.py) ─────────────────────────────────────────

def build_signals(closes: pd.DataFrame, t1: str, t2: str,
                  beta: float, half_life: float) -> pd.DataFrame:
    spread = closes[t1] - beta * closes[t2]
    window = max(20, min(int(half_life), 200))
    zscore = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    return pd.DataFrame({
        f"{t1}_close": closes[t1],
        f"{t2}_close": closes[t2],
        "spread":      spread,
        "zscore":      zscore,
    }).dropna().between_time(SIGNAL_START, RTH_END)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    """Load closes, regimes, and pairs metadata."""
    # Closes
    path = DATA_DIR / CLOSES_FILE
    if not path.exists():
        path = DATA_DIR / "closes_15min.csv"
    closes = pd.read_csv(path, index_col=0, parse_dates=True)
    closes.index = pd.to_datetime(closes.index, utc=True).tz_convert("US/Eastern")
    closes = closes.between_time(RTH_START, RTH_END)

    # Regimes (per-pair HMM labels)
    regimes_path = DATA_DIR / "regimes.csv"
    if not regimes_path.exists():
        raise FileNotFoundError(
            "regimes.csv not found — run hmm.py (step 3b) first")
    regimes = pd.read_csv(regimes_path, index_col=0, parse_dates=True)
    if not isinstance(regimes.index, pd.DatetimeIndex):
        regimes.index = pd.to_datetime(regimes.index, utc=True)
    if regimes.index.tz is None:
        regimes.index = regimes.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        regimes.index = regimes.index.tz_convert("US/Eastern")

    # Pairs metadata
    pairs = pd.read_csv(DATA_DIR / "pairs_selected.csv")

    return closes, regimes, pairs


def split_train_test(closes, pairs):
    """Split closes into train/test using same logic as grid.py."""
    if "test_start_date" in pairs.columns and not pairs.empty:
        test_start = pd.Timestamp(pairs["test_start_date"].iloc[0]).tz_localize("US/Eastern")
        train = closes[closes.index < test_start]
        test  = closes[closes.index >= test_start]
        if len(train) > 200 and len(test) > 200:
            return train, test

    n     = len(closes)
    split = int(n * TRAIN_RATIO)
    return closes.iloc[:split], closes.iloc[split:]


# ── Main profiling loop ──────────────────────────────────────────────────────

def main():
    closes, regimes, pairs = load_data()
    closes_train, closes_test = split_train_test(closes, pairs)

    days_train = (closes_train.index[-1] - closes_train.index[0]).days
    days_test  = (closes_test.index[-1]  - closes_test.index[0]).days

    print("=" * 90)
    print("REGIME-CONDITIONED Z-SCORE PROFILING (RCDP)")
    print("=" * 90)
    print(f"TRAIN: {closes_train.index[0].date()} → {closes_train.index[-1].date()}  "
          f"({days_train} days)")
    print(f"TEST:  {closes_test.index[0].date()}  → {closes_test.index[-1].date()}   "
          f"({days_test} days)")
    print(f"Grid:  {len(RCDP_ENTRY_GRID)} entry × {len(RCDP_EXIT_GRID)} exit × "
          f"{len(RCDP_STOP_GRID)} stop = {N_COMBOS} combos per regime")
    print(f"Pairs with HMM data: {len([c for c in regimes.columns if c in [r['pair'] for _, r in pairs.iterrows()]])}")
    print()

    all_results = []
    pair_details = {}  # for visualization

    for _, row in pairs.iterrows():
        pair_name = row["pair"]
        t1, t2    = pair_name.split("-")
        beta      = float(row.get("beta_daily", row["beta"]) or row["beta"])
        half_life = float(row["half_life_bars"])

        if t1 not in closes_train.columns or t2 not in closes_train.columns:
            continue

        # Check if we have HMM regime data for this pair
        if pair_name not in regimes.columns:
            print(f"  SKIP {pair_name}: no HMM regime data")
            continue

        # Build signals on TRAIN
        sig_train = build_signals(closes_train, t1, t2, beta, half_life)
        sig_test  = build_signals(closes_test, t1, t2, beta, half_life)

        if len(sig_train) < 200:
            print(f"  SKIP {pair_name}: too few train bars ({len(sig_train)})")
            continue

        # Get regime labels for TRAIN bars (align by timestamp)
        pair_regime = regimes[pair_name]
        regime_aligned = pair_regime.reindex(sig_train.index, method="ffill").fillna(0).astype(int)

        # Split TRAIN into Normal and Volatile slices
        mask_normal   = regime_aligned == 0
        mask_volatile = regime_aligned == 1

        sig_normal   = sig_train[mask_normal]
        sig_volatile = sig_train[mask_volatile]

        n_normal   = len(sig_normal)
        n_volatile = len(sig_volatile)
        pct_vol    = n_volatile / max(len(sig_train), 1) * 100

        print(f"  {pair_name:12s}  Normal={n_normal:5d}  Volatile={n_volatile:5d} "
              f"({pct_vol:.0f}%)", end="", flush=True)

        if n_normal < 100:
            print("  — too few Normal bars, skipping")
            continue

        # ── Grid search: Normal regime ────────────────────────────────────
        df_normal = run_grid(sig_normal, t1, t2, beta, COMBOS,
                             days_train, min_trades=MIN_TRADES_NORMAL)

        if df_normal.empty:
            print("  — no valid Normal combos")
            continue

        best_normal = df_normal.iloc[0]

        # ── Grid search: Volatile regime ──────────────────────────────────
        best_volatile = None
        vol_validated = False

        if n_volatile >= 50:  # need at least 50 bars to attempt grid search
            df_volatile = run_grid(sig_volatile, t1, t2, beta, COMBOS,
                                   days_train,
                                   min_trades=RCDP_MIN_TRADES_VOLATILE)

            if not df_volatile.empty:
                best_volatile = df_volatile.iloc[0]

                # Validate on TEST volatile bars
                regime_test = pair_regime.reindex(sig_test.index, method="ffill").fillna(0).astype(int)
                sig_test_vol = sig_test[regime_test == 1]

                if len(sig_test_vol) >= 20:
                    best_combo = [(float(best_volatile["entry_z"]),
                                   float(best_volatile["exit_z"]),
                                   float(best_volatile["stop_z"]))]
                    df_test_vol = run_grid(sig_test_vol, t1, t2, beta,
                                          best_combo, days_test,
                                          min_trades=3)
                    if not df_test_vol.empty and df_test_vol.iloc[0]["sharpe"] > 0:
                        vol_validated = True

        # ── Fallback for volatile regime ──────────────────────────────────
        if best_volatile is None or not vol_validated:
            # Use normal entry + ENTRY_Z_VOLATILE offset as fallback
            fallback_entry = min(float(best_normal["entry_z"]) +
                                 (ENTRY_Z_VOLATILE - ENTRY_Z),
                                 max(RCDP_ENTRY_GRID))
            best_volatile = pd.Series({
                "entry_z":   fallback_entry,
                "exit_z":    float(best_normal["exit_z"]),
                "stop_z":    min(float(best_normal["stop_z"]) + 0.4,
                                 max(RCDP_STOP_GRID)),
                "sharpe":    0.0,
                "trades":    0,
                "win_rate":  0.0,
                "total_pnl": 0.0,
                "stop_rate": 0.0,
            })
            vol_source = "fallback"
        else:
            vol_source = "grid" + (" ✓" if vol_validated else " ✗")

        print(f"  →  Normal: e={best_normal['entry_z']:.1f} x={best_normal['exit_z']:+.1f} "
              f"s={best_normal['stop_z']:.1f} Sh={best_normal['sharpe']:.2f}"
              f"  |  Volatile: e={best_volatile['entry_z']:.1f} "
              f"x={best_volatile['exit_z']:+.1f} "
              f"s={best_volatile['stop_z']:.1f} "
              f"Sh={best_volatile['sharpe']:.2f} ({vol_source})")

        # Save results for both regimes
        for regime_id, best, source in [
            (0, best_normal, "grid"),
            (1, best_volatile, vol_source),
        ]:
            all_results.append({
                "pair":      pair_name,
                "regime":    regime_id,
                "entry_z":   float(best["entry_z"]),
                "exit_z":    float(best["exit_z"]),
                "stop_z":    float(best["stop_z"]),
                "sharpe":    float(best["sharpe"]),
                "trades":    int(best["trades"]),
                "win_rate":  float(best["win_rate"]),
                "source":    source,
            })

        pair_details[pair_name] = {
            "normal": best_normal,
            "volatile": best_volatile,
            "n_normal": n_normal,
            "n_volatile": n_volatile,
            "vol_source": vol_source,
        }

    if not all_results:
        print("\nNo pairs profiled. Ensure hmm.py (step 3b) has run.")
        return

    # ── Save ──────────────────────────────────────────────────────────────────
    df_out = pd.DataFrame(all_results)
    out_path = DATA_DIR / "regime_thresholds.csv"
    df_out.to_csv(out_path, index=False)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print(f"{'REGIME-CONDITIONED THRESHOLDS':^100}")
    print(f"{'=' * 100}")
    print(f"{'Pair':<12} {'── Normal ──':^35} {'── Volatile ──':^35} {'Δentry':>7} {'Source':>10}")
    print(f"{'':12} {'entry':>6} {'exit':>6} {'stop':>6} {'Sh':>6} {'Tr':>5}  "
          f"{'entry':>6} {'exit':>6} {'stop':>6} {'Sh':>6} {'Tr':>5}")
    print("-" * 100)

    for pair_name, detail in pair_details.items():
        n = detail["normal"]
        v = detail["volatile"]
        delta = float(v["entry_z"]) - float(n["entry_z"])
        print(f"{pair_name:<12} "
              f"{n['entry_z']:>6.1f} {n['exit_z']:>+6.1f} {n['stop_z']:>6.1f} "
              f"{n['sharpe']:>6.2f} {int(n['trades']):>5}  "
              f"{v['entry_z']:>6.1f} {v['exit_z']:>+6.1f} {v['stop_z']:>6.1f} "
              f"{v['sharpe']:>6.2f} {int(v['trades']):>5}  "
              f"{delta:>+6.1f}  {detail['vol_source']:>10}")
    print("=" * 100)

    n_profiled = len(pair_details)
    n_grid     = sum(1 for d in pair_details.values() if "grid" in d["vol_source"])
    n_fallback = n_profiled - n_grid
    print(f"\nProfiled: {n_profiled} pairs  "
          f"(Volatile from grid: {n_grid}, fallback: {n_fallback})")
    print(f"Saved → {out_path}")

    # ── Visualization ─────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)

    if pair_details:
        names = list(pair_details.keys())
        n_e_normal   = [float(pair_details[p]["normal"]["entry_z"]) for p in names]
        n_e_volatile = [float(pair_details[p]["volatile"]["entry_z"]) for p in names]

        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        # Panel 1: entry_z comparison (Normal vs Volatile)
        ax = axes[0]
        x = np.arange(len(names))
        w = 0.35
        bars1 = ax.bar(x - w/2, n_e_normal,   w, label="Normal entry_z",
                        color="steelblue", alpha=0.8)
        bars2 = ax.bar(x + w/2, n_e_volatile, w, label="Volatile entry_z",
                        color="salmon", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Entry Z-score")
        ax.set_title("RCDP: Normal vs Volatile Entry Thresholds per Pair",
                      fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        # Add delta labels
        for i, (n, v) in enumerate(zip(n_e_normal, n_e_volatile)):
            delta = v - n
            ax.text(i + w/2, v + 0.05, f"Δ{delta:+.1f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")

        # Panel 2: volatile bar % and source
        ax2 = axes[1]
        pct_vol = [pair_details[p]["n_volatile"] /
                   max(pair_details[p]["n_normal"] + pair_details[p]["n_volatile"], 1) * 100
                   for p in names]
        colors = ["salmon" if "grid" in pair_details[p]["vol_source"] else "lightgray"
                  for p in names]
        ax2.bar(x, pct_vol, color=colors, edgecolor="white")
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("Volatile bars (%)")
        ax2.set_title("% Volatile Bars per Pair  (colored = grid-optimized, gray = fallback)",
                       fontsize=11, fontweight="bold")
        ax2.axhline(10, color="red", lw=1, ls="--", alpha=0.5, label="10% threshold")
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        chart_path = OUTPUT_DIR / "regime_profiler.png"
        plt.savefig(chart_path, dpi=150)
        plt.close(fig)
        print(f"Chart saved to {chart_path}")


if __name__ == "__main__":
    main()
