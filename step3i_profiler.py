"""
profiler.py — Z-Bounce Density Profiler.

Replaces brute-force grid search with a 1D statistical scan.

Math:
  R:R constraint (exit=0.0):
      Reward = entry_z - 0 = entry_z
      Risk   = stop_z - entry_z
      R:R = 1.3  →  stop_z = entry_z × 1.77

  For each entry_z level tested:
      EV = WinRate × 1.3 - LossRate × 1.0
      (positive EV = edge exists at this level)

Output:
  data/profiler_results.csv  — per-pair optimal entry_z by EV
  output/profiler_*.png      — EV curves per pair
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from config import (
    DATA_DIR, OUTPUT_DIR, RTH_START, RTH_END, SIGNAL_START,
    CLOSES_FILE, TRAIN_RATIO, BARS_PER_DAY,
)
from kalman import kalman_hedge
from step3f_grid import load_closes_split

# ── Parameters ────────────────────────────────────────────────────────────────
ENTRY_LEVELS = np.arange(1.5, 3.05, 0.05)   # 1.50 … 3.00 in steps of 0.05
RR_RATIO     = 1.3                            # target reward:risk
EXIT_Z       = 0.0                            # always exit at mean
MIN_TRADES   = 20                             # skip if fewer crossings found
MAX_HOLD_BARS = BARS_PER_DAY * 10            # 10 trading days max hold


def stop_from_entry(entry_z: float, rr: float = RR_RATIO, exit_z: float = EXIT_Z) -> float:
    """
    Derive stop_z from entry_z given fixed R:R.
    R:R = (entry_z - exit_z) / (stop_z - entry_z) = rr
    → stop_z = entry_z + (entry_z - exit_z) / rr
    """
    return entry_z + (entry_z - exit_z) / rr


def profile_pair(z: np.ndarray, entry_z: float) -> dict | None:
    """
    For one entry_z level, scan all crossings in z and record
    whether exit_z=0.0 or stop_z is hit first.

    Returns dict with win_rate, n_trades, ev — or None if too few trades.
    """
    stop_z = stop_from_entry(entry_z)
    wins = losses = 0

    i = 0
    while i < len(z):
        # Long entry: z drops below -entry_z
        if z[i] <= -entry_z:
            end = min(i + MAX_HOLD_BARS, len(z))
            future = z[i + 1: end]
            hit_exit = np.where(future >= EXIT_Z)[0]
            hit_stop = np.where(future <= -stop_z)[0]

            first_exit = hit_exit[0] if len(hit_exit) > 0 else MAX_HOLD_BARS
            first_stop = hit_stop[0] if len(hit_stop) > 0 else MAX_HOLD_BARS

            if first_exit <= first_stop:
                wins += 1
            else:
                losses += 1
            i += first_exit + 1 if first_exit < MAX_HOLD_BARS else MAX_HOLD_BARS
            continue

        # Short entry: z rises above +entry_z
        if z[i] >= entry_z:
            end = min(i + MAX_HOLD_BARS, len(z))
            future = z[i + 1: end]
            hit_exit = np.where(future <= EXIT_Z)[0]
            hit_stop = np.where(future >= stop_z)[0]

            first_exit = hit_exit[0] if len(hit_exit) > 0 else MAX_HOLD_BARS
            first_stop = hit_stop[0] if len(hit_stop) > 0 else MAX_HOLD_BARS

            if first_exit <= first_stop:
                wins += 1
            else:
                losses += 1
            i += first_exit + 1 if first_exit < MAX_HOLD_BARS else MAX_HOLD_BARS
            continue

        i += 1

    n = wins + losses
    if n < MIN_TRADES:
        return None

    wr  = wins / n
    lr  = losses / n
    ev  = wr * RR_RATIO - lr * 1.0
    return {
        "entry_z": round(entry_z, 2),
        "stop_z":  round(stop_from_entry(entry_z), 2),
        "exit_z":  EXIT_Z,
        "n_trades": n,
        "wins":    wins,
        "losses":  losses,
        "win_rate": round(wr * 100, 1),
        "ev":      round(ev, 4),
    }


def build_zscore(closes: pd.DataFrame, t1: str, t2: str,
                 beta: float, half_life: float) -> np.ndarray:
    _, _, innov, _ = kalman_hedge(
        closes[t1].values, closes[t2].values,
        delta=3e-6, beta_init=float(beta),
    )
    spread = pd.Series(innov, index=closes.index)
    window = max(20, min(int(half_life), 200))
    zscore = (spread / spread.rolling(window).std()).dropna()
    zscore = zscore.between_time(SIGNAL_START, RTH_END)
    return zscore.values.astype(np.float64)


# ── Load data ─────────────────────────────────────────────────────────────────
closes_train, closes_test, days_train = load_closes_split()
pairs = pd.read_csv(DATA_DIR / "pairs_selected.csv")

if pairs.empty:
    raise SystemExit("pairs_selected.csv is empty — run pairs.py first")

OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Z-Bounce Profiler  |  R:R={RR_RATIO}  exit={EXIT_Z}  "
      f"levels={ENTRY_LEVELS[0]:.2f}…{ENTRY_LEVELS[-1]:.2f}")
print(f"TRAIN: {closes_train.index[0].date()} → {closes_train.index[-1].date()}\n")

all_best: list[dict] = []

for _, row in pairs.iterrows():
    pair = row["pair"]
    t1, t2 = pair.split("-")
    beta = float(row.get("beta_daily", row["beta"]) or row["beta"])
    hl   = float(row["half_life_bars"])

    if t1 not in closes_train.columns or t2 not in closes_train.columns:
        print(f"  SKIP {pair}: missing data")
        continue

    z_train = build_zscore(closes_train, t1, t2, beta, hl)
    z_test  = build_zscore(closes_test,  t1, t2, beta, hl)

    if len(z_train) < 500:
        print(f"  SKIP {pair}: too few bars ({len(z_train)})")
        continue

    # ── Scan all entry levels on TRAIN ────────────────────────────────────
    train_rows = []
    for ez in ENTRY_LEVELS:
        r = profile_pair(z_train, ez)
        if r:
            r["split"] = "train"
            r["pair"]  = pair
            train_rows.append(r)

    if not train_rows:
        print(f"  {pair}: no valid levels on train")
        continue

    df_train = pd.DataFrame(train_rows).sort_values("ev", ascending=False)
    best     = df_train.iloc[0]

    # ── Validate best level on TEST ───────────────────────────────────────
    test_r = profile_pair(z_test, float(best["entry_z"]))
    if test_r:
        test_ev   = test_r["ev"]
        test_wr   = test_r["win_rate"]
        test_n    = test_r["n_trades"]
        oos_flag  = "✓" if test_ev > 0 else "✗"
    else:
        test_ev = test_wr = test_n = 0.0
        oos_flag = "—"

    print(f"  {pair:<12}  "
          f"best entry={best['entry_z']}  stop={best['stop_z']}  "
          f"TRAIN EV={best['ev']:+.4f}  WR={best['win_rate']:.1f}%  n={best['n_trades']}  "
          f"│  TEST EV={test_ev:+.4f}  WR={test_wr:.1f}%  {oos_flag}")

    all_best.append({
        "pair":       pair,
        "entry_z":    float(best["entry_z"]),
        "exit_z":     EXIT_Z,
        "stop_z":     float(best["stop_z"]),
        "train_ev":   float(best["ev"]),
        "train_wr":   float(best["win_rate"]),
        "train_n":    int(best["n_trades"]),
        "test_ev":    test_ev,
        "test_wr":    test_wr,
        "test_n":     int(test_n),
    })

    # ── EV curve chart ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, df_rows, label in [
        (axes[0], train_rows, "TRAIN"),
    ]:
        df_plt = pd.DataFrame(df_rows)
        color  = ["green" if v > 0 else "red" for v in df_plt["ev"]]
        ax.bar(df_plt["entry_z"], df_plt["ev"], width=0.04, color=color, alpha=0.8)
        ax.axhline(0, color="black", lw=1)
        ax.set_xlabel("Entry Z-score")
        ax.set_ylabel("Expected Value (EV)")
        ax.set_title(f"{pair}  {label}  EV by entry level  (stop = entry × 1.77)")
        ax.axvline(float(best["entry_z"]), color="gold", lw=2, ls="--",
                   label=f"Best entry={best['entry_z']}")
        ax.legend(fontsize=9)

    # Win rate curve
    ax2 = axes[1]
    df_plt = pd.DataFrame(train_rows)
    ax2.plot(df_plt["entry_z"], df_plt["win_rate"], "o-", color="steelblue", lw=2)
    ax2.axhline(100 / (1 + RR_RATIO), color="red", ls="--",
                label=f"Break-even WR={100/(1+RR_RATIO):.1f}%")
    ax2.axvline(float(best["entry_z"]), color="gold", lw=2, ls="--")
    ax2.set_xlabel("Entry Z-score")
    ax2.set_ylabel("Win Rate %")
    ax2.set_title(f"{pair}  TRAIN Win Rate by entry level")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"{pair}  │  R:R={RR_RATIO}  exit={EXIT_Z}  "
                 f"best stop={best['stop_z']}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = OUTPUT_DIR / f"profiler_{pair.replace('-', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print(f"{'PROFILER RESULTS':^90}")
print(f"{'='*90}")
print(f"{'Pair':<12} {'Entry':>6} {'Stop':>6}  "
      f"{'── TRAIN ──':^28}  {'── TEST ──':^28}")
print(f"{'':>26}  {'EV':>8} {'WR':>7} {'n':>5}  "
      f"{'EV':>8} {'WR':>7} {'n':>5}  {'OOS':>4}")
print("-" * 90)

positive_ev = 0
for r in sorted(all_best, key=lambda x: -x["train_ev"]):
    flag = "✓" if r["test_ev"] > 0 else "✗"
    if r["train_ev"] > 0:
        positive_ev += 1
    print(f"{r['pair']:<12} {r['entry_z']:>6.2f} {r['stop_z']:>6.2f}  "
          f"{r['train_ev']:>+8.4f} {r['train_wr']:>6.1f}% {r['train_n']:>5}  "
          f"{r['test_ev']:>+8.4f} {r['test_wr']:>6.1f}% {r['test_n']:>5}  {flag}")

print("=" * 90)
print(f"Pairs with positive TRAIN EV: {positive_ev} / {len(all_best)}")

# Save results
if all_best:
    df_out = pd.DataFrame(all_best)
    df_out.to_csv(DATA_DIR / "profiler_results.csv", index=False)
    print(f"Saved → data/profiler_results.csv")

    # Overwrite optimal_params.csv for backtest.py to use
    opt_cols = ["pair", "entry_z", "exit_z", "stop_z"]
    df_opt = df_out[df_out["train_ev"] > 0][opt_cols].copy()
    if not df_opt.empty:
        df_opt.to_csv(DATA_DIR / "optimal_params.csv", index=False)
        print(f"Updated optimal_params.csv with {len(df_opt)} EV-positive pairs")
    else:
        print("WARNING: no pairs with positive EV — optimal_params.csv not updated")
