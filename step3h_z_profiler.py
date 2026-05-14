"""
z_profiler.py — Z-Bounce Density Profiler (Expected Value Edition).

ИДЕЯ:
Вместо 616-комбинаций 3D-сетки (entry × exit × stop) — один математический радар.
Фиксируем R:R = 1.3.  Выход всегда на Z=0 (возврат к среднему).
Стоп автоматически вычисляется: stop_z = entry_z / target_rr + entry_z.
Единственный параметр оптимизации — entry_z.

Для каждого исторического пробоя порога entry_z запускается "гонка":
  → Кто наступит первым: take-profit (Z≤0) или stop-loss (Z≥stop_z)?
  → Win Rate → Expected Value: EV = WR × rr - (1-WR) × 1.0
  → Лучший entry_z = max(EV)

РЕЗУЛЬТАТ:
  data/z_profiles.csv   — оптимальные параметры по каждой паре
  output/z_density/     — графики EV-кривой для каждой пары

Pipeline position: step 3h (replaces blind grid — run before backtest.py)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from config import (
    CLOSES_FILE, DATA_DIR, OUTPUT_DIR,
    RTH_START, RTH_END, SIGNAL_START,
    KALMAN_DELTA, TRAIN_RATIO,
)
from utils import fast_read

# ── Tunable parameters ────────────────────────────────────────────────────────
TARGET_RR        = 1.3    # Fixed Reward / Risk ratio (Gatev-style constraint)
MIN_ENTRY_Z      = 1.5    # Scan range start
MAX_ENTRY_Z      = 3.5    # Scan range end
ENTRY_Z_STEP     = 0.1
MIN_TRADES       = 15     # Minimum crossings for a valid estimate
MIN_EV           = 0.0    # Only use entry_z with positive expected value
FALLBACK_ENTRY_Z = 2.0    # Fallback if no positive EV found


# ── Kalman spread builder (mirrors backtest.py exactly) ──────────────────────

def _kalman_hedge(p1: np.ndarray, p2: np.ndarray,
                  delta: float, beta_init: float):
    """Scalar Kalman filter for dynamic hedge ratio."""
    n       = len(p1)
    beta    = np.empty(n); alpha = np.empty(n)
    innov   = np.empty(n); Q = delta / (1.0 - delta)
    R_var   = 1.0
    P       = np.eye(2)
    state   = np.array([0.0, beta_init])

    for i in range(n):
        F       = np.array([1.0, p2[i]])
        pred    = state.copy()
        P_pred  = P + Q * np.eye(2)
        err     = p1[i] - F @ pred
        S       = float(F @ P_pred @ F) + R_var
        K       = P_pred @ F / S
        state   = pred + K * err
        P       = P_pred - np.outer(K, F) @ P_pred
        alpha[i] = state[0]; beta[i] = state[1]; innov[i] = err

    return alpha, beta, innov


def build_zscore(closes: pd.DataFrame, t1: str, t2: str,
                 beta_init: float, half_life: float) -> pd.Series | None:
    """Return Kalman-based Z-score series aligned to intraday RTH bars."""
    s1 = closes[t1].values
    s2 = closes[t2].values
    if len(s1) < 200:
        return None

    _, _, innov = _kalman_hedge(s1, s2, delta=KALMAN_DELTA, beta_init=beta_init)
    spread = pd.Series(innov, index=closes.index)
    window = max(20, min(int(half_life), 200))
    spread_std = spread.rolling(window).std()
    zscore     = (spread / spread_std).dropna()
    return zscore.between_time(SIGNAL_START, RTH_END)


# ── Core profiling engine ─────────────────────────────────────────────────────

def profile_z_bounces(z_series: pd.Series,
                      min_z: float = MIN_ENTRY_Z,
                      max_z: float = MAX_ENTRY_Z,
                      step: float  = ENTRY_Z_STEP,
                      target_rr: float = TARGET_RR,
                      min_trades: int  = MIN_TRADES) -> pd.DataFrame:
    """
    Scans the Z-score history and finds the optimal entry_z by Expected Value.

    Both directions (LONG when Z < -entry_z, SHORT when Z > entry_z) are
    treated symmetrically — crossings from both sides are counted.

    Returns a DataFrame sorted by EV descending, filtered to min_trades.
    """
    z_vals = np.asarray(z_series.values, dtype=np.float64)
    n      = len(z_vals)
    rows   = []

    for entry_z in np.round(np.arange(min_z, max_z + step / 2, step), 2):
        # Auto-compute stop from fixed R:R
        # Reward = entry_z - 0 = entry_z  (exit at Z=0)
        # Risk   = stop_z - entry_z
        # RR     = Reward / Risk  →  stop_z = entry_z × (1 + 1/RR)
        stop_z = round(entry_z * (1.0 + 1.0 / target_rr), 3)

        wins = losses = 0

        # ── SHORT side: Z crosses UP through +entry_z ─────────────────────
        cross_short = np.where(
            (z_vals[:-1] < entry_z) & (z_vals[1:] >= entry_z)
        )[0] + 1

        for idx in cross_short:
            future = z_vals[idx:]
            wi = np.where(future <= 0.0)[0]
            li = np.where(future >= stop_z)[0]
            fw = wi[0] if len(wi) else np.inf
            fl = li[0] if len(li) else np.inf
            if fw < fl:    wins   += 1
            elif fl < fw:  losses += 1

        # ── LONG side: Z crosses DOWN through -entry_z ────────────────────
        cross_long = np.where(
            (z_vals[:-1] > -entry_z) & (z_vals[1:] <= -entry_z)
        )[0] + 1

        for idx in cross_long:
            future = z_vals[idx:]
            wi = np.where(future >= 0.0)[0]
            li = np.where(future <= -stop_z)[0]
            fw = wi[0] if len(wi) else np.inf
            fl = li[0] if len(li) else np.inf
            if fw < fl:    wins   += 1
            elif fl < fw:  losses += 1

        total = wins + losses
        if total < min_trades:
            continue

        wr = wins / total
        ev = wr * target_rr - (1.0 - wr) * 1.0

        rows.append({
            "entry_z":    entry_z,
            "stop_z":     stop_z,
            "exit_z":     0.0,
            "trades":     total,
            "wins":       wins,
            "losses":     losses,
            "win_rate":   round(wr * 100, 1),
            "ev":         round(ev, 4),
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("ev", ascending=False).reset_index(drop=True)
    return df


# ── Load data ─────────────────────────────────────────────────────────────────

def load_closes() -> pd.DataFrame:
    path = DATA_DIR / CLOSES_FILE
    if not path.exists():
        path = DATA_DIR / "closes_15min.csv"
    closes = fast_read(path, log_label=path.name)
    closes.index = pd.to_datetime(closes.index, utc=True).tz_convert("US/Eastern")
    return closes.between_time(RTH_START, RTH_END)


def get_train_closes(closes: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Return TRAIN slice (same split logic as backtest.py)."""
    if "test_start_date" in pairs.columns and not pairs.empty:
        ts = pd.Timestamp(pairs["test_start_date"].iloc[0]).tz_localize("US/Eastern")
        train = closes[closes.index < ts]
        if len(train) > 200:
            return train
    n = len(closes)
    return closes.iloc[:int(n * TRAIN_RATIO)]


# ── Visualization for one pair ────────────────────────────────────────────────

def _plot_pair(pair_name: str, df: pd.DataFrame, out_dir: Path):
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Z-Bounce Density Profile — {pair_name}  (R:R={TARGET_RR})",
                 fontsize=13, fontweight="bold")

    # Left: EV curve
    ax = axes[0]
    colors = ["#2ecc71" if ev >= 0 else "#e74c3c" for ev in df["ev"]]
    bars = ax.bar(df["entry_z"], df["ev"], width=ENTRY_Z_STEP * 0.8,
                  color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", lw=1.2)
    ax.set_xlabel("Entry Z-score")
    ax.set_ylabel("Expected Value (EV)")
    ax.set_title("EV per Entry Z  (green = profitable)")
    ax.grid(True, alpha=0.3, axis="y")

    # Mark best EV
    best = df.iloc[0]
    ax.annotate(f"BEST\nZ={best['entry_z']:.1f}\nEV={best['ev']:.3f}",
                xy=(best["entry_z"], best["ev"]),
                xytext=(best["entry_z"] + 0.3, best["ev"] + 0.05),
                fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black"))

    # Right: Win Rate vs entry_z with EV breakeven line
    ax2 = axes[1]
    ax2.plot(df["entry_z"], df["win_rate"], "o-",
             color="steelblue", lw=2, markersize=5, label="Win Rate %")
    breakeven_wr = 1.0 / (1.0 + TARGET_RR) * 100
    ax2.axhline(breakeven_wr, color="orange", ls="--", lw=1.5,
                label=f"Breakeven WR = {breakeven_wr:.1f}%")
    ax2.set_xlabel("Entry Z-score")
    ax2.set_ylabel("Win Rate (%)")
    ax2.set_title(f"Win Rate  (breakeven at {breakeven_wr:.1f}% for R:R={TARGET_RR})")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    # Mark trade count on secondary axis
    ax3 = ax2.twinx()
    ax3.plot(df["entry_z"], df["trades"], "s--",
             color="gray", lw=1, markersize=4, alpha=0.6, label="# Trades")
    ax3.set_ylabel("# Trades (sample size)", color="gray")
    ax3.tick_params(axis="y", labelcolor="gray")
    ax3.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    out_path = out_dir / f"density_{pair_name.replace('-', '_')}.png"
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    closes = load_closes()
    pairs = fast_read(DATA_DIR / "pairs_selected.csv", prefer_parquet=False, fast_bars=0, index_col=None, parse_dates=False, log_label="pairs_selected.csv")

    if pairs.empty:
        raise SystemExit("pairs_selected.csv is empty — run pairs.py first")

    closes_train = get_train_closes(closes, pairs)
    days_train   = (closes_train.index[-1] - closes_train.index[0]).days

    out_dir = OUTPUT_DIR / "z_density"
    out_dir.mkdir(parents=True, exist_ok=True)

    breakeven_wr = 1.0 / (1.0 + TARGET_RR) * 100

    print("=" * 90)
    print(f"Z-BOUNCE DENSITY PROFILER  (R:R={TARGET_RR}, exit@Z=0)")
    print("=" * 90)
    print(f"TRAIN: {closes_train.index[0].date()} → {closes_train.index[-1].date()}"
          f"  ({days_train} days)")
    print(f"Entry scan: Z={MIN_ENTRY_Z:.1f}→{MAX_ENTRY_Z:.1f} step={ENTRY_Z_STEP}")
    print(f"Breakeven WR at R:R={TARGET_RR}: {breakeven_wr:.1f}%")
    print(f"Pairs: {len(pairs)}\n")

    header = (f"{'Pair':<12} {'Best Entry':>10} {'Stop':>7} "
              f"{'Trades':>7} {'WR%':>6} {'EV':>7}  {'Signal':>10}")
    print(header)
    print("-" * len(header))

    all_profiles = []
    summary_rows = []

    for _, row in pairs.iterrows():
        pair_name = row["pair"]
        t1, t2    = pair_name.split("-")
        beta      = float(row.get("beta_daily", row.get("beta", 1.0)) or 1.0)
        half_life = float(row.get("half_life_bars", 100))

        if t1 not in closes_train.columns or t2 not in closes_train.columns:
            continue

        z = build_zscore(closes_train, t1, t2, beta, half_life)
        if z is None or len(z) < 500:
            print(f"  SKIP {pair_name}: insufficient data")
            continue

        df_profile = profile_z_bounces(z)

        if df_profile.empty:
            print(f"  SKIP {pair_name}: no crossings with ≥{MIN_TRADES} trades")
            continue

        # Save all rows tagged with pair
        df_profile["pair"] = pair_name
        all_profiles.append(df_profile)

        # Best row (max EV)
        best = df_profile.iloc[0]
        has_positive_ev = best["ev"] > MIN_EV
        signal = "✓ TRADE" if has_positive_ev else "✗ SKIP"

        # Use best if positive EV, else fallback
        chosen_entry = float(best["entry_z"]) if has_positive_ev else FALLBACK_ENTRY_Z
        chosen_stop  = round(chosen_entry * (1.0 + 1.0 / TARGET_RR), 3)
        chosen_ev    = float(best["ev"]) if has_positive_ev else 0.0

        print(f"  {pair_name:<12} "
              f"Z={chosen_entry:>5.2f}  "
              f"stop={chosen_stop:>6.3f}  "
              f"n={int(best['trades']):>4}  "
              f"WR={best['win_rate']:>5.1f}%  "
              f"EV={chosen_ev:>+7.4f}  {signal}")

        summary_rows.append({
            "pair":        pair_name,
            "entry_z":     chosen_entry,
            "exit_z":      0.0,
            "stop_z":      chosen_stop,
            "trades":      int(best["trades"]),
            "win_rate":    float(best["win_rate"]),
            "ev":          chosen_ev,
            "tradeable":   has_positive_ev,
            "source":      "density_profiler",
        })

        # Plot
        _plot_pair(pair_name, df_profile, out_dir)

    if not all_profiles:
        print("\nNo pairs profiled.")
        return

    # ── Save ──────────────────────────────────────────────────────────────────
    df_all = pd.concat(all_profiles, ignore_index=True)
    df_all.to_csv(DATA_DIR / "z_density_full.csv", index=False)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(DATA_DIR / "z_profiles.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_tradeable = df_summary["tradeable"].sum()
    n_skip      = len(df_summary) - n_tradeable

    print(f"\n{'='*90}")
    print(f"RESULT: {n_tradeable} TRADEABLE  |  {n_skip} SKIP  "
          f"(EV > {MIN_EV})")
    print(f"Saved → data/z_profiles.csv  ({len(df_summary)} pairs)")
    print(f"Saved → data/z_density_full.csv  ({len(df_all)} rows)")
    print(f"Charts → output/z_density/  ({len(df_summary)} plots)")

    # Best EV pairs at the top
    print(f"\n{'Top pairs by EV':}")
    top = df_summary[df_summary["tradeable"]].sort_values("ev", ascending=False)
    for _, r in top.head(10).iterrows():
        print(f"  {r['pair']:<12}  entry={r['entry_z']:.1f}  "
              f"stop={r['stop_z']:.3f}  WR={r['win_rate']:.1f}%  EV={r['ev']:+.4f}")


if __name__ == "__main__":
    main()
