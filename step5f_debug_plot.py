"""
debug_plot.py — Z-score debug visualizer.

Matches the actual data structures from backtest.py:
  sig_df  : DataFrame with 'zscore' column (index = DatetimeIndex)
  trades  : DataFrame from backtest_pair() with entry_time / exit_time /
            direction / exit_reason columns

Usage after backtest_pair():
    from step5f_debug_plot import plot_zscore_debug
    plot_zscore_debug(sig_df, trades_df, "GS-MS", entry_z=2.2, exit_z=0.3, stop_z=3.4)
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_zscore_debug(
    sig_df: pd.DataFrame,
    trades: pd.DataFrame,
    pair_name: str,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    tail: int = 1500,
    save_dir: str = "output/debug",
) -> Path:
    """
    Plot z-score with fill zones and markers for ACTUAL trades from backtest_pair().

    Parameters
    ----------
    sig_df    : signals DataFrame — must have 'zscore' column, DatetimeIndex
    trades    : trades DataFrame from backtest_pair() — needs entry_time,
                exit_time, direction, exit_reason
    pair_name : used for title and filename
    entry_z   : entry threshold (absolute value)
    exit_z    : exit threshold (can be negative)
    stop_z    : stop-loss threshold
    tail      : last N bars to display (keeps chart readable)
    save_dir  : output directory

    Returns
    -------
    Path to saved PNG.
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Slice to last tail bars
    plot_df = sig_df.tail(tail).copy()
    z = plot_df["zscore"].values
    x = np.arange(len(z))
    ts = plot_df.index  # DatetimeIndex for trade matching

    fig, ax = plt.subplots(figsize=(18, 6))

    # ── 1. Z-score line ───────────────────────────────────────────────────────
    ax.plot(x, z, color="gray", alpha=0.6, lw=1.0, label="Z-score")

    # ── 2. Threshold lines ────────────────────────────────────────────────────
    ax.axhline(0,        color="black",     lw=1.0,  label="Mean (0)")
    ax.axhline( entry_z, color="red",       lw=1.2,  ls="--", label=f"+entry {entry_z}")
    ax.axhline(-entry_z, color="green",     lw=1.2,  ls="--", label=f"−entry {entry_z}")
    ax.axhline( exit_z,  color="steelblue", lw=0.8,  ls=":",  label=f"exit {exit_z:+.1f}")
    ax.axhline(-exit_z,  color="steelblue", lw=0.8,  ls=":")
    ax.axhline( stop_z,  color="darkred",   lw=0.8,  ls="-.", label=f"stop ±{stop_z}")
    ax.axhline(-stop_z,  color="darkred",   lw=0.8,  ls="-.")

    # ── 3. Fill zones ─────────────────────────────────────────────────────────
    ax.fill_between(x, z, entry_z,
                    where=(z >= entry_z),
                    color="red", alpha=0.25, label="Short spread zone")
    ax.fill_between(x, z, -entry_z,
                    where=(z <= -entry_z),
                    color="green", alpha=0.25, label="Long spread zone")

    # ── 4. Trade markers (from real trades, not reconstructed position) ───────
    if trades is not None and not trades.empty:
        entry_times = pd.to_datetime(trades["entry_time"])
        exit_times  = pd.to_datetime(trades["exit_time"])
        ts_arr      = pd.to_datetime(ts)

        long_ex_x,  long_ex_z  = [], []
        short_ex_x, short_ex_z = [], []
        exit_sig_x, exit_sig_z = [], []
        exit_stop_x, exit_stop_z = [], []

        for _, tr in trades.iterrows():
            et = pd.Timestamp(tr["entry_time"])
            xt = pd.Timestamp(tr["exit_time"])
            reason = tr.get("exit_reason", "SIGNAL")
            direction = tr.get("direction", "LONG")

            # Find nearest bar index for entry / exit
            i_entry = _nearest_bar(ts_arr, et)
            i_exit  = _nearest_bar(ts_arr, xt)

            if i_entry is not None:
                if direction == "LONG":
                    long_ex_x.append(i_entry); long_ex_z.append(z[i_entry])
                else:
                    short_ex_x.append(i_entry); short_ex_z.append(z[i_entry])

            if i_exit is not None:
                if reason == "STOP":
                    exit_stop_x.append(i_exit); exit_stop_z.append(z[i_exit])
                else:
                    exit_sig_x.append(i_exit); exit_sig_z.append(z[i_exit])

        if long_ex_x:
            ax.scatter(long_ex_x, long_ex_z,
                       marker="^", color="lime", s=90, zorder=5,
                       edgecolors="darkgreen", lw=0.8, label=f"Entry Long ({len(long_ex_x)})")
        if short_ex_x:
            ax.scatter(short_ex_x, short_ex_z,
                       marker="v", color="red", s=90, zorder=5,
                       edgecolors="darkred", lw=0.8, label=f"Entry Short ({len(short_ex_x)})")
        if exit_sig_x:
            ax.scatter(exit_sig_x, exit_sig_z,
                       marker="x", color="black", s=60, zorder=5,
                       lw=1.5, label=f"Exit Signal ({len(exit_sig_x)})")
        if exit_stop_x:
            ax.scatter(exit_stop_x, exit_stop_z,
                       marker="x", color="crimson", s=90, zorder=5,
                       lw=2.0, label=f"Exit STOP ({len(exit_stop_x)})")

    # ── 5. Styling ────────────────────────────────────────────────────────────
    n_tr = len(trades) if trades is not None and not trades.empty else 0
    n_stop = int((trades["exit_reason"] == "STOP").sum()) if n_tr > 0 else 0
    wr = ((trades["net_pnl"] > 0).mean() * 100) if n_tr > 0 else 0.0

    ax.set_title(
        f"{pair_name}  |  Z-score debug  "
        f"[entry±{entry_z}  exit{exit_z:+.1f}  stop±{stop_z}]  "
        f"last {len(plot_df)} bars  "
        f"|  {n_tr} trades  WR={wr:.0f}%  stops={n_stop}",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Bar index")
    ax.set_ylabel("Z-score")
    ax.legend(loc="upper left", fontsize=8, ncol=4, framealpha=0.7)
    ax.grid(True, alpha=0.2)

    y_clip = max(entry_z * 2.5, stop_z + 0.5, 5.0)
    ax.set_ylim(-y_clip, y_clip)

    plt.tight_layout()
    out = out_dir / f"debug_{pair_name.replace('-', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Debug plot → {out}")
    return out


def _nearest_bar(ts_arr: pd.DatetimeIndex, target: pd.Timestamp) -> int | None:
    """Return index of the bar nearest to target timestamp, or None if out of range."""
    if len(ts_arr) == 0:
        return None
    delta_sec = np.abs((ts_arr - target).total_seconds().values)
    idx = int(delta_sec.argmin())
    # Only accept if within 2 trading days (2 * 26 bars on 15-min)
    if delta_sec[idx] > 2 * 26 * 15 * 60:
        return None
    return idx
