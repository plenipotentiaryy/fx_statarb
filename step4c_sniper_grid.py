import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from config import (
    STOP_Z as DEFAULT_STOP_Z,
    COST_PER_SIDE, BORROW_RATE_ANNUAL,
    RTH_START, RTH_END, SIGNAL_START, RECENT_BARS,
    DATA_DIR, OUTPUT_DIR,
)

ENTRY_Z_GRID = [3.5, 3.3, 3.2, 3.0]
EXIT_Z_GRID  = [0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3]
STOP_Z_GRID  = [3.8, 4.2, 4.4]

BARS_PER_DAY = 26
COMBOS       = list(itertools.product(ENTRY_Z_GRID, EXIT_Z_GRID, STOP_Z_GRID))
print(f"Sniper grid: {len(ENTRY_Z_GRID)} entry × {len(EXIT_Z_GRID)} exit × "
      f"{len(STOP_Z_GRID)} stop = {len(COMBOS)} combinations\n")


def load_closes() -> pd.DataFrame:
    closes = pd.read_csv(DATA_DIR / "closes_15min.csv", index_col=0, parse_dates=True)
    if closes.index.tz is None:
        closes.index = closes.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        closes.index = closes.index.tz_convert("US/Eastern")
    return closes.between_time(RTH_START, RTH_END).dropna().tail(RECENT_BARS)


def build_signals(closes, t1, t2, beta, half_life) -> pd.DataFrame:
    spread = closes[t1] - beta * closes[t2]
    window = max(20, min(int(half_life), 200))
    zscore = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    return pd.DataFrame({
        f"{t1}_close": closes[t1],
        f"{t2}_close": closes[t2],
        "spread": spread,
        "zscore": zscore,
    }).dropna().between_time(SIGNAL_START, RTH_END)


def backtest(df, t1, t2, beta, entry_z, exit_z, stop_z) -> list:
    t1_col, t2_col = f"{t1}_close", f"{t2}_close"
    position = 0
    entry_spread = entry_t1 = entry_t2 = 0.0
    entry_bar = 0
    pnls = []

    for i in range(len(df)):
        z          = df["zscore"].iloc[i]
        spread_now = df["spread"].iloc[i]
        t1_price   = df[t1_col].iloc[i]
        t2_price   = df[t2_col].iloc[i]

        if position != 0:
            exit_sig = (position == 1 and z > -exit_z) or (position == -1 and z < exit_z)
            stop_sig = (position == 1 and z < -stop_z) or (position == -1 and z > stop_z)

            if exit_sig or stop_sig:
                gross          = position * (spread_now - entry_spread)
                notional       = entry_t1 + beta * entry_t2
                tx_cost        = 2 * notional * COST_PER_SIDE
                holding_days   = (i - entry_bar) / BARS_PER_DAY
                short_notional = (beta * entry_t2) if position == 1 else entry_t1
                borrow_cost    = short_notional * BORROW_RATE_ANNUAL * holding_days / 252
                pnls.append(gross - tx_cost - borrow_cost)
                position = 0

        if position == 0:
            if z < -entry_z:
                position = 1
            elif z > entry_z:
                position = -1
            if position != 0:
                entry_spread = spread_now
                entry_t1     = t1_price
                entry_t2     = t2_price
                entry_bar    = i

    return pnls


# ── Load ──────────────────────────────────────────────────────────────────────
closes = load_closes()
pairs  = pd.read_csv(DATA_DIR / "pairs_selected.csv")

if pairs.empty:
    raise SystemExit("pairs_selected.csv is empty — run step2_pairs.py first")

# Pre-build signals for each pair
signals = {}
for _, row in pairs.iterrows():
    t1, t2 = row["pair"].split("-")
    if t1 not in closes.columns or t2 not in closes.columns:
        continue
    signals[row["pair"]] = (
        build_signals(closes, t1, t2, row["beta"], row["half_life_bars"]),
        t1, t2, row["beta"]
    )

print(f"Pairs: {list(signals.keys())}")
print(f"Running {len(COMBOS)} combinations...\n")

days_total = (closes.index[-1] - closes.index[0]).days

# ── Grid search ───────────────────────────────────────────────────────────────
results = []

for entry_z, exit_z, stop_z in COMBOS:
    all_pnl = []

    for pair_name, (df_sig, t1, t2, beta) in signals.items():
        all_pnl.extend(backtest(df_sig, t1, t2, beta, entry_z, exit_z, stop_z))

    if not all_pnl:
        continue

    pnl   = np.array(all_pnl)
    n     = len(pnl)
    curve = np.cumsum(pnl)
    wr    = (pnl > 0).mean() * 100
    tpy   = n / (days_total / 365.25)
    sh    = pnl.mean() / pnl.std() * np.sqrt(tpy) if pnl.std() > 0 else 0.0
    dd    = (curve - np.maximum.accumulate(curve)).min()
    pf_w  = pnl[pnl > 0].sum()
    pf_l  = abs(pnl[pnl <= 0].sum())
    pf    = pf_w / pf_l if pf_l > 0 else float("inf")

    results.append({
        "entry_z":   entry_z,
        "exit_z":    exit_z,
        "stop_z":    stop_z,
        "trades":    n,
        "win_rate":  wr,
        "sharpe":    sh,
        "total_pnl": pnl.sum(),
        "avg_pnl":   pnl.mean(),
        "max_dd":    dd,
        "pf":        pf,
        "curve":     curve,
    })

df_res = pd.DataFrame([{k: v for k, v in r.items() if k != "curve"}
                        for r in results])
df_res = df_res.sort_values("sharpe", ascending=False).reset_index(drop=True)

# ── Print full table ──────────────────────────────────────────────────────────
print("=" * 85)
print(f"{'#':>3} {'entry':>6} {'exit':>6} {'stop':>6} {'trades':>7} "
      f"{'WR':>6} {'sharpe':>7} {'total_pnl':>11} {'max_dd':>9} {'PF':>6}")
print("=" * 85)

for i, row in df_res.iterrows():
    marker = " ◄ BEST" if i == 0 else (" ◄ top3" if i < 3 else "")
    print(f"{i+1:>3} {row['entry_z']:>6.1f} {row['exit_z']:>+6.1f} {row['stop_z']:>6.1f} "
          f"{row['trades']:>7.0f} {row['win_rate']:>5.1f}% {row['sharpe']:>7.2f} "
          f"{row['total_pnl']:>+11.4f} {row['max_dd']:>9.4f} {row['pf']:>6.2f}{marker}")

best = results[df_res.index[0]]
print(f"\n{'='*85}")
print(f"BEST: entry={best['entry_z']}  exit={best['exit_z']:+.1f}  stop={best['stop_z']}  "
      f"→  Sharpe={best['sharpe']:.2f}  WR={best['win_rate']:.1f}%  "
      f"Trades={best['trades']}  P&L={best['total_pnl']:+.4f}")

# ── Heatmaps: Sharpe by entry_z × exit_z for each stop_z ─────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)

fig, axes = plt.subplots(1, len(STOP_Z_GRID), figsize=(6 * len(STOP_Z_GRID), 5))
if len(STOP_Z_GRID) == 1:
    axes = [axes]

for ax, stop_z in zip(axes, STOP_Z_GRID):
    sub = df_res[df_res["stop_z"] == stop_z]
    pivot = sub.pivot(index="entry_z", columns="exit_z", values="sharpe")
    pivot = pivot.reindex(index=sorted(pivot.index, reverse=True))

    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=df_res["sharpe"].min(), vmax=df_res["sharpe"].max())
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:+.1f}" for v in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.1f}" for v in pivot.index], fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="white" if abs(val) > df_res["sharpe"].std() else "black")

    ax.set_title(f"Sharpe  |  stop_z = {stop_z}", fontsize=11, fontweight="bold")
    ax.set_xlabel("EXIT_Z")
    ax.set_ylabel("ENTRY_Z")
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle("Sniper Grid — Sharpe Heatmap (entry × exit × stop)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sniper_grid.png", dpi=150)
print(f"\nHeatmap saved to {OUTPUT_DIR / 'sniper_grid.png'}")

# ── Top-10 equity curves ──────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(14, 6))
top10_idx = df_res.head(10).index.tolist()
colors    = plt.cm.RdYlGn(np.linspace(0.2, 0.9, 10))

for rank, (orig_idx, color) in enumerate(zip(top10_idx, colors)):
    r   = results[orig_idx]
    lbl = (f"#{rank+1} entry={r['entry_z']} exit={r['exit_z']:+.1f} "
           f"stop={r['stop_z']}  Sh={r['sharpe']:.2f}")
    lw  = 2.5 if rank == 0 else 1.2
    ax2.plot(r["curve"], label=lbl, color=color, lw=lw, alpha=0.85)

ax2.axhline(0, color="black", lw=0.8)
ax2.set_title("Top-10 Sniper Combinations — Equity Curves", fontsize=12, fontweight="bold")
ax2.set_xlabel("Trade #")
ax2.set_ylabel("Cumulative net P&L")
ax2.legend(fontsize=7, loc="upper left")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sniper_top10.png", dpi=150)
print(f"Top-10 curves saved to {OUTPUT_DIR / 'sniper_top10.png'}")
# plt.show()
