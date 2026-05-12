import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import (
    CLOSES_FILE,
    ENTRY_Z, STOP_Z,
    COST_PER_SIDE, BORROW_RATE_ANNUAL,
    RTH_START, RTH_END, SIGNAL_START, RECENT_BARS,
    DATA_DIR, OUTPUT_DIR,
)

EXIT_Z_GRID = [0.5, 0.3, 0.1, 0.0, -0.1, -0.2, -0.3]
from config import BARS_PER_DAY
BARS_PER_TRADING_DAY = BARS_PER_DAY


def load_closes() -> pd.DataFrame:
    closes = pd.read_csv(DATA_DIR / CLOSES_FILE, index_col=0, parse_dates=True)
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


def backtest_pair(df, t1, t2, beta, exit_z) -> pd.DataFrame:
    t1_col, t2_col = f"{t1}_close", f"{t2}_close"
    position = 0
    entry_spread = entry_t1 = entry_t2 = 0.0
    entry_bar = 0
    trades = []

    for i in range(len(df)):
        z          = df["zscore"].iloc[i]
        spread_now = df["spread"].iloc[i]
        t1_price   = df[t1_col].iloc[i]
        t2_price   = df[t2_col].iloc[i]

        if position != 0:
            exit_signal = (position == 1 and z > -exit_z) or (position == -1 and z < exit_z)
            stop_signal = (position == 1 and z < -STOP_Z) or (position == -1 and z > STOP_Z)

            if exit_signal or stop_signal:
                gross_pnl      = position * (spread_now - entry_spread)
                notional       = entry_t1 + beta * entry_t2
                tx_cost        = 2 * notional * COST_PER_SIDE
                holding_days   = (i - entry_bar) / BARS_PER_TRADING_DAY
                short_notional = (beta * entry_t2) if position == 1 else entry_t1
                borrow_cost    = short_notional * BORROW_RATE_ANNUAL * holding_days / 252
                trades.append(gross_pnl - tx_cost - borrow_cost)
                position = 0

        if position == 0:
            if z < -ENTRY_Z:
                position = 1
            elif z > ENTRY_Z:
                position = -1
            if position != 0:
                entry_spread = spread_now
                entry_t1     = t1_price
                entry_t2     = t2_price
                entry_bar    = i

    return trades


# ── Load ──────────────────────────────────────────────────────────────────────
closes = load_closes()
pairs  = pd.read_csv(DATA_DIR / "pairs_selected.csv")

if pairs.empty:
    raise SystemExit("pairs_selected.csv is empty — run step2_pairs.py first")

print(f"Grid search EXIT_Z = {EXIT_Z_GRID}")
print(f"Pairs: {list(pairs['pair'])}\n")

# ── Grid search ───────────────────────────────────────────────────────────────
rows = []

for exit_z in EXIT_Z_GRID:
    all_pnl = []

    for _, row in pairs.iterrows():
        t1, t2    = row["pair"].split("-")
        beta      = row["beta"]
        half_life = row["half_life_bars"]

        if t1 not in closes.columns or t2 not in closes.columns:
            continue

        df_sig = build_signals(closes, t1, t2, beta, half_life)
        trade_pnl = backtest_pair(df_sig, t1, t2, beta, exit_z)
        all_pnl.extend(trade_pnl)

    if not all_pnl:
        continue

    pnl   = pd.Series(all_pnl)
    n     = len(pnl)
    curve = pnl.cumsum()
    wr    = (pnl > 0).mean() * 100
    tpy   = n / (closes.index[-1] - closes.index[0]).days * 365.25
    sh    = pnl.mean() / pnl.std() * np.sqrt(tpy) if pnl.std() > 0 else 0.0
    dd    = (curve - curve.cummax()).min()

    rows.append({
        "exit_z":    exit_z,
        "trades":    n,
        "win_rate":  wr,
        "sharpe":    sh,
        "total_pnl": pnl.sum(),
        "avg_pnl":   pnl.mean(),
        "max_dd":    dd,
        "curve":     curve.values,
    })

# ── Table ─────────────────────────────────────────────────────────────────────
print("=" * 70)
print(f"{'exit_z':>8} {'trades':>7} {'win_rate':>9} {'sharpe':>7} "
      f"{'total_pnl':>11} {'avg_pnl':>9} {'max_dd':>10}")
print("=" * 70)
for r in rows:
    print(f"{r['exit_z']:>+8.1f} {r['trades']:>7} {r['win_rate']:>8.1f}% "
          f"{r['sharpe']:>7.2f} {r['total_pnl']:>+11.4f} "
          f"{r['avg_pnl']:>+9.4f} {r['max_dd']:>10.4f}")

best = max(rows, key=lambda r: r["sharpe"])
print(f"\nBest EXIT_Z by Sharpe: {best['exit_z']:+.1f}  →  "
      f"Sharpe={best['sharpe']:.2f}, WR={best['win_rate']:.1f}%, "
      f"Trades={best['trades']}, P&L={best['total_pnl']:+.4f}")

# ── Charts ────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)
colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(rows)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

for r, color in zip(rows, colors):
    lw    = 2.5 if r["exit_z"] == best["exit_z"] else 1.2
    alpha = 1.0 if r["exit_z"] == best["exit_z"] else 0.6
    label = f"exit_z={r['exit_z']:+.1f}  Sh={r['sharpe']:.2f}  WR={r['win_rate']:.0f}%"
    ax1.plot(r["curve"], label=label, color=color, lw=lw, alpha=alpha)
ax1.axhline(0, color="black", lw=0.8)
ax1.set_title("Portfolio Equity Curves by EXIT_Z")
ax1.set_xlabel("Trade #")
ax1.set_ylabel("Cumulative net P&L")
ax1.legend(fontsize=7)

sharpes   = [r["sharpe"]  for r in rows]
exit_zs   = [f"{r['exit_z']:+.1f}" for r in rows]
bars = ax2.bar(exit_zs, sharpes, color=colors, edgecolor="white")
ax2.axhline(0, color="black", lw=0.8)
ax2.set_title("Sharpe by EXIT_Z (all pairs combined)")
ax2.set_xlabel("EXIT_Z")
ax2.set_ylabel("Sharpe")
for bar, val in zip(bars, sharpes):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.02 * max(sharpes),
             f"{val:.2f}", ha="center", va="bottom", fontsize=9)

plt.suptitle("EXIT_Z Grid Search — All Pairs Combined", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "exit_z_grid.png", dpi=150)
print(f"\nChart saved to {OUTPUT_DIR / 'exit_z_grid.png'}")
# plt.show()
