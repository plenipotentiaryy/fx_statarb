import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import (
    CLOSES_FILE,
    COST_TAKER, BORROW_RATE_ANNUAL, PAIR_MAX_LOSS,
    RTH_START, RTH_END, SIGNAL_START, RECENT_BARS,
    DATA_DIR, OUTPUT_DIR,
)

# ── Strategy parameters (overriding config defaults) ─────────────────────────
ENTRY_Z = 3.2   # optimal from sniper grid (Sharpe 4.20, WR 83.9%)
EXIT_Z  = -0.2  # exit slightly past mean reversion
STOP_Z  = 4.4   # wide stop — prevents premature exits

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


def backtest_pair(df, t1, t2, beta) -> pd.DataFrame:
    t1_col, t2_col = f"{t1}_close", f"{t2}_close"
    position = 0
    entry_spread = entry_t1 = entry_t2 = 0.0
    entry_bar = 0
    cumulative_pnl = 0.0
    trades = []

    for i in range(len(df)):
        z          = df["zscore"].iloc[i]
        spread_now = df["spread"].iloc[i]
        t1_price   = df[t1_col].iloc[i]
        t2_price   = df[t2_col].iloc[i]

        if position != 0:
            exit_signal = (position == 1 and z > -EXIT_Z) or (position == -1 and z < EXIT_Z)
            stop_signal = (position == 1 and z < -STOP_Z) or (position == -1 and z > STOP_Z)

            if exit_signal or stop_signal:
                gross_pnl      = position * (spread_now - entry_spread)
                notional       = entry_t1 + beta * entry_t2
                tx_cost        = 2 * notional * COST_TAKER
                holding_days   = (i - entry_bar) / BARS_PER_TRADING_DAY
                short_notional = (beta * entry_t2) if position == 1 else entry_t1
                borrow_cost    = short_notional * BORROW_RATE_ANNUAL * holding_days / 252
                net_pnl        = gross_pnl - tx_cost - borrow_cost
                cumulative_pnl += net_pnl

                trades.append({
                    "pair":         f"{t1}-{t2}",
                    "entry_time":   df.index[entry_bar],
                    "exit_time":    df.index[i],
                    "direction":    "LONG" if position == 1 else "SHORT",
                    "holding_bars": i - entry_bar,
                    "gross_pnl":    round(gross_pnl, 4),
                    "tx_cost":      round(tx_cost, 4),
                    "borrow_cost":  round(borrow_cost, 4),
                    "net_pnl":      round(net_pnl, 4),
                    "cum_pnl":      round(cumulative_pnl, 4),
                    "exit_reason":  "STOP" if stop_signal else "SIGNAL",
                    "entry_z":      round(df["zscore"].iloc[entry_bar], 2),
                    "exit_z":       round(z, 2),
                })
                position = 0

                if cumulative_pnl < PAIR_MAX_LOSS:
                    break

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

    return pd.DataFrame(trades)


# ── Load ──────────────────────────────────────────────────────────────────────
closes = load_closes()
pairs  = pd.read_csv(DATA_DIR / "pairs_selected.csv")

if pairs.empty:
    raise SystemExit("pairs_selected.csv is empty — run step2_pairs.py first")

print(f"STRICT STRATEGY  entry={ENTRY_Z}  exit={EXIT_Z}  stop={STOP_Z}")
print(f"Trading {len(pairs)} pairs | {closes.shape[0]} bars per ticker")
print(f"Period: {closes.index[0]} — {closes.index[-1]}\n")

pair_results = {}

for _, row in pairs.iterrows():
    t1, t2    = row["pair"].split("-")
    beta      = row["beta"]
    half_life = row["half_life_bars"]

    if t1 not in closes.columns or t2 not in closes.columns:
        print(f"  SKIP {row['pair']}: missing ticker data")
        continue

    df_sig = build_signals(closes, t1, t2, beta, half_life)
    trades = backtest_pair(df_sig, t1, t2, beta)

    if trades.empty:
        print(f"  {row['pair']:12s}  0 trades")
        continue

    pnl      = trades["net_pnl"]
    win_rate = (pnl > 0).mean() * 100
    disabled = trades["cum_pnl"].iloc[-1] < PAIR_MAX_LOSS
    status   = " [DISABLED]" if disabled else ""

    pair_results[row["pair"]] = {"trades": trades}
    print(f"  {row['pair']:12s}  trades={len(trades):3d}  "
          f"WR={win_rate:4.1f}%  net P&L={pnl.sum():+.4f}{status}")

if not pair_results:
    raise SystemExit("No trades generated — signals too rare at entry_z=3.0")

df_trades = (pd.concat([v["trades"] for v in pair_results.values()])
               .sort_values("exit_time")
               .reset_index(drop=True))

pnl             = df_trades["net_pnl"]
winning         = df_trades[pnl > 0]
losing          = df_trades[pnl <= 0]
stops           = df_trades[df_trades["exit_reason"] == "STOP"]
cumulative      = pnl.cumsum()
max_drawdown    = (cumulative - cumulative.cummax()).min()
days_total      = pd.to_datetime(df_trades["exit_time"].iloc[-1]) - pd.to_datetime(df_trades["exit_time"].iloc[0])
trades_per_year = len(df_trades) / (days_total.days / 365.25)
sharpe          = pnl.mean() / pnl.std() * np.sqrt(trades_per_year) if pnl.std() > 0 else 0.0
profit_factor   = (winning["net_pnl"].sum() / abs(losing["net_pnl"].sum())
                   if len(losing) > 0 and losing["net_pnl"].sum() != 0 else float("inf"))

print(f"\n{'='*60}")
print(f"STRICT PORTFOLIO  entry={ENTRY_Z}  exit={EXIT_Z}  stop={STOP_Z}")
print(f"{'='*60}")
print(f"Trades:        {len(df_trades)}  ({trades_per_year:.0f}/yr)")
print(f"Win rate:      {len(winning)/len(df_trades)*100:.1f}%")
print(f"Stops:         {len(stops)}")
print(f"Gross P&L:     {df_trades['gross_pnl'].sum():+.4f}")
print(f"Costs:         {(df_trades['tx_cost']+df_trades['borrow_cost']).sum():.4f}")
print(f"Net P&L:       {pnl.sum():+.4f}")
print(f"Avg trade:     {pnl.mean():+.4f}")
print(f"Profit factor: {profit_factor:.2f}")
print(f"Max drawdown:  {max_drawdown:.4f}")
print(f"Sharpe:        {sharpe:.2f}")
print(f"Avg hold:      {df_trades['holding_bars'].mean():.0f} bars "
      f"({df_trades['holding_bars'].mean()/BARS_PER_TRADING_DAY:.1f} days)")

print(f"\n{'─'*60}")
print(f"{'Pair':<12} {'Trades':>6} {'WR':>6} {'Net P&L':>10} {'Sharpe':>7}")
print(f"{'─'*60}")
for pair_name, data in pair_results.items():
    t   = data["trades"]
    p   = t["net_pnl"]
    wr  = (p > 0).mean() * 100
    tpy = len(t) / (days_total.days / 365.25)
    sh  = p.mean() / p.std() * np.sqrt(tpy) if p.std() > 0 else 0.0
    print(f"{pair_name:<12} {len(t):>6} {wr:>5.1f}% {p.sum():>+10.4f} {sh:>7.2f}")

df_trades.to_csv(DATA_DIR / "trades_strict.csv", index=False)
print(f"\nSaved to {DATA_DIR / 'trades_strict.csv'}")

OUTPUT_DIR.mkdir(exist_ok=True)
exit_times = pd.to_datetime(df_trades["exit_time"])
colors = plt.cm.tab10(np.linspace(0, 1, len(pair_results)))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

ax1.plot(exit_times, cumulative.values, color="darkgreen", lw=2)
ax1.axhline(0, color="black", lw=0.8)
ax1.set_title(f"STRICT — Portfolio Equity  (entry={ENTRY_Z}, exit={EXIT_Z}, stop={STOP_Z})")
ax1.set_ylabel("Cumulative net P&L")

for (pair_name, data), color in zip(pair_results.items(), colors):
    t  = data["trades"]
    et = pd.to_datetime(t["exit_time"])
    ax2.plot(et, t["net_pnl"].cumsum().values, label=pair_name, color=color, lw=1.5)
ax2.axhline(0, color="black", lw=0.8)
ax2.set_title("Per-pair Equity Curves")
ax2.set_ylabel("Cumulative net P&L")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "backtest_strict.png", dpi=150)
print(f"Chart saved to {OUTPUT_DIR / 'backtest_strict.png'}")
# plt.show()
