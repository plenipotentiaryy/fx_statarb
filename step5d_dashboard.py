import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from config import DATA_DIR, OUTPUT_DIR, BARS_PER_DAY

ROLLING_WINDOW = 20   # trades for rolling Sharpe

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / "trades.csv")
df["exit_time"]  = pd.to_datetime(df["exit_time"],  utc=True)
df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)

if df.empty:
    raise SystemExit("trades.csv is empty — run step4_backtest.py first")

df = df.sort_values("exit_time").reset_index(drop=True)
pnl        = df["net_pnl"]
cumulative = pnl.cumsum()
drawdown   = cumulative - cumulative.cummax()
pairs      = df["pair"].unique()
exit_times = df["exit_time"]

days_total      = (df["exit_time"].iloc[-1] - df["exit_time"].iloc[0]).days
trades_per_year = len(df) / (days_total / 365.25)
sharpe          = pnl.mean() / pnl.std() * np.sqrt(trades_per_year) if pnl.std() > 0 else 0
win_rate        = (pnl > 0).mean() * 100
profit_factor   = (pnl[pnl > 0].sum() / abs(pnl[pnl <= 0].sum())
                   if pnl[pnl <= 0].sum() != 0 else float("inf"))
max_dd          = drawdown.min()
total_costs     = (df["tx_cost"] + df["borrow_cost"]).sum()

print(f"Dashboard: {len(df)} trades across {len(pairs)} pairs")

# ── Layout ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 26))
fig.patch.set_facecolor("#f8f9fa")

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

TITLE_SIZE = 10
colors_pairs = plt.cm.tab10(np.linspace(0, 1, len(pairs)))
pair_color   = {p: c for p, c in zip(pairs, colors_pairs)}

# ═════════════════════════════════════════════════════════════════════════════
# ROW 1
# ═════════════════════════════════════════════════════════════════════════════

# 1. Portfolio equity curve
ax = fig.add_subplot(gs[0, :2])
ax.plot(exit_times, cumulative.values, color="#2196F3", lw=2, label="Net P&L")
ax.plot(exit_times, df["gross_pnl"].cumsum().values,
        color="#2196F3", lw=1, linestyle="--", alpha=0.4, label="Gross P&L")
ax.fill_between(exit_times, cumulative.values, 0,
                where=(cumulative.values >= 0), color="#4CAF50", alpha=0.1)
ax.fill_between(exit_times, cumulative.values, 0,
                where=(cumulative.values < 0),  color="#F44336", alpha=0.1)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Portfolio Equity Curve (net vs gross)", fontsize=TITLE_SIZE, fontweight="bold")
ax.set_ylabel("Cumulative P&L")
ax.legend(fontsize=8)
ax.set_facecolor("#ffffff")

# 2. Per-pair equity
ax = fig.add_subplot(gs[0, 2])
for pair in pairs:
    t = df[df["pair"] == pair]
    et = pd.to_datetime(t["exit_time"])
    ax.plot(et, t["net_pnl"].cumsum().values,
            label=pair, color=pair_color[pair], lw=1.5)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Per-pair Equity", fontsize=TITLE_SIZE, fontweight="bold")
ax.legend(fontsize=7)
ax.set_facecolor("#ffffff")

# ═════════════════════════════════════════════════════════════════════════════
# ROW 2
# ═════════════════════════════════════════════════════════════════════════════

# 3. P&L distribution
ax = fig.add_subplot(gs[1, 0])
wins   = pnl[pnl > 0]
losses = pnl[pnl <= 0]
ax.hist(losses, bins=25, color="#F44336", alpha=0.7, label=f"Losses ({len(losses)})")
ax.hist(wins,   bins=25, color="#4CAF50", alpha=0.7, label=f"Wins ({len(wins)})")
ax.axvline(pnl.mean(), color="black", lw=1.5, linestyle="--",
           label=f"Mean {pnl.mean():+.3f}")
ax.axvline(0, color="black", lw=0.8)
ax.set_title("Trade P&L Distribution", fontsize=TITLE_SIZE, fontweight="bold")
ax.set_xlabel("Net P&L per trade")
ax.legend(fontsize=7)
ax.set_facecolor("#ffffff")

# 4. Drawdown
ax = fig.add_subplot(gs[1, 1])
ax.fill_between(exit_times, drawdown.values, 0, color="#F44336", alpha=0.6)
ax.plot(exit_times, drawdown.values, color="#B71C1C", lw=1)
ax.axhline(max_dd, color="darkred", lw=1, linestyle="--",
           label=f"Max DD {max_dd:.3f}")
ax.set_title("Drawdown Curve", fontsize=TITLE_SIZE, fontweight="bold")
ax.set_ylabel("Drawdown")
ax.legend(fontsize=8)
ax.set_facecolor("#ffffff")

# 5. Rolling Sharpe
ax = fig.add_subplot(gs[1, 2])
rolling_sh = [
    (pnl.iloc[max(0, i - ROLLING_WINDOW):i].mean()
     / pnl.iloc[max(0, i - ROLLING_WINDOW):i].std()
     * np.sqrt(ROLLING_WINDOW)
     if pnl.iloc[max(0, i - ROLLING_WINDOW):i].std() > 0 else 0)
    for i in range(1, len(pnl) + 1)
]
ax.plot(exit_times, rolling_sh, color="#9C27B0", lw=1.5)
ax.axhline(0, color="black", lw=0.8)
ax.axhline(1, color="green", lw=0.8, linestyle="--", alpha=0.5)
ax.fill_between(exit_times, rolling_sh, 0,
                where=np.array(rolling_sh) >= 0, color="#4CAF50", alpha=0.15)
ax.fill_between(exit_times, rolling_sh, 0,
                where=np.array(rolling_sh) < 0,  color="#F44336", alpha=0.15)
ax.set_title(f"Rolling Sharpe ({ROLLING_WINDOW} trades)", fontsize=TITLE_SIZE, fontweight="bold")
ax.set_facecolor("#ffffff")

# ═════════════════════════════════════════════════════════════════════════════
# ROW 3
# ═════════════════════════════════════════════════════════════════════════════

# 6. Win rate per pair
ax = fig.add_subplot(gs[2, 0])
pair_wr = {p: (df[df["pair"] == p]["net_pnl"] > 0).mean() * 100 for p in pairs}
bar_colors = ["#4CAF50" if v >= 50 else "#F44336" for v in pair_wr.values()]
bars = ax.bar(pair_wr.keys(), pair_wr.values(), color=bar_colors, edgecolor="white")
ax.axhline(50, color="black", lw=1, linestyle="--")
ax.set_title("Win Rate per Pair", fontsize=TITLE_SIZE, fontweight="bold")
ax.set_ylabel("Win rate (%)")
ax.set_ylim(0, 100)
for bar, val in zip(bars, pair_wr.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.0f}%", ha="center", va="bottom", fontsize=8)
ax.set_facecolor("#ffffff")

# 7. Hold time distribution
ax = fig.add_subplot(gs[2, 1])
bars_per_day = BARS_PER_DAY
hold_days = df["holding_bars"] / bars_per_day
ax.hist(hold_days, bins=30, color="#FF9800", alpha=0.8, edgecolor="white")
ax.axvline(hold_days.mean(), color="black", lw=1.5, linestyle="--",
           label=f"Mean {hold_days.mean():.1f}d")
ax.set_title("Holding Time Distribution", fontsize=TITLE_SIZE, fontweight="bold")
ax.set_xlabel("Days in trade")
ax.legend(fontsize=8)
ax.set_facecolor("#ffffff")

# 8. Long vs Short P&L
ax = fig.add_subplot(gs[2, 2])
directions = df.groupby("direction")["net_pnl"].agg(["sum", "count", "mean"])
dir_colors = {"LONG": "#4CAF50", "SHORT": "#2196F3"}
for i, (direction, row_d) in enumerate(directions.iterrows()):
    color = dir_colors.get(direction, "gray")
    ax.bar(direction, row_d["sum"], color=color, alpha=0.8, label=f"{direction} ({int(row_d['count'])})")
    ax.text(i, row_d["sum"] + (1 if row_d["sum"] >= 0 else -3),
            f"{row_d['sum']:+.1f}", ha="center", fontsize=8)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("P&L by Direction", fontsize=TITLE_SIZE, fontweight="bold")
ax.set_ylabel("Total net P&L")
ax.legend(fontsize=8)
ax.set_facecolor("#ffffff")

# ═════════════════════════════════════════════════════════════════════════════
# ROW 4
# ═════════════════════════════════════════════════════════════════════════════

# 9. Monthly P&L heatmap
ax = fig.add_subplot(gs[3, :2])
df["month"]    = df["exit_time"].dt.to_period("M")
monthly_pnl    = df.groupby(["pair", "month"])["net_pnl"].sum().unstack(fill_value=0)
monthly_pnl.columns = monthly_pnl.columns.astype(str)

cmap = LinearSegmentedColormap.from_list("rg", ["#F44336", "white", "#4CAF50"])
im = ax.imshow(monthly_pnl.values, aspect="auto", cmap=cmap,
               vmin=-abs(monthly_pnl.values).max(),
               vmax=abs(monthly_pnl.values).max())
ax.set_xticks(range(len(monthly_pnl.columns)))
ax.set_xticklabels(monthly_pnl.columns, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(len(monthly_pnl.index)))
ax.set_yticklabels(monthly_pnl.index, fontsize=8)
for i in range(monthly_pnl.shape[0]):
    for j in range(monthly_pnl.shape[1]):
        val = monthly_pnl.values[i, j]
        if val != 0:
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=7, color="black")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Monthly P&L Heatmap (per pair)", fontsize=TITLE_SIZE, fontweight="bold")
ax.set_facecolor("#ffffff")

# 10. Costs breakdown + summary metrics
ax = fig.add_subplot(gs[3, 2])
ax.axis("off")

gross_total  = df["gross_pnl"].sum()
tx_total     = df["tx_cost"].sum()
borrow_total = df["borrow_cost"].sum()
net_total    = pnl.sum()
stops        = (df["exit_reason"] == "STOP").sum()

metrics = [
    ("━━━ PORTFOLIO SUMMARY ━━━", "", "bold"),
    ("", "", "normal"),
    ("Trades",        f"{len(df)}  ({trades_per_year:.0f}/yr)", "normal"),
    ("Win rate",      f"{win_rate:.1f}%", "normal"),
    ("Stops",         f"{stops}", "normal"),
    ("Avg hold",      f"{(df['holding_bars']/BARS_PER_DAY).mean():.1f} days", "normal"),
    ("", "", "normal"),
    ("━━━ P&L ━━━", "", "bold"),
    ("", "", "normal"),
    ("Gross P&L",     f"{gross_total:+.2f}", "normal"),
    ("  tx costs",    f"−{tx_total:.2f}", "normal"),
    ("  borrow",      f"−{borrow_total:.2f}", "normal"),
    ("Net P&L",       f"{net_total:+.2f}", "bold"),
    ("", "", "normal"),
    ("━━━ RISK ━━━", "", "bold"),
    ("", "", "normal"),
    ("Sharpe",        f"{sharpe:.2f}", "normal"),
    ("Profit factor", f"{profit_factor:.2f}", "normal"),
    ("Max drawdown",  f"{max_dd:.3f}", "normal"),
    ("Cost drag",     f"{total_costs/gross_total*100:.0f}% of gross", "normal"),
]

y = 0.98
for label, value, weight in metrics:
    if label.startswith("━"):
        ax.text(0.05, y, label, transform=ax.transAxes, fontsize=8,
                fontweight="bold", color="#333333", va="top")
    else:
        ax.text(0.05, y, label,  transform=ax.transAxes, fontsize=8.5,
                fontweight=weight, color="#555555", va="top")
        ax.text(0.70, y, value,  transform=ax.transAxes, fontsize=8.5,
                fontweight=weight, color="#111111", va="top", ha="right")
    y -= 0.05

ax.set_facecolor("#f0f4f8")
ax.set_title("Key Metrics", fontsize=TITLE_SIZE, fontweight="bold")

# ── Save ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)
plt.suptitle("AFES — Statistical Arbitrage Dashboard",
             fontsize=16, fontweight="bold", y=0.995, color="#1a237e")

out_path = OUTPUT_DIR / "dashboard.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Dashboard saved to {out_path}")
# plt.show()
