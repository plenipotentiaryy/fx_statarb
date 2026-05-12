import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import DATA_DIR, OUTPUT_DIR

N_SIMS = 10_000

df_trades = pd.read_csv(DATA_DIR / "trades.csv")
if df_trades.empty:
    raise SystemExit("No trades found — run step4_backtest.py first.")

pnl = df_trades["net_pnl"].values
n = len(pnl)

real_curve = np.cumsum(pnl)
real_total = real_curve[-1]
real_sharpe = pnl.mean() / pnl.std() * np.sqrt(n) if pnl.std() > 0 else 0.0
real_dd = (real_curve - np.maximum.accumulate(real_curve)).min()

print(f"Loaded {n} trades. Running {N_SIMS:,} simulations...\n")

# ── Bootstrap resampling ──────────────────────────────────────────────────────
# Resample trades with replacement → N possible equity curves
idx = np.random.randint(0, n, size=(N_SIMS, n))
boot_samples = pnl[idx]                                   # (N_SIMS, n)
boot_curves  = np.cumsum(boot_samples, axis=1)            # (N_SIMS, n)
boot_totals  = boot_curves[:, -1]
boot_sharpes = (boot_samples.mean(axis=1)
                / boot_samples.std(axis=1)
                * np.sqrt(n))
running_max  = np.maximum.accumulate(boot_curves, axis=1)
boot_drawdowns = (boot_curves - running_max).min(axis=1)

# ── Random sign test ──────────────────────────────────────────────────────────
# Keep trade magnitudes, randomly flip signs → baseline of pure chance
signs        = np.random.choice([-1, 1], size=(N_SIMS, n))
sign_samples = np.abs(pnl) * signs
sign_totals  = sign_samples.sum(axis=1)
sign_sharpes = (sign_samples.mean(axis=1)
                / sign_samples.std(axis=1)
                * np.sqrt(n))

p_value_total  = np.mean(sign_totals  >= real_total)
p_value_sharpe = np.mean(sign_sharpes >= real_sharpe)

# ── Print results ─────────────────────────────────────────────────────────────
def ci(arr, lo=5, hi=95):
    return np.percentile(arr, lo), np.percentile(arr, hi)

print("=" * 60)
print("MONTE CARLO RESULTS")
print("=" * 60)

print(f"\nReal strategy:")
print(f"  Total P&L : {real_total:+.4f}")
print(f"  Sharpe    : {real_sharpe:.2f}")
print(f"  Max DD    : {real_dd:.4f}")

lo, hi = ci(boot_totals)
print(f"\nBootstrap 90% CI  ({N_SIMS:,} samples with replacement):")
print(f"  Total P&L : [{lo:+.4f}, {hi:+.4f}]")
lo, hi = ci(boot_sharpes)
print(f"  Sharpe    : [{lo:.2f}, {hi:.2f}]")
lo, hi = ci(boot_drawdowns)
print(f"  Max DD    : [{lo:.4f}, {hi:.4f}]")

print(f"\nRandom sign test  (is the result non-random?):")
sig_total  = "SIGNIFICANT" if p_value_total  < 0.05 else "not significant"
sig_sharpe = "SIGNIFICANT" if p_value_sharpe < 0.05 else "not significant"
print(f"  p (P&L)   : {p_value_total:.4f}  → {sig_total}")
print(f"  p (Sharpe): {p_value_sharpe:.4f}  → {sig_sharpe}")

# ── Visualisation ─────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
x = np.arange(1, n + 1)

# ── 1. Equity curve fan (the "branches") ─────────────────────────────────────
ax = axes[0, 0]

# Draw 300 individual paths as light branches
sample_idx = np.random.choice(N_SIMS, size=300, replace=False)
for i in sample_idx:
    ax.plot(x, boot_curves[i], color="steelblue", alpha=0.04, lw=0.8)

# Percentile bands
p5, p25, p50, p75, p95 = np.percentile(boot_curves, [5, 25, 50, 75, 95], axis=0)
ax.fill_between(x, p5,  p95, alpha=0.15, color="steelblue", label="5–95%")
ax.fill_between(x, p25, p75, alpha=0.35, color="steelblue", label="25–75%")
ax.plot(x, p50, color="steelblue", lw=1.5, linestyle="--", label="Median path")
ax.plot(x, real_curve, color="red", lw=2.5, label="Real strategy", zorder=5)
ax.axhline(0, color="black", lw=0.8)

ax.set_title("Equity Curve — Bootstrap Fan (branches)")
ax.set_xlabel("Trade #")
ax.set_ylabel("Cumulative P&L")
ax.legend(fontsize=8)

# ── 2. Distribution of final P&L ─────────────────────────────────────────────
ax = axes[0, 1]
ax.hist(boot_totals, bins=80, color="steelblue", alpha=0.7, edgecolor="none")
ax.axvline(real_total, color="red",    lw=2,   label=f"Real: {real_total:+.3f}")
ax.axvline(np.percentile(boot_totals, 5),  color="orange", lw=1.5,
           linestyle="--", label="5th pct")
ax.axvline(np.percentile(boot_totals, 95), color="green",  lw=1.5,
           linestyle="--", label="95th pct")
ax.axvline(0, color="black", lw=0.8)
ax.set_title("Distribution of Final P&L (bootstrap)")
ax.set_xlabel("Total P&L")
ax.legend(fontsize=8)

# ── 3. Distribution of Sharpe ─────────────────────────────────────────────────
ax = axes[1, 0]
ax.hist(boot_sharpes, bins=80, color="mediumseagreen", alpha=0.7, edgecolor="none")
ax.axvline(real_sharpe, color="red", lw=2, label=f"Real: {real_sharpe:.2f}")
ax.axvline(0, color="black", lw=0.8)
ax.set_title("Distribution of Sharpe Ratio (bootstrap)")
ax.set_xlabel("Sharpe")
ax.legend(fontsize=8)

# ── 4. Random sign test ───────────────────────────────────────────────────────
ax = axes[1, 1]
ax.hist(sign_totals, bins=80, color="gray", alpha=0.6, edgecolor="none",
        label="Random (sign-flipped)")
ax.axvline(real_total, color="red", lw=2,
           label=f"Real: {real_total:+.3f}  (p={p_value_total:.3f})")
ax.axvline(0, color="black", lw=0.8)
ax.set_title("Random Sign Test — is the result luck?")
ax.set_xlabel("Total P&L")
ax.legend(fontsize=8)

plt.suptitle(
    f"Monte Carlo Analysis  |  {n} trades  |  {N_SIMS:,} simulations",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "montecarlo_results.png", dpi=150)
print(f"\nChart saved to {OUTPUT_DIR / 'montecarlo_results.png'}")
# plt.show()
