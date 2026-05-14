import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_DIR, OUTPUT_DIR
from regime_block_bootstrap import RegimeBlockBootstrap, attach_regime_to_trades


N_SIMS = 10_000
BLOCK_SIZE = 20


def load_global_hmm() -> pd.Series | None:
    path = DATA_DIR / "global_hmm_regime.csv"
    if not path.exists():
        return None
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    s.index = pd.to_datetime(s.index, utc=True).tz_localize(None)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


df_trades = pd.read_csv(DATA_DIR / "trades.csv")
if df_trades.empty:
    raise SystemExit("No trades found — run step4a_backtest.py first.")

global_hmm = load_global_hmm()
if global_hmm is not None:
    df_trades = attach_regime_to_trades(df_trades, global_hmm, time_col="exit_time", regime_col="hmm_regime")
else:
    df_trades["hmm_regime"] = 0

bootstrap = RegimeBlockBootstrap(
    block_size=BLOCK_SIZE,
    n_bootstrap=N_SIMS,
    pnl_col="net_pnl",
    regime_col="hmm_regime",
    random_seed=42,
)
result = bootstrap.bootstrap_metrics(df_trades)
boot_samples = result["samples"]

pnl = pd.to_numeric(df_trades["net_pnl"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
n = len(pnl)
real_curve = np.cumsum(pnl)
real_total = float(real_curve[-1]) if len(real_curve) > 0 else 0.0
real_sharpe = float(result["realized"]["sharpe"])
real_dd = float(result["realized"]["max_drawdown"])

print(f"Loaded {n} trades. Running {N_SIMS:,} regime-aware block bootstrap simulations...\n")
print("=" * 60)
print("REGIME-AWARE BLOCK BOOTSTRAP RESULTS")
print("=" * 60)
print(f"\nReal strategy:")
print(f"  Total P&L : {real_total:+.4f}")
print(f"  Sharpe    : {real_sharpe:.2f}")
print(f"  Max DD    : {real_dd:.4f}")

for metric in ("total_pnl", "sharpe", "max_drawdown"):
    lo, hi = result["confidence_intervals"][metric]
    print(f"  {metric:10s}: [{lo:+.4f}, {hi:+.4f}]")

OUTPUT_DIR.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].hist(boot_samples["total_pnl"], bins=80, color="steelblue", alpha=0.75, edgecolor="none")
axes[0].axvline(real_total, color="red", lw=2, label=f"Real: {real_total:+.3f}")
axes[0].set_title("Final P&L Distribution")
axes[0].set_xlabel("Total P&L")
axes[0].legend(fontsize=8)

axes[1].hist(boot_samples["sharpe"], bins=80, color="mediumseagreen", alpha=0.75, edgecolor="none")
axes[1].axvline(real_sharpe, color="red", lw=2, label=f"Real: {real_sharpe:.2f}")
axes[1].set_title("Sharpe Distribution")
axes[1].set_xlabel("Sharpe")
axes[1].legend(fontsize=8)

axes[2].hist(boot_samples["max_drawdown"], bins=80, color="slategray", alpha=0.75, edgecolor="none")
axes[2].axvline(real_dd, color="red", lw=2, label=f"Real: {real_dd:.3f}")
axes[2].set_title("Max Drawdown Distribution")
axes[2].set_xlabel("Max Drawdown")
axes[2].legend(fontsize=8)

plt.suptitle(
    f"Regime-Aware Block Bootstrap  |  {n} trades  |  {N_SIMS:,} samples  |  block={BLOCK_SIZE}",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "regime_block_bootstrap.png", dpi=150)
print(f"\nChart saved to {OUTPUT_DIR / 'regime_block_bootstrap.png'}")
