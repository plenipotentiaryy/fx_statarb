import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
from config import (
    CLOSES_FILE,
    ENTRY_Z, EXIT_Z, STOP_Z,
    COST_TAKER, BORROW_RATE_ANNUAL,
    RTH_START, RTH_END, RECENT_BARS,
    BARS_PER_DAY,
    DATA_DIR, OUTPUT_DIR,
)

parser = argparse.ArgumentParser()
parser.add_argument("--pairs", type=str, default="pairs_selected.csv")
args = parser.parse_args()

N_SIMS = 10_000
DT     = 1        # one bar


# ── OU parameter estimation ───────────────────────────────────────────────────

def estimate_ou(spread: pd.Series) -> dict:
    """
    Fit OU process: dX = θ(μ - X)dt + σdW
    Using discrete OLS: ΔX = a + b·X_lag + ε
    → θ = -b, μ = -a/b, σ = std(ε)
    """
    dx   = spread.diff().dropna()
    x_lag = spread.shift(1).dropna()
    aligned = pd.concat([dx, x_lag], axis=1).dropna()
    aligned.columns = ["dx", "x_lag"]

    model = sm.OLS(aligned["dx"], sm.add_constant(aligned["x_lag"])).fit()
    a, b  = model.params.iloc[0], model.params.iloc[1]
    sigma = model.resid.std()

    theta     = -b                            # mean reversion speed (per bar)
    mu        = -a / b if b != 0 else spread.mean()
    half_life = np.log(2) / theta if theta > 0 else float("inf")

    return {"theta": theta, "mu": mu, "sigma": sigma,
            "half_life": half_life, "r2": model.rsquared}


# ── Vectorised OU simulation ──────────────────────────────────────────────────

def simulate_ou(theta, mu, sigma, X0, n_steps, n_sims) -> np.ndarray:
    """Returns shape (n_sims, n_steps+1)."""
    paths       = np.empty((n_sims, n_steps + 1))
    paths[:, 0] = X0
    noise = np.random.normal(0, sigma, size=(n_sims, n_steps))
    for t in range(n_steps):
        paths[:, t + 1] = (paths[:, t]
                           + theta * (mu - paths[:, t]) * DT
                           + noise[:, t])
    return paths


# ── P&L for each simulated path ───────────────────────────────────────────────

def compute_pnl(paths, mu, sigma, beta,
                entry_level, exit_level, stop_level,
                direction=1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    direction = +1 for long spread (entry below mean)
               = -1 for short spread (entry above mean)

    Returns: (net_pnl, exit_bar, exit_reason)
             exit_reason: 0=signal, 1=stop, 2=timeout
    """
    n_sims, n_steps = paths.shape[0], paths.shape[1] - 1
    pnl        = np.full(n_sims, np.nan)
    exit_bar   = np.full(n_sims, n_steps, dtype=int)
    exit_reason = np.full(n_sims, 2, dtype=int)  # default: timeout

    for i in range(n_sims):
        path = paths[i]
        for t in range(1, n_steps + 1):
            x = path[t]
            # Exit signal: spread recovered
            if direction == 1 and x >= exit_level:
                pnl[i]         = x - entry_level
                exit_bar[i]    = t
                exit_reason[i] = 0
                break
            elif direction == -1 and x <= exit_level:
                pnl[i]         = entry_level - x
                exit_bar[i]    = t
                exit_reason[i] = 0
                break
            # Stop loss: spread diverged further
            if direction == 1 and x <= stop_level:
                pnl[i]         = x - entry_level
                exit_bar[i]    = t
                exit_reason[i] = 1
                break
            elif direction == -1 and x >= stop_level:
                pnl[i]         = entry_level - x
                exit_bar[i]    = t
                exit_reason[i] = 1
                break

        if np.isnan(pnl[i]):  # timeout: close at last bar
            x      = path[-1]
            pnl[i] = (x - entry_level) * direction

    # Apply transaction costs
    notional    = abs(entry_level) * (1 + beta)
    tx_cost     = 2 * notional * COST_TAKER
    holding_days = exit_bar / BARS_PER_DAY
    borrow_cost  = notional * 0.5 * BORROW_RATE_ANNUAL * holding_days / 252

    net_pnl = pnl - tx_cost - borrow_cost
    return net_pnl, exit_bar, exit_reason


def var_cvar(pnl: np.ndarray, alpha=0.05) -> tuple[float, float]:
    v = np.percentile(pnl, alpha * 100)
    c = pnl[pnl <= v].mean()
    return float(v), float(c)


# ── Load data ─────────────────────────────────────────────────────────────────
def _data_file():
    p = DATA_DIR / CLOSES_FILE
    if not p.exists():
        fb = DATA_DIR / "closes_15min.csv"
        if fb.exists():
            return fb
        raise FileNotFoundError(f"No data: {CLOSES_FILE}")
    return p
closes = pd.read_csv(_data_file(), index_col=0, parse_dates=True)
closes.index = pd.to_datetime(closes.index, utc=True)
closes = closes.between_time(RTH_START, RTH_END).dropna().tail(RECENT_BARS)

pairs_path = DATA_DIR / args.pairs
if not pairs_path.exists():
    raise FileNotFoundError(f"Pairs file {pairs_path} not found.")
pairs = pd.read_csv(pairs_path)
if pairs.empty:
    raise SystemExit("pairs_selected.csv is empty — run step2_pairs.py first")

np.random.seed(42)
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"OU Monte Carlo  |  {N_SIMS:,} simulations per trade")
print(f"Strategy: entry={ENTRY_Z}  exit={EXIT_Z:+.1f}  stop={STOP_Z}\n")

summary_rows = []

for _, row in pairs.iterrows():
    t1, t2 = row["pair"].split("-")
    beta   = row["beta"]

    if t1 not in closes.columns or t2 not in closes.columns:
        continue

    spread = (closes[t1] - beta * closes[t2]).dropna()
    ou     = estimate_ou(spread)
    theta, mu, sigma = ou["theta"], ou["mu"], ou["sigma"]
    half_life = ou["half_life"]

    if half_life == float("inf") or half_life > 5000:
        print(f"  SKIP {row['pair']}: half_life=inf (OU mean-reversion too slow to simulate)")
        continue
    horizon = max(int(half_life * 2), 50)

    print(f"{'─'*55}")
    print(f"  {row['pair']}   β={beta:.4f}")
    print(f"  OU params:  θ={theta:.5f}  μ={mu:.4f}  σ={sigma:.4f}")
    print(f"  Half-life:  {half_life:.0f} bars ({half_life/BARS_PER_DAY:.1f} days)")
    print(f"  Horizon:    {horizon} bars ({horizon/BARS_PER_DAY:.1f} days)")
    print(f"  R²:         {ou['r2']:.4f}")

    # Entry levels in spread units
    entry_level_long  = mu - ENTRY_Z * sigma
    exit_level_long   = mu - EXIT_Z  * sigma   # EXIT_Z can be negative → exit past mean
    stop_level_long   = mu - STOP_Z  * sigma

    entry_level_short = mu + ENTRY_Z * sigma
    exit_level_short  = mu + EXIT_Z  * sigma
    stop_level_short  = mu + STOP_Z  * sigma

    all_pnl = []
    all_exit_bars = []
    all_reasons = []

    for direction, entry_level, exit_level, stop_level in [
        (1,  entry_level_long,  exit_level_long,  stop_level_long),
        (-1, entry_level_short, exit_level_short, stop_level_short),
    ]:
        paths = simulate_ou(theta, mu, sigma, entry_level, horizon, N_SIMS // 2)
        pnl, exit_bars, reasons = compute_pnl(
            paths, mu, sigma, beta,
            entry_level, exit_level, stop_level, direction
        )
        all_pnl.append(pnl)
        all_exit_bars.append(exit_bars)
        all_reasons.append(reasons)

    pnl       = np.concatenate(all_pnl)
    exit_bars = np.concatenate(all_exit_bars)
    reasons   = np.concatenate(all_reasons)

    mean_pnl  = pnl.mean()
    std_pnl   = pnl.std()
    win_rate  = (pnl > 0).mean() * 100
    p_signal  = (reasons == 0).mean() * 100
    p_stop    = (reasons == 1).mean() * 100
    p_timeout = (reasons == 2).mean() * 100
    avg_hold  = exit_bars.mean() / BARS_PER_DAY
    v5, cv5   = var_cvar(pnl, 0.05)
    tpy       = BARS_PER_DAY * 252 / max(exit_bars.mean(), 1)
    sharpe    = mean_pnl / std_pnl * np.sqrt(tpy) if std_pnl > 0 else 0

    print(f"\n  Results ({N_SIMS:,} simulations):")
    print(f"    Win rate:      {win_rate:.1f}%")
    print(f"    Exit signal:   {p_signal:.0f}%  |  Stop: {p_stop:.0f}%  |  Timeout: {p_timeout:.0f}%")
    print(f"    Mean P&L:      {mean_pnl:+.4f}")
    print(f"    VaR  (5%):     {v5:+.4f}")
    print(f"    CVaR (5%):     {cv5:+.4f}")
    print(f"    Sharpe (ann):  {sharpe:.2f}")
    print(f"    Avg hold:      {avg_hold:.1f} days")

    summary_rows.append({
        "pair":      row["pair"],
        "theta":     round(theta, 5),
        "half_life": round(half_life, 0),
        "win_rate":  round(win_rate, 1),
        "mean_pnl":  round(mean_pnl, 4),
        "VaR_5":     round(v5, 4),
        "CVaR_5":    round(cv5, 4),
        "sharpe":    round(sharpe, 2),
        "p_signal":  round(p_signal, 1),
        "p_stop":    round(p_stop, 1),
    })

    # ── Plot: paths fan + P&L distribution ────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: OU paths fan (long side)
    ax = axes[0]
    sample_paths = simulate_ou(theta, mu, sigma, entry_level_long, horizon, 300)
    x_axis = np.arange(horizon + 1)

    for path in sample_paths:
        color = "steelblue" if path[-1] >= exit_level_long else "salmon"
        ax.plot(x_axis, path, color=color, alpha=0.04, lw=0.8)

    p5, p25, p50, p75, p95 = np.percentile(sample_paths, [5, 25, 50, 75, 95], axis=0)
    ax.fill_between(x_axis, p5, p95, alpha=0.15, color="steelblue")
    ax.fill_between(x_axis, p25, p75, alpha=0.30, color="steelblue")
    ax.plot(x_axis, p50, color="steelblue", lw=2, label="Median path")

    ax.axhline(mu,              color="black",  lw=1, linestyle="--", label=f"μ = {mu:.3f}")
    ax.axhline(exit_level_long, color="green",  lw=1.5, linestyle="--",
               label=f"Exit z={EXIT_Z:+.1f}")
    ax.axhline(stop_level_long, color="red",    lw=1.5, linestyle="--",
               label=f"Stop z=-{STOP_Z:.1f}")
    ax.axhline(entry_level_long, color="orange", lw=1.5, linestyle="-",
               label=f"Entry z=-{ENTRY_Z:.1f}")

    ax.set_title(f"{row['pair']} — OU Simulation Paths (Long)")
    ax.set_xlabel("Bars")
    ax.set_ylabel("Spread")
    ax.legend(fontsize=7)

    # Panel 2: P&L distribution
    ax = axes[1]
    ax.hist(pnl[pnl > 0], bins=60, color="steelblue", alpha=0.6, label="Wins")
    ax.hist(pnl[pnl <= 0], bins=40, color="salmon",    alpha=0.6, label="Losses")
    ax.axvline(mean_pnl, color="black", lw=2, label=f"Mean {mean_pnl:+.4f}")
    ax.axvline(v5,  color="orange", lw=1.5, linestyle="--", label=f"VaR(5%) {v5:+.4f}")
    ax.axvline(cv5, color="red",    lw=1.5, linestyle="--", label=f"CVaR(5%) {cv5:+.4f}")
    ax.axvline(0,   color="black",  lw=0.8)
    ax.set_title(f"{row['pair']} — P&L Distribution ({N_SIMS:,} sims)")
    ax.set_xlabel("Net P&L per trade")
    ax.legend(fontsize=7)

    # Panel 3: Cumulative P&L distribution (confidence fan)
    ax = axes[2]
    sorted_pnl = np.sort(pnl)
    ax.plot(sorted_pnl, np.linspace(0, 1, len(sorted_pnl)), color="steelblue", lw=2)
    ax.axhline(0.05, color="red",    linestyle="--", lw=1, label="5th pct (VaR)")
    ax.axhline(0.50, color="black",  linestyle="--", lw=1, label="Median")
    ax.axvline(v5,  color="orange", linestyle="--", lw=1.5, label=f"VaR = {v5:+.4f}")
    ax.axvline(0,   color="black",  lw=0.8)
    ax.set_title(f"{row['pair']} — CDF of P&L")
    ax.set_xlabel("Net P&L")
    ax.set_ylabel("Cumulative probability")
    ax.legend(fontsize=7)

    plt.suptitle(
        f"OU Monte Carlo — {row['pair']}  "
        f"θ={theta:.5f}  μ={mu:.3f}  σ={sigma:.4f}  HL={half_life:.0f} bars",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    fname = f"ou_montecarlo_{row['pair'].replace('-','_')}.png"
    plt.savefig(OUTPUT_DIR / fname, dpi=150)
    print(f"  Chart saved → {OUTPUT_DIR / fname}")
    plt.close()

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SUMMARY — OU MONTE CARLO")
print(f"{'='*70}")
df_summary = pd.DataFrame(summary_rows)
print(df_summary.to_string(index=False))
df_summary.to_csv(DATA_DIR / "ou_montecarlo_summary.csv", index=False)
print(f"\nSaved to {DATA_DIR / 'ou_montecarlo_summary.csv'}")
