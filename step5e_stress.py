"""
step_stress.py — Strategy stress test & live trade decision.

Three analyses:
  1. Walk-forward: 6-month rolling windows over full intraday history
     → shows if performance is consistent across time or lucky in one period
  2. Regime breakdown: performance split by VIX level (calm / normal / elevated)
     → shows how strategy behaves in different market conditions
  3. Trade today? — GO / CAUTION / STOP signal per pair
     → combines current VIX, regime state, recent correlation, MC confidence
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import itertools
from config import (
    COST_MAKER, COST_TAKER, CIRCUIT_BREAKER_Z,
    CLOSES_FILE, ENTRY_Z, EXIT_Z, STOP_Z, ENTRY_Z_VOLATILE,
    BORROW_RATE_ANNUAL,
    RTH_START, RTH_END, SIGNAL_START,
    BARS_PER_DAY, DATA_DIR, OUTPUT_DIR,
    HURST_ENTRY_WINDOW,
)
from filters import HurstFilter
from step3j_wfo import check_coint_johansen, compute_half_life

WINDOW_MONTHS  = 6    # walk-forward window size
STEP_MONTHS    = 2    # how far each window advances

VIX_LOW    = 15.0
VIX_HIGH   = 25.0

OUTPUT_DIR.mkdir(exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_daily() -> pd.DataFrame:
    path = DATA_DIR / "closes_daily.csv"
    if not path.exists(): return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")
    return df

def load_closes() -> pd.DataFrame:
    path = DATA_DIR / CLOSES_FILE
    if not path.exists():
        path = DATA_DIR / "closes_15min.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")
    return df.between_time(RTH_START, RTH_END)


def load_vix_daily() -> pd.Series | None:
    iv_path = DATA_DIR / "iv_filter.csv"
    if not iv_path.exists():
        return None
    df = pd.read_csv(iv_path, index_col=0, parse_dates=True)
    if "vix" in df.columns:
        return df["vix"].dropna()
    return None


# ── Backtest engine (lean, no sizing) ────────────────────────────────────────

def run_backtest(df: pd.DataFrame, t1: str, t2: str, beta: float,
                 entry_z: float, exit_z: float, stop_z: float,
                 hurst_filter: HurstFilter = None,
                 spread_daily: pd.Series = None) -> list[dict]:
    t1c, t2c = f"{t1}_close", f"{t2}_close"
    pos = entry_spread = entry_t1 = entry_t2 = entry_sma = entry_std = 0.0
    entry_bar = 0
    trades = []
    pair_blocked = False

    for i in range(len(df)):
        if pair_blocked: continue
        z  = df["zscore"].iloc[i]
        s  = df["spread"].iloc[i]
        p1 = df[t1c].iloc[i]
        p2 = df[t2c].iloc[i]
        
        z_active = z
        if pos != 0 and entry_std > 0:
            z_active = (s - entry_sma) / entry_std

        if pos != 0:
            ex = (pos == 1 and z_active >= exit_z)  or (pos == -1 and z_active <= -exit_z)
            st = (pos == 1 and z_active <= -stop_z) or (pos == -1 and z_active >= stop_z)
            cb = abs(z_active) >= CIRCUIT_BREAKER_Z
            if ex or st or cb:
                if cb:
                    pair_blocked = True
                    st = True
                gross      = pos * (s - entry_spread)
                notional   = entry_t1 + beta * entry_t2
                tx         = notional * COST_MAKER + notional * (COST_MAKER if ex and not cb else COST_TAKER)
                hold_days  = (i - entry_bar) / BARS_PER_DAY
                borrow     = (beta * entry_t2 if pos == 1 else entry_t1) * BORROW_RATE_ANNUAL * hold_days / 252
                trades.append({
                    "bar":    i,
                    "time":   df.index[i],
                    "net":    gross - tx - borrow,
                    "reason": "STOP" if st else "EXIT",
                })
                pos = 0

        if pos == 0:
            if z < -entry_z: pos = 1
            elif z > entry_z: pos = -1
            if pos != 0:
                # ── Hurst drift guard ─────────────────────────────
                if hurst_filter is not None and spread_daily is not None:
                    ts = df.index[i]
                    d_prev = (ts - pd.Timedelta(days=1)).normalize()
                    h_tail_daily = spread_daily.loc[:d_prev].tail(HURST_ENTRY_WINDOW - 1)
                    h_tail = pd.concat([h_tail_daily, pd.Series({ts: df["spread"].iloc[i]})])
                    
                    h_blocked, _ = hurst_filter.should_block(h_tail, ts)
                    if h_blocked:
                        pos = 0
                        continue
                entry_spread = s; entry_t1 = p1; entry_t2 = p2; entry_bar = i
                entry_sma = df["spread_mean"].iloc[i]
                entry_std = df["spread_std"].iloc[i]

    return trades


def build_signals(closes: pd.DataFrame, t1: str, t2: str,
                  beta: float, half_life: float) -> pd.DataFrame:
    spread = closes[t1] - beta * closes[t2]
    window = max(20, min(int(half_life), 200))
    spread_mean = spread.rolling(window).mean()
    spread_std = spread.rolling(window).std()
    zscore = (spread - spread_mean) / spread_std
    return pd.DataFrame({
        f"{t1}_close": closes[t1],
        f"{t2}_close": closes[t2],
        "spread": spread, "zscore": zscore,
        "spread_mean": spread_mean, "spread_std": spread_std
    }).dropna().between_time(SIGNAL_START, RTH_END)


def metrics(trades: list[dict], days: float) -> dict:
    if not trades:
        return {"n": 0, "wr": 0, "sharpe": 0, "pnl": 0, "dd": 0}
    pnl   = np.array([t["net"] for t in trades])
    curve = np.cumsum(pnl)
    wr    = float((pnl > 0).mean() * 100)
    tpy   = len(pnl) / max(days / 365.25, 0.01)
    sh    = float(pnl.mean() / pnl.std() * np.sqrt(tpy)) if pnl.std() > 0 else 0.0
    dd    = float((curve - np.maximum.accumulate(curve)).min())
    return {"n": len(pnl), "wr": round(wr, 1), "sharpe": round(sh, 2),
            "pnl": round(float(pnl.sum()), 4), "dd": round(dd, 4)}


# ── Load data ─────────────────────────────────────────────────────────────────

closes_all = load_closes()
daily_all = load_daily()
pairs      = pd.read_csv(DATA_DIR / "pairs_selected.csv")
vix_daily  = load_vix_daily()

if pairs.empty:
    raise SystemExit("pairs_selected.csv is empty — run step2_pairs.py first")

print(f"Stress test  |  {len(pairs)} pairs  |  "
      f"{closes_all.index[0].date()} → {closes_all.index[-1].date()}")
print(f"VIX data:    {'available' if vix_daily is not None else 'not found'}\n")

print("=" * 70)
print("WALK-FORWARD TEST  (Dynamic Beta)")
print("=" * 70)

wf_rows = []

for _, row in pairs.iterrows():
    pair_name = row["pair"]
    t1, t2    = pair_name.split("-")
    if t1 not in closes_all.columns or t2 not in closes_all.columns:
        continue
        
    start = closes_all.index[0]
    end   = closes_all.index[-1]
    w     = pd.DateOffset(months=WINDOW_MONTHS)
    step  = pd.DateOffset(months=STEP_MONTHS)
    cur   = start

    windows = []
    hf = HurstFilter()
    
    while cur + w <= end + pd.DateOffset(days=1):
        win_end = cur + w
        
        # Dynamic beta calculation over trailing 3 years
        train_start = cur - pd.DateOffset(years=3)
        daily_train = daily_all[(daily_all.index >= train_start) & (daily_all.index < cur)]
        
        dyn_beta = float(row.get("beta_daily", row["beta"]))
        dyn_hl = float(row["half_life_bars"])
        
        if len(daily_train) >= 100:
            is_coint, b = check_coint_johansen(daily_train, t1, t2, crit_level=0.90)
            if is_coint and b is not None and b > 0:
                dyn_beta = b
                spread_train = daily_train[t1] - dyn_beta * daily_train[t2]
                hl = compute_half_life(spread_train) * BARS_PER_DAY
                if 20 <= hl <= 500:
                    dyn_hl = hl

        # Build signals for the whole history (fast enough) to get correct rolling Z
        df_sig = build_signals(closes_all, t1, t2, dyn_beta, dyn_hl)
        win_df = df_sig[(df_sig.index >= cur) & (df_sig.index < win_end)]
        
        if len(win_df) > 200:
            # Daily spread using dynamic beta for Hurst
            spread_daily = (daily_all[t1] - dyn_beta * daily_all[t2]).dropna()
            windows.append((cur.date(), (win_end - pd.Timedelta(days=1)).date(), win_df, dyn_beta, spread_daily))
            
        cur += step

    if not windows:
        continue

    print(f"\n  {pair_name}  ({len(windows)} windows)")
    print(f"  {'Window':<24} {'Trades':>7} {'WR':>6} {'Sharpe':>8} {'P&L':>10}")
    print(f"  {'─'*58}")

    for w_start, w_end, win_df, dyn_beta, spread_daily in windows:
        days_w = (pd.Timestamp(w_end) - pd.Timestamp(w_start)).days
        trades = run_backtest(win_df, t1, t2, dyn_beta, ENTRY_Z, EXIT_Z, STOP_Z, hf, spread_daily)
        m      = metrics(trades, days_w)
        flag   = "  ◄ GOOD" if m["sharpe"] > 1 else ("  ✗ BAD" if m["sharpe"] < -1 else "")
        print(f"  {str(w_start)} → {str(w_end)}  "
              f"{m['n']:>7}  {m['wr']:>5.1f}%  {m['sharpe']:>8.2f}  "
              f"{m['pnl']:>+10.4f}{flag}")
        wf_rows.append({"pair": pair_name, "window_start": str(w_start),
                        "window_end": str(w_end), **m})

wf_df = pd.DataFrame(wf_rows)
wf_df.to_csv(DATA_DIR / "stress_walkforward.csv", index=False)

# Walk-forward chart
if not wf_df.empty:
    fig, axes = plt.subplots(len(pairs), 1,
                             figsize=(13, 4 * len(pairs)), squeeze=False)
    for ax_row, (_, row) in zip(axes, pairs.iterrows()):
        pair_name = row["pair"]
        ax  = ax_row[0]
        sub = wf_df[wf_df["pair"] == pair_name].reset_index(drop=True)
        if sub.empty:
            ax.set_visible(False)
            continue
        colors = ["green" if s > 0 else "red" for s in sub["sharpe"]]
        ax.bar(range(len(sub)), sub["sharpe"], color=colors, alpha=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(1, color="green", lw=1, ls="--", alpha=0.5)
        ax.axhline(-1, color="red",  lw=1, ls="--", alpha=0.5)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([r["window_start"][:7] for _, r in sub.iterrows()],
                           rotation=30, fontsize=8)
        ax.set_title(f"{pair_name}  —  Sharpe per {WINDOW_MONTHS}-month window", fontweight="bold")
        ax.set_ylabel("Sharpe")

    plt.suptitle("Walk-Forward Consistency", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "stress_walkforward.png", dpi=150)
    plt.close()
    print(f"\nWalk-forward chart saved → {OUTPUT_DIR / 'stress_walkforward.png'}")


# ═══════════════════════════════════════════════════════════════════════
# 2. VIX REGIME BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("VIX REGIME BREAKDOWN")
print(f"{'='*70}")

if vix_daily is not None:
    # Map VIX regime to intraday bars
    vix_tz = vix_daily.copy()
    if vix_tz.index.tz is None:
        vix_tz.index = vix_tz.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        vix_tz.index = vix_tz.index.tz_convert("US/Eastern")

    def vix_regime(date) -> str:
        d = pd.Timestamp(date).tz_localize("US/Eastern") if date.tzinfo is None else date
        candidates = vix_tz[vix_tz.index <= d]
        if candidates.empty:
            return "normal"
        v = candidates.iloc[-1]
        if v < VIX_LOW:   return "calm"
        if v > VIX_HIGH:  return "elevated"
        return "normal"

    regime_labels = ["calm", "normal", "elevated"]

    for _, row in pairs.iterrows():
        pair_name = row["pair"]
        t1, t2 = pair_name.split("-")
        beta = float(row.get("beta_daily", row["beta"]))
        half_life = float(row["half_life_bars"])
        if t1 not in closes_all.columns or t2 not in closes_all.columns:
            continue
        df_sig = build_signals(closes_all, t1, t2, beta, half_life)
        spread_daily = (daily_all[t1] - beta * daily_all[t2]).dropna()
        print(f"\n  {pair_name}")
        print(f"  {'Regime':<12} {'Trades':>7} {'WR':>6} {'Sharpe':>8} {'P&L':>10} {'VIX range'}")
        print(f"  {'─'*58}")
        for regime in regime_labels:
            mask = df_sig.index.map(lambda ts: vix_regime(ts) == regime)
            sub  = df_sig[mask]
            if len(sub) < 50:
                print(f"  {regime:<12}  (not enough data)")
                continue
            days = (sub.index[-1] - sub.index[0]).days
            hf   = HurstFilter()
            tr   = run_backtest(sub, t1, t2, beta, ENTRY_Z, EXIT_Z, STOP_Z, hf)
            m    = metrics(tr, days)
            vix_range = {"calm": f"< {VIX_LOW}",
                         "normal": f"{VIX_LOW}–{VIX_HIGH}",
                         "elevated": f"> {VIX_HIGH}"}[regime]
            print(f"  {regime:<12}  {m['n']:>7}  {m['wr']:>5.1f}%  "
                  f"{m['sharpe']:>8.2f}  {m['pnl']:>+10.4f}  VIX {vix_range}")
else:
    print("  VIX data not available — run step7_iv.py first")


# ═══════════════════════════════════════════════════════════════════════
# 3. "TRADE TODAY?" DECISION
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("TRADE TODAY? — LIVE SIGNAL")
print(f"{'='*70}")

# Load supporting data
mc_path = DATA_DIR / "ou_montecarlo_summary.csv"
mc_conf: dict[str, float] = {}
if mc_path.exists():
    mc_df = pd.read_csv(mc_path)
    for _, r in mc_df.iterrows():
        mc_conf[r["pair"]] = float(r["win_rate"]) / 100

regimes_path = DATA_DIR / "regimes.csv"
current_regime: dict[str, str] = {}
if regimes_path.exists():
    reg_df = pd.read_csv(regimes_path, index_col=0, parse_dates=True)
    for col in reg_df.columns:
        last_val = reg_df[col].dropna().iloc[-1] if not reg_df[col].dropna().empty else 0
        current_regime[col] = "volatile" if last_val == 1 else "normal"

current_vix = None
if vix_daily is not None and len(vix_daily) > 0:
    current_vix = float(vix_daily.iloc[-1])

# Walk-forward consistency score: fraction of windows with Sharpe > 0
wf_consistency: dict[str, float] = {}
if not wf_df.empty:
    for pair_name in pairs["pair"]:
        sub = wf_df[wf_df["pair"] == pair_name]
        if len(sub) > 0:
            wf_consistency[pair_name] = float((sub["sharpe"] > 0).mean())

print(f"\n  Current VIX: {f'{current_vix:.1f}' if current_vix else 'n/a'}")
print()
print(f"  {'Pair':<12} {'Regime':<10} {'MC_conf':>8} {'WF_ok%':>7} "
      f"{'Recent_corr':>12} {'Signal':<10} {'Reason'}")
print(f"  {'─'*80}")

signals_today = []

for _, row in pairs.iterrows():
    pair_name = row["pair"]
    regime    = current_regime.get(pair_name, "unknown")
    mc        = mc_conf.get(pair_name, 1.0)
    wf_ok     = wf_consistency.get(pair_name, 0.5)
    rc        = float(row.get("recent_corr_120d", 1.0))

    # Score each dimension
    issues    = []
    score     = 0

    if mc < 0.45:
        issues.append(f"MC={mc:.2f}<0.45")
    else:
        score += 1

    if wf_ok < 0.5:
        issues.append(f"WF_ok={wf_ok:.0%}<50%")
    else:
        score += 1

    if rc < 0.55:
        issues.append(f"rc120={rc:.2f}<0.55")
    else:
        score += 1

    if regime == "volatile":
        issues.append("volatile regime")
    else:
        score += 1

    if current_vix is not None and current_vix > VIX_HIGH:
        issues.append(f"VIX={current_vix:.0f}>{VIX_HIGH}")
    else:
        score += 1

    if score == 5:
        signal = "✅ GO"
    elif score >= 3:
        signal = "⚠️  CAUTION"
    else:
        signal = "🔴 STOP"

    reason = ", ".join(issues) if issues else "all clear"
    print(f"  {pair_name:<12} {regime:<10} {mc:>8.2f} {wf_ok:>6.0%}  "
          f"{rc:>11.3f}  {signal:<12}  {reason}")

    signals_today.append({
        "pair": pair_name, "regime": regime, "mc_conf": mc,
        "wf_consistency": wf_ok, "recent_corr": rc, "signal": signal,
        "issues": reason,
    })

pd.DataFrame(signals_today).to_csv(DATA_DIR / "stress_signal_today.csv", index=False)

# VIX context
if current_vix is not None:
    vix_regime_now = ("calm" if current_vix < VIX_LOW
                      else "elevated" if current_vix > VIX_HIGH
                      else "normal")
    print(f"\n  Market regime: VIX={current_vix:.1f} → {vix_regime_now.upper()}")

print(f"\nSaved → data/stress_signal_today.csv")
print(f"        data/stress_walkforward.csv")
print(f"        output/stress_walkforward.png")
