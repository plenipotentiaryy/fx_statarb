import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm
from scipy.optimize import brentq
from config import (
    IV_LOOKBACK, IV_THRESHOLD, IV_SIZE_HIGH, IV_SIZE_NORM,
    DATA_DIR, OUTPUT_DIR,
)

RISK_FREE_RATE = 0.045    # ~4.5% (approximate current US risk-free rate)
MIN_OPTION_VOLUME = 10    # skip options with very low volume


# ── Black-Scholes & IV ────────────────────────────────────────────────────────

def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, sigma):
    call = bs_call(S, K, T, r, sigma)
    return call - S + K * np.exp(-r * T)   # put-call parity


def implied_vol(market_price, S, K, T, r, option_type="call"):
    if market_price <= 0 or T <= 0:
        return np.nan
    pricer = bs_call if option_type == "call" else bs_put
    intrinsic = max(S - K * np.exp(-r * T), 0.0) if option_type == "call" \
                else max(K * np.exp(-r * T) - S, 0.0)
    if market_price <= intrinsic:
        return np.nan
    try:
        return brentq(
            lambda sigma: pricer(S, K, T, r, sigma) - market_price,
            1e-6, 10.0, xtol=1e-6, maxiter=100,
        )
    except (ValueError, RuntimeError):
        return np.nan


def atm_iv_for_ticker(ticker_str: str) -> float | None:
    try:
        tk = yf.Ticker(ticker_str)
        S  = tk.fast_info.get("lastPrice") or tk.fast_info.get("regularMarketPrice")
        if S is None or S <= 0:
            return None

        expirations = tk.options
        if not expirations:
            return None

        # Pick nearest expiry with >= 7 days to avoid pinning
        today = pd.Timestamp.today().normalize()
        valid = [e for e in expirations
                 if (pd.Timestamp(e) - today).days >= 7]
        if not valid:
            return None
        expiry = valid[0]

        T_days = (pd.Timestamp(expiry) - today).days
        T      = T_days / 365.0

        chain = tk.option_chain(expiry)
        calls = chain.calls
        puts  = chain.puts

        # ATM = nearest strike to spot
        strikes   = calls["strike"].values
        atm_idx   = np.argmin(np.abs(strikes - S))
        atm_strike = strikes[atm_idx]

        ivs = []
        # Try call
        call_row = calls[calls["strike"] == atm_strike]
        if not call_row.empty:
            row = call_row.iloc[0]
            volume = row.get("volume", 0) or 0
            if volume >= MIN_OPTION_VOLUME:
                mid = (row["bid"] + row["ask"]) / 2
                iv  = implied_vol(mid, S, atm_strike, T, RISK_FREE_RATE, "call")
                if iv and not np.isnan(iv):
                    ivs.append(iv)
                # Fallback to yfinance's own IV
                if not ivs and row.get("impliedVolatility", 0) > 0:
                    ivs.append(row["impliedVolatility"])

        # Try put
        put_row = puts[puts["strike"] == atm_strike]
        if not put_row.empty:
            row = put_row.iloc[0]
            volume = row.get("volume", 0) or 0
            if volume >= MIN_OPTION_VOLUME:
                mid = (row["bid"] + row["ask"]) / 2
                iv  = implied_vol(mid, S, atm_strike, T, RISK_FREE_RATE, "put")
                if iv and not np.isnan(iv):
                    ivs.append(iv)

        return float(np.mean(ivs)) if ivs else None

    except Exception as e:
        print(f"    {ticker_str}: option error — {e}")
        return None


# ── Download VIX / VIX9D history ─────────────────────────────────────────────
if os.getenv("ALLOW_NETWORK") != "1":
    cached = DATA_DIR / "iv_filter.csv"
    if cached.exists():
        print(f"Network disabled; using cached {cached}")
        raise SystemExit(0)

    daily_path = DATA_DIR / "closes_daily.csv"
    if not daily_path.exists():
        raise FileNotFoundError("No cached iv_filter.csv or closes_daily.csv available.")

    daily = pd.read_csv(daily_path, index_col=0, parse_dates=True)
    idx = pd.to_datetime(daily.index, utc=True)
    neutral = pd.DataFrame({
        "vix": 20.0,
        "vix9d": 19.0,
        "term_ratio": 0.95,
        "vix_pct": 20.0,
        "position_size": IV_SIZE_NORM,
        "monday_no_crush": 0,
        "macro_alert": 0,
    }, index=idx)
    neutral.to_csv(cached)
    print(f"Network disabled; saved neutral IV filter to {cached}")
    raise SystemExit(0)

import yfinance as yf

print("Downloading VIX / VIX9D historical data...")
vix_raw   = yf.download("^VIX",  period="2y", interval="1d", progress=False)
vix9d_raw = yf.download("^VIX9D", period="2y", interval="1d", progress=False)

vix = vix_raw["Close"].squeeze().dropna()
vix.index = pd.to_datetime(vix.index, utc=True)

vix9d = vix9d_raw["Close"].squeeze().dropna()
vix9d.index = pd.to_datetime(vix9d.index, utc=True)
vix9d = vix9d.reindex(vix.index).ffill()   # align to VIX calendar

print(f"VIX:   {len(vix)} daily bars  ({vix.index[0].date()} — {vix.index[-1].date()})")
print(f"VIX9D: {len(vix9d.dropna())} daily bars\n")

# ── Get current ATM IV for each ticker in our pairs ───────────────────────────
pairs = pd.read_csv(DATA_DIR / "pairs_selected.csv")

print("Fetching current ATM IV from option chains:")
ticker_iv: dict[str, float] = {}

all_tickers = set()
for _, row in pairs.iterrows():
    t1, t2 = row["pair"].split("-")
    all_tickers.update([t1, t2])

for ticker_str in sorted(all_tickers):
    iv = atm_iv_for_ticker(ticker_str)
    ticker_iv[ticker_str] = iv
    iv_str = f"{iv*100:.1f}%" if iv else "N/A"
    print(f"  {ticker_str:6s}  ATM IV = {iv_str}")

# Pair-level IV = average of both legs
print("\nPair ATM IV:")
pair_iv_now: dict[str, float] = {}
for _, row in pairs.iterrows():
    t1, t2 = row["pair"].split("-")
    iv1 = ticker_iv.get(t1)
    iv2 = ticker_iv.get(t2)
    vals = [v for v in [iv1, iv2] if v is not None]
    if vals:
        pair_iv_now[row["pair"]] = float(np.mean(vals))
        print(f"  {row['pair']:12s}  ATM IV = {np.mean(vals)*100:.1f}%")
    else:
        print(f"  {row['pair']:12s}  ATM IV = N/A")

# ── Rolling IV percentile filter (VIX-based) ──────────────────────────────────
vix_pct = vix.rolling(IV_LOOKBACK).quantile(IV_THRESHOLD / 100)

# Position size signal: 1.0 = normal, IV_SIZE_HIGH = elevated
iv_signal = pd.Series(
    np.where(vix >= vix_pct, IV_SIZE_HIGH, IV_SIZE_NORM),
    index=vix.index,
    name="position_size",
)

# ── Term structure signal (VIX9D / VIX ratio) ────────────────────────────────
# Ratio > 1.0 → backwardation: short-term fear > medium-term → panic signal
term_ratio = (vix9d / vix).rename("term_ratio")

# Monday no-crush: Monday AND VIX9D did not fall from prior Friday close
is_monday     = pd.Series(vix.index.dayofweek == 0, index=vix.index)
vix9d_chg     = vix9d.diff()
monday_no_crush = (is_monday & (vix9d_chg >= 0)).rename("monday_no_crush")

# macro_alert = 1 when term structure inverted OR monday no-crush
# → backtest / sizing layers should block new entries when macro_alert == 1
macro_alert = ((term_ratio > 1.0) | monday_no_crush).astype(int).rename("macro_alert")

# ── Current market status ─────────────────────────────────────────────────────
current_vix      = float(vix.iloc[-1])
current_vix9d    = float(vix9d.iloc[-1]) if not pd.isna(vix9d.iloc[-1]) else None
current_pct_rank = float((vix.tail(IV_LOOKBACK) <= current_vix).mean() * 100)
current_size     = IV_SIZE_HIGH if current_vix >= float(vix_pct.iloc[-1]) else IV_SIZE_NORM
current_ratio    = float(term_ratio.iloc[-1]) if not pd.isna(term_ratio.iloc[-1]) else None
current_alert    = int(macro_alert.iloc[-1])

print(f"\n{'='*55}")
print(f"CURRENT IV STATUS")
print(f"{'='*55}")
print(f"VIX today:          {current_vix:.2f}")
if current_vix9d:
    print(f"VIX9D today:        {current_vix9d:.2f}")
    print(f"VIX9D/VIX ratio:    {current_ratio:.3f}  "
          f"({'BACKWARDATION ⚠' if current_ratio and current_ratio > 1.0 else 'normal contango'})")
print(f"VIX {IV_LOOKBACK}d {IV_THRESHOLD}th pct:   {float(vix_pct.iloc[-1]):.2f}")
print(f"Current percentile: {current_pct_rank:.0f}th")
print(f"Position size:      {current_size:.1f}x  "
      f"({'REDUCED — high IV' if current_size < 1 else 'FULL — normal IV'})")
print(f"Macro alert:        {'⚠  BLOCK NEW ENTRIES' if current_alert else 'clear'}")

pct_elevated = (iv_signal < IV_SIZE_NORM).mean() * 100
pct_alert    = macro_alert.mean() * 100
print(f"\nHistorically elevated ({IV_THRESHOLD}th pct): {pct_elevated:.0f}% of days")
print(f"Macro alert active:               {pct_alert:.0f}% of days")

# ── Save IV filter ────────────────────────────────────────────────────────────
iv_out = pd.DataFrame({
    "vix":             vix,
    "vix9d":           vix9d,
    "term_ratio":      term_ratio,
    "vix_pct":         vix_pct,
    "position_size":   iv_signal,
    "monday_no_crush": monday_no_crush.astype(int),
    "macro_alert":     macro_alert,
})
iv_out.to_csv(DATA_DIR / "iv_filter.csv")
print(f"\nSaved {DATA_DIR / 'iv_filter.csv'}")

# ── Visualisation ─────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)

fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True)

# Panel 1: VIX with percentile threshold
ax = axes[0]
ax.plot(vix.index, vix.values, color="black", lw=1, label="VIX")
ax.plot(vix_pct.index, vix_pct.values, color="red", lw=1.5,
        linestyle="--", label=f"{IV_THRESHOLD}th percentile ({IV_LOOKBACK}d)")
is_high = vix >= vix_pct
ax.fill_between(vix.index, vix.values, vix_pct.values,
                where=is_high, color="salmon", alpha=0.4, label="High IV zone")
ax.fill_between(vix.index, vix.values, vix_pct.values,
                where=~is_high, color="lightgreen", alpha=0.2)
ax.set_title(f"VIX — Market Implied Volatility with {IV_THRESHOLD}th Percentile Filter",
             fontsize=11, fontweight="bold")
ax.set_ylabel("VIX")
ax.legend(fontsize=8)

# Panel 2: Position size signal
ax = axes[1]
ax.step(iv_signal.index, iv_signal.values, color="navy", lw=1.5, where="post")
ax.fill_between(iv_signal.index, iv_signal.values, IV_SIZE_NORM,
                where=(iv_signal < IV_SIZE_NORM),
                color="salmon", alpha=0.5, step="post", label="Reduced size")
ax.fill_between(iv_signal.index, iv_signal.values, IV_SIZE_NORM,
                where=(iv_signal >= IV_SIZE_NORM),
                color="lightgreen", alpha=0.3, step="post", label="Full size")
ax.set_ylim(0, 1.2)
ax.set_yticks([IV_SIZE_HIGH, IV_SIZE_NORM])
ax.set_yticklabels([f"Reduced ({IV_SIZE_HIGH}x)", f"Full ({IV_SIZE_NORM}x)"])
ax.set_title("Position Size Signal", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)

# Panel 3: Current ATM IV per pair (bar chart, last date only)
ax = axes[2]
if pair_iv_now:
    pair_names = [str(k) for k in pair_iv_now.keys()]   # ensure plain strings, not dates
    pair_ivs   = [v * 100 for v in pair_iv_now.values()]
    x_pos      = range(len(pair_names))
    colors     = ["salmon" if v > current_vix else "lightgreen" for v in pair_ivs]
    bars = ax.bar(x_pos, pair_ivs, color=colors, edgecolor="white")
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(pair_names)
    for bar, val in zip(bars, pair_ivs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.axhline(current_vix, color="red", lw=1.5, linestyle="--",
               label=f"VIX = {current_vix:.1f}")
    ax.set_ylabel("ATM IV (%)")
    ax.set_title("Current ATM Implied Volatility per Pair (today's snapshot)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, "No option data available", transform=ax.transAxes,
            ha="center", va="center", fontsize=12, color="gray")
    ax.set_title("Current ATM IV — no data", fontsize=11)

# Panel 4: Term structure (VIX9D/VIX ratio) + macro_alert
ax = axes[3]
valid_ratio = term_ratio.dropna()
ax.plot(valid_ratio.index, valid_ratio.values, color="purple", lw=1.2, label="VIX9D / VIX")
ax.axhline(1.0, color="red", lw=1.5, linestyle="--", label="Backwardation threshold (1.0)")
ax.fill_between(valid_ratio.index, valid_ratio.values, 1.0,
                where=(valid_ratio > 1.0), color="salmon", alpha=0.5, label="Backwardation (panic)")
ax.fill_between(valid_ratio.index, valid_ratio.values, 1.0,
                where=(valid_ratio <= 1.0), color="lightblue", alpha=0.2)

monday_alerts = macro_alert[monday_no_crush & (macro_alert == 1)]
if not monday_alerts.empty:
    ax.scatter(monday_alerts.index,
               term_ratio.reindex(monday_alerts.index),
               color="orange", zorder=5, s=40, label="Monday no-crush")

ax.set_ylabel("VIX9D / VIX")
ax.set_title("Term Structure (VIX9D/VIX) — Backwardation = Macro Alert",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8)

plt.suptitle("Implied Volatility Filter (Layer 3)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "iv_filter.png", dpi=150)
print(f"Chart saved to {OUTPUT_DIR / 'iv_filter.png'}")
# plt.show()
