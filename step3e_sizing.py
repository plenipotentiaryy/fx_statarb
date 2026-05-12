import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from config import (
    REGIME_MULT_NORMAL, REGIME_MULT_VOLATILE, HMM_PANIC_MULT,
    IV_MULT_MAX, IV_MULT_MIN, IV_LOOKBACK,
    MIN_POSITION_SIZE,
    DATA_DIR, OUTPUT_DIR,
)

# ── Load all sources ──────────────────────────────────────────────────────────

def load_regimes() -> pd.DataFrame | None:
    path = DATA_DIR / "regimes.csv"
    if not path.exists():
        print("  regimes.csv not found — regime_mult = 1.0 for all bars")
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    return df   # wide: timestamp × pair, values 0/1


def load_iv() -> tuple[pd.Series | None, pd.Series | None]:
    path = DATA_DIR / "iv_filter.csv"
    if not path.exists():
        print("  iv_filter.csv not found — iv_mult = 1.0 for all days")
        return None, None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    vix = df["vix"].dropna()
    macro_alert = df["macro_alert"].reindex(df.index).fillna(0).astype(int) \
        if "macro_alert" in df.columns else pd.Series(0, index=df.index)
    return vix, macro_alert


def load_global_hmm() -> pd.Series | None:
    path = DATA_DIR / "global_hmm_regime.csv"
    if not path.exists():
        print("  global_hmm_regime.csv not found — global HMM panic mult = 1.0")
        return None
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    s.index = pd.to_datetime(s.index, utc=True)
    return s.rename("global_hmm")


def load_mc_confidence() -> dict[str, float]:
    path = DATA_DIR / "ou_montecarlo_summary.csv"
    if not path.exists():
        print("  ou_montecarlo_summary.csv not found — mc_conf = 1.0 for all pairs")
        return {}
    df = pd.read_csv(path)
    return {row["pair"]: row["win_rate"] / 100 for _, row in df.iterrows()}


# ── Compute multipliers ───────────────────────────────────────────────────────

def regime_multiplier(regimes: pd.DataFrame | None, pair: str, ts) -> float:
    if regimes is None or pair not in regimes.columns:
        return REGIME_MULT_NORMAL
    try:
        val = regimes.loc[ts, pair] if ts in regimes.index else 0
        return REGIME_MULT_VOLATILE if val == 1 else REGIME_MULT_NORMAL
    except Exception:
        return REGIME_MULT_NORMAL


def iv_multiplier_series(vix: pd.Series | None) -> pd.Series:
    """Returns daily iv_mult series (0.5 – 1.0)."""
    if vix is None:
        return pd.Series(dtype=float)
    pct_rank = vix.rolling(IV_LOOKBACK).rank(pct=True)
    iv_mult  = IV_MULT_MAX - (IV_MULT_MAX - IV_MULT_MIN) * pct_rank
    return iv_mult.fillna(IV_MULT_MAX)


def iv_multiplier(iv_mult_series: pd.Series, date) -> float:
    if iv_mult_series.empty:
        return IV_MULT_MAX
    key = pd.Timestamp(date).normalize().tz_localize(None)
    try:
        return float(iv_mult_series.asof(key))
    except Exception:
        return IV_MULT_MAX


def global_hmm_multiplier(global_hmm: pd.Series | None, date) -> float:
    if global_hmm is None or global_hmm.empty:
        return 1.0
    key = pd.Timestamp(date).normalize().tz_localize("UTC")
    try:
        val = global_hmm.asof(key)
        return HMM_PANIC_MULT if (not pd.isna(val) and int(val) == 1) else 1.0
    except Exception:
        return 1.0


def macro_alert_active(macro_alert: pd.Series | None, date) -> bool:
    if macro_alert is None or macro_alert.empty:
        return False
    key = pd.Timestamp(date).normalize().tz_localize("UTC")
    try:
        return bool(macro_alert.asof(key))
    except Exception:
        return False


def mc_confidence(mc_conf: dict, pair: str) -> float:
    return mc_conf.get(pair, 1.0)


def position_size(pair: str, ts, regimes, iv_mult_s, mc_conf: dict,
                  macro_alert_s: pd.Series | None = None,
                  global_hmm_s: pd.Series | None = None) -> float:
    date = ts.date() if hasattr(ts, "date") else ts
    if macro_alert_active(macro_alert_s, date):
        return 0.0   # block new entries during macro panic (VIX9D backwardation)
    r = regime_multiplier(regimes, pair, ts)      # per-pair HMM (0.3 in volatile)
    g = global_hmm_multiplier(global_hmm_s, date) # global SPY HMM (0.333 in panic)
    i = iv_multiplier(iv_mult_s, date)            # VIX percentile (0.5–1.0)
    m = mc_confidence(mc_conf, pair)              # OU Monte Carlo win rate
    return r * g * i * m


# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data sources...\n")

regimes        = load_regimes()
vix, macro_alert_s = load_iv()
global_hmm_s   = load_global_hmm()
mc_conf        = load_mc_confidence()
iv_mult_s      = iv_multiplier_series(vix)

pairs = pd.read_csv(DATA_DIR / "pairs_selected.csv")
pair_names = list(pairs["pair"])

# ── Build position size grid ──────────────────────────────────────────────────
# Use timestamps from regimes (if available), otherwise from VIX dates
if regimes is not None:
    timestamps = regimes.index
elif vix is not None:
    timestamps = pd.date_range(vix.index[0], vix.index[-1], freq="B", tz="UTC")
else:
    print("No time-series data available — cannot build sizing grid.")
    raise SystemExit(0)

sizes = {}
for pair in pair_names:
    col = []
    for ts in timestamps:
        col.append(position_size(pair, ts, regimes, iv_mult_s, mc_conf, macro_alert_s, global_hmm_s))
    sizes[pair] = col

df_sizes = pd.DataFrame(sizes, index=timestamps)

# ── Stats ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("POSITION SIZING SUMMARY")
print("=" * 60)

for pair in pair_names:
    if pair not in df_sizes.columns:
        continue
    s   = df_sizes[pair]
    mc  = mc_conf.get(pair, 1.0)
    pct_skipped = (s < MIN_POSITION_SIZE).mean() * 100

    print(f"\n  {pair}")
    print(f"    MC confidence:   {mc:.2f}  ({mc*100:.0f}% sim win rate)")
    print(f"    Size mean:       {s.mean():.2f}")
    print(f"    Size min / max:  {s.min():.2f} / {s.max():.2f}")
    print(f"    Trades skipped:  {pct_skipped:.0f}%  (size < {MIN_POSITION_SIZE})")

    # Decompose a single bar (latest)
    latest = timestamps[-1]
    r    = regime_multiplier(regimes, pair, latest)
    i    = iv_multiplier(iv_mult_s, latest.date() if hasattr(latest, "date") else latest)
    date_latest = latest.date() if hasattr(latest, "date") else latest
    alert = macro_alert_active(macro_alert_s, date_latest)
    g     = global_hmm_multiplier(global_hmm_s, date_latest)
    print(f"    Latest (current):")
    print(f"      regime_mult={r:.1f}  global_hmm={g:.3f}  iv_mult={i:.2f}  "
          f"mc_conf={mc:.2f}  macro_alert={'BLOCK' if alert else 'ok'}"
          f"  → size={0.0 if alert else r*g*i*mc:.2f}")

# ── Save ──────────────────────────────────────────────────────────────────────
df_sizes.to_csv(DATA_DIR / "position_sizes.csv")
print(f"\nSaved {DATA_DIR / 'position_sizes.csv'}  {df_sizes.shape}")

# ── Visualisation ─────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)

n_pairs = len(pair_names)
fig = plt.figure(figsize=(18, 4 + 3 * n_pairs))
gs  = gridspec.GridSpec(n_pairs + 2, 2, figure=fig, hspace=0.5, wspace=0.3)

# Top row: multiplier components
ax_reg = fig.add_subplot(gs[0, 0])
ax_iv  = fig.add_subplot(gs[0, 1])

# Regime multiplier over time (first pair as example)
if regimes is not None and pair_names:
    example = pair_names[0]
    if example in regimes.columns:
        reg_vals = regimes[example].map({0: REGIME_MULT_NORMAL, 1: REGIME_MULT_VOLATILE})
        ax_reg.step(reg_vals.index, reg_vals.values, color="steelblue", lw=1, where="post")
        ax_reg.fill_between(reg_vals.index, reg_vals.values,
                            REGIME_MULT_NORMAL, step="post",
                            where=(reg_vals.values < REGIME_MULT_NORMAL),
                            color="salmon", alpha=0.4)
        ax_reg.set_title(f"Regime Multiplier — {example}", fontsize=9, fontweight="bold")
        ax_reg.set_ylim(0, 1.2)
        ax_reg.set_ylabel("Multiplier")

# IV multiplier over time
if not iv_mult_s.empty:
    ax_iv.plot(iv_mult_s.index, iv_mult_s.values, color="darkorange", lw=1.5)
    ax_iv.fill_between(iv_mult_s.index, iv_mult_s.values, IV_MULT_MAX,
                       color="salmon", alpha=0.3)
    ax_iv.set_title("IV Multiplier (VIX-based, smooth)", fontsize=9, fontweight="bold")
    ax_iv.set_ylim(0, 1.1)
    ax_iv.set_ylabel("Multiplier")

# MC confidence bar chart
ax_mc = fig.add_subplot(gs[1, :])
if mc_conf:
    labels = list(mc_conf.keys())
    vals   = [mc_conf[p] for p in labels]
    colors = ["#4CAF50" if v >= 0.6 else "#FF9800" if v >= 0.5 else "#F44336"
              for v in vals]
    bars = ax_mc.bar(labels, vals, color=colors, edgecolor="white")
    ax_mc.axhline(MIN_POSITION_SIZE, color="red", lw=1.5, linestyle="--",
                  label=f"Min size threshold ({MIN_POSITION_SIZE})")
    ax_mc.axhline(1.0, color="gray", lw=0.8, linestyle="--")
    for bar, val in zip(bars, vals):
        ax_mc.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + 0.01, f"{val:.2f}",
                   ha="center", va="bottom", fontsize=9)
    ax_mc.set_title("MC Confidence per Pair (OU Monte Carlo win rate)", fontsize=9, fontweight="bold")
    ax_mc.set_ylabel("Win rate fraction")
    ax_mc.set_ylim(0, 1.15)
    ax_mc.legend(fontsize=8)

# Combined size per pair
colors_p = plt.cm.tab10(np.linspace(0, 1, n_pairs))
for idx, (pair, color) in enumerate(zip(pair_names, colors_p)):
    if pair not in df_sizes.columns:
        continue
    ax = fig.add_subplot(gs[2 + idx, :])
    s  = df_sizes[pair]

    ax.fill_between(s.index, s.values, 0,
                    where=(s.values >= MIN_POSITION_SIZE),
                    color=color, alpha=0.5, step="post")
    ax.fill_between(s.index, s.values, 0,
                    where=(s.values < MIN_POSITION_SIZE),
                    color="red", alpha=0.3, step="post", label="Skipped (too small)")
    ax.step(s.index, s.values, color=color, lw=1.2, where="post")
    ax.axhline(MIN_POSITION_SIZE, color="red", lw=1, linestyle="--")
    ax.axhline(1.0, color="gray", lw=0.5, linestyle="--")
    ax.set_title(f"{pair}  —  Combined Position Size", fontsize=9, fontweight="bold")
    ax.set_ylabel("Size")
    ax.set_ylim(0, 1.1)
    if (s < MIN_POSITION_SIZE).any():
        ax.legend(fontsize=7)

plt.suptitle("Dynamic Position Sizing — Layer 5\n"
             "size = regime_mult × iv_mult × mc_confidence",
             fontsize=12, fontweight="bold")
plt.savefig(OUTPUT_DIR / "position_sizing.png", dpi=150, bbox_inches="tight")
print(f"Chart saved to {OUTPUT_DIR / 'position_sizing.png'}")
# plt.show()
