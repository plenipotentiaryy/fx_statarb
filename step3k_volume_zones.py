"""
step3k_volume_zones.py — Volume-zone (HVN/LVN) profiler for the FX spread.

Pipeline position: step 3k — run after pairs are selected, before the backtest.

WHAT IT DOES
------------
For every selected pair it:
  1. Builds the spread Z-score (return-spread + VW-Z, mirroring step4a.build_signals).
  2. Builds a Volume-at-Z profile from the chosen volume source
     (CME futures volume if available, else Dukascopy tick volume) → POC / VA / HVN / LVN.
  3. Runs the mean-reversion "race" (TP at Z=0 vs stop) SEPARATELY for entries whose
     trigger Z sits in a High-Volume Node vs a Low-Volume Node — so you can SEE which
     hypothesis carries the edge and pick ZONE_MODE:
        HVN edge → fade acceptance (range);  LVN edge → fade rejection (snapback).

OUTPUT
------
  data/volume_zones.csv         per-pair POC/VA/HVN-LVN win-rates + recommended mode
  output/volume_zones/*.png     Volume-at-Z histogram with POC/VA/HVN/LVN shaded
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from config import (
    CLOSES_FILE, DATA_DIR, OUTPUT_DIR, RTH_START, RTH_END, SIGNAL_START,
    KALMAN_DELTA, TRAIN_RATIO, RETURN_WINDOW, USE_RETURN_SPREAD,
    ENTRY_Z, TARGET_RISK_USD,
)
from utils import fast_read
from volume_profile import build_profile, pick_volume_source

# ── Profile parameters (overridable via config if present) ────────────────────
from config import (
    VOLUME_ZONE_BINS as N_BINS,
    VOLUME_ZONE_VA_PCT as VA_PCT,
    VOLUME_ZONE_HVN_Q as HVN_Q,
    VOLUME_ZONE_LVN_Q as LVN_Q,
)

TARGET_RR  = 1.3
MIN_BUCKET = 20     # min triggers in a zone bucket to trust its win-rate


# ── Spread Z construction (mirrors step4a.build_signals return-spread mode) ────

def _kalman_beta(p1: np.ndarray, p2: np.ndarray, delta: float, beta_init: float):
    n = len(p1)
    beta = np.empty(n)
    Q = delta / (1.0 - delta)
    R_var, P = 1.0, np.eye(2)
    state = np.array([0.0, beta_init])
    for i in range(n):
        F = np.array([1.0, p2[i]])
        P_pred = P + Q * np.eye(2)
        err = p1[i] - F @ state
        S = float(F @ P_pred @ F) + R_var
        K = P_pred @ F / S
        state = state + K * err
        P = P_pred - np.outer(K, F) @ P_pred
        beta[i] = state[1]
    return beta


def build_zscore_and_vol(closes: pd.DataFrame, vol_df: pd.DataFrame | None,
                         t1: str, t2: str, beta_init: float
                         ) -> tuple[pd.Series, pd.Series] | None:
    if t1 not in closes.columns or t2 not in closes.columns or len(closes) < 500:
        return None
    p1, p2 = closes[t1], closes[t2]

    if vol_df is not None and t1 in vol_df.columns and t2 in vol_df.columns:
        vol = (vol_df[t1].reindex(closes.index).fillna(0.0)
               + vol_df[t2].reindex(closes.index).fillna(0.0)).clip(lower=1.0)
    else:
        vol = pd.Series(1.0, index=closes.index)

    if USE_RETURN_SPREAD:
        r1 = p1.pct_change().fillna(0)
        r2 = p2.pct_change().fillna(0)
        cum_r1 = r1.rolling(RETURN_WINDOW).sum()
        cum_r2 = r2.rolling(RETURN_WINDOW).sum()
        beta_arr = _kalman_beta(cum_r1.fillna(0).values, cum_r2.fillna(0).values,
                                KALMAN_DELTA, beta_init)
        beta_s = pd.Series(beta_arr, index=closes.index)
        spread = cum_r1 - beta_s * cum_r2
        window = RETURN_WINDOW * 3
    else:
        beta_arr = _kalman_beta(p1.values, p2.values, KALMAN_DELTA, beta_init)
        spread = p1 - pd.Series(beta_arr, index=closes.index) * p2
        window = 60

    # Volume-weighted Z (mirrors _volume_weighted_zscore)
    w_sum = vol.rolling(window, min_periods=window).sum()
    mean = (spread * vol).rolling(window, min_periods=window).sum() / w_sum
    var = (((spread - mean) ** 2) * vol).rolling(window, min_periods=window).sum() / w_sum
    z = ((spread - mean) / var.pow(0.5).replace(0, np.nan))
    z = z.dropna().between_time(SIGNAL_START, RTH_END)
    vol = vol.reindex(z.index)
    return z, vol


# ── Zone-bucketed mean-reversion race ─────────────────────────────────────────

def trigger_analysis(z: pd.Series, vol: pd.Series, profile, entry_z: float,
                     target_rr: float = TARGET_RR,
                     rvol_window: int = 200) -> dict:
    """Analyse mean-reversion entries triggered at |Z| >= entry_z.

    For a mean-reverting spread the entry extreme almost always lands OUTSIDE the
    high-volume core, so the static HVN/LVN label of the extreme is not the edge.
    What IS actionable is whether volume is BUILDING at the extreme (acceptance →
    breakout risk, fade fails) or DRYING UP (rejection → snapback, fade works).
    We bucket triggers by local relative volume (rvol) and compare reversion EV,
    and we also report the raw zone distribution so the inertness is visible.

    Returns: zone_dist, baseline EV, and EV for low-rvol vs high-rvol entries.
    """
    zv = z.to_numpy(dtype="float64")
    rvol = (vol / vol.rolling(rvol_window, min_periods=rvol_window).mean()).to_numpy()
    stop_z = entry_z * (1.0 + 1.0 / target_rr)

    zone_dist: dict[str, int] = {}
    recs: list[tuple[float, bool]] = []   # (rvol_at_entry, won)

    def _race(idx_list, long: bool):
        for idx in idx_list:
            zone = profile.zone_of(zv[idx])
            zone_dist[zone] = zone_dist.get(zone, 0) + 1
            future = zv[idx:]
            if long:
                wi = np.where(future >= 0.0)[0]
                li = np.where(future <= -stop_z)[0]
            else:
                wi = np.where(future <= 0.0)[0]
                li = np.where(future >= stop_z)[0]
            fw = wi[0] if wi.size else np.inf
            fl = li[0] if li.size else np.inf
            rv = rvol[idx]
            if fw < fl:
                recs.append((rv, True))
            elif fl < fw:
                recs.append((rv, False))

    cross_short = np.where((zv[:-1] < entry_z) & (zv[1:] >= entry_z))[0] + 1
    cross_long  = np.where((zv[:-1] > -entry_z) & (zv[1:] <= -entry_z))[0] + 1
    _race(cross_short, long=False)
    _race(cross_long, long=True)

    def _ev(rows):
        if not rows:
            return np.nan, 0
        wr = sum(w for _, w in rows) / len(rows)
        return wr * target_rr - (1 - wr), len(rows)

    valid = [(rv, w) for rv, w in recs if np.isfinite(rv)]
    base_ev, base_n = _ev([(0, w) for _, w in recs])
    if valid:
        med = float(np.median([rv for rv, _ in valid]))
        low_ev,  low_n  = _ev([(rv, w) for rv, w in valid if rv <= med])  # volume drying
        high_ev, high_n = _ev([(rv, w) for rv, w in valid if rv > med])   # volume building
    else:
        med = np.nan; low_ev = high_ev = np.nan; low_n = high_n = 0

    return {
        "zone_dist": zone_dist,
        "base_ev": base_ev, "base_n": base_n,
        "low_rvol_ev": low_ev, "low_rvol_n": low_n,    # REJECT bucket (empty/drying)
        "high_rvol_ev": high_ev, "high_rvol_n": high_n,  # ACCEPT bucket (churn/building)
        "rvol_median": med,
    }


# ── Plot ──────────────────────────────────────────────────────────────────────

def _plot(pair_name: str, profile, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(profile.centers, profile.vol, height=(profile.edges[1] - profile.edges[0]) * 0.9,
            color="#3498db", alpha=0.7)
    ax.axhline(profile.poc, color="#e67e22", lw=2, label=f"POC {profile.poc:+.2f}")
    ax.axhspan(profile.va_low, profile.va_high, color="#2ecc71", alpha=0.12,
               label=f"Value Area [{profile.va_low:+.2f},{profile.va_high:+.2f}]")
    hvn = profile.centers[profile.vol >= profile.hvn_thresh]
    lvn = profile.centers[profile.vol <= profile.lvn_thresh]
    for h in hvn:
        ax.axhline(h, color="#c0392b", lw=0.4, alpha=0.5)
    ax.scatter(np.zeros_like(lvn), lvn, color="black", s=8, marker="x",
               label="LVN nodes", zorder=5)
    ax.set_xlabel("Volume at Z"); ax.set_ylabel("Spread Z-score")
    ax.set_title(f"Volume-at-Z Profile — {pair_name}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(out_dir / f"vzones_{pair_name.replace('-', '_')}.png", dpi=130)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def _load_closes() -> pd.DataFrame:
    path = DATA_DIR / CLOSES_FILE
    closes = fast_read(path, log_label=path.name)
    closes.index = pd.to_datetime(closes.index, utc=True)
    return closes.between_time(RTH_START, RTH_END)


def main() -> None:
    closes = _load_closes()
    pairs = fast_read(DATA_DIR / "pairs_selected.csv", prefer_parquet=False,
                      fast_bars=0, index_col=None, parse_dates=False,
                      log_label="pairs_selected.csv")
    if pairs.empty:
        raise SystemExit("pairs_selected.csv is empty — run pairs.py first")

    n = len(closes)
    closes_train = closes.iloc[:int(n * TRAIN_RATIO)]

    needed = sorted({t for p in pairs["pair"] for t in p.split("-")})
    vol_df, src = pick_volume_source(needed, start=str(closes_train.index[0].date()))
    print("=" * 78)
    print(f"VOLUME-ZONE PROFILER   volume source: {src.upper()}")
    print(f"  ({'REAL CME futures volume' if src == 'cme_futures' else 'tick-volume proxy' if src == 'tick' else 'NO VOLUME — flat weights'})")
    print("=" * 78)
    print(f"TRAIN: {closes_train.index[0].date()} → {closes_train.index[-1].date()}")
    print(f"{'Pair':<14}{'POC':>7}{'magnet':>8}{'peak':>7}"
          f"{'REJECT_EV':>10}{'ACCEPT_EV':>10}  {'MODE':>7}")
    print("  (magnet = % volume in value area;  peak = POC dominance;  "
          "REJECT=low-vol extreme, ACCEPT=high-vol extreme)")
    print("-" * 78)

    out_dir = OUTPUT_DIR / "volume_zones"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for _, row in pairs.iterrows():
        pair_name = row["pair"]
        t1, t2 = pair_name.split("-")
        beta = float(row.get("beta_daily", row.get("beta", 1.0)) or 1.0)

        built = build_zscore_and_vol(closes_train, vol_df, t1, t2, beta)
        if built is None:
            print(f"  SKIP {pair_name}: insufficient data")
            continue
        z, vol = built

        profile = build_profile(z, vol, n_bins=N_BINS, va_pct=VA_PCT,
                                hvn_q=HVN_Q, lvn_q=LVN_Q)
        if profile is None:
            print(f"  SKIP {pair_name}: no profile")
            continue

        magnet = profile.va_volume_share
        peak   = profile.poc_dominance
        ta     = trigger_analysis(z, vol, profile, entry_z=ENTRY_Z)
        rej_ev, acc_ev = ta["low_rvol_ev"], ta["high_rvol_ev"]

        # Per-bar mode: prefer the volume bucket with the higher reversion EV,
        # only if both buckets carry enough samples and the edge is positive.
        mode = "NONE"
        if (ta["low_rvol_n"] >= MIN_BUCKET and ta["high_rvol_n"] >= MIN_BUCKET
                and (np.isfinite(rej_ev) or np.isfinite(acc_ev))):
            if (rej_ev if np.isfinite(rej_ev) else -9) >= (acc_ev if np.isfinite(acc_ev) else -9):
                mode = "REJECT" if rej_ev > 0 else "NONE"
            else:
                mode = "ACCEPT" if acc_ev > 0 else "NONE"

        rows.append({
            "pair": pair_name, "volume_source": src,
            "poc": round(profile.poc, 3),
            "va_low": round(profile.va_low, 3), "va_high": round(profile.va_high, 3),
            "magnet_va_share": round(magnet, 4),
            "poc_dominance": round(peak, 2),
            "base_ev": round(ta["base_ev"], 4) if np.isfinite(ta["base_ev"]) else np.nan,
            "reject_ev": round(rej_ev, 4) if np.isfinite(rej_ev) else np.nan,
            "reject_n": ta["low_rvol_n"],
            "accept_ev": round(acc_ev, 4) if np.isfinite(acc_ev) else np.nan,
            "accept_n": ta["high_rvol_n"],
            "trigger_zone_dist": str(ta["zone_dist"]),
            "recommended_mode": mode,
        })
        _plot(pair_name, profile, out_dir)
        _f = lambda x: x if np.isfinite(x) else float("nan")  # noqa: E731
        print(f"{pair_name:<14}{profile.poc:>7.2f}{magnet:>8.1%}{peak:>7.1f}"
              f"{_f(rej_ev):>10.3f}{_f(acc_ev):>10.3f}  {mode:>7}")

    if rows:
        df = pd.DataFrame(rows)
        out_csv = DATA_DIR / "volume_zones.csv"
        df.to_csv(out_csv, index=False)
        print("-" * 78)
        print("Trigger-zone distribution (first pair):",
              rows[0]["trigger_zone_dist"])
        print("→ NOTE: entry extremes land mostly in NEUTRAL; HVN sits at the mean")
        print("        (the reversion TARGET), LVN at the far tails (stop region).")
        print(f"Saved {out_csv}  ({len(df)} pairs)  +  plots in {out_dir}/")
        strong = df[df["magnet_va_share"] >= 0.70]["pair"].tolist()
        if strong:
            print(f"Strong-magnet pairs (≥70% vol in VA): {strong}")
        agg = df[df["recommended_mode"] != "NONE"]["recommended_mode"].value_counts()
        if not agg.empty:
            print(f"Per-bar lean: {agg.to_dict()}  → set ZONE_MODE accordingly")
    else:
        print("No pairs profiled.")


if __name__ == "__main__":
    main()
