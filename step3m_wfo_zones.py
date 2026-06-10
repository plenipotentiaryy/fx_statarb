"""
step3m_wfo_zones.py — Out-of-sample validation of the volume-zone gate.

Pipeline position: step 3m — run after step3j_wfo, before flipping USE_VOLUME_ZONES.

QUESTION IT ANSWERS
-------------------
Does gating entries by volume behaviour at the extreme add *out-of-sample* value,
or is it just train-set overfit? For every walk-forward window and pair we:

  1. Build train/test signals (reusing step3j.build_signals — identical Z/rvol logic).
  2. For each zone mode — OFF (0), REJECT (1), ACCEPT (2) — pick the best
     (entry_z, exit_z, stop_z) by Sharpe on TRAIN only.
  3. Trade exactly those params on the next, unseen TEST slice and record OOS Sharpe.

Comparing OOS Sharpe of OFF vs REJECT vs ACCEPT across all windows tells us whether
the zone gate genuinely helps once the parameters are frozen — the only test that
matters. Volume is sourced from CME futures if present, else Dukascopy tick volume.

OUTPUT
------
  data/wfo_zone_validation.csv   per-(window,pair,mode) train/OOS Sharpe + trades
  console verdict                aggregate OOS Sharpe uplift of zones vs OFF
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import (
    DATA_DIR, ZONE_RVOL_SPLIT, USE_RETURN_SPREAD, RETURN_WINDOW, USE_VWZ,
    RVOL_WINDOW, VELOCITY_WINDOW, SIGNAL_START, RTH_END,
)
from utils import fast_read
from volume_profile import pick_volume_source
from step3j_wfo import (
    load_closes, make_windows, run_grid, COMBOS,
    _combined_volume, _volume_weighted_zscore,
)

MODES = {0: "OFF", 1: "REJECT", 2: "ACCEPT"}
MIN_TRAIN_BARS = 200
MIN_TEST_BARS  = 50


def _build_sig(closes: pd.DataFrame, vol_df: pd.DataFrame | None,
               t1: str, t2: str, beta: float) -> pd.DataFrame:
    """Minimal signal frame for run_grid (mirrors step3j.build_signals' return-spread
    + VW-Z logic, but without the fragile all-NaN spread_vwap_mtf column that the
    production dropna() trips on when USE_VWAP_MTF is off). run_grid supplies a NaN
    spread_vwap_mtf itself, so omitting it is safe."""
    p1, p2 = closes[t1], closes[t2]
    vol = _combined_volume(vol_df, closes.index, t1, t2)

    if USE_RETURN_SPREAD:
        cum_r1 = p1.pct_change().fillna(0).rolling(RETURN_WINDOW).sum()
        cum_r2 = p2.pct_change().fillna(0).rolling(RETURN_WINDOW).sum()
        spread = (cum_r1 - beta * cum_r2).rename("spread")
        window = RETURN_WINDOW * 3
    else:
        spread = (p1 - beta * p2).rename("spread")
        window = 60

    if USE_VWZ and vol is not None:
        zscore, _, _ = _volume_weighted_zscore(spread, vol, window)
    else:
        m = spread.rolling(window).mean()
        zscore = (spread - m) / (spread - m).rolling(window).std().replace(0, np.nan)

    velocity = zscore.diff(VELOCITY_WINDOW)
    if vol is not None:
        rvol = vol / vol.rolling(RVOL_WINDOW, min_periods=RVOL_WINDOW).mean().replace(0, np.nan)
    else:
        rvol = pd.Series(1.0, index=closes.index)

    df = pd.DataFrame({
        f"{t1}_close": p1, f"{t2}_close": p2,
        "spread": spread, "zscore": zscore,
        "velocity": velocity, "rvol": rvol,
    }).dropna(subset=["zscore"]).between_time(SIGNAL_START, RTH_END)
    return df


def _ols_beta(closes: pd.DataFrame, t1: str, t2: str) -> float:
    """Simple OLS hedge ratio on log levels over the train slice."""
    a = np.log(closes[t1].to_numpy())
    b = np.log(closes[t2].to_numpy())
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return 1.0
    var = np.var(b[m])
    if var <= 0:
        return 1.0
    return float(np.cov(a[m], b[m])[0, 1] / var)


def validate() -> None:
    pairs = fast_read(DATA_DIR / "pairs_selected.csv", prefer_parquet=False,
                      fast_bars=0, index_col=None, parse_dates=False,
                      log_label="pairs_selected.csv")
    if pairs.empty:
        raise SystemExit("pairs_selected.csv is empty — run pairs.py first")

    needed = sorted({t for p in pairs["pair"] for t in p.split("-")})
    closes, _daily, _vol, _vwaps = load_closes(needed)

    # Prefer REAL futures volume; fall back to tick volume.
    vol_df, src = pick_volume_source(needed)
    if vol_df is not None:
        vol_df = vol_df.reindex(closes.index).fillna(0.0)
    windows = make_windows(closes)

    print("=" * 80)
    print(f"WFO ZONE VALIDATION   volume source: {src.upper()}   "
          f"rvol_split={ZONE_RVOL_SPLIT}")
    print(f"windows: {len(windows)}   pairs: {len(pairs)}")
    print("=" * 80)

    rows = []
    for w_i, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        c_tr = closes.loc[str(tr_s):str(tr_e)]
        c_te = closes.loc[str(te_s):str(te_e)]
        v_tr = vol_df.loc[str(tr_s):str(tr_e)] if vol_df is not None else None
        v_te = vol_df.loc[str(te_s):str(te_e)] if vol_df is not None else None
        if len(c_tr) < MIN_TRAIN_BARS or len(c_te) < MIN_TEST_BARS:
            continue
        tr_days = (tr_e - tr_s).days
        te_days = (te_e - te_s).days

        for _, prow in pairs.iterrows():
            pair = prow["pair"]
            t1, t2 = pair.split("-")
            if t1 not in c_tr.columns or t2 not in c_tr.columns:
                continue
            beta = _ols_beta(c_tr, t1, t2)

            sig_tr = _build_sig(c_tr, v_tr, t1, t2, beta)
            sig_te = _build_sig(c_te, v_te, t1, t2, beta)
            if len(sig_tr) < MIN_TRAIN_BARS or len(sig_te) < MIN_TEST_BARS:
                continue

            for mode, name in MODES.items():
                # Pick best params on TRAIN under this zone mode …
                best = run_grid(sig_tr, t1, t2, beta, COMBOS, tr_days,
                                zone_mode=mode, rvol_split=ZONE_RVOL_SPLIT)
                if best is None:
                    continue
                # … then trade exactly those params on the unseen TEST slice.
                oos = run_grid(sig_te, t1, t2, beta,
                               [(best["entry_z"], best["exit_z"], best["stop_z"])],
                               te_days, min_trades=1,
                               zone_mode=mode, rvol_split=ZONE_RVOL_SPLIT)
                rows.append({
                    "window": w_i, "pair": pair, "mode": name,
                    "entry_z": best["entry_z"], "exit_z": best["exit_z"],
                    "stop_z": best["stop_z"],
                    "train_sharpe": best["sharpe"], "train_trades": best["trades"],
                    "oos_sharpe": oos["sharpe"] if oos else np.nan,
                    "oos_trades": oos["trades"] if oos else 0,
                    "oos_pnl": oos["total_pnl"] if oos else np.nan,
                })

    if not rows:
        print("No valid (window, pair) results — check data coverage.")
        return

    df = pd.DataFrame(rows)
    out_csv = DATA_DIR / "wfo_zone_validation.csv"
    df.to_csv(out_csv, index=False)

    # ── Verdict: aggregate OOS Sharpe by mode ─────────────────────────────────
    print(f"\n{'mode':<8}{'n':>5}{'OOS Sharpe (median)':>22}{'OOS Sharpe (mean)':>20}"
          f"{'OOS trades':>12}")
    print("-" * 67)
    stats = {}
    for name in ("OFF", "REJECT", "ACCEPT"):
        sub = df[df["mode"] == name].dropna(subset=["oos_sharpe"])
        if sub.empty:
            continue
        med = sub["oos_sharpe"].median()
        mean = sub["oos_sharpe"].mean()
        stats[name] = med
        print(f"{name:<8}{len(sub):>5}{med:>22.3f}{mean:>20.3f}{sub['oos_trades'].sum():>12}")

    print("-" * 67)
    if "OFF" in stats:
        best_zone = max((m for m in ("REJECT", "ACCEPT") if m in stats),
                        key=lambda m: stats[m], default=None)
        if best_zone:
            uplift = stats[best_zone] - stats["OFF"]
            verdict = ("ADDS OOS value" if uplift > 0.05
                       else "NO meaningful OOS edge" if abs(uplift) <= 0.05
                       else "HURTS OOS")
            print(f"Best zone mode: {best_zone}  "
                  f"(median OOS Sharpe {stats[best_zone]:+.3f} vs OFF {stats['OFF']:+.3f}, "
                  f"Δ={uplift:+.3f}) → {verdict}")
            print(f"  → if it adds value, set ZONE_MODE='{best_zone}', USE_VOLUME_ZONES=True")
    # Per-pair winning mode (paired comparison removes window/pair noise)
    piv = df.pivot_table(index=["window", "pair"], columns="mode",
                         values="oos_sharpe", aggfunc="first")
    if {"OFF", "REJECT", "ACCEPT"}.issubset(piv.columns):
        piv = piv.dropna()
        if len(piv):
            win = piv[["OFF", "REJECT", "ACCEPT"]].idxmax(axis=1).value_counts()
            print(f"Per-(window,pair) winner counts: {win.to_dict()}  (n={len(piv)})")
    print(f"\nSaved {out_csv}  ({len(df)} rows)")


if __name__ == "__main__":
    validate()
