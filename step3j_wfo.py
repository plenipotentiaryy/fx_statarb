"""
wfo.py — Walk-Forward Optimization (WFO) Engine.

Implements the Gatev, Goetzmann, Rouwenhorst (2006) baseline framework:
  - Formation (train) window:  12 months
  - Trading (OOS) window:       6 months
  - Step:                       6 months (non-overlapping OOS windows)

Workflow per window:
  1. Build signals on TRAIN slice (rolling zscore)
  2. Grid search (entry_z, exit_z, stop_z) → pick best by Sharpe
  3. Trade those params on the NEXT 6-month OOS slice (zero look-ahead)
  4. Accumulate OOS trades into a continuous equity curve

Output:
  data/wfo_results.csv      — OOS trades from all windows
  data/wfo_params.csv       — best params per pair per window
  output/wfo_equity.png     — continuous 18-year OOS equity curve
  output/wfo_stability.png  — train vs test Sharpe per window per pair

Pipeline position: step 3i (optional, after grid.py)
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import itertools
import numpy as np
import pandas as pd
import scipy.optimize as opt
from sklearn.ensemble import HistGradientBoostingClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm

from config import (
    TAIL_HEDGE_DRAG_ANNUAL, TAIL_HEDGE_PAYOUT_MULT, INITIAL_CAPITAL,
    COST_MAKER, COST_TAKER, CIRCUIT_BREAKER_Z,
    CLOSES_FILE, VOLUMES_FILE, BORROW_RATE_ANNUAL,
    RTH_START, RTH_END, SIGNAL_START,
    BARS_PER_DAY, DATA_DIR, OUTPUT_DIR,
    WFO_TRAIN_MONTHS, WFO_TEST_MONTHS, WFO_STEP_MONTHS, WFO_MIN_TRADES,
    WFO_EXPANDING, HMM_PANIC_MULT,
    HURST_ENTRY_WINDOW,
    USE_VWZ, VWZ_MIN_VOLUME, USE_VWAP,
    USE_RETURN_SPREAD, RETURN_WINDOW,
    USE_VELOCITY_GATE, VELOCITY_WINDOW,
    USE_RVOL_GATE, RVOL_THRESHOLD, RVOL_WINDOW,
    USE_VWAP_MTF, VWAP_MTF_TF, KALMAN_DELTA
)
from filters import HurstFilter, get_hurst_multiplier, validate_kde_density

# Grid definition (same as grid.py)
ENTRY_Z_GRID = [1.65, 1.7, 1.8, 2.0, 2.2]
STOP_Z_GRID  = [3.0, 3.2, 3.5]
EXIT_Z_GRID  = [-0.1, 0.0, 0.1]
COMBOS       = [(e, x, s) for e, x, s in
                itertools.product(ENTRY_Z_GRID, EXIT_Z_GRID, STOP_Z_GRID)
                if s > e and x < e]
N_COMBOS = len(COMBOS)

# Half-life gate: skip pairs whose OU mean-reversion is too slow.
# A 12m OOS window gives ~6 months to trade; a pair with HL > 90 days
# will barely complete one full cycle — it's a 'lazy' cointegration.
HL_MAX_DAYS = 90   # calendar days (will be converted to bars inside)


# ── Grid kernel (same logic as grid.py, no numba dependency) ─────────────────

import numba
from numba import njit

@njit
def _grid_kernel(zscore, spread, velocity, rvol, sv_mtf, t1_p, t2_p,
                 entry_arr, exit_arr, stop_arr,
                 beta, cost_taker, borrow_rate, bars_per_day,
                 use_vel_gate, use_rvol_gate, rvol_thresh, use_vwap_mtf):
    n_bars, n_c = len(zscore), len(entry_arr)
    pos   = np.zeros(n_c, dtype=numba.int64)
    e_sp  = np.zeros(n_c); e_bar = np.zeros(n_c, dtype=numba.int64)
    res   = np.zeros((n_c, 6)); cum = np.zeros(n_c); pk = np.zeros(n_c)

    for i in range(n_bars):
        z = zscore[i]; s = spread[i]; v = velocity[i]
        rv = rvol[i]; vwap = sv_mtf[i]
        
        for c in range(n_c):
            ez, xz, sz, pc = entry_arr[c], exit_arr[c], stop_arr[c], pos[c]
            
            if pc != 0:
                is_exit = (pc == 1 and z >= xz) or (pc == -1 and z <= -xz)
                is_stop = (pc == 1 and z <= -sz) or (pc == -1 and z >= sz)
                if is_exit or is_stop:
                    gross = pc * (s - e_sp[c])
                    tx    = (1.0 + beta) * cost_taker * 2.0
                    hd    = (i - e_bar[c]) / bars_per_day
                    brw   = (1.0 + beta) * (borrow_rate / 252.0) * hd
                    net   = gross - tx - brw
                    if np.isnan(net):
                        pos[c] = 0; continue
                    res[c, 0] += net; res[c, 1] += 1
                    if net > 0: res[c, 2] += 1
                    res[c, 3] += net * net
                    if is_stop: res[c, 5] += 1
                    cum[c] += net
                    if cum[c] > pk[c]: pk[c] = cum[c]
                    dd = cum[c] - pk[c]
                    if dd < res[c, 4]: res[c, 4] = dd
                    pos[c] = 0

            if pos[c] == 0:
                if use_rvol_gate and (np.isnan(rv) or rv < rvol_thresh): continue
                if z < -ez:
                    if use_vel_gate and (np.isnan(v) or v < 0): continue
                    if use_vwap_mtf and not np.isnan(vwap) and s < vwap: continue
                    pos[c] = 1; e_sp[c] = s; e_bar[c] = i
                elif z > ez:
                    if use_vel_gate and (np.isnan(v) or v > 0): continue
                    if use_vwap_mtf and not np.isnan(vwap) and s > vwap: continue
                    pos[c] = -1; e_sp[c] = s; e_bar[c] = i
    return res


def run_grid(df, t1, t2, beta, combos, days, min_trades=WFO_MIN_TRADES):
    """Run grid search. Returns best row (entry_z, exit_z, stop_z, sharpe) or None."""
    valid = [(e, x, s) for e, x, s in combos if s > e and x < e]
    if not valid or len(df) < 50:
        return None

    ea = np.array([c[0] for c in valid], dtype=np.float64)
    xa = np.array([c[1] for c in valid], dtype=np.float64)
    sa = np.array([c[2] for c in valid], dtype=np.float64)

    res = _grid_kernel(
        np.ascontiguousarray(df["zscore"].to_numpy(np.float64)),
        np.ascontiguousarray(df["spread"].to_numpy(np.float64)),
        np.ascontiguousarray(df["velocity"].to_numpy(np.float64)) if "velocity" in df.columns else np.zeros(len(df)),
        np.ascontiguousarray(df["rvol"].to_numpy(np.float64))     if "rvol" in df.columns     else np.ones(len(df)),
        np.ascontiguousarray(df["spread_vwap_mtf"].to_numpy(np.float64)) if "spread_vwap_mtf" in df.columns else np.full(len(df), np.nan),
        np.ascontiguousarray(df[f"{t1}_close"].to_numpy(np.float64)),
        np.ascontiguousarray(df[f"{t2}_close"].to_numpy(np.float64)),
        ea, xa, sa, float(beta),
        float(COST_TAKER), float(BORROW_RATE_ANNUAL), float(BARS_PER_DAY),
        bool(USE_VELOCITY_GATE), bool(USE_RVOL_GATE), float(RVOL_THRESHOLD), bool(USE_VWAP_MTF)
    )

    years = max(days / 365.25, 1e-9)
    best_sh, best_row = -np.inf, None
    for i, (ez, xz, sz) in enumerate(valid):
        n = int(res[i, 1])
        if n < min_trades:
            continue
        mean = res[i, 0] / n
        var  = max(res[i, 3] / n - mean**2, 0.0)
        std  = np.sqrt(var)
        tpy  = n / years
        sh   = mean / std * np.sqrt(tpy) if std > 0 else 0.0
        if sh > best_sh:
            best_sh  = sh
            best_row = {"entry_z": ez, "exit_z": xz, "stop_z": sz,
                        "sharpe": round(sh, 3), "trades": n,
                        "win_rate": round(res[i, 2] / n * 100, 1),
                        "total_pnl": round(res[i, 0], 4)}
    return best_row


def _combined_volume(volumes, index, t1, t2):
    if volumes is None or t1 not in volumes.columns or t2 not in volumes.columns:
        return None
    vol = volumes[t1].reindex(index).fillna(0.0) + volumes[t2].reindex(index).fillna(0.0)
    return vol.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=VWZ_MIN_VOLUME)


def _volume_weighted_zscore(spread, vol, window):
    w_sum = vol.rolling(window=window, min_periods=window).sum()
    mean = (spread * vol).rolling(window=window, min_periods=window).sum() / w_sum
    var = (((spread - mean) ** 2) * vol).rolling(window=window, min_periods=window).sum() / w_sum
    std = var.pow(0.5)
    zscore = (spread - mean) / std.replace(0, np.nan)
    return zscore, mean, std


def build_signals(closes, volumes, vwaps, t1, t2, beta, half_life):
    p1, p2 = closes[t1], closes[t2]
    vol = _combined_volume(volumes, closes.index, t1, t2)
    
    # 1. Return-spread mode (Standard for 1-min FX)
    if USE_RETURN_SPREAD:
        r1 = p1.pct_change().fillna(0)
        r2 = p2.pct_change().fillna(0)
        cum_r1 = r1.rolling(RETURN_WINDOW).sum()
        cum_r2 = r2.rolling(RETURN_WINDOW).sum()
        
        # Simple beta for WFO windows
        spread = (cum_r1 - beta * cum_r2).rename("spread")
        window = RETURN_WINDOW * 3
        if USE_VWZ and vol is not None:
            zscore, spread_mean, spread_std = _volume_weighted_zscore(spread, vol, window)
        else:
            spread_mean = spread.rolling(window).mean()
            spread_std  = (spread - spread_mean).rolling(window).std()
            zscore      = (spread - spread_mean) / spread_std.replace(0, np.nan)
    else:
        spread = p1 - beta * p2
        window = max(20, min(int(half_life), 200))
        
        if USE_VWZ and vol is not None:
            zscore, spread_mean, spread_std = _volume_weighted_zscore(spread, vol, window)
        else:
            spread_mean = spread.rolling(window).mean()
            spread_std = spread.rolling(window).std()
            zscore = (spread - spread_mean) / spread_std
            
    velocity = zscore.diff(VELOCITY_WINDOW)
    
    # 2. RVOL: Relative Volume
    if vol is not None:
        rvol = vol / vol.rolling(RVOL_WINDOW, min_periods=RVOL_WINDOW).mean().replace(0, np.nan)
    else:
        rvol = pd.Series(1.0, index=closes.index)

    # 3. MTF VWAP Anchor
    if USE_VWAP_MTF and vwaps is not None and t1 in vwaps.columns and t2 in vwaps.columns:
        p1_v = vwaps[t1].reindex(closes.index).fillna(closes[t1])
        p2_v = vwaps[t2].reindex(closes.index).fillna(closes[t2])
        spread_v = p1_v - beta * p2_v
        spread_vwap_mtf = spread_v.resample(VWAP_MTF_TF).mean().reindex(closes.index, method="ffill")
    else:
        spread_vwap_mtf = pd.Series(np.nan, index=closes.index)

    return pd.DataFrame({
        f"{t1}_close": p1,
        f"{t2}_close": p2,
        "spread":          spread,
        "spread_mean":     spread_mean,
        "spread_std":      spread_std,
        "zscore":          zscore,
        "velocity":        velocity,
        "rvol":            rvol,
        "spread_vwap_mtf": spread_vwap_mtf
    }).dropna().between_time(SIGNAL_START, RTH_END)

# ── Dynamic Cointegration ────────────────────────────────────────────────────

_JOH_CRIT_IDX = {0.90: 0, 0.95: 1, 0.99: 2}

def check_coint_johansen(df_daily, t1, t2, crit_level=0.95):
    """Run Johansen on daily slice; returns (is_coint, beta)."""
    pc = df_daily[[t1, t2]].dropna()
    if len(pc) < 100:
        return False, None
    try:
        res = coint_johansen(pc, det_order=0, k_ar_diff=1)
        trace = float(res.lr1[0])
        crit = float(res.cvt[0, _JOH_CRIT_IDX[crit_level]])
        if trace <= crit:
            return False, None
        evec = res.evec[:, 0]
        beta = -evec[1] / evec[0]
        return True, float(beta)
    except Exception:
        return False, None

def compute_half_life(spread_daily: pd.Series) -> float:
    aligned = pd.concat([spread_daily.diff(), spread_daily.shift(1)], axis=1).dropna()
    aligned.columns = ["diff", "lag"]
    try:
        theta = sm.OLS(aligned["diff"], sm.add_constant(aligned["lag"])).fit().params["lag"]
        return -np.log(2) / theta if theta < 0 else 200.0
    except Exception:
        return 200.0


def backtest_oos(df, t1, t2, beta, entry_z, exit_z, stop_z,
                 hmm_regime=None, hurst_filter=None, spread_daily=None,
                 limit_rebate=0.05, limit_ttl=3, session_window=None):
    """
    Tier-1 Backtest: 
    - Limit Order Simulation (TTL & Rebate)
    - VW-Z Signal Processing
    - HMM & Hurst Regime Filters
    """
    t1c, t2c = f"{t1}_close", f"{t2}_close"
    pos = es = et1 = et2 = entry_sma = entry_std = 0.0
    ebar = 0; trades = []; hmm_blocked = 0; hurst_blocked = 0
    pair_blocked = False
    
    # Limit Order State
    pending_pos = 0  # 1 for Long limit, -1 for Short limit
    pending_ttl = 0
    limit_z = 0.0
    pending_h_mult = 1.0
    current_h_mult = 1.0

    for i in range(len(df)):
        if pair_blocked: continue
        z = df["zscore"].iloc[i]; s = df["spread"].iloc[i]
        try:
            p1 = df[t1c].iloc[i];    p2 = df[t2c].iloc[i]
        except KeyError as e:
            print(f"KeyError in backtest_oos! df.columns: {df.columns.tolist()}, t1c: {t1c}, t2c: {t2c}")
            raise e
        vr = df["rvol"].iloc[i] if "rvol" in df.columns else 1.0
        vel = df["velocity"].iloc[i] if "velocity" in df.columns else 0.0
        sv_mtf = df["spread_vwap_mtf"].iloc[i] if "spread_vwap_mtf" in df.columns else np.nan
        
        z_active = z
        if pos != 0 and entry_std > 0:
            z_active = (s - entry_sma) / entry_std
            
        # ── Exit Logic ────────────────────────────────────────────────────────
        if pos != 0:
            ex = (pos == 1 and z_active >= exit_z)  or (pos == -1 and z_active <= -exit_z)
            st = (pos == 1 and z_active <= -stop_z) or (pos == -1 and z_active >= stop_z)
            cb = abs(z_active) >= CIRCUIT_BREAKER_Z
            if ex or st or cb:
                if cb:
                    pair_blocked = True
                    st = True # force taker cost
                gross = pos * (s - es)
                notl  = et1 + beta * et2
                tx    = notl * COST_MAKER + notl * (COST_MAKER if ex and not cb else COST_TAKER)
                hd    = (i - ebar) / BARS_PER_DAY
                brw   = (beta * et2 if pos == 1 else et1) * BORROW_RATE_ANNUAL * hd / 252
                # Apply Soft Hurst Multiplier
                final_pnl = (gross - tx - brw) * current_h_mult
                trades.append({
                    "entry_time": df.index[ebar],
                    "exit_time":  df.index[i],
                    "direction":  "LONG" if pos == 1 else "SHORT",
                    "net_pnl":    round(final_pnl, 4),
                    "exit_reason": "STOP" if st else "SIGNAL",
                    "holding_bars": i - ebar,
                    "gross_pnl":  round(gross, 4),
                    "tx_cost":    round(tx, 4),
                    "borrow_cost": round(brw, 4),
                    "features":   e_features if 'e_features' in locals() else [abs(z), vr, entry_std, df.index[i].hour + df.index[i].minute / 60.0]
                })
                pos = 0

        # ── Limit Order Matching ──────────────────────────────────────────────
        if pos == 0 and pending_pos != 0:
            pending_ttl -= 1
            is_filled = (pending_pos == -1 and z <= limit_z) or (pending_pos == 1 and z >= limit_z)
            
            if is_filled:
                pos = pending_pos
                current_h_mult = pending_h_mult
                pending_pos = 0
                es = s; et1 = p1; et2 = p2; ebar = i
                entry_sma = df["spread_mean"].iloc[i]
                entry_std = df["spread_std"].iloc[i]
                e_features = [abs(z), vr, entry_std, df.index[i].hour + df.index[i].minute / 60.0]
            elif pending_ttl <= 0:
                pending_pos = 0

        # ── Entry Logic (Signal Detection) ────────────────────────────────────
        if pos == 0 and pending_pos == 0:
            # 1. RVOL check
            if USE_RVOL_GATE and (np.isnan(vr) or vr < RVOL_THRESHOLD):
                continue
            # ── Macro HMM gate ──
            if hmm_regime is not None:
                d = df.index[i].normalize().tz_localize(None)
                try:
                    if int(hmm_regime.asof(d)) == 1:
                        hmm_blocked += 1; continue
                except: pass
            
            # ── Session Filter ──
            if session_window:
                h = df.index[i].hour
                start, end = session_window["best_start_utc"], session_window["best_end_utc"]
                is_in = (h >= start and h < end) if start < end else (h >= start or h < end)
                if not is_in:
                    continue

            detect_long = z < -entry_z
            detect_short = z > entry_z
            
            # 2. Velocity and VWAP Gates
            if detect_long:
                if USE_VELOCITY_GATE and (np.isnan(vel) or vel < 0): detect_long = False
                if USE_VWAP_MTF and not np.isnan(sv_mtf) and s < sv_mtf: detect_long = False
            if detect_short:
                if USE_VELOCITY_GATE and (np.isnan(vel) or vel > 0): detect_short = False
                if USE_VWAP_MTF and not np.isnan(sv_mtf) and s > sv_mtf: detect_short = False

            if detect_long or detect_short:
                h_val = 0.5
                if hurst_filter is not None and spread_daily is not None:
                    d_prev = (df.index[i] - pd.Timedelta(days=1)).normalize()
                    h_tail_daily = spread_daily.loc[:d_prev].tail(HURST_ENTRY_WINDOW - 1)
                    h_tail = pd.concat([h_tail_daily, pd.Series({df.index[i]: df["spread"].iloc[i]})])
                    h_blocked, h_val = hurst_filter.should_block(h_tail, df.index[i])
                    if h_blocked:
                        hurst_blocked += 1
                        continue
                
                pending_pos = 1 if detect_long else -1
                pending_ttl = limit_ttl
                limit_z = (-entry_z + limit_rebate) if detect_long else (entry_z - limit_rebate)
                pending_h_mult = get_hurst_multiplier(h_val)

    # Note: current_h_mult is used inside the loop at exit to scale 'final_pnl'
    return trades, hmm_blocked, hurst_blocked


def optimize_portfolio_weights(returns_df: pd.DataFrame, target_return: float = 0.10, max_weight: float = 0.20) -> dict:
    """
    V9: Mean-Variance Optimization with Target Return constraint (μ_target).
    returns_df: rows=dates, columns=pairs, values=daily PnL.
    """
    if returns_df.empty or returns_df.shape[1] == 0:
        return {}
    if returns_df.shape[1] == 1:
        return {returns_df.columns[0]: 1.0}
        
    mu = returns_df.mean().values
    Sigma = returns_df.cov().values
    n = len(mu)
    target_daily = target_return / 252.0
    
    def portfolio_variance(w, S):
        return np.dot(w.T, np.dot(S, w))
        
    init_w = np.full(n, 1.0 / n)
    bounds = tuple((0.0, max_weight) for _ in range(n))
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'ineq', 'fun': lambda w: np.sum(mu * w) - target_daily}
    ]
    
    res = opt.minimize(portfolio_variance, init_w, args=(Sigma,), method='SLSQP', bounds=bounds, constraints=constraints)
    if not res.success:
        # Fallback to Min-Variance if target unreachable
        res = opt.minimize(portfolio_variance, init_w, args=(Sigma,), method='SLSQP', bounds=bounds, 
                           constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}])
        
    weights = res.x
    weights[weights < 1e-4] = 0.0
    weights /= np.sum(weights)
    return {col: float(w) for col, w in zip(returns_df.columns, weights)}

def calculate_systemic_risk_scaler(returns_df: pd.DataFrame, threshold: float = 0.7) -> float:
    """
    Module 4: Tail Risk Guard (Eigenvalue Spiking).
    Scales down position size when systemic correlation spikes.
    """
    if returns_df.shape[1] < 2:
        return 1.0
    try:
        # Use correlation matrix for spectral analysis
        corr_matrix = returns_df.corr().fillna(0)
        eigenvalues = np.linalg.eigvals(corr_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        # Absorption Ratio = Top Eigenvalue / Total Trace
        absorption_ratio = eigenvalues[0] / np.sum(eigenvalues)
        if absorption_ratio > threshold:
            # Linear scale-down to 0.2
            scaler = max(0.2, 1.0 - (absorption_ratio - threshold) / (1.0 - threshold))
            return float(scaler)
        return 1.0
    except Exception:
        return 1.0

def _trades_to_daily_pnl(trades: list, dates: pd.DatetimeIndex) -> pd.Series:
    pnl = pd.Series(0.0, index=dates.normalize().unique())
    for t in trades:
        d = pd.Timestamp(t["exit_time"]).normalize()
        if d in pnl.index:
            pnl[d] += t["net_pnl"]
    return pnl



# ── Load data ─────────────────────────────────────────────────────────────────

def load_closes():
    path = DATA_DIR / CLOSES_FILE
    if not path.exists():
        path = DATA_DIR / "closes_15min.csv"
    closes = pd.read_csv(path, index_col=0)
    closes.index = pd.to_datetime(closes.index, utc=True)
    closes = closes.between_time(RTH_START, RTH_END)
    
    vol_path = DATA_DIR / VOLUMES_FILE
    volumes = None
    if vol_path.exists():
        volumes = pd.read_csv(vol_path, index_col=0)
        volumes.index = pd.to_datetime(volumes.index, utc=True)
        volumes = volumes.reindex(closes.index).fillna(0.0)

    daily_path = DATA_DIR / "closes_daily.csv"
    if daily_path.exists():
        daily = pd.read_csv(daily_path, index_col=0)
        daily.index = pd.to_datetime(daily.index, utc=True)
    else:
        daily = None

    vwap_path = DATA_DIR / VWAPS_FILE
    vwaps = None
    if vwap_path.exists():
        vwaps = pd.read_csv(vwap_path, index_col=0)
        vwaps.index = pd.to_datetime(vwaps.index, utc=True)
        vwaps = vwaps.reindex(closes.index).ffill()

    return closes, daily, volumes, vwaps


def make_windows(closes, expanding=WFO_EXPANDING):
    """
    Generate (train_start, train_end, test_start, test_end) tuples.

    expanding=True  (Expanding Window):
      train_start is ANCHORED to the first date in closes.
      Each subsequent window grows: 12m, 18m, 24m, …
      The algorithm accumulates memory of all past regimes.

    expanding=False (Rolling Window, classic Gatev):
      train_start slides forward by WFO_STEP_MONTHS each iteration.
      Fixed-width train window (12m).
    """
    anchor = closes.index[0].date()
    end    = closes.index[-1].date()
    windows = []
    cur = anchor
    while True:
        train_start = anchor if expanding else cur
        train_end   = cur + relativedelta(months=WFO_TRAIN_MONTHS)
        test_start  = train_end
        test_end    = test_start + relativedelta(months=WFO_TEST_MONTHS)
        if test_end > end:
            break
        windows.append((train_start, train_end, test_start, test_end))
        cur = cur + relativedelta(months=WFO_STEP_MONTHS)
    return windows


# ── Main ──────────────────────────────────────────────────────────────────────

# ── Load Global Macro HMM Regime ──────────────────────────────────────────────

def load_global_hmm() -> pd.Series | None:
    """Load global_hmm_regime.csv (date → 0/1). Returns None if missing."""
    path = DATA_DIR / "global_hmm_regime.csv"
    if not path.exists():
        return None
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.rename("global_hmm")


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Optimization")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date for WFO (YYYY-MM-DD). Needs 12m before first OOS.")
    parser.add_argument("--end",   type=str, default=None,
                        help="End date for WFO (YYYY-MM-DD).")
    parser.add_argument("--hl-max", type=int, default=HL_MAX_DAYS,
                        help=f"Max half-life in calendar days (default {HL_MAX_DAYS}).")
    parser.add_argument("--rolling", action="store_true",
                        help="Force rolling window mode (override config WFO_EXPANDING).")
    parser.add_argument("--no-hmm", action="store_true",
                        help="Disable global Macro-HMM filter.")
    parser.add_argument("--no-hurst", action="store_true",
                        help="Disable Hurst exponent entry filter.")
    parser.add_argument("--pairs", type=str, default="pairs_selected.csv",
                        help="CSV file in data/ directory to read pairs from.")
    parser.add_argument("--full-universe", action="store_true",
                        help="Generate all possible pair combinations (2145) from tickers.")
    args = parser.parse_args()

    use_expanding = (not args.rolling) and WFO_EXPANDING

    closes, daily, volumes, vwaps = load_closes()
    if args.full_universe:
        from itertools import combinations
        ticker_list = sorted(daily.columns.tolist())
        combos = list(combinations(ticker_list, 2))
        pairs = pd.DataFrame({"pair": [f"{t1}-{t2}" for t1, t2 in combos]})
        print(f"Full Universe enabled: generated {len(pairs)} pair combinations.")
    else:
        pairs_file = DATA_DIR / args.pairs
        if not pairs_file.exists():
            raise SystemExit(f"Pairs file {pairs_file} not found.")
        pairs  = pd.read_csv(pairs_file)

    if pairs.empty:
        raise SystemExit("No pairs to process.")
    if daily is None:
        raise SystemExit("closes_daily.csv is required for WFO cointegration testing.")

    # ── Load EV-optimal params from Z-Bounce Density Profiler (step 3h) ──────────
    _z_profiles: dict[str, dict] = {}
    _zp_path = DATA_DIR / "z_profiles.csv"
    if _zp_path.exists():
        _zp_df = pd.read_csv(_zp_path)
        for _, _r in _zp_df[_zp_df["tradeable"] == True].iterrows():
            _z_profiles[_r["pair"]] = {
                "entry_z": float(_r["entry_z"]),
                "exit_z":  float(_r["exit_z"]),
                "stop_z":  float(_r["stop_z"]),
                "ev":      float(_r["ev"]),
            }
        print(f"Z-Bounce profiles loaded for {len(_z_profiles)} pairs "
              f"(EV-optimal, R:R=1.3)")
    else:
        print("No z_profiles.csv — using grid search (run z_profiler.py for faster WFO)")

    # ── Load Global Macro-HMM Regime ──────────────────────────────────────────
    if args.no_hmm:
        global_hmm = None
        print("Macro-HMM filter: DISABLED (--no-hmm flag)")
    else:
        global_hmm = load_global_hmm()
        if global_hmm is not None:
            n_panic = int(global_hmm.sum())
            pct     = n_panic / len(global_hmm) * 100
            print(f"Macro-HMM loaded: {len(global_hmm)} days, "
                  f"panic={n_panic} ({pct:.0f}%) — entries BLOCKED during panic")
        else:
            print("Macro-HMM filter: global_hmm_regime.csv not found — DISABLED")

    # ── Apply date range filter ───────────────────────────────────────────────
    if args.start:
        start_ts = pd.Timestamp(args.start, tz="UTC")
        closes = closes[closes.index >= start_ts]
        daily  = daily[daily.index  >= start_ts]
        print(f"Date filter applied: start={args.start}")
    if args.end:
        end_ts = pd.Timestamp(args.end, tz="UTC")
        closes = closes[closes.index <= end_ts]
        daily  = daily[daily.index  <= end_ts]
        print(f"Date filter applied: end={args.end}")

    hl_max_bars = args.hl_max * BARS_PER_DAY  # convert days → intraday bars

    windows = make_windows(closes, expanding=use_expanding)
    n_wins  = len(windows)

    window_mode = "EXPANDING" if use_expanding else "ROLLING"
    print("=" * 80)
    print(f"WALK-FORWARD OPTIMIZATION  ({window_mode} Window)")
    print("=" * 80)
    print(f"Initial train:  {WFO_TRAIN_MONTHS}m  |  "
          f"OOS:  {WFO_TEST_MONTHS}m  |  Step: {WFO_STEP_MONTHS}m")
    if use_expanding:
        first_train = (windows[0][1] - windows[0][0]).days // 30 if windows else 0
        last_train  = (windows[-1][1] - windows[-1][0]).days // 30 if windows else 0
        print(f"Train window: {first_train}m → {last_train}m (expanding)")
    print(f"Total windows: {n_wins}")
    print(f"Full span: {closes.index[0].date()} → {closes.index[-1].date()}")
    print(f"Pairs: {len(pairs)}")
    print(f"Half-life gate: ≤ {args.hl_max} calendar days")
    print(f"Macro-HMM: {'ACTIVE' if global_hmm is not None else 'OFF'}")
    use_hurst = not args.no_hurst
    from config import HURST_ENTRY_MAX
    hurst_label = f"ACTIVE (H>{HURST_ENTRY_MAX})" if use_hurst else "OFF"
    print(f"Hurst gate: {hurst_label}\n")

    # ── Load Session Profiles ──
    sessions = {}
    sess_path = DATA_DIR / "pair_sessions.csv"
    if sess_path.exists():
        _sess_df = pd.read_csv(sess_path)
        sessions = _sess_df.set_index("pair").to_dict(orient="index")
        print(f"Session profiles loaded for {len(sessions)} pairs")

    all_oos_trades = []   # accumulate across all windows
    wfo_params     = []   # best params log per pair per window
    total_hmm_blocked   = 0 # count of entries blocked by Macro-HMM
    total_hurst_blocked = 0 # count of entries blocked by Hurst drift guard

    for w_idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        tr_s_ts = pd.Timestamp(tr_s, tz="UTC")
        tr_e_ts = pd.Timestamp(tr_e, tz="UTC")
        te_s_ts = pd.Timestamp(te_s, tz="UTC")
        te_e_ts = pd.Timestamp(te_e, tz="UTC")

        closes_train = closes[(closes.index >= tr_s_ts) & (closes.index < tr_e_ts)]
        closes_test  = closes[(closes.index >= te_s_ts) & (closes.index < te_e_ts)]

        # Johansen uses ROLLING window (last 3 years of train) even in expanding
        # mode — long histories mask structural breaks like COST-WMT.
        _joh_lookback = relativedelta(years=3)
        joh_start = max(tr_s, (tr_e - _joh_lookback))
        joh_start_ts = pd.Timestamp(joh_start, tz="UTC")
        daily_train = daily[(daily.index >= joh_start_ts) & (daily.index < tr_e_ts)]

        if len(closes_train) < 500 or len(closes_test) < 100 or len(daily_train) < 100:
            continue

        days_train = (tr_e - tr_s).days
        days_test  = (te_e - te_s).days

        train_months = round(days_train / 30.44)
        print(f"  Window {w_idx+1:02d}/{n_wins}  "
              f"TRAIN {tr_s} → {tr_e} ({train_months}m)  |  "
              f"OOS {te_s} → {te_e}", end="")

        window_oos_pnl    = 0.0
        window_trades     = 0
        window_coint      = 0
        window_hmm_blocks = 0
        window_hurst_blocks = 0
        window_best = {}
        optimal_weights = {}
        window_best       = {}
        optimal_weights   = {}

        for _, row in pairs.iterrows():
            pair_name = row["pair"]
            t1, t2    = pair_name.split("-")

            if t1 not in closes_train.columns or t2 not in closes_train.columns:
                continue
            if t1 not in daily_train.columns or t2 not in daily_train.columns:
                continue

            # 1. Dynamic Cointegration Test (Johansen on daily train slice)
            is_coint, dynamic_beta = check_coint_johansen(daily_train, t1, t2, crit_level=0.95)
            if not is_coint or dynamic_beta < 0 or not (0.1 <= dynamic_beta <= 15.0):
                continue

            # 2. Dynamic Half-life
            spread_daily = daily_train[t1] - dynamic_beta * daily_train[t2]
            dynamic_hl   = compute_half_life(spread_daily) * BARS_PER_DAY

            # ── OU Half-life gate: reject lazy pairs ──────────────────────────
            if dynamic_hl > hl_max_bars:
                continue  # mean-reversion too slow for the OOS window
            
            window_coint += 1

            # 3. Build signals with window-specific beta and hl
            sig_train = build_signals(closes_train, volumes, vwaps, t1, t2, dynamic_beta, dynamic_hl)
            sig_test  = build_signals(closes_test,  volumes, vwaps, t1, t2, dynamic_beta, dynamic_hl)

            if len(sig_train) < 100 or len(sig_test) < 20:
                continue

            # 4. Get entry params — prefer EV-optimal profile, fall back to grid
            if pair_name in _z_profiles:
                zp = _z_profiles[pair_name]
                best = {
                    "entry_z": zp["entry_z"],
                    "exit_z":  zp["exit_z"],
                    "stop_z":  zp["stop_z"],
                    "sharpe":  zp["ev"],    # EV stands in for sharpe in the log
                    "trades":  0,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                }
                source = "EV-profile"
            else:
                best = run_grid(sig_train, t1, t2, dynamic_beta, COMBOS, days_train)
                source = "grid"
            if best is None:
                continue

            # ── KDE Structural Filter (Quality Check) ─────────────
            from filters import validate_kde_density
            is_kde_valid = validate_kde_density(sig_train["zscore"], best["entry_z"], threshold_ratio=0.5)
            if not is_kde_valid:
                print(f"  {pair_name:<10} TRAIN: REJECTED by KDE (Low Density Node at Z={best['entry_z']})")
                continue

            # 5. Trade OOS with the best params found on TRAIN
            pair_hurst_f = HurstFilter() if use_hurst else None
            spread_daily_full = (daily[t1] - dynamic_beta * daily[t2]).dropna() if use_hurst else None

            # ── ML Model Training (Train Trades) ─────────────
            train_trades, _, _ = backtest_oos(
                sig_train, t1, t2, dynamic_beta,
                best["entry_z"], best["exit_z"], best["stop_z"],
                hmm_regime=global_hmm,
                hurst_filter=pair_hurst_f,
                spread_daily=spread_daily_full,
                session_window=sessions.get(pair_name)
            )
            best["ml_model"] = None
            if len(train_trades) >= 20:
                X_list = [t["features"] for t in train_trades if "features" in t]
                y_list = [1 if t["net_pnl"] > 0 else 0 for t in train_trades if "features" in t]
                if len(X_list) > 0 and len(np.unique(y_list)) > 1:
                    from sklearn.ensemble import HistGradientBoostingClassifier
                    clf = HistGradientBoostingClassifier(max_depth=3, min_samples_leaf=5, learning_rate=0.05, max_iter=50)
                    clf.fit(np.array(X_list), np.array(y_list))
                    best["ml_model"] = clf

            # Save for MVO calculation
            best["dynamic_beta"] = dynamic_beta
            best["dynamic_hl"] = dynamic_hl
            best["source"] = source
            best["train_returns"] = _trades_to_daily_pnl(train_trades, daily_train.index)
            window_best[pair_name] = best
            
        # ── 6. Portfolio Optimization (MVO) ───────────
        active_pairs = list(window_best.keys())
        if active_pairs:
            train_returns_df = pd.DataFrame({p: window_best[p]["train_returns"] for p in active_pairs}).fillna(0)
            optimal_weights = optimize_portfolio_weights(train_returns_df, target_return=0.10)
            
            # --- Module 4: Tail Risk Guard ---
            sys_scaler = calculate_systemic_risk_scaler(train_returns_df, threshold=0.7)
            if sys_scaler < 1.0:
                print(f"  [TAIL RISK] Absorption Ratio spike ({sys_scaler:.2f}). Scaling risk.")
                for p in optimal_weights:
                    optimal_weights[p] *= sys_scaler
        else:
            optimal_weights = {}

        # ── 7. OOS Execution (Second Pass with Weights) ───────────
        for pair_name in active_pairs:
            t1, t2 = pair_name.split("-")
            best = window_best[pair_name]
            weight = optimal_weights.get(pair_name, 0.0)
            if weight <= 0: continue

            # Re-build sig_test (or use cached if memory allowed, but for simplicity we rebuild)
            sig_test = build_signals(closes_test, volumes, vwaps, t1, t2, best["dynamic_beta"], best["dynamic_hl"])
            
            pair_hurst_f = HurstFilter() if use_hurst else None
            spread_daily_full = (daily[t1] - best["dynamic_beta"] * daily[t2]).dropna() if use_hurst else None
            
            oos_trades, pair_hmm_blocked, pair_hurst_blocked = backtest_oos(
                sig_test, t1, t2, best["dynamic_beta"],
                best["entry_z"], best["exit_z"], best["stop_z"],
                hmm_regime=global_hmm,
                hurst_filter=pair_hurst_f,
                spread_daily=spread_daily_full,
                session_window=sessions.get(pair_name)
            )
            window_hmm_blocks += pair_hmm_blocked
            window_hurst_blocks += pair_hurst_blocked

            if len(oos_trades) < WFO_MIN_TRADES:
                continue
                
            # ── ML Sizing (OOS) ─────────────
            clf = best.get("ml_model")
            for t in oos_trades:
                ml_mult = 1.0
                if clf is not None and "features" in t:
                    prob = clf.predict_proba(np.array([t["features"]]))[0][1]
                    ml_mult = max(0.0, 2.0 * (prob - 0.5))
                
                # Apply Portfolio Weight AND ML Multiplier
                final_size = weight * ml_mult
                t["net_pnl"] *= final_size
                t["gross_pnl"] *= final_size
                t["tx_cost"] *= final_size

            # Filter out 0 size trades
            oos_trades = [t for t in oos_trades if t["net_pnl"] != 0 or t["tx_cost"] != 0]
            if len(oos_trades) == 0:
                continue

            oos_pnl = sum(t["net_pnl"] for t in oos_trades)

            for t in oos_trades:
                t["pair"]     = pair_name
                t["window"]   = w_idx + 1
                t["entry_z"]  = best["entry_z"]
                t["exit_z"]   = best["exit_z"]
                t["stop_z"]   = best["stop_z"]
                t["source"]   = best["source"]
            all_oos_trades.extend(oos_trades)

            oos_arr = np.array([t["net_pnl"] for t in oos_trades])
            oos_sh  = (oos_arr.mean() / oos_arr.std() *
                       np.sqrt(len(oos_arr) / max(days_test / 365.25, 0.01))
                       ) if oos_arr.std() > 0 else 0.0

            wfo_params.append({
                "window":      w_idx + 1,
                "train_start": str(tr_s), "train_end": str(tr_e),
                "oos_start":   str(te_s), "oos_end":   str(te_e),
                "pair":        pair_name,
                "beta":        round(best["dynamic_beta"], 4),
                "half_life":   round(best["dynamic_hl"], 1),
                "entry_z":     best["entry_z"],
                "exit_z":      best["exit_z"],
                "stop_z":      best["stop_z"],
                "train_sharpe": best["sharpe"],
                "oos_sharpe":  round(oos_sh, 3),
                "oos_trades":  len(oos_trades),
                "oos_pnl":     round(oos_pnl, 4),
            })

            window_oos_pnl += oos_pnl
            window_trades  += len(oos_trades)

        total_hmm_blocked += window_hmm_blocks
        total_hurst_blocked += window_hurst_blocks
        hmm_info   = f"  HMM blocked={window_hmm_blocks}" if global_hmm is not None else ""
        hurst_info = f"  Hurst blocked={window_hurst_blocks}" if use_hurst else ""
        print(f"  →  {window_coint:2d} pairs passed Johansen  |  "
              f"{window_trades:3d} OOS trades   P&L={window_oos_pnl:+.4f}{hmm_info}{hurst_info}")

        if len(optimal_weights) > 0:
            import joblib
            try:
                live_state = {
                    "window_best": window_best,
                    "optimal_weights": optimal_weights
                }
                OUTPUT_DIR.mkdir(exist_ok=True)
                joblib.dump(live_state, OUTPUT_DIR / "live_state.pkl")
            except Exception:
                pass



    if not all_oos_trades:
        print("\nNo OOS trades accumulated. "
              "Check that pairs_selected.csv matches closes data.")
        return

    # ── Save ──────────────────────────────────────────────────────────────────
    df_trades = pd.DataFrame(all_oos_trades)
    df_params = pd.DataFrame(wfo_params)

    df_trades.to_csv(DATA_DIR / "wfo_results.csv", index=False)
    df_params.to_csv(DATA_DIR / "wfo_params.csv",  index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    pnl       = df_trades["net_pnl"]
    win_rate  = (pnl > 0).mean() * 100
    total_pnl = pnl.sum()
    n_trades  = len(pnl)
    wins      = pnl[pnl > 0].sum()
    losses    = abs(pnl[pnl <= 0].sum())
    pf        = wins / losses if losses > 0 else float("inf")

    # Annualised Sharpe across entire OOS chain
    span_years = (pd.Timestamp(windows[-1][3]) -
                  pd.Timestamp(windows[0][2])).days / 365.25
    tpy  = n_trades / max(span_years, 0.01)
    sh   = pnl.mean() / pnl.std() * np.sqrt(tpy) if pnl.std() > 0 else 0.0
    cum  = pnl.cumsum()
    dd   = float((cum - cum.cummax()).min())

    print(f"\n{'='*70}")
    print(f"WFO PORTFOLIO SUMMARY  ({n_wins} windows,  "
          f"{len(df_params['pair'].unique())} pairs)")
    print(f"{'='*70}")
    print(f"OOS Trades:      {n_trades}")
    print(f"Win Rate:        {win_rate:.1f}%")
    print(f"Total OOS P&L:   {total_pnl:+.4f}  (spread units)")
    print(f"Profit Factor:   {pf:.2f}")
    print(f"Max Drawdown:    {dd:.4f}")
    print(f"Sharpe (chain):  {sh:.2f}")
    if global_hmm is not None:
        print(f"HMM blocked:     {total_hmm_blocked} potential entries")
    if use_hurst:
        print(f"Hurst blocked:   {total_hurst_blocked} potential entries")
    print(f"Window mode:     {window_mode}")
    print(f"\nSaved → data/wfo_results.csv  ({n_trades} trades)")
    print(f"Saved → data/wfo_params.csv   ({len(df_params)} rows)")

    # ── Per-pair WFO stability summary ────────────────────────────────────────
    print(f"\n{'Pair':<12}  {'Windows':>7}  {'OOS Sh':>7}  {'Train Sh':>9}  "
          f"{'OOS Trades':>10}  {'OOS P&L':>10}")
    print("-" * 65)
    for pair in sorted(df_params["pair"].unique()):
        sub   = df_params[df_params["pair"] == pair]
        n_w   = len(sub)
        avg_oos_sh   = sub["oos_sharpe"].mean()
        avg_train_sh = sub["train_sharpe"].mean()
        tot_tr = sub["oos_trades"].sum()
        tot_pnl = sub["oos_pnl"].sum()
        print(f"{pair:<12}  {n_w:>7}  {avg_oos_sh:>7.2f}  "
              f"{avg_train_sh:>9.2f}  {tot_tr:>10}  {tot_pnl:>+10.4f}")

    # ── Visualization ─────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Panel 1: Continuous OOS equity curve
    ax = axes[0]
    df_trades_sorted = df_trades.sort_values("exit_time")
    equity = df_trades_sorted["net_pnl"].cumsum()

    ax.plot(range(len(equity)), equity.values,
            color="steelblue", lw=1.5, label="WFO OOS equity")
    ax.fill_between(range(len(equity)), equity.values, 0,
                    where=(equity.values >= 0),
                    color="steelblue", alpha=0.15)
    ax.fill_between(range(len(equity)), equity.values, 0,
                    where=(equity.values < 0),
                    color="salmon", alpha=0.3)
    ax.axhline(0, color="black", lw=0.8)

    # Mark window boundaries
    window_starts = {}
    for _, r in df_params.iterrows():
        w = r["window"]
        oos_s = r["oos_start"]
        if w not in window_starts:
            # Find trade index nearest to this OOS start
            mask = df_trades_sorted["exit_time"] >= oos_s
            if mask.any():
                idx = mask.idxmax()
                pos = df_trades_sorted.index.get_loc(idx)
                window_starts[w] = pos

    for w, pos in window_starts.items():
        ax.axvline(pos, color="gray", lw=0.7, ls="--", alpha=0.5)
        ax.text(pos + 1, ax.get_ylim()[0] * 0.9 if ax.get_ylim()[0] < 0 else 0,
                f"W{w}", fontsize=7, color="gray")

    ax.set_title(f"WFO Continuous OOS Equity Curve  |  "
                 f"{n_trades} trades  Sharpe={sh:.2f}  P&L={total_pnl:+.4f}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative Net P&L (spread units)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Panel 2: Train vs OOS Sharpe per window (scatter per pair)
    ax2 = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(df_params["pair"].unique())))
    pair_colors = {p: c for p, c in zip(sorted(df_params["pair"].unique()), colors)}

    for pair in df_params["pair"].unique():
        sub = df_params[df_params["pair"] == pair]
        ax2.scatter(sub["train_sharpe"], sub["oos_sharpe"],
                    color=pair_colors[pair], alpha=0.7, s=50,
                    label=pair, zorder=3)

    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.axvline(0, color="gray", lw=0.8, ls="--")
    ax2.set_xlabel("TRAIN Sharpe")
    ax2.set_ylabel("OOS Sharpe")
    ax2.set_title("WFO Stability: Train vs OOS Sharpe per Window per Pair",
                   fontsize=11, fontweight="bold")
    ax2.legend(fontsize=7, ncol=3, loc="upper left")
    ax2.grid(True, alpha=0.2)

    # Add correlation annotation
    if len(df_params) > 5:
        corr = df_params[["train_sharpe", "oos_sharpe"]].corr().iloc[0, 1]
        ax2.text(0.02, 0.97, f"r(train, OOS) = {corr:.2f}",
                 transform=ax2.transAxes, fontsize=10,
                 va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    plt.tight_layout()
    equity_path = OUTPUT_DIR / "wfo_equity.png"
    plt.savefig(equity_path, dpi=150)
    plt.close(fig)
    print(f"Chart saved → {equity_path}")


if __name__ == "__main__":
    main()
