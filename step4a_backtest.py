import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
from pathlib import Path
from collections import Counter
from config import (
    ENTRY_Z, EXIT_Z, STOP_Z, ENTRY_Z_VOLATILE,
    COST_MAKER, COST_TAKER, CIRCUIT_BREAKER_Z, BORROW_RATE_ANNUAL, PAIR_MAX_LOSS,
    IV_SIZE_NORM, MIN_POSITION_SIZE, TRAIN_RATIO,
    RTH_START, RTH_END, SIGNAL_START, RECENT_BARS,
    DATA_DIR, OUTPUT_DIR, INITIAL_CAPITAL, LEVERAGE,
    ACCOUNT_MODEL,
    ALLOCATION_METHOD, MAX_PAIR_WEIGHT, TARGET_RISK_USD,
    ZERO_MAX_TRAILING_LOSS_PCT, ZERO_MAX_DAILY_LOSS_PCT,
    ZERO_MIN_PROFIT_DAYS_30D,
    ZERO_MAX_INACTIVE_DAYS,
    BARS_PER_DAY, CLOSES_FILE, KALMAN_DELTA,
    COINT_WINDOW_DAYS, COINT_BREAK_P, COINT_RECHECK_DAYS,
    USE_VWZ, VWZ_MIN_VOLUME, USE_RVOL_GATE, RVOL_THRESHOLD, RVOL_WINDOW,
    USE_VWAP, USE_VWAP_MTF, VWAP_MTF_TF, USE_VELOCITY_GATE, VELOCITY_WINDOW,
    MTF_CONFIRM, MTF_Z_MIN, MTF_RESAMPLE,
    USE_RETURN_SPREAD, RETURN_WINDOW,
    RVR_FILTER, RVR_WINDOW_SHORT, RVR_WINDOW_LONG, RVR_MAX,
    LIVE_CORR_FILTER, LIVE_CORR_WINDOW, LIVE_CORR_MIN,
    SESSION_FILTER,
    KALMAN_INNOV_FILTER, KALMAN_INNOV_WINDOW, KALMAN_INNOV_MAX,
    USE_COPULA, COPULA_WINDOW, COPULA_Z_MIN, COPULA_LAMBDA_MIN,
    EVENT_FILTER, EVENT_BARS_BEFORE, EVENT_BARS_AFTER, BAR_MINUTES,
    ENTRY_Z_MIN,
)
from kalman import kalman_hedge
from step3e_sizing import (
    load_regimes, load_iv, load_global_hmm, load_mc_confidence,
    iv_multiplier_series, position_size,
)
from filters import CointegrationFilter, MacroFilter, HurstFilter, _LAZY_WINDOW_BARS
from copula_signals import compute_copula_signals
from macro_calendar import build_event_blackout
from config import HURST_ENTRY_WINDOW

BARS_PER_TRADING_DAY = BARS_PER_DAY


def _pairs_universe_path() -> str:
    zero_path = DATA_DIR / "pairs_zero_universe.csv"
    if ACCOUNT_MODEL == "Zero" and zero_path.exists():
        try:
            if not pd.read_csv(zero_path, nrows=1).empty:
                return str(zero_path)
        except Exception:
            pass
    return str(DATA_DIR / "pairs_selected.csv")


def _data_path() -> str:
    p = DATA_DIR / CLOSES_FILE
    if not p.exists():
        fallback = DATA_DIR / "closes_15min.csv"
        if fallback.exists():
            return str(fallback)
        raise FileNotFoundError(f"No data file: {CLOSES_FILE}")
    return str(p)


def load_closes() -> pd.DataFrame:
    """Load ALL available intraday data.

    Pair selection is done on 8-year daily data (step2), so there is no
    look-ahead: using the full intraday history is valid and maximises
    the number of trades for statistical evaluation.
    """
    pairs_path = Path(_pairs_universe_path())
    tickers = None
    if pairs_path.exists():
        meta = pd.read_csv(pairs_path)
        if not meta.empty:
            tickers = sorted({t for p in meta["pair"] for t in str(p).split("-")})

    try:
        from data_loader import load_closes as _load_closes
        closes = _load_closes(tickers, rth=True)
    except Exception:
        closes = pd.read_csv(_data_path(), index_col=0, parse_dates=True)
        if tickers is not None:
            closes = closes[[c for c in tickers if c in closes.columns]]
        closes.index = pd.to_datetime(closes.index, utc=True)
        closes = closes.between_time(RTH_START, RTH_END)

    closes.index = pd.to_datetime(closes.index, utc=True).tz_convert("US/Eastern")

    # Remove corrupted price values: any bar where |pct_change| > 5% is a bad tick.
    # FX pairs never move 5% in one minute; such values are Dukascopy data artifacts.
    for col in closes.columns:
        col_ret = closes[col].pct_change(fill_method=None).abs()
        closes.loc[col_ret > 0.05, col] = np.nan

    if pairs_path.exists():
        meta = pd.read_csv(pairs_path)
        if not meta.empty:
            needed    = {t for p in meta["pair"] for t in p.split("-")}
            available = [t for t in needed if t in closes.columns]
            closes    = closes[available].dropna()
            recent_bars = int(os.getenv("BACKTEST_RECENT_BARS", "0") or 0)
            if recent_bars > 0:
                closes = closes.tail(recent_bars)
            return closes

    closes = closes.dropna()
    recent_bars = int(os.getenv("BACKTEST_RECENT_BARS", "0") or 0)
    if recent_bars > 0:
        closes = closes.tail(recent_bars)
    return closes


def _combined_volume(volumes: pd.DataFrame | None,
                     index: pd.DatetimeIndex,
                     t1: str,
                     t2: str) -> pd.Series | None:
    if volumes is None or t1 not in volumes.columns or t2 not in volumes.columns:
        return None
    vol = (
        volumes[t1].reindex(index).fillna(0.0)
        + volumes[t2].reindex(index).fillna(0.0)
    )
    vol = vol.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return vol.clip(lower=VWZ_MIN_VOLUME)


def _volume_weighted_zscore(spread: pd.Series,
                            vol: pd.Series,
                            window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    w_sum = vol.rolling(window=window, min_periods=window).sum()
    mean = (spread * vol).rolling(window=window, min_periods=window).sum() / w_sum
    var = (((spread - mean) ** 2) * vol).rolling(window=window, min_periods=window).sum() / w_sum
    std = var.pow(0.5)
    zscore = (spread - mean) / std.replace(0, np.nan)
    return zscore, mean, std


def _relative_volume(vol: pd.Series | None, index: pd.DatetimeIndex) -> pd.Series:
    if vol is None:
        return pd.Series(1.0, index=index)
    baseline = vol.rolling(RVOL_WINDOW, min_periods=RVOL_WINDOW).mean()
    return (vol / baseline.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def build_signals(closes, t1, t2, beta, half_life,
                  delta: float = KALMAN_DELTA,
                  volumes: pd.DataFrame | None = None,
                  vwaps:   pd.DataFrame | None = None) -> pd.DataFrame:
    """Kalman-filter spread signals with VW-Z, VWAP prices, and velocity.

    VWAP:     uses VWAP bar prices instead of closes as Kalman observations,
              giving a volume-weighted "fair price" per bar (less noise at
              illiquid open/close ticks).  Falls back to closes if unavailable.

    VW-Z:     rolling mean and std of the spread are weighted by combined bar
              volume of both legs — liquid-session bars dominate the Z-score.

    Velocity: Δzscore over VELOCITY_WINDOW bars.  Used as an entry gate:
              only enter when Z-score is already moving back toward mean,
              i.e., mean reversion is confirmed rather than just triggered.
    """
    # ── Price input: VWAP or close ────────────────────────────────────────────
    if USE_VWAP and vwaps is not None and t1 in vwaps.columns and t2 in vwaps.columns:
        p1 = vwaps[t1].reindex(closes.index).fillna(closes[t1])
        p2 = vwaps[t2].reindex(closes.index).fillna(closes[t2])
    else:
        p1, p2 = closes[t1], closes[t2]

    # ── Return-spread mode ────────────────────────────────────────────────────
    # We use the RAW cumulative-return spread (not Kalman innovations).
    # Rationale: Kalman innovations are approximately i.i.d. by construction
    # (the filter removes predictable signal).  The RAW spread
    #   S[t] = cum_r1[t] - beta * cum_r2[t]
    # is an MA(RETURN_WINDOW) process with natural mean reversion at the window
    # boundary — it IS the signal we want to trade.
    # The Kalman filter is retained for beta adaptation and innov_var diagnostics
    # but its innovation sequence is NOT used as the tradeable spread.
    vol = _combined_volume(volumes, closes.index, t1, t2)

    if USE_RETURN_SPREAD:
        r1 = p1.pct_change().fillna(0)
        r2 = p2.pct_change().fillna(0)
        cum_r1 = r1.rolling(RETURN_WINDOW).sum()
        cum_r2 = r2.rolling(RETURN_WINDOW).sum()
        # Kalman for time-varying beta (diagnostic + innov_var)
        alpha_arr, beta_arr, innov, innov_var_arr = kalman_hedge(
            cum_r1.fillna(0).values, cum_r2.fillna(0).values,
            delta=delta, beta_init=float(beta),
        )
        beta_s    = pd.Series(beta_arr,      index=closes.index, name="beta")
        alpha_s   = pd.Series(alpha_arr,     index=closes.index, name="alpha")
        innov_var = pd.Series(innov_var_arr, index=closes.index, name="innov_var")
        # Raw spread: use time-varying Kalman beta (smoothed, avoids OLS look-ahead)
        spread = (cum_r1 - beta_s * cum_r2).rename("spread")
        # Z-score window = 3× return window so we see 3 full oscillations
        window = RETURN_WINDOW * 3
        if USE_VWZ and vol is not None:
            zscore, spread_mean, spread_std = _volume_weighted_zscore(spread, vol, window)
        else:
            spread_mean = spread.rolling(window).mean()
            spread_std  = (spread - spread_mean).rolling(window).std()
            zscore      = (spread - spread_mean) / spread_std.replace(0, np.nan)
    else:
        p1_kalman, p2_kalman = p1, p2
        alpha_arr, beta_arr, innov, innov_var_arr = kalman_hedge(
            p1_kalman.values, p2_kalman.values, delta=delta, beta_init=float(beta),
        )
        spread    = pd.Series(innov,         index=closes.index, name="spread")
        beta_s    = pd.Series(beta_arr,      index=closes.index, name="beta")
        alpha_s   = pd.Series(alpha_arr,     index=closes.index, name="alpha")
        innov_var = pd.Series(innov_var_arr, index=closes.index, name="innov_var")
        window  = max(20, min(int(half_life), 200))

    # ── Z-score: volume-weighted or standard (price-spread mode only for VW-Z) ─
    if not USE_RETURN_SPREAD:
        if USE_VWZ and vol is not None:
            zscore, spread_mean, spread_std = _volume_weighted_zscore(spread, vol, window)
        else:
            spread_mean = spread.rolling(window=window).mean()
            spread_std = spread.rolling(window=window).std()
            zscore     = spread / spread_std.replace(0, np.nan)

    # ── Velocity: Δzscore over N bars ────────────────────────────────────────
    velocity = zscore.diff(VELOCITY_WINDOW)

    # ── MTF: resample Z to higher TF (5-min on 1-min bars, 1H on 15-min) ────
    zscore_1h = (zscore.resample(MTF_RESAMPLE).last()
                       .reindex(closes.index, method="ffill"))

    # ── RVR: short-term / long-term variance of spread diff ──────────────────
    spread_diff = spread.diff()
    rv_short    = spread_diff.rolling(RVR_WINDOW_SHORT).var()
    rv_long     = spread_diff.rolling(RVR_WINDOW_LONG).var()
    rvr         = rv_short / rv_long.replace(0, np.nan)

    # ── Live correlation: rolling return correlation of both legs ─────────────
    r1          = closes[t1].pct_change()
    r2          = closes[t2].pct_change()
    corr_live   = r1.rolling(LIVE_CORR_WINDOW).corr(r2)

    # ── Kalman innovation variance ratio ──────────────────────────────────────
    innov_var_baseline = innov_var.rolling(KALMAN_INNOV_WINDOW).mean()
    innov_var_ratio    = innov_var / innov_var_baseline.replace(0, np.nan)

    # ── RVOL: Relative Volume Gate ──────────────────────────────────────────
    rvol = _relative_volume(vol, closes.index)

    # ── MTF VWAP: 15-min anchor for 1-min strategy ──────────────────────────
    if USE_VWAP_MTF and vwaps is not None and t1 in vwaps.columns and t2 in vwaps.columns:
        # Use VWAP-based spread
        p1_v = vwaps[t1].reindex(closes.index).fillna(closes[t1])
        p2_v = vwaps[t2].reindex(closes.index).fillna(closes[t2])
        spread_v = p1_v - beta_s * p2_v
        spread_vwap_mtf = spread_v.resample(VWAP_MTF_TF).mean().reindex(closes.index, method="ffill")
    else:
        spread_vwap_mtf = pd.Series(np.nan, index=closes.index)

    copula_df = pd.DataFrame(index=closes.index)
    if USE_COPULA:
        copula_df = compute_copula_signals(p1, p2, COPULA_WINDOW)

    out = pd.DataFrame({
        f"{t1}_close":     closes[t1],
        f"{t2}_close":     closes[t2],
        "spread":          spread,
        "beta":            beta_s,
        "alpha":           alpha_s,
        "spread_std":      spread_std,
        "zscore":          zscore,
        "velocity":        velocity,
        "zscore_1h":       zscore_1h,
        "rvr":             rvr,
        "corr_live":       corr_live,
        "rvol":            rvol,
        "spread_vwap_mtf": spread_vwap_mtf,
        "innov_var":       innov_var,
        "innov_var_ratio": innov_var_ratio,
    })
    if USE_COPULA and not copula_df.empty:
        out = out.join(copula_df, how="left")

    return out.dropna(subset=["zscore"]).between_time(SIGNAL_START, RTH_END)




def load_kmeans_regime() -> pd.Series | None:
    """Load daily K-Means macro regime labels (0=Trend, 1=Sideways, 2=Panic)."""
    path = DATA_DIR / "kmeans_regimes.csv"
    if not path.exists():
        return None
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    return s


def min_viable_entry_z(sigma_spread: float, avg_notional: float,
                        exit_z: float, safety: float = 2.5) -> float:
    """
    Minimum entry Z-score for expected gross P&L to exceed transaction costs.

    Expected gross per trade = (entry_z + |exit_z|) × sigma_spread
    Break-even condition:
        (entry_z + |exit_z|) × sigma >= safety × 2 × COST_MAKER, COST_TAKER, CIRCUIT_BREAKER_Z × notional
        → entry_z_min = safety × cost_fraction − |exit_z|

    safety=2.5 means expected profit must be 2.5× the transaction cost.
    If the grid/MC optimised entry_z is below this floor, we raise it.
    """
    cost_frac = (2 * COST_TAKER * avg_notional) / max(sigma_spread, 1e-8)
    return max(safety * cost_frac - abs(exit_z), 0.0)


def ou_params_from_spread(spread: pd.Series) -> tuple[float, float]:
    """OLS on ΔS = -θ·S_{t-1} + ε  →  returns (theta_per_bar, residual_std)."""
    ds  = spread.diff().dropna()
    lag = spread.shift(1).dropna()
    idx = ds.index.intersection(lag.index)
    reg = sm.OLS(ds[idx], lag[idx]).fit()
    theta = float(-reg.params.iloc[0])
    return max(theta, 1e-6), float(reg.resid.std())


def optimal_thresholds(theta: float, sigma_roll: float, notional: float,
                       n_sim: int = 250, max_bars: int = 1500
                       ) -> tuple[float, float, float]:
    """
    Joint Monte Carlo grid search for (entry_z, exit_thresh, stop_thresh).

    OU in z-score space: Z_{t+1} = Z_t·(1-θ) + √(2θ)·ε
    LONG: enter at -z_e, exit when Z >= z_x, stop when Z <= -z_s.
    P&L_exit = z_e + z_x - c_z   (spread moved from -z_e to z_x)
    P&L_stop = z_e - z_s - c_z   (spread moved against us to -z_s)
    """
    rng   = np.random.default_rng(42)
    c_z   = 2 * COST_TAKER * notional / max(sigma_roll, 1e-8)
    sig_z = np.sqrt(2 * theta)

    entry_grid = np.arange(1.5, 3.75, 0.25)  # [1.5 … 3.5]  (8 values)
    exit_grid  = np.arange(-0.3, 1.05, 0.1)  # [-0.3 … 1.0] fine step for exit-past-zero
    stop_grid  = np.arange(3.0,  5.5,  0.5)  # [3.0 … 5.0]  (5 values)

    best_rate = -np.inf
    best      = (float(ENTRY_Z), float(EXIT_Z), float(STOP_Z))

    for z_e in entry_grid:
        for z_x in exit_grid:
            if z_x >= z_e:          # exit must be less extreme than entry
                continue
            for z_s in stop_grid:
                if z_s <= z_e:      # stop must be more extreme than entry
                    continue

                z     = np.full(n_sim, -z_e)
                done  = np.zeros(n_sim, bool)
                pnl   = np.zeros(n_sim)
                t_end = np.full(n_sim, float(max_bars))

                for t in range(1, max_bars + 1):
                    if done.all():
                        break
                    z = np.where(done, z, z * (1 - theta) + sig_z * rng.standard_normal(n_sim))
                    he = (~done) & (z >= z_x)
                    hs = (~done) & (z <= -z_s)
                    pnl   = np.where(he,       z_e + z_x - c_z,  pnl)
                    pnl   = np.where(hs & ~he, z_e - z_s  - c_z, pnl)
                    t_end = np.where((he | hs) & ~done, float(t), t_end)
                    done  = done | he | hs

                pnl = np.where(~done, -c_z, pnl)   # timed-out: pay cost, no profit

                rate = float(pnl.mean()) / float(t_end.mean())
                if rate > best_rate:
                    best_rate = rate
                    best      = (z_e, z_x, z_s)

    if best_rate <= 0:
        return float(ENTRY_Z), float(EXIT_Z), float(STOP_Z)
    return round(best[0], 2), round(best[1], 2), round(best[2], 2)


def backtest_pair(df, t1, t2, beta, pair_name: str = "",
                  regime_dict: dict | None = None,
                  sizing_args: dict | None = None,
                  entry_z: float = ENTRY_Z,
                  exit_thresh: float = EXIT_Z,
                  stop_thresh: float = STOP_Z,
                  coint_filter: CointegrationFilter | None = None,
                  macro_filter: MacroFilter | None = None,
                  hurst_filter: HurstFilter | None = None,
                  spread_daily: pd.Series | None = None,
                  oos_start: pd.Timestamp | None = None,
                  max_notional: float = 1e9,
                  max_hold_bars: int = 9999,
                  session_window: tuple[int, int] | None = None) -> pd.DataFrame:
    """
    oos_start    : first timestamp of OOS — Kalman warms up on full history,
                   entries blocked before this date.
    coint_filter : CointegrationFilter — O(1) daily coint validity + lazy ADF
                   on Z-trigger. Force-closes and suspends when coint breaks.
    macro_filter : MacroFilter — entry blocking (VIX9D / K-Means) and
                   force-close (K-Means Panic / global HMM panic).
    """
    t1_col, t2_col = f"{t1}_close", f"{t2}_close"
    position       = 0
    entry_spread   = entry_t1 = entry_t2 = entry_beta = entry_alpha = entry_std = 0.0
    entry_bar      = 0
    entry_n_shares = 1.0
    cumulative_pnl = 0.0
    trades         = []
    hurst_blocked  = 0       # count of entries blocked by Hurst drift guard
    suspended      = False   # True when daily coint check says broken
    _cb_cooldown   = 0       # circuit-breaker cooldown counter (bars remaining)
    active_exit_thresh = exit_thresh   # exit/stop thresholds active for current trade
    active_stop_thresh = stop_thresh   # (may change per regime at entry time)
    diag = Counter()

    for i in range(len(df)):
        ts         = df.index[i]
        z          = df["zscore"].iloc[i]
        spread_now = df["spread"].iloc[i]
        p1         = df[t1_col].iloc[i]
        p2         = df[t2_col].iloc[i]

        # ── Phase 1: cointegration validity (O(1) daily lookup) ───────────
        if coint_filter is not None:
            coint_ok = coint_filter.is_valid(ts)
            if not coint_ok and not suspended:
                suspended = True
            elif coint_ok and suspended:
                suspended = False
        elif USE_RETURN_SPREAD and _cb_cooldown > 0:
            # Return-spread: circuit-breaker uses a timed cooldown instead of
            # permanent suspension (no coint filter to reset it).
            _cb_cooldown -= 1
            if _cb_cooldown == 0:
                suspended = False

        # ── Active Z-score for exit / stop / circuit-breaker ────────────
        z_active = z
        if position != 0 and entry_std > 0:
            if USE_RETURN_SPREAD:
                # Use current rolling z-score directly.
                # Rolling std normalizes against news spikes; static entry_std
                # is too small after a period of low volatility, causing false CBs.
                z_active = z
            else:
                # Price-spread: frozen beta + alpha to combat Kalman illusion.
                static_spread = p1 - (entry_alpha + entry_beta * p2)
                z_active = static_spread / entry_std
            
        # ── Idiosyncratic Circuit Breaker ────────────────────────────────
        if position != 0 and abs(z_active) >= CIRCUIT_BREAKER_Z:
            force_close = True
            if USE_RETURN_SPREAD:
                # Timed cooldown (60 bars = 1 hour on 1-min) — don't suspend permanently.
                # Return spread spikes are transient (news); the pair stays valid.
                suspended = True
                _cb_cooldown = 60
            else:
                suspended = True  # Block pair permanently for this window
        else:
            # ── Macro force-close (K-Means Panic or global HMM panic) ────────
            force_close = (
                suspended
                or (macro_filter is not None and macro_filter.is_force_close(ts))
            )
        if position != 0 and force_close:
            n              = entry_n_shares
            gross_pnl      = position * (spread_now - entry_spread) * n
            notional       = (entry_t1 + abs(entry_beta) * entry_t2) * n
            tx_cost        = notional * COST_MAKER + notional * COST_TAKER
            holding_days   = (i - entry_bar) / BARS_PER_TRADING_DAY
            short_notional = n * ((abs(entry_beta) * entry_t2) if position == 1 else entry_t1)
            borrow_cost    = short_notional * BORROW_RATE_ANNUAL * holding_days / 252
            net_pnl        = gross_pnl - tx_cost - borrow_cost
            cumulative_pnl += net_pnl
            reason = ("COINT_BREAK" if suspended
                      else macro_filter.force_close_reason(ts))
            trades.append({
                "pair":         f"{t1}-{t2}",
                "entry_time":   df.index[entry_bar],
                "exit_time":    ts,
                "direction":    "LONG" if position == 1 else "SHORT",
                "holding_bars": i - entry_bar,
                "n_shares":     round(n, 2),
                "size":         1.0,
                "gross_pnl":    round(gross_pnl, 4),
                "tx_cost":      round(tx_cost, 4),
                "borrow_cost":  round(borrow_cost, 4),
                "net_pnl":      round(net_pnl, 4),
                "cum_pnl":      round(cumulative_pnl, 4),
                "exit_reason":  reason,
                "entry_z":      round(df["zscore"].iloc[entry_bar], 2),
                "exit_z":       round(z, 2),
            })
            position       = 0
            entry_n_shares = 1.0
            if cumulative_pnl < PAIR_MAX_LOSS:
                break
            continue

        # ── Normal exit / stop / time-stop ───────────────────────────────
        if position != 0:
            bars_held   = i - entry_bar
            time_stop   = bars_held >= max_hold_bars
            exit_signal = (not time_stop) and (
                (position == 1 and z_active >= active_exit_thresh) or (position == -1 and z_active <= -active_exit_thresh))
            stop_signal = (not time_stop) and (
                (position == 1 and z_active <= -active_stop_thresh) or (position == -1 and z_active >= active_stop_thresh))

            if exit_signal or stop_signal or time_stop:
                n              = entry_n_shares
                gross_pnl      = position * (spread_now - entry_spread) * n
                notional       = (entry_t1 + abs(entry_beta) * entry_t2) * n
                # Maker entry + Maker exit (if TP), otherwise Taker exit
                tx_cost        = notional * COST_MAKER + notional * (COST_MAKER if exit_signal else COST_TAKER)
                holding_days   = (i - entry_bar) / BARS_PER_TRADING_DAY
                short_notional = n * ((abs(entry_beta) * entry_t2) if position == 1 else entry_t1)
                borrow_cost    = short_notional * BORROW_RATE_ANNUAL * holding_days / 252
                size = position_size(
                    pair_name, df.index[entry_bar],
                    sizing_args["regimes"],
                    sizing_args["iv_mult_s"],
                    sizing_args["mc_conf"],
                    sizing_args.get("macro_alert_s"),
                    sizing_args.get("global_hmm_s"),
                ) if sizing_args else IV_SIZE_NORM
                net_pnl        = (gross_pnl - tx_cost - borrow_cost) * size
                cumulative_pnl += net_pnl
                trades.append({
                    "pair":         f"{t1}-{t2}",
                    "entry_time":   df.index[entry_bar],
                    "exit_time":    ts,
                    "direction":    "LONG" if position == 1 else "SHORT",
                    "holding_bars": i - entry_bar,
                    "n_shares":     round(n, 2),
                    "size":         round(size, 4),
                    "gross_pnl":    round(gross_pnl, 4),
                    "tx_cost":      round(tx_cost, 4),
                    "borrow_cost":  round(borrow_cost, 4),
                    "net_pnl":      round(net_pnl, 4),
                    "cum_pnl":      round(cumulative_pnl, 4),
                    "exit_reason":  "STOP" if stop_signal else ("TIME_STOP" if time_stop else "SIGNAL"),
                    "entry_z":      round(df["zscore"].iloc[entry_bar], 2),
                    "exit_z":       round(z, 2),
                })
                position       = 0
                entry_n_shares = 1.0
                if cumulative_pnl < PAIR_MAX_LOSS:
                    break

        # ── Entry gate ────────────────────────────────────────────────────
        if position == 0:
            if oos_start is not None and ts < oos_start:
                diag["pre_oos"] += 1
                continue
            if suspended:
                diag["suspended"] += 1
                continue
            if macro_filter is not None and macro_filter.is_entry_blocked(ts):
                diag["macro_entry_blocked"] += 1
                continue
            if sizing_args:
                sz = position_size(
                    pair_name, ts,
                    sizing_args["regimes"],
                    sizing_args["iv_mult_s"],
                    sizing_args["mc_conf"],
                    sizing_args.get("macro_alert_s"),
                    sizing_args.get("global_hmm_s"),
                )
                if sz < MIN_POSITION_SIZE:
                    diag["size_too_small"] += 1
                    continue

            is_volatile = regime_dict.get(ts, 0) == 1 if regime_dict else False
            if is_volatile and pair_name in _regime_thresholds:
                rt = _regime_thresholds[pair_name]
                threshold   = rt["vol_entry"]
                # Override exit/stop for this bar's entry decision
                exit_thresh_live = rt["vol_exit"]
                stop_thresh_live = rt["vol_stop"]
            else:
                threshold = entry_z
                exit_thresh_live = exit_thresh
                stop_thresh_live = stop_thresh

            if z < -threshold:
                position = 1
            elif z > threshold:
                position = -1
            else:
                diag["no_z_trigger"] += 1

            if position != 0:
                diag["z_trigger"] += 1
                # ── Event blackout (NFP / CPI / FOMC / ECB / BOE) ────────
                if EVENT_FILTER and "event_blackout" in df.columns:
                    if df["event_blackout"].iloc[i]:
                        diag["event_blackout"] += 1
                        position = 0
                        continue

                # ── Session filter ────────────────────────────────────────
                if SESSION_FILTER and session_window is not None:
                    utc_hour = ts.tz_convert("UTC").hour
                    s_start, s_end = session_window
                    if s_start < s_end:
                        in_session = s_start <= utc_hour < s_end
                    else:  # wraps midnight
                        in_session = utc_hour >= s_start or utc_hour < s_end
                    if not in_session:
                        diag["session"] += 1
                        position = 0
                        continue

                # ── Velocity gate: confirm mean reversion direction ────────
                if USE_VELOCITY_GATE and "velocity" in df.columns:
                    vel = df["velocity"].iloc[i]
                    if np.isnan(vel):
                        diag["velocity_nan"] += 1
                        position = 0
                        continue
                    if position == 1 and vel < 0:
                        diag["velocity"] += 1
                        position = 0
                        continue
                    elif position == -1 and vel > 0:
                        diag["velocity"] += 1
                        position = 0
                        continue

                # ── RVOL Gate: Avoid thin markets ─────────────────────────
                if USE_RVOL_GATE and "rvol" in df.columns:
                    rvol_val = df["rvol"].iloc[i]
                    if np.isnan(rvol_val) or rvol_val < RVOL_THRESHOLD:
                        diag["rvol"] += 1
                        position = 0
                        continue

                # ── VWAP MTF: Higher TF fair-value anchor ─────────────────
                if USE_VWAP_MTF and "spread_vwap_mtf" in df.columns:
                    sv_mtf = df["spread_vwap_mtf"].iloc[i]
                    if not np.isnan(sv_mtf):
                        # Don't buy if spread is already above 15m VWAP (expensive)
                        if position == 1 and spread_now > sv_mtf:
                            diag["vwap_mtf"] += 1
                            position = 0
                            continue
                        # Don't sell if spread is already below 15m VWAP (cheap)
                        elif position == -1 and spread_now < sv_mtf:
                            diag["vwap_mtf"] += 1
                            position = 0
                            continue

                # ── MTF: 1H Z-score must agree in direction ───────────────
                if MTF_CONFIRM and "zscore_1h" in df.columns:
                    z1h = df["zscore_1h"].iloc[i]
                    if not np.isnan(z1h) and abs(z1h) >= MTF_Z_MIN:
                        if position == 1 and z1h >= 0:   # 1H not confirming long
                            diag["mtf"] += 1
                            position = 0
                            continue
                        elif position == -1 and z1h <= 0: # 1H not confirming short
                            diag["mtf"] += 1
                            position = 0
                            continue

                # ── RVR: block entry when spread is trending ──────────────
                if RVR_FILTER and "rvr" in df.columns:
                    rvr_val = df["rvr"].iloc[i]
                    if not np.isnan(rvr_val) and rvr_val > RVR_MAX:
                        diag["rvr"] += 1
                        position = 0
                        continue

                # ── Live correlation: require legs still moving together ───
                if LIVE_CORR_FILTER and "corr_live" in df.columns:
                    c_live = df["corr_live"].iloc[i]
                    if not np.isnan(c_live) and c_live < LIVE_CORR_MIN:
                        diag["live_corr"] += 1
                        position = 0
                        continue

                # ── Kalman innovation variance spike ──────────────────────
                # Leading indicator: innov_var spikes before ADF/EG catches
                # the cointegration breakdown. Blocks entries on unstable pairs.
                if KALMAN_INNOV_FILTER and "innov_var_ratio" in df.columns:
                    ivr = df["innov_var_ratio"].iloc[i]
                    if not np.isnan(ivr) and ivr > KALMAN_INNOV_MAX:
                        diag["innov_var"] += 1
                        position = 0
                        continue

                # ── Copula confirmation ───────────────────────────────────
                # Gaussian copula_z must agree in direction with Kalman zscore.
                # Clayton lambda_L gates LONG quality (lower-tail dependence).
                if USE_COPULA and "copula_z" in df.columns:
                    c_z = df["copula_z"].iloc[i]
                    if not np.isnan(c_z) and abs(c_z) >= COPULA_Z_MIN:
                        # copula and Kalman must agree on direction
                        if position == 1 and c_z > -COPULA_Z_MIN:
                            diag["copula_z"] += 1
                            position = 0
                            continue
                        elif position == -1 and c_z < COPULA_Z_MIN:
                            diag["copula_z"] += 1
                            position = 0
                            continue
                    if position == 1 and "lambda_L" in df.columns:
                        l_L = df["lambda_L"].iloc[i]
                        if not np.isnan(l_L) and l_L < COPULA_LAMBDA_MIN:
                            diag["copula_lambda"] += 1
                            position = 0
                            continue

                # ── Tier 2: lazy ADF check on Z-trigger ──────────────────
                # Runs ADF on recent intraday spread — cached per day,
                # so at most one ADF call per pair per trading day.
                if coint_filter is not None:
                    spread_tail = df["spread"].iloc[max(0, i - _LAZY_WINDOW_BARS): i + 1]
                    if not coint_filter.lazy_check(spread_tail, ts):
                        diag["lazy_coint"] += 1
                        position = 0
                        continue

                # ── Tier 3: Hurst drift guard ─────────────────────────────
                # Blocks entry if the spread is trending (H > 0.55),
                # regardless of macro regime.  Catches H1 2021-style
                # structural drift where one leg gets bid up by
                # retail/passive flows while the other is ignored.
                if hurst_filter is not None and spread_daily is not None:
                    d_prev = (ts - pd.Timedelta(days=1)).normalize()
                    # Tail of daily spread up to yesterday
                    h_tail_daily = spread_daily.loc[:d_prev].tail(HURST_ENTRY_WINDOW - 1)
                    # Append today's intraday spread to simulate the full tail
                    h_tail = pd.concat([h_tail_daily, pd.Series({ts: df["spread"].iloc[i]})])
                    
                    h_blocked, h_val = hurst_filter.should_block(h_tail, ts)
                    if h_blocked:
                        hurst_blocked += 1
                        diag["hurst"] += 1
                        position = 0
                        continue

                # Execute at NEXT bar (signal on close i, fill on bar i+1)
                next_i = i + 1
                if next_i >= len(df):
                    diag["no_next_bar"] += 1
                    position = 0
                    continue
                entry_spread   = df["spread"].iloc[next_i]
                entry_t1       = df[t1_col].iloc[next_i]
                entry_t2       = df[t2_col].iloc[next_i]
                entry_beta     = df["beta"].iloc[next_i]
                entry_alpha    = df["alpha"].iloc[next_i]
                entry_std      = df["spread_std"].iloc[next_i]
                entry_bar      = next_i

                # Volatility scaling: n_shares = TARGET_RISK / spread_std
                spread_vol     = float(df["spread_std"].iloc[next_i])
                raw_n          = TARGET_RISK_USD / max(spread_vol, 1e-8)
                # Cap by two-leg notional (both legs combined must not exceed max_notional)
                two_leg        = max(entry_t1 + abs(entry_beta) * entry_t2, 1e-8)
                cap_n          = max_notional / two_leg
                entry_n_shares = min(raw_n, cap_n)

                # Lock in the regime-conditioned thresholds for this trade
                active_exit_thresh = exit_thresh_live
                active_stop_thresh = stop_thresh_live
                diag["entries"] += 1

    out = pd.DataFrame(trades)
    out.attrs["diag"] = dict(diag)
    return out


# ── Load ─────────────────────────────────────────────────────────────────────
closes = load_closes()
pairs_path = Path(_pairs_universe_path())
pairs  = pd.read_csv(pairs_path)
needed_tickers = sorted({t for p in pairs["pair"] for t in str(p).split("-")})

# Load volumes for VW-Z and VWAP prices (both optional — fallback to close/equal-weight)
_volumes: pd.DataFrame | None = None
_vwaps:   pd.DataFrame | None = None

if USE_VWZ:
    try:
        from data_loader import load_volumes
        _volumes = load_volumes(needed_tickers, start=str(closes.index[0].date()))
        _volumes = _volumes.reindex(closes.index)
        print(f"VW-Z enabled  — volumes loaded  ({_volumes.shape[1]} tickers)")
    except Exception as _e:
        print(f"VW-Z: volumes unavailable ({_e}) — using equal-weight Z")

if USE_VWAP:
    try:
        from data_loader import load_vwaps
        _vwaps = load_vwaps(needed_tickers, start=str(closes.index[0].date()))
        _vwaps = _vwaps.reindex(closes.index)
        print(f"VWAP enabled  — vwaps loaded    ({_vwaps.shape[1]} tickers)")
    except Exception as _e:
        print(f"VWAP: vwaps unavailable ({_e}) — using close prices")

# Pre-compute macro event blackout mask for the full intraday index (once, all pairs share it)
_event_blackout_arr: np.ndarray | None = None
if EVENT_FILTER:
    _event_blackout_arr = build_event_blackout(
        closes.index, DATA_DIR,
        bars_before=EVENT_BARS_BEFORE,
        bars_after=EVENT_BARS_AFTER,
        bar_minutes=BAR_MINUTES,
    )
    n_blocked = int(_event_blackout_arr.sum())
    pct       = n_blocked / max(len(closes), 1) * 100
    print(f"Event filter — {n_blocked:,} bars blocked ({pct:.1f}% of history)  "
          f"[{EVENT_BARS_BEFORE} bars before / {EVENT_BARS_AFTER} after]")

# Load session windows per pair (optimal UTC hours from gen_full_sessions.py)
_sessions: dict[str, tuple[int, int]] = {}
if SESSION_FILTER:
    _sess_path = DATA_DIR / "pair_sessions.csv"
    if _sess_path.exists():
        _sess_df = pd.read_csv(_sess_path).set_index("pair")
        for _pair, _row in _sess_df.iterrows():
            _sessions[str(_pair)] = (int(_row["best_start_utc"]), int(_row["best_end_utc"]))
        print(f"Session filter — {len(_sessions)} pair windows loaded")
    else:
        print("Session filter — pair_sessions.csv missing, run gen_full_sessions.py")

# Phase 1: load daily data for rolling cointegration validity
_daily_cache = DATA_DIR / "closes_daily.csv"
_daily: pd.DataFrame | None = None
if _daily_cache.exists():
    _daily = pd.read_csv(_daily_cache, index_col=0, parse_dates=True)
    _daily.index = pd.to_datetime(_daily.index, utc=True)
    print(f"Daily cache loaded for rolling coint check  ({len(_daily)} days)")
else:
    print("No closes_daily.csv — rolling coint check disabled (run step0_download.py)")

# Pre-compute CointegrationFilter per pair (daily EG, O(1) lookup + lazy ADF)
# Skip in return-spread mode: pairs are selected by return correlation, not
# price cointegration, so the daily Johansen/EG filter would falsely suspend them.
_coint_filters: dict[str, CointegrationFilter] = {}
if not USE_RETURN_SPREAD:
    print("Building CointegrationFilters …", end=" ", flush=True)
    for _, _row in pairs.iterrows():
        _t1, _t2 = _row["pair"].split("-")
        _coint_filters[_row["pair"]] = CointegrationFilter(_daily, _t1, _t2)
    print(f"{len(_coint_filters)} pairs")
else:
    print("CointegrationFilter skipped (return-spread mode)")

# Phase 4: load K-Means macro regime
_kmeans_regime = load_kmeans_regime()
if _kmeans_regime is not None:
    sideways_pct = (_kmeans_regime == 1).mean() * 100
    panic_pct    = (_kmeans_regime == 2).mean() * 100
    print(f"K-Means regimes loaded  "
          f"(Sideways {sideways_pct:.0f}%  Panic {panic_pct:.0f}%  "
          f"Trend {100-sideways_pct-panic_pct:.0f}%)")
else:
    print("No kmeans_regimes.csv — K-Means gate disabled (run step5_kmeans.py)")

# Full history for Kalman warmup (train + test, no date split)
# Load dynamic sizing components (step5 + step7 + step8 → step9)
_regimes             = load_regimes()
_vix, _macro_alert_s = load_iv()
_global_hmm_s        = load_global_hmm()
_mc_conf             = load_mc_confidence()

# Build single MacroFilter (shared across all pairs — market-wide signal)
_macro_filter = MacroFilter(_macro_alert_s, _global_hmm_s, _kmeans_regime)
_iv_mult_s           = iv_multiplier_series(_vix)

sizing_args = {
    "regimes":        _regimes,
    "iv_mult_s":      _iv_mult_s,
    "mc_conf":        _mc_conf,
    "macro_alert_s":  _macro_alert_s,
    "global_hmm_s":   _global_hmm_s,
}
layers_active = sum([
    _regimes is not None,
    _vix is not None,
    bool(_mc_conf),
])
print(f"Dynamic sizing: {layers_active}/3 layers active "
      f"(regime={'✓' if _regimes is not None else '✗'}  "
      f"IV={'✓' if _vix is not None else '✗'}  "
      f"MC={'✓' if _mc_conf else '✗'})")

# Load per-pair regimes (from step5_regime.py)
# Format: wide CSV — index=timestamp, columns=pair names, values=0/1
regime_data: dict[str, dict] = {}
regimes_path = DATA_DIR / "regimes.csv"
if regimes_path.exists():
    reg_df = pd.read_csv(regimes_path, index_col=0, parse_dates=True)
    try:
        if reg_df.index.tz is None:
            reg_df.index = reg_df.index.tz_localize("UTC").tz_convert("US/Eastern")
        else:
            reg_df.index = reg_df.index.tz_convert("US/Eastern")
    except AttributeError:
        reg_df.index = pd.to_datetime(reg_df.index, utc=True).tz_convert("US/Eastern")
    for col in reg_df.columns:
        regime_data[col] = reg_df[col].to_dict()
    avg_vol = reg_df.mean().mean() * 100
    print(f"Regimes loaded for {len(regime_data)} pairs "
          f"(avg {avg_vol:.0f}% volatile bars) → entry_z={ENTRY_Z_VOLATILE} when volatile")
else:
    print("No regimes.csv — fixed entry_z (run step5_regime.py to enable HMM filter)")

if pairs.empty:
    raise SystemExit("pairs_selected.csv is empty — run step2_pairs.py first")

# ── Load per-pair optimal params (from step3j WFO or step4d grid search) ──────
_opt_params: dict[str, tuple[float, float, float]] = {}
_wfo_path = DATA_DIR / "wfo_params.csv"
_opt_path = DATA_DIR / "optimal_params.csv"

# When using return-spread mode, WFO params from daily price-spread runs are
# incompatible (different strategy, different units, different half-life).
# Skip them to avoid applying wrong thresholds.
if USE_RETURN_SPREAD:
    print("Return-spread mode — WFO/grid params skipped (use OU Monte Carlo defaults)")
else:
    # Load order: WFO baseline first, then Grid Search overrides per-pair.
    # optimal_params.csv (step3f) takes priority — it's fitted on the current pair
    # universe with the current filter stack, so it's more relevant than old WFO runs.
    if _wfo_path.exists():
        _opt_df = pd.read_csv(_wfo_path)
        for _, _r in _opt_df.sort_values("oos_end").iterrows():
            _opt_params[_r["pair"]] = (float(_r["entry_z"]),
                                       float(_r["exit_z"]),
                                       float(_r["stop_z"]))
        print(f"WFO baseline loaded   (wfo_params.csv):      {len(_opt_params)} pairs")

    if _opt_path.exists():
        _grid_df = pd.read_csv(_opt_path)
        n_before = len(_opt_params)
        for _, _r in _grid_df.iterrows():
            _opt_params[_r["pair"]] = (float(_r["entry_z"]),
                                       float(_r["exit_z"]),
                                       float(_r["stop_z"]))
        n_overridden = len(_grid_df)
        print(f"Grid search override  (optimal_params.csv):  {n_overridden} pairs  "
              f"[priority over WFO]")

if not _opt_params:
    print("No optimized params found — will use OU Monte Carlo defaults.")

# ── Load regime-conditioned thresholds (from regime_profiler.py) ──────────────
_regime_thresholds: dict[str, dict] = {}
_rt_path = DATA_DIR / "regime_thresholds.csv"
if _rt_path.exists():
    _rt_df = pd.read_csv(_rt_path)
    for _, _r in _rt_df[_rt_df["regime"] == 1].iterrows():
        _regime_thresholds[_r["pair"]] = {
            "vol_entry": float(_r["entry_z"]),
            "vol_exit":  float(_r["exit_z"]),
            "vol_stop":  float(_r["stop_z"]),
        }
    print(f"Regime thresholds loaded for {len(_regime_thresholds)} pairs (RCDP volatile)")
else:
    print("No regime_thresholds.csv — using static ENTRY_Z_VOLATILE offset  "
          "(run regime_profiler.py for dynamic thresholds)")

# ── Capital allocation weights per pair ───────────────────────────────────────
def compute_pair_weights(pairs_df: pd.DataFrame,
                         opt_path,
                         method: str = ALLOCATION_METHOD,
                         max_w: float = MAX_PAIR_WEIGHT) -> dict[str, float]:
    """
    Returns normalised weight per pair (sum = 1.0).

    "equal"  — $INITIAL_CAPITAL / n_pairs each (baseline)
    "sharpe" — proportional to max(train_sharpe, 0) from optimal_params.csv,
               capped at MAX_PAIR_WEIGHT to avoid concentration.
               Falls back to equal if train_sharpe unavailable.
    """
    n = len(pairs_df)
    equal = {p: 1.0 / n for p in pairs_df["pair"]}

    if method == "equal" or not opt_path.exists():
        return equal

    opt = pd.read_csv(opt_path)
    if "train_sharpe" not in opt.columns:
        return equal

    raw = {r["pair"]: max(float(r["train_sharpe"]), 0.0) for _, r in opt.iterrows()}

    # A sparse/partially failed optimization file can otherwise assign 0% to
    # selected pairs and silently skip them in the backtest. If every selected
    # pair does not have a positive in-sample Sharpe, keep the universe alive
    # with equal weights and let the trade filters decide.
    if any(raw.get(p, 0.0) <= 0.0 for p in pairs_df["pair"]):
        return equal

    # If all Sharpes are ≤ 0 (bad training data), fall back to equal
    total = sum(raw.values())
    if total <= 0:
        return equal

    # Normalise, then apply cap iteratively (excess redistributed to others)
    weights = {p: raw.get(p, 0.0) / total for p in pairs_df["pair"]}
    for _ in range(20):   # iterative cap: redistribute excess
        over   = {p: w for p, w in weights.items() if w > max_w}
        if not over:
            break
        excess = sum(w - max_w for w in over.values())
        under  = {p: w for p, w in weights.items() if w < max_w}
        total_under = sum(under.values()) or 1.0
        for p in over:
            weights[p] = max_w
        for p in under:
            weights[p] += excess * (weights[p] / total_under)

    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {p: w / total_weight for p, w in weights.items()}

    return weights


def zero_account_report(df_trades: pd.DataFrame,
                        initial_capital: float) -> dict[str, object]:
    """Approximate FundingPips Zero compliance from realized trade exits."""
    if df_trades.empty:
        return {
            "trading_days": 0,
            "max_inactive_days": None,
            "trailing_loss_breached": False,
            "daily_loss_breached": False,
            "max_trade_days_30d": 0,
            "max_profit_days_30d": 0,
            "max_30d_pnl": 0.0,
        }

    trades = df_trades.copy()
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True).dt.tz_convert("US/Eastern")
    trades = trades.sort_values("exit_time")
    trades["trade_day"] = trades["exit_time"].dt.normalize()

    daily_pnl = trades.groupby("trade_day")["dollar_pnl"].sum().sort_index()
    all_days = pd.date_range(daily_pnl.index.min(), daily_pnl.index.max(), freq="D", tz=daily_pnl.index.tz)
    daily_pnl = daily_pnl.reindex(all_days, fill_value=0.0)
    daily_equity = initial_capital + daily_pnl.cumsum()

    # Daily loss limit: 3% of day-start equity, checked on each realized trade exit.
    daily_loss_breached = False
    trailing_loss_breached = False
    peak_equity = initial_capital
    prev_day = None
    day_start_equity = initial_capital
    intraday_equity = initial_capital

    for _, row in trades.iterrows():
        day = row["trade_day"]
        if prev_day is None or day != prev_day:
            day_start_equity = intraday_equity
        intraday_equity += float(row["dollar_pnl"])
        peak_equity = max(peak_equity, intraday_equity)

        daily_floor = day_start_equity * (1 - ZERO_MAX_DAILY_LOSS_PCT)
        if intraday_equity < daily_floor:
            daily_loss_breached = True

        if peak_equity < initial_capital * 1.05:
            trailing_floor = peak_equity - initial_capital * ZERO_MAX_TRAILING_LOSS_PCT
        else:
            trailing_floor = initial_capital
        if intraday_equity < trailing_floor:
            trailing_loss_breached = True

        prev_day = day

    trade_day = daily_pnl != 0
    profit_day = daily_pnl > 0
    rolling_trade_days = trade_day.rolling(30, min_periods=30).sum()
    rolling_profit_days = profit_day.rolling(30, min_periods=30).sum()
    trade_max = rolling_trade_days.max()
    profit_max = rolling_profit_days.max()
    pnl_max = daily_pnl.rolling(30, min_periods=30).sum().max()
    max_trade_days_30d = int(trade_max) if pd.notna(trade_max) else 0
    max_profit_days_30d = int(profit_max) if pd.notna(profit_max) else 0
    max_30d_pnl = float(pnl_max) if pd.notna(pnl_max) else 0.0

    trade_days = trades["trade_day"].drop_duplicates().sort_values()
    inactive_days = None
    if len(trade_days) >= 2:
        gaps = trade_days.diff().dt.days.dropna()
        inactive_days = int(gaps.max() - 1) if not gaps.empty else 0
    elif len(trade_days) == 1:
        inactive_days = 0

    return {
        "trading_days": int(len(trade_days)),
        "max_inactive_days": inactive_days,
        "trailing_loss_breached": bool(trailing_loss_breached),
        "daily_loss_breached": bool(daily_loss_breached),
        "max_trade_days_30d": int(max_trade_days_30d),
        "max_profit_days_30d": int(max_profit_days_30d),
        "max_30d_pnl": round(max_30d_pnl, 4),
    }


def zero_universe_ok(zero: dict[str, object]) -> bool:
    """Universe curation gate for Zero profile."""
    return (
        int(zero.get("max_trade_days_30d", 0)) >= ZERO_MIN_PROFIT_DAYS_30D
        and int(zero.get("max_profit_days_30d", 0)) >= ZERO_MIN_PROFIT_DAYS_30D
    )

_pair_weights = compute_pair_weights(pairs, _opt_path)
print(f"\nCapital allocation  (method={ALLOCATION_METHOD}, leverage={LEVERAGE}x):")
for _p, _w in sorted(_pair_weights.items(), key=lambda x: -x[1]):
    margin   = _w * INITIAL_CAPITAL
    notional = margin * LEVERAGE
    print(f"  {_p:<12}  {_w*100:5.1f}%  margin=${margin:,.0f}  notional=${notional:,.0f}")
print()

# ── OOS start date — read from pairs_selected.csv (set by pairs.py via TRAIN_RATIO)
# Kalman warms up on full history; trading begins only from this date.
_oos_start: pd.Timestamp | None = None
if "test_start_date" in pairs.columns:
    _oos_start = pd.Timestamp(pairs["test_start_date"].iloc[0]).tz_localize("US/Eastern")
    print(f"OOS start: {_oos_start.date()}  (Kalman warms up on full history before this)")
else:
    print("WARNING: test_start_date not in pairs_selected.csv — running on full period (in-sample!)")

print(f"Trading {len(pairs)} pairs | {closes.shape[0]} bars per ticker")
print(f"Full data: {closes.index[0]} — {closes.index[-1]}")
if _oos_start:
    oos_bars = (closes.index >= _oos_start).sum()
    print(f"OOS bars: {oos_bars} / {len(closes.index)}  ({oos_bars/len(closes.index)*100:.0f}% of total)")
print(f"Pair max loss cutoff: {PAIR_MAX_LOSS}\n")

# ── Run backtest per pair ─────────────────────────────────────────────────────
pair_results = {}
zero_universe_rows: list[dict[str, object]] = []

for _, row in pairs.iterrows():
    t1, t2    = row["pair"].split("-")
    beta      = float(row.get("beta_daily", row["beta"]) or row["beta"])
    half_life = row["half_life_bars"]

    if t1 not in closes.columns or t2 not in closes.columns:
        print(f"  SKIP {row['pair']}: missing ticker data")
        continue

    pair_weight = _pair_weights.get(row["pair"], 0.0)
    if pair_weight == 0.0:
        print(f"  SKIP {row['pair']}: 0% capital allocation (train Sharpe ≤ 0)")
        continue

    df_sig = build_signals(closes, t1, t2, beta, half_life,
                           volumes=_volumes, vwaps=_vwaps)

    # Attach event blackout column (aligned by position in closes.index)
    if EVENT_FILTER and _event_blackout_arr is not None:
        sig_pos = closes.index.get_indexer(df_sig.index, method="nearest")
        df_sig["event_blackout"] = _event_blackout_arr[sig_pos]

    spread_daily = df_sig["spread"].resample('D').last().dropna()

    # Use per-pair optimal params if available; otherwise fall back to config defaults.
    # In return-spread mode: skip MC optimization entirely — OU params computed on
    # Kalman innovations don't reflect the return-spread half-life correctly, and MC
    # can produce negative exit thresholds or unreachable entry levels.
    if row["pair"] in _opt_params:
        opt_entry, opt_exit, opt_stop = _opt_params[row["pair"]]
        src = "grid"
    elif USE_RETURN_SPREAD:
        opt_entry, opt_exit, opt_stop = ENTRY_Z, EXIT_Z, STOP_Z
        src = "cfg"
    else:
        theta_ou     = np.log(2) / max(float(half_life), 1.0)
        sigma_roll   = float(df_sig["spread"].std())
        avg_notional = float(closes[t1].mean() + beta * closes[t2].mean())
        opt_entry, opt_exit, opt_stop = optimal_thresholds(
            theta_ou, sigma_roll, avg_notional)
        src = "MC"

    # Microstructure floor + global minimum entry_z (price-spread only; skip for return-spread)
    if not USE_RETURN_SPREAD:
        sigma_spread = float(df_sig["spread"].std())
        avg_notional = float(closes[t1].mean() + abs(beta) * closes[t2].mean())
        z_floor = max(min_viable_entry_z(sigma_spread, avg_notional, opt_exit), ENTRY_Z_MIN)
        if z_floor > opt_entry:
            tag = "micro" if min_viable_entry_z(sigma_spread, avg_notional, opt_exit) >= ENTRY_Z_MIN else "floor"
            print(f"  {row['pair']:12s}  [{src}→{tag}]  "
                  f"entry {opt_entry}→{z_floor:.2f}  exit={opt_exit:+.1f}  stop={opt_stop}")
            opt_entry = round(z_floor, 2)
        else:
            print(f"  {row['pair']:12s}  [{src}]  "
                  f"entry={opt_entry}  exit={opt_exit:+.1f}  stop={opt_stop}")
    else:
        print(f"  {row['pair']:12s}  [{src}]  "
              f"entry={opt_entry}  exit={opt_exit:+.1f}  stop={opt_stop}")

    avg_notional  = float(closes[t1].mean() + abs(beta) * closes[t2].mean())
    pair_regime   = regime_data.get(row["pair"])
    pair_weight   = _pair_weights.get(row["pair"], 1.0 / len(pairs))
    pair_max_notl = pair_weight * INITIAL_CAPITAL * LEVERAGE
    pair_half_life  = int(row.get("half_life_bars", 200))
    pair_hurst_f    = HurstFilter()   # fresh cache per pair
    # Return-spread pairs are selected by return correlation, not price cointegration.
    # The daily Johansen filter will falsely suspend them — skip it.
    _coint_f = None if USE_RETURN_SPREAD else _coint_filters.get(row["pair"])
    trades = backtest_pair(df_sig, t1, t2, beta,
                           pair_name=row["pair"],
                           regime_dict=pair_regime,
                           sizing_args=sizing_args,
                           entry_z=opt_entry,
                           exit_thresh=opt_exit,
                           stop_thresh=opt_stop,
                           coint_filter=_coint_f,
                           macro_filter=_macro_filter,
                           hurst_filter=pair_hurst_f,
                           spread_daily=spread_daily,
                           oos_start=_oos_start,
                           max_notional=pair_max_notl,
                           max_hold_bars=pair_half_life * 2,
                           session_window=_sessions.get(row["pair"]))
    diag = trades.attrs.get("diag", {})
    top_blocks = ", ".join(
        f"{k}={v}" for k, v in sorted(diag.items(), key=lambda kv: -kv[1])[:5]
        if k not in {"no_z_trigger", "pre_oos"}
    )

    if trades.empty:
        detail = f"  blocks: {top_blocks}" if top_blocks else ""
        print(f"  {row['pair']:12s}  0 trades{detail}")
        continue

    # net_pnl is already dollar-denominated: n_shares = max_notional / (price1 + beta*price2)
    # so gross_pnl = spread_change * n_shares = return * notional = dollars directly.
    # The old u = capital_now / notional_now formula was double-converting the units.
    dollar_pnls, dollar_grosses, dollar_costses, units_list = [], [], [], []

    for idx in range(len(trades)):
        gp  = trades["gross_pnl"].iloc[idx]
        tc  = trades["tx_cost"].iloc[idx]
        bc  = trades["borrow_cost"].iloc[idx]
        np_ = trades["net_pnl"].iloc[idx]

        dollar_pnls.append(round(np_, 2))
        dollar_grosses.append(round(gp, 2))
        dollar_costses.append(round(tc + bc, 2))
        units_list.append(trades["n_shares"].iloc[idx])

    trades["dollar_pnl"]     = dollar_pnls
    trades["dollar_gross"]   = dollar_grosses
    trades["dollar_costs"]   = dollar_costses
    trades["units_per_pair"] = units_list

    pnl      = trades["net_pnl"]
    win_rate = (pnl > 0).mean() * 100
    disabled = trades["cum_pnl"].iloc[-1] < PAIR_MAX_LOSS
    status   = " [DISABLED — max loss hit]" if disabled else ""

    zero = zero_account_report(trades, INITIAL_CAPITAL)
    zero_ok = zero_universe_ok(zero)
    zero_universe_rows.append({
        **row.to_dict(),
        "pair": row["pair"],
        "net_pnl": round(float(trades["dollar_pnl"].sum()), 4),
        "trades": int(len(trades)),
        "win_rate": round(float(win_rate), 1),
        "max_trade_days_30d": int(zero["max_trade_days_30d"]),
        "max_profit_days_30d": int(zero["max_profit_days_30d"]),
        "max_30d_pnl": float(zero["max_30d_pnl"]),
        "max_inactive_days": zero["max_inactive_days"],
        "zero_universe_ok": bool(zero_ok),
    })

    if ACCOUNT_MODEL == "Zero" and not zero_ok:
        print(f"  SKIP {row['pair']}: Zero universe filter "
              f"(trade30={zero['max_trade_days_30d']}, profit30={zero['max_profit_days_30d']}, "
              f"gap={zero['max_inactive_days']})")
        continue

    pair_results[row["pair"]] = {"trades": trades, "signals": df_sig}
    hurst_blk = sum(1 for v in pair_hurst_f._cache.values() if v[0])
    hurst_tag = f"  H_blk={hurst_blk}" if hurst_blk > 0 else ""
    diag_tag = f"  blocks: {top_blocks}" if top_blocks else ""
    print(f"  {row['pair']:12s}  trades={len(trades):3d}  "
          f"WR={win_rate:4.1f}%  net P&L={pnl.sum():+.4f}{status}{hurst_tag}{diag_tag}")

if ACCOUNT_MODEL == "Zero":
    zero_df = pd.DataFrame([r for r in zero_universe_rows if r["zero_universe_ok"]])
    zero_path = DATA_DIR / "pairs_zero_universe.csv"
    zero_cols = [c for c in [
            "pair", "t1", "t2", "corr", "beta", "hurst", "half_life_bars",
            "joh_trace", "joh_crit_95", "joh_margin", "net_pnl", "trades",
            "win_rate", "max_trade_days_30d", "max_profit_days_30d",
            "max_30d_pnl", "max_inactive_days",
    ] if c in (zero_df.columns if not zero_df.empty else zero_universe_rows[0].keys() if zero_universe_rows else [])]
    if zero_df.empty:
        pd.DataFrame(columns=zero_cols).to_csv(zero_path, index=False)
        print(f"\nZero universe saved: 0 pairs -> {zero_path}")
    else:
        zero_df = zero_df[zero_cols].sort_values(["net_pnl", "max_30d_pnl"], ascending=False)
        zero_df.to_csv(zero_path, index=False)
        print(f"\nZero universe saved: {len(zero_df)} pairs -> {zero_path}")

if not pair_results:
    raise SystemExit("No trades generated.")

# ── Combine ───────────────────────────────────────────────────────────────────
df_trades = (pd.concat([v["trades"] for v in pair_results.values()])
               .sort_values("exit_time")
               .reset_index(drop=True))

pnl             = df_trades["net_pnl"]
winning         = df_trades[pnl > 0]
losing          = df_trades[pnl <= 0]
stops           = df_trades[df_trades["exit_reason"] == "STOP"]
coint_breaks    = df_trades[df_trades["exit_reason"] == "COINT_BREAK"]
panic_exits     = df_trades[df_trades["exit_reason"] == "PANIC"]
cumulative      = pnl.cumsum()
max_drawdown    = (cumulative - cumulative.cummax()).min()
days_total      = pd.to_datetime(df_trades["exit_time"].iloc[-1]) - pd.to_datetime(df_trades["exit_time"].iloc[0])
trades_per_year = len(df_trades) / max(days_total.days / 365.25, 1 / 365.25)
sharpe          = (pnl.mean() / pnl.std() * np.sqrt(trades_per_year)
                   if pnl.std() > 0 else 0.0)
profit_factor   = (winning["net_pnl"].sum() / abs(losing["net_pnl"].sum())
                   if len(losing) > 0 and losing["net_pnl"].sum() != 0 else float("inf"))

# ── Dollar P&L summary ───────────────────────────────────────────────────────
dollar_net      = df_trades["dollar_pnl"].sum()
dollar_gross    = df_trades["dollar_gross"].sum()
dollar_costs    = df_trades["dollar_costs"].sum()
final_balance   = INITIAL_CAPITAL + dollar_net
total_return    = dollar_net / INITIAL_CAPITAL * 100
dollar_drawdown = (df_trades["dollar_pnl"].cumsum()
                   - df_trades["dollar_pnl"].cumsum().cummax()).min()
avg_dollar_trade = df_trades["dollar_pnl"].mean()

print(f"\n{'='*60}")
print(f"PORTFOLIO  ({len(pair_results)} pairs)  —  ${INITIAL_CAPITAL:,.0f} starting capital")
print(f"{'='*60}")
print(f"Trades:        {len(df_trades)}  ({trades_per_year:.0f}/yr)")
print(f"Win rate:      {len(winning)/len(df_trades)*100:.1f}%")
print(f"Stops:         {len(stops)}")
print(f"Coint breaks:  {len(coint_breaks)}")
print(f"Panic exits:   {len(panic_exits)}")
print(f"")
print(f"Gross P&L:     ${dollar_gross:>+8.2f}   ({df_trades['gross_pnl'].sum():+.4f} spread units)")
print(f"Costs:         ${dollar_costs:>8.2f}   ({(df_trades['tx_cost']+df_trades['borrow_cost']).sum():.4f} spread units)")
print(f"Net P&L:       ${dollar_net:>+8.2f}   ({pnl.sum():+.4f} spread units)")
print(f"")
print(f"Starting:      ${INITIAL_CAPITAL:>8,.2f}")
print(f"Final balance: ${final_balance:>8,.2f}")
print(f"Total return:  {total_return:>+7.2f}%")
print(f"")
print(f"Avg trade:     ${avg_dollar_trade:>+7.2f}   ({pnl.mean():+.4f} spread units)")
print(f"Profit factor: {profit_factor:.2f}")
print(f"Max drawdown:  ${dollar_drawdown:>8.2f}   ({max_drawdown:.4f} spread units)")
print(f"Sharpe:        {sharpe:.2f}")
print(f"Avg hold:      {df_trades['holding_bars'].mean():.0f} bars "
      f"({df_trades['holding_bars'].mean()/BARS_PER_TRADING_DAY:.1f} days)")

zero = zero_account_report(df_trades, INITIAL_CAPITAL)
print(f"\n{'='*60}")
print("FUNDINGPIPS ZERO CHECK (realized-exit approximation)")
print(f"{'='*60}")
print(f"Account size:      ${INITIAL_CAPITAL:,.0f}")
print(f"Trailing loss 5%:   {'BREACH' if zero['trailing_loss_breached'] else 'OK'}")
print(f"Daily loss 3%:      {'BREACH' if zero['daily_loss_breached'] else 'OK'}")
print(f"30d trade days:     max {zero['max_trade_days_30d']} in any 30d window "
      f"(need {ZERO_MIN_PROFIT_DAYS_30D})")
print(f"30d profit days:    max {zero['max_profit_days_30d']} in any 30d window "
      f"(need {ZERO_MIN_PROFIT_DAYS_30D})")
if zero["max_inactive_days"] is not None:
    print(f"Max inactivity gap: {zero['max_inactive_days']} days "
          f"(limit {ZERO_MAX_INACTIVE_DAYS})")
    print(f"Inactivity rule:    {'OK' if zero['max_inactive_days'] <= ZERO_MAX_INACTIVE_DAYS else 'FAIL'}")

print(f"\n{'─'*75}")
print(f"{'Pair':<12} {'Trades':>6} {'WR':>6} {'Net $':>9} {'Net P&L':>10} {'Sharpe':>7} {'AvgHold':>8} {'Status':>10}")
print(f"{'─'*75}")
for pair_name, data in pair_results.items():
    t  = data["trades"]
    p  = t["net_pnl"]
    wr = (p > 0).mean() * 100
    tpy = len(t) / max(days_total.days / 365.25, 1 / 365.25)
    sh  = p.mean() / p.std() * np.sqrt(tpy) if p.std() > 0 else 0.0
    ah  = t["holding_bars"].mean() / BARS_PER_TRADING_DAY
    disabled  = t["cum_pnl"].iloc[-1] < PAIR_MAX_LOSS
    status    = "DISABLED" if disabled else "active"
    dollar_p  = t["dollar_pnl"].sum() if "dollar_pnl" in t.columns else 0.0
    print(f"{pair_name:<12} {len(t):>6} {wr:>5.1f}% {dollar_p:>+8.2f}$ {p.sum():>+10.4f} "
          f"{sh:>7.2f} {ah:>6.1f}d {status:>10}")

df_trades.to_csv(DATA_DIR / "trades.csv", index=False)
print(f"\nSaved {len(df_trades)} trades to {DATA_DIR / 'trades.csv'}")

# ── SPY benchmark ─────────────────────────────────────────────────────────────
spy_return = None
spy_sharpe = None
try:
    test_start = closes.index[0].tz_convert("UTC").tz_localize(None)
    test_end   = closes.index[-1].tz_convert("UTC").tz_localize(None)
    daily_path = DATA_DIR / "closes_daily.csv"
    if daily_path.exists():
        daily = pd.read_csv(daily_path, index_col=0, parse_dates=True)
        if "spxusd" in daily.columns:
            spy_close = daily["spxusd"].dropna()
            spy_close.index = pd.to_datetime(spy_close.index).tz_localize(None)
            spy_close = spy_close[(spy_close.index >= test_start) & (spy_close.index <= test_end)]
        else:
            spy_close = pd.Series(dtype=float)
    elif os.getenv("ALLOW_NETWORK") == "1":
        import yfinance as yf
        spy_raw = yf.download("SPY", start=test_start, end=test_end,
                              interval="1d", progress=False, auto_adjust=True)
        spy_close = spy_raw["Close"].squeeze().dropna()
    else:
        spy_close = pd.Series(dtype=float)

    if len(spy_close) > 5:
        spy_ret        = spy_close.pct_change().dropna()
        spy_cum        = (1 + spy_ret).cumprod()
        spy_return     = float(spy_cum.iloc[-1] - 1) * 100
        spy_anndays    = (spy_close.index[-1] - spy_close.index[0]).days
        spy_sharpe     = (spy_ret.mean() / spy_ret.std() * np.sqrt(252)
                          if spy_ret.std() > 0 else 0)
        print(f"\nSPY benchmark ({spy_close.index[0].date()} → {spy_close.index[-1].date()}):")
        print(f"  Return: {spy_return:+.1f}%  |  Sharpe: {spy_sharpe:.2f}")
    else:
        print("\nSPY: insufficient data (yfinance returned < 5 bars)")
except Exception as e:
    print(f"\nSPY benchmark unavailable: {e}")

# ── OOS summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("OUT-OF-SAMPLE COMPARISON")
print(f"{'='*60}")
test_period = (pd.to_datetime(df_trades['exit_time'].iloc[-1])
               - pd.to_datetime(df_trades['exit_time'].iloc[0])).days
_oos_display = _oos_start.date() if _oos_start else closes.index[0].date()
print(f"Test period:       {_oos_display} → {closes.index[-1].date()} "
      f"({test_period} days)")
print(f"Strategy  Sharpe:  {sharpe:.2f}")
print(f"Strategy  Net P&L: {pnl.sum():+.4f} (spread units)")
if spy_return is not None:
    print(f"SPY       Return:  {spy_return:+.1f}%")
    print(f"SPY       Sharpe:  {spy_sharpe:.2f}")
    alpha = sharpe - spy_sharpe
    print(f"Alpha (Sharpe):    {alpha:+.2f}")

# ── Charts ────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)
exit_times = pd.to_datetime(df_trades["exit_time"])

n_rows = 3 if spy_return is not None else 2
fig, axes = plt.subplots(n_rows, 1, figsize=(14, 5 * n_rows))

ax = axes[0]
ax.plot(exit_times, cumulative.values, color="blue", lw=2, label="Portfolio net P&L")
ax.plot(exit_times, df_trades["gross_pnl"].cumsum().values,
        color="blue", lw=1, linestyle="--", alpha=0.35, label="Gross P&L")
ax.axhline(PAIR_MAX_LOSS, color="red", linestyle=":", lw=1, alpha=0.5,
           label=f"Max loss cutoff ({PAIR_MAX_LOSS})")
ax.axhline(0, color="black", lw=0.8)
ax.set_title(f"Portfolio Equity Curve  [OUT-OF-SAMPLE: "
             f"{closes.index[0].date()} → {closes.index[-1].date()}]")
ax.set_ylabel("Cumulative net P&L")
ax.legend()

ax = axes[1]
colors = plt.cm.tab10(np.linspace(0, 1, len(pair_results)))
for (pair_name, data), color in zip(pair_results.items(), colors):
    t  = data["trades"]
    et = pd.to_datetime(t["exit_time"])
    ax.plot(et, t["net_pnl"].cumsum().values, label=pair_name, color=color, lw=1.5)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Per-pair Equity Curves")
ax.set_ylabel("Cumulative net P&L")
ax.legend(fontsize=8)

if spy_return is not None and n_rows == 3:
    ax = axes[2]
    ax2 = ax.twinx()

    # Strategy: normalise cumulative to % starting from 0
    first_trade_val = cumulative.values[0]
    strat_norm = (cumulative.values - first_trade_val) / max(abs(first_trade_val), 1) * 100

    ax.plot(exit_times, strat_norm, color="blue", lw=2, label="Strategy (normalised %)")
    ax2.plot(spy_cum.index, (spy_cum.values - 1) * 100, color="orange",
             lw=2, linestyle="--", label=f"SPY buy & hold")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Strategy return (%)", color="blue")
    ax2.set_ylabel("SPY return (%)", color="orange")
    ax.set_title(f"Strategy vs SPY  |  Strategy Sharpe={sharpe:.2f}  "
                 f"SPY Sharpe={spy_sharpe:.2f}")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "backtest_results.png", dpi=150)
print(f"Chart saved to {OUTPUT_DIR / 'backtest_results.png'}")

# ── Z-score debug plots per pair ──────────────────────────────────────────────
try:
    from step5f_debug_plot import plot_zscore_debug
    print("\nGenerating Z-score debug plots …")
    for pair_name, data in pair_results.items():
        t1, t2 = pair_name.split("-")
        ez, xz, sz = _opt_params.get(pair_name, (ENTRY_Z, EXIT_Z, STOP_Z))
        plot_zscore_debug(
            data["signals"], data["trades"],
            pair_name, entry_z=ez, exit_z=xz, stop_z=sz,
        )
except Exception as _e:
    print(f"Debug plots skipped: {_e}")
