"""
filters.py — CointegrationFilter and MacroFilter for backtest_pair.

CointegrationFilter
    Tier 1 (daily):   pre-computed EG p-value on daily closes, sampled every
                      COINT_RECHECK_DAYS days and forward-filled. O(1) lookup.
    Tier 2 (lazy):    ADF on recent intraday spread, triggered only when
                      |z| >= entry threshold. Cached per calendar day —
                      at most one ADF call per pair per day.

MacroFilter
    is_entry_blocked: VIX9D backwardation (macro_alert) OR
                      K-Means != Sideways OR global SPY HMM panic
    is_force_close:   K-Means Panic (regime 2)
    force_close_reason: label written to trade record
"""

import pandas as pd
import numpy as np
from datetime import date as _date
from statsmodels.tsa.stattools import coint, adfuller
from scipy.stats import gaussian_kde

from config import (
    COINT_WINDOW_DAYS, COINT_BREAK_P, COINT_RECHECK_DAYS,
    BARS_PER_DAY,
    HURST_ENTRY_WINDOW, HURST_ENTRY_SOFT_MAX, HURST_ENTRY_MAX,
)

# Intraday window for lazy ADF: same calendar span as daily pre-compute
_LAZY_WINDOW_BARS = COINT_WINDOW_DAYS * BARS_PER_DAY   # 90d × 26 bars = 2 340


# ── CointegrationFilter ───────────────────────────────────────────────────────

class CointegrationFilter:
    """
    Usage:
        cf = CointegrationFilter(daily_df, "JPM", "BAC")
        if not cf.is_valid(today):          # daily pre-computed check
            ...
        if not cf.lazy_check(spread_tail, today):   # intraday ADF on trigger
            ...
    """

    def __init__(self, daily_df: pd.DataFrame | None, t1: str, t2: str,
                 window: int = COINT_WINDOW_DAYS,
                 p_thresh: float = COINT_BREAK_P,
                 step: int = COINT_RECHECK_DAYS):
        self._p_thresh   = p_thresh
        self._daily_dict: dict[_date, bool] = {}
        self._lazy_cache: dict[_date, bool] = {}

        if daily_df is None:
            return
        if t1 not in daily_df.columns or t2 not in daily_df.columns:
            return

        pc = daily_df[[t1, t2]].dropna()
        if len(pc) < window:
            return

        sampled: dict = {}
        for i in range(window, len(pc) + 1, step):
            chunk = pc.iloc[i - window:i]
            try:
                _, pval, _ = coint(chunk[t1], chunk[t2])
                sampled[pc.index[i - 1]] = pval < p_thresh
            except Exception:
                sampled[pc.index[i - 1]] = False

        if sampled:
            s = pd.Series(sampled).reindex(pc.index).ffill().fillna(True)
            self._daily_dict = {
                (ts.date() if hasattr(ts, "date") else ts): bool(v)
                for ts, v in s.items()
            }

    # ── Tier 1: O(1) daily lookup ─────────────────────────────────────────
    def is_valid(self, d) -> bool:
        """True = cointegrated per daily pre-compute. Defaults to True if no data."""
        key = d if isinstance(d, _date) else pd.Timestamp(d).date()
        return self._daily_dict.get(key, True)

    # ── Tier 2: lazy intraday ADF on Z-trigger ────────────────────────────
    def lazy_check(self, spread_tail: pd.Series, d) -> bool:
        """
        Run ADF on the most recent intraday spread window.
        Returns True = spread still stationary (OK to enter).
        Cached per calendar day — runs at most once per pair per day.
        Call only when |z| >= entry threshold to keep the hot loop fast.
        """
        key = d if isinstance(d, _date) else pd.Timestamp(d).date()
        if key in self._lazy_cache:
            return self._lazy_cache[key]

        s = spread_tail.dropna()
        if len(s) < 60:
            self._lazy_cache[key] = True
            return True

        try:
            _, pval, _, _, _, _ = adfuller(s.values, maxlag=1,
                                           regression="c", autolag=None)
            ok = pval < self._p_thresh
        except Exception:
            ok = True   # be permissive on numerical failure

        self._lazy_cache[key] = ok
        return ok


# ── MacroFilter ───────────────────────────────────────────────────────────────

class MacroFilter:
    """
    Unified macro regime decision gate.

    Entry blocking (is_entry_blocked):
        — VIX9D > VIX (backwardation) from iv.py  → macro_alert_s
        — K-Means regime != 1 (Sideways)           → kmeans_series
        — Global SPY HMM panic                    → global_hmm_s

    Force-close (is_force_close):
        — K-Means regime == 2 (Panic)

    All three input Series are optional; missing → conservative default
    (no block, no force-close).
    """

    def __init__(self,
                 macro_alert_s: pd.Series | None,   # VIX9D backwardation (0/1)
                 global_hmm_s:  pd.Series | None,   # global HMM on SPY (0/1)
                 kmeans_series: pd.Series | None):  # K-Means regime (0/1/2)
        self._alert: dict[_date, bool] = {}
        self._hmm:   dict[_date, int]  = {}
        self._km:    dict[_date, int]  = {}

        if macro_alert_s is not None and not macro_alert_s.empty:
            for ts, v in macro_alert_s.items():
                self._alert[_to_date(ts)] = bool(v)

        if global_hmm_s is not None and not global_hmm_s.empty:
            for ts, v in global_hmm_s.items():
                self._hmm[_to_date(ts)] = int(v)

        if kmeans_series is not None and not kmeans_series.empty:
            for ts, v in kmeans_series.items():
                self._km[_to_date(ts)] = int(v)

    # ── Public API ────────────────────────────────────────────────────────
    def is_entry_blocked(self, ts) -> bool:
        """True → do not open new positions on this bar.

        K-Means gate is a hard block only in Panic=2. Trend=0 stays tradable
        for FX stat-arb and is handled via a soft size multiplier instead.
        """
        d = _to_date(ts)
        if self._alert.get(d, False):          # VIX9D backwardation
            return True
        if self._km and self._km.get(d, 1) == 2:  # K-Means: only Panic=2 blocks
            return True
        if self._hmm.get(d, 0) == 1:           # global SPY HMM panic
            return True
        return False

    def km_size_multiplier(self, ts) -> float:
        """Soft position size scaling by K-Means regime."""
        d = _to_date(ts)
        regime = self._km.get(d, 1)
        if regime == 1:
            return 1.0
        if regime == 0:
            return 0.5
        return 0.0

    def is_force_close(self, ts) -> bool:
        """True → immediately close any open position on this bar."""
        d = _to_date(ts)
        if self._km.get(d, 1) == 2:   # K-Means Panic
            return True
        return False

    def force_close_reason(self, ts) -> str:
        """Human-readable exit reason for the trade record."""
        d = _to_date(ts)
        if self._km.get(d, 1) == 2:
            return "PANIC"
        return "FORCE_CLOSE"


# ── HurstFilter ───────────────────────────────────────────────────────────────

def get_hurst_multiplier(h_val: float) -> float:
    """Soft size penalty in the guarded Hurst band."""
    if h_val <= HURST_ENTRY_SOFT_MAX:
        return 1.0
    if h_val >= HURST_ENTRY_MAX:
        return 0.0

    penalty = (h_val - HURST_ENTRY_SOFT_MAX) / (HURST_ENTRY_MAX - HURST_ENTRY_SOFT_MAX)
    return max(0.3, 1.0 - penalty * 0.7)

class HurstFilter:
    """
    Fast Vectorized Hurst Exponent (Variance Ratio Proxy).
    H \approx 0.5 * [log(Var(P_t - P_{t-tau})) / log(tau)]

    Purpose: catch structural drift where the spread is trending.
    """

    def __init__(self, h_soft_max: float = HURST_ENTRY_SOFT_MAX,
                 h_max: float = HURST_ENTRY_MAX,
                 window: int = HURST_ENTRY_WINDOW,
                 hurst_lag: int = 10):
        self._h_soft_max = h_soft_max
        self._h_max  = h_max
        self._window = window
        self._hurst_lag = hurst_lag
        self._cache: dict[_date, tuple[bool, float, bool]] = {}

    def compute_hurst(self, spread_series: pd.Series) -> pd.Series:
        if len(spread_series) < self._window + self._hurst_lag:
            return pd.Series(0.5, index=spread_series.index)
            
        diff_1 = spread_series.diff(1)
        diff_tau = spread_series.diff(self._hurst_lag)
        
        var_1 = diff_1.rolling(window=self._window).var()
        var_tau = diff_tau.rolling(window=self._window).var()
        
        var_1 = var_1.replace(0, np.nan)
        hurst = 0.5 * (np.log(var_tau / var_1) / np.log(self._hurst_lag))
        return hurst.fillna(0.5)

    def should_block(self, spread_tail: pd.Series, ts) -> tuple[bool, float, bool]:
        """Returns (should_block, hurst_value, guarded_mode). Cached per day."""
        key = _to_date(ts)
        if key in self._cache:
            return self._cache[key]

        s = spread_tail.dropna()
        if len(s) < self._window + self._hurst_lag:
            self._cache[key] = (False, 0.5, False)
            return False, 0.5, False

        # Calculate only the last value for performance in hot loop
        diff_1 = s.diff(1).tail(self._window)
        diff_tau = s.diff(self._hurst_lag).tail(self._window)
        
        v1 = diff_1.var()
        vt = diff_tau.var()
        
        if v1 == 0 or np.isnan(v1):
            h = 0.5
        else:
            h = 0.5 * (np.log(vt / v1) / np.log(self._hurst_lag))

        blocked = h > self._h_max
        guarded = self._h_soft_max < h <= self._h_max
        self._cache[key] = (blocked, h, guarded)
        return blocked, h, guarded


# ── Helper ────────────────────────────────────────────────────────────────────

def _to_date(ts) -> _date:
    if isinstance(ts, _date) and not isinstance(ts, pd.Timestamp):
        return ts
    return pd.Timestamp(ts).date()


def validate_kde_density(z_series: pd.Series, entry_z: float, threshold_ratio: float = 0.5) -> bool:
    """
    KDE Structural Filter.
    Checks if the historical density at entry_z is at least `threshold_ratio` of the 
    theoretical Gaussian density at entry_z.
    If KDE(entry_z) < Gaussian(entry_z) * threshold_ratio, it's a Low Density Node (void).
    Returns True if valid (HVN or normal), False if invalid (LDN / void).
    """
    z = z_series.dropna().values
    if len(z) < 100:
        return True # Not enough data
        
    try:
        kde = gaussian_kde(z)
        
        # We evaluate empirical density at entry_z and -entry_z
        d_pos = kde(entry_z)[0]
        d_neg = kde(-entry_z)[0]
        d_empirical = (d_pos + d_neg) / 2.0
        
        # Theoretical Gaussian PDF at entry_z
        # phi(z) = (1 / sqrt(2*pi)) * e^(-0.5 * z^2)
        d_theoretical = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (entry_z ** 2))
        
        return bool(d_empirical >= (d_theoretical * threshold_ratio))
    except Exception:
        # e.g., singular matrix if variance is zero
        return True
