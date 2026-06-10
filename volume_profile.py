"""
volume_profile.py — Volume-at-Price / Volume-at-Z zone engine for FX stat-arb.

WHERE THE VOLUME COMES FROM
---------------------------
Spot FX has no centralised traded volume — it is a decentralised OTC market, so
there is no single "tape". This engine is deliberately *source-agnostic*: it only
needs (value, weight) pairs. Two weight sources are supported, in priority order:

  1. CME FX futures volume  (REAL exchange volume)
     data/futures_volumes_{bar}min.parquet — built by download_futures_volume.py.
     6E↔EURUSD, 6B↔GBPUSD, 6J↔USDJPY, ... This is the only genuinely *traded*
     volume available for FX and is the gold standard for a volume profile.

  2. Dukascopy tick volume  (PROXY)
     data/volumes_{bar}min.parquet — number of ticks per bar. Empirically
     ~0.85-0.90 correlated with real volume in liquid majors (Karpoff; CLS
     studies), so it is a usable fallback when futures volume is unavailable.

pick_volume_source() resolves which one is present and returns a labelled series.

TWO PROFILE FLAVOURS
--------------------
  * Price profile (per leg):  Volume-at-Price histogram → POC / VAH / VAL / HVN / LVN.
    Used for STOP placement — don't park a stop inside a churn HVN where the spread
    oscillates; place it just beyond a low-volume node (LVN) the price travels through.

  * Z-profile (spread):       Volume-at-Z histogram of the pair spread Z-score → HVN-Z.
    Used for the ENTRY gate. In volume-profile theory price reverts toward HVN/POC
    (a magnet) and accelerates through LVN (a vacuum). Two tradeable hypotheses:
      - ZONE_MODE="HVN": fade extremes that sit in a high-volume node (range/acceptance).
      - ZONE_MODE="LVN": fade extremes that sit in a low-volume node (rejection → snapback).

NO LEAKAGE
----------
All rolling profiles are built CAUSALLY from a trailing window of *past* bars only.
build_rolling_zone_series() returns a per-bar zone label and per-bar LVN-stop level
computed strictly from data *before* that bar, so the output can be dropped straight
into the backtest as a feature column with no look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ── Profile container ─────────────────────────────────────────────────────────

@dataclass
class VolumeProfile:
    """A volume histogram over a 1-D value axis (price levels or spread Z-levels)."""
    edges:   np.ndarray            # bin edges, length n_bins + 1
    vol:     np.ndarray            # volume in each bin, length n_bins
    poc:     float                 # Point of Control — value of the max-volume bin
    va_low:  float                 # Value-Area low edge
    va_high: float                 # Value-Area high edge
    hvn_thresh: float              # bins with vol >= this are High-Volume Nodes
    lvn_thresh: float              # bins with vol <= this are Low-Volume Nodes
    centers: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])

    # ── Classification ────────────────────────────────────────────────────────

    def _bin_index(self, x: float) -> int | None:
        if not np.isfinite(x) or x < self.edges[0] or x > self.edges[-1]:
            return None
        idx = int(np.searchsorted(self.edges, x, side="right") - 1)
        return min(max(idx, 0), len(self.vol) - 1)

    def zone_of(self, x: float) -> str:
        """Classify a value as 'HVN', 'LVN', 'VALUE', or 'OUTSIDE'."""
        idx = self._bin_index(x)
        if idx is None:
            return "OUTSIDE"
        v = self.vol[idx]
        if v >= self.hvn_thresh:
            return "HVN"
        if v <= self.lvn_thresh:
            return "LVN"
        if self.va_low <= x <= self.va_high:
            return "VALUE"
        return "NEUTRAL"

    def nearest_lvn(self, x: float, direction: int) -> float | None:
        """Value of the nearest Low-Volume Node strictly beyond ``x``.

        direction = +1 searches upward (x increasing), -1 searches downward.
        Returns the bin center of the first LVN found, or None if there is none.
        Used to place a stop just past the vacuum the price would run through.
        """
        if direction not in (1, -1) or not np.isfinite(x):
            return None
        lvn = self.vol <= self.lvn_thresh
        if direction == 1:
            cand = np.where(lvn & (self.centers > x))[0]
            return float(self.centers[cand[0]]) if cand.size else None
        cand = np.where(lvn & (self.centers < x))[0]
        return float(self.centers[cand[-1]]) if cand.size else None


# ── Profile builder ───────────────────────────────────────────────────────────

def build_profile(values: np.ndarray | pd.Series,
                  weights: np.ndarray | pd.Series,
                  n_bins: int = 60,
                  va_pct: float = 0.70,
                  hvn_q: float = 0.80,
                  lvn_q: float = 0.20,
                  value_range: tuple[float, float] | None = None) -> VolumeProfile | None:
    """Build a weighted Volume Profile.

    values        : the axis to bin (price levels, or spread Z-scores).
    weights       : volume per observation (futures volume or tick volume).
    n_bins        : histogram resolution.
    va_pct        : Value-Area fraction (classic Market Profile = 70%).
    hvn_q / lvn_q : per-bin volume quantiles defining High / Low Volume Nodes.
    value_range   : optional (lo, hi) clip for the axis; default = data min/max.

    Returns None when there is not enough valid data to form a profile.
    """
    v = np.asarray(values, dtype="float64")
    w = np.asarray(weights, dtype="float64")
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[mask], w[mask]
    if v.size < n_bins or w.sum() <= 0:
        return None

    lo, hi = (float(np.min(v)), float(np.max(v))) if value_range is None else value_range
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None

    vol, edges = np.histogram(v, bins=n_bins, range=(lo, hi), weights=w)
    total = vol.sum()
    if total <= 0:
        return None

    centers = 0.5 * (edges[:-1] + edges[1:])
    poc_idx = int(np.argmax(vol))
    poc = float(centers[poc_idx])

    # Value area: expand outward from the POC bin until va_pct of volume is covered.
    lo_i = hi_i = poc_idx
    covered = vol[poc_idx]
    target = va_pct * total
    while covered < target and (lo_i > 0 or hi_i < len(vol) - 1):
        left  = vol[lo_i - 1] if lo_i > 0 else -1.0
        right = vol[hi_i + 1] if hi_i < len(vol) - 1 else -1.0
        if right >= left:
            hi_i += 1
            covered += vol[hi_i]
        else:
            lo_i -= 1
            covered += vol[lo_i]
    va_low  = float(edges[lo_i])
    va_high = float(edges[hi_i + 1])

    # HVN / LVN thresholds from the distribution of *non-empty* bin volumes.
    nz = vol[vol > 0]
    hvn_thresh = float(np.quantile(nz, hvn_q)) if nz.size else float("inf")
    lvn_thresh = float(np.quantile(nz, lvn_q)) if nz.size else 0.0

    return VolumeProfile(
        edges=edges, vol=vol, poc=poc,
        va_low=va_low, va_high=va_high,
        hvn_thresh=hvn_thresh, lvn_thresh=lvn_thresh,
    )


# ── Causal rolling zone labels (backtest feature) ─────────────────────────────

def build_rolling_zone_series(values: pd.Series,
                              weights: pd.Series,
                              window: int,
                              step: int,
                              n_bins: int = 60,
                              va_pct: float = 0.70,
                              hvn_q: float = 0.80,
                              lvn_q: float = 0.20) -> pd.DataFrame:
    """Per-bar zone label + LVN-stop level, computed causally (no look-ahead).

    The profile is rebuilt every ``step`` bars from the *trailing* ``window`` bars
    that end strictly before the current chunk, then used to classify the bars in
    the next ``step`` chunk. Output columns:

      zone        : {'HVN','LVN','VALUE','NEUTRAL','OUTSIDE'} or NaN during warm-up.
      lvn_up      : nearest Low-Volume-Node value above the bar (NaN if none).
      lvn_down    : nearest Low-Volume-Node value below the bar (NaN if none).
      poc         : Point of Control of the active profile.

    For the spread Z-profile, pass values=zscore, weights=combined volume; lvn_up /
    lvn_down then come back in Z units, ready to feed an adaptive stop.
    """
    idx = values.index
    zone = pd.Series(index=idx, dtype="object")
    lvn_up = pd.Series(np.nan, index=idx, dtype="float64")
    lvn_down = pd.Series(np.nan, index=idx, dtype="float64")
    poc_s = pd.Series(np.nan, index=idx, dtype="float64")

    v_arr = values.to_numpy(dtype="float64")
    w_arr = weights.reindex(idx).to_numpy(dtype="float64")
    n = len(idx)

    start = window
    while start < n:
        end = min(start + step, n)
        prof = build_profile(
            v_arr[start - window:start], w_arr[start - window:start],
            n_bins=n_bins, va_pct=va_pct, hvn_q=hvn_q, lvn_q=lvn_q,
        )
        if prof is not None:
            for j in range(start, end):
                x = v_arr[j]
                zone.iat[j] = prof.zone_of(x)
                up = prof.nearest_lvn(x, +1)
                dn = prof.nearest_lvn(x, -1)
                if up is not None:
                    lvn_up.iat[j] = up
                if dn is not None:
                    lvn_down.iat[j] = dn
                poc_s.iat[j] = prof.poc
        start = end

    return pd.DataFrame({"zone": zone, "lvn_up": lvn_up,
                         "lvn_down": lvn_down, "poc": poc_s})


# ── Volume source resolver ────────────────────────────────────────────────────

def pick_volume_source(tickers: list[str] | None = None,
                       start: str | None = None,
                       end: str | None = None,
                       rth: bool = True,
                       freq: str | None = None) -> tuple[pd.DataFrame | None, str]:
    """Return (volume_df, source_label) preferring REAL futures volume over tick volume.

    source_label ∈ {"cme_futures", "tick", "none"} so callers can log provenance.
    Mapping from spot pairs to futures columns is applied here, so downstream code
    keeps using spot ticker names regardless of which source backs the volume.
    """
    from futures_map import load_futures_volume_as_spot

    fut = load_futures_volume_as_spot(tickers, start=start, end=end, rth=rth, freq=freq)
    if fut is not None and not fut.empty:
        return fut, "cme_futures"

    try:
        from data_loader import load_volumes
        tick = load_volumes(tickers, start=start, end=end, rth=rth, freq=freq)
        if tick is not None and not tick.empty:
            return tick, "tick"
    except Exception:
        pass
    return None, "none"


# ── Synthetic self-test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(7)
    # A realistic spread Z: mean-reverting, so it spends most of its volume near 0
    # (HVN at the mean) and only thin tails reach the extremes (LVN). Volume is
    # heavier in the dense regime, lighter at the extremes — the shape we exploit.
    z = rng.normal(0.0, 1.0, 20000)
    vol = 50.0 + 50.0 * np.exp(-0.5 * z ** 2) + rng.uniform(0, 5, z.size)
    prof = build_profile(z, vol, n_bins=60)
    assert prof is not None
    print(f"POC={prof.poc:+.2f}  VA=[{prof.va_low:+.2f}, {prof.va_high:+.2f}]")
    print(f"zone(0.0)   = {prof.zone_of(0.0)}   (expect HVN/VALUE)")
    print(f"zone(+3.5)  = {prof.zone_of(3.5)}   (expect LVN/OUTSIDE)")
    print(f"nearest LVN above +0.0 = {prof.nearest_lvn(0.0, +1):+.2f}")
    assert prof.zone_of(0.0) in ("HVN", "VALUE")
    assert prof.zone_of(3.5) in ("LVN", "OUTSIDE")

    s = pd.Series(z); w = pd.Series(vol)
    s.index = pd.date_range("2020-01-01", periods=len(s), freq="min")
    w.index = s.index
    out = build_rolling_zone_series(s, w, window=2000, step=500, n_bins=40)
    print("rolling zone counts:\n", out["zone"].value_counts(dropna=False))
    print("self-test OK")
