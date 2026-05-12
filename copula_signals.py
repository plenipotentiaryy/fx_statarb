"""
copula_signals.py — Copula-based dependency signals for pairs trading.

Why copulas over linear cointegration:
  Engle-Granger / Johansen test captures only LINEAR dependence between price levels.
  At 2–3σ entry zones the actual joint distribution is non-linear and fat-tailed.
  Copulas model the dependence structure independently of marginal distributions,
  giving a more precise signal exactly where we trade.

Two copulas implemented:

  Gaussian copula → copula_z
    Symmetric non-linear signal. Used as direction-confirmation of Kalman zscore.
    If Kalman says "short" but copula_z < 0 → signals disagree → skip.

  Clayton copula → lambda_L  (lower-tail dependence coefficient)
    Asymmetric: designed for joint extreme-LOW moves. λ_L ∈ [0, 1].
    Higher λ_L means the pair's extreme down-moves are more correlated
    → spread is more likely to revert from extreme lows.
    Used as a quality gate on LONG entries specifically.

Usage:
    from copula_signals import gaussian_copula_z, clayton_tail_dependence
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


# ── Gaussian Copula ───────────────────────────────────────────────────────────

def gaussian_copula_z(p1: pd.Series,
                      p2: pd.Series,
                      window: int) -> pd.Series:
    """
    Compute a rolling Gaussian copula residual Z-score.

    Steps:
      1. Rolling empirical CDF via rank-percentile (pandas built-in, O(n log w))
      2. Probability integral transform: u → Φ⁻¹(u) → copula-space normals
      3. Rolling Gaussian copula correlation ρ̂
      4. Residual = n1 − ρ̂·n2, normalised by its rolling std

    The resulting copula_z has the same sign convention as the Kalman zscore:
      copula_z > 0  →  leg1 expensive relative to leg2  →  SHORT signal
      copula_z < 0  →  leg1 cheap  relative to leg2  →  LONG  signal
    """
    u1 = p1.rolling(window).rank(pct=True).clip(0.01, 0.99)
    u2 = p2.rolling(window).rank(pct=True).clip(0.01, 0.99)

    # Vectorised Φ⁻¹ — much faster than .apply(norm.ppf)
    n1 = pd.Series(norm.ppf(u1.values), index=p1.index, name="n1")
    n2 = pd.Series(norm.ppf(u2.values), index=p2.index, name="n2")

    rho   = n1.rolling(window).corr(n2)
    resid = n1 - rho * n2
    std   = resid.rolling(window).std().replace(0, np.nan)

    return (resid / std).rename("copula_z")


# ── Clayton Copula ────────────────────────────────────────────────────────────

def clayton_tail_dependence(p1: pd.Series,
                             p2: pd.Series,
                             window: int) -> pd.Series:
    """
    Rolling Clayton lower-tail dependence coefficient: λ_L = 2^(−1/θ)

    Clayton copula C(u,v) = (u⁻θ + v⁻θ − 1)^(−1/θ) has positive lower-tail
    dependence and zero upper-tail dependence — ideal for FX pairs that
    tend to co-crash (risk-off sell-offs) but diverge asymmetrically on rallies.

    θ estimation:
      1. Rolling Spearman ρ_s on empirical ranks
      2. Convert to Kendall's τ via τ ≈ (2/π)·arcsin(ρ_s)   [bivariate normal approx]
      3. Method of moments: θ = 2τ / (1 − τ)

    Interpretation:
      λ_L ~ 0    pair tails are independent (weak dependence in extremes)
      λ_L ~ 0.3  moderate co-crash behavior
      λ_L ~ 0.5  strong joint lower-tail dependence
      λ_L → 1    perfect co-movement in tails (never reached in practice)

    Use as quality gate: prefer LONG entries when λ_L is high
    (greater probability that the spread will revert from extreme lows).
    """
    u1 = p1.rolling(window).rank(pct=True).clip(1e-4, 1 - 1e-4)
    u2 = p2.rolling(window).rank(pct=True).clip(1e-4, 1 - 1e-4)

    # Rolling Spearman on uniform marginals
    spearman = u1.rolling(window).corr(u2)

    # Kendall's τ approximation
    tau = (2.0 / np.pi) * np.arcsin(spearman.values)
    tau = np.clip(tau, 1e-4, 0.9999)

    # Clayton θ (must be positive; enforce floor)
    theta = np.where(tau > 0, 2.0 * tau / (1.0 - tau), 1e-4)
    theta = np.maximum(theta, 1e-4)

    lambda_L = 2.0 ** (-1.0 / theta)

    return pd.Series(lambda_L, index=p1.index, name="lambda_L")


# ── Convenience wrapper ───────────────────────────────────────────────────────

def compute_copula_signals(p1: pd.Series,
                           p2: pd.Series,
                           window: int) -> pd.DataFrame:
    """Return both copula signals as a DataFrame for easy concat."""
    return pd.DataFrame({
        "copula_z":  gaussian_copula_z(p1, p2, window),
        "lambda_L":  clayton_tail_dependence(p1, p2, window),
    })
