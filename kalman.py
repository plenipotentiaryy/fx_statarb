from __future__ import annotations

import numpy as np


def kalman_hedge(
    p1: np.ndarray,
    p2: np.ndarray,
    delta: float = 1e-4,
    beta_init: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Kalman Filter for time-varying hedge ratio.
    Observation model: p1[t] = alpha[t] + beta[t]*p2[t] + noise
    State transition:  [alpha, beta] ~ random walk  (Q = delta/(1-delta) * I)

    Scalar-loop implementation: avoids numpy micro-matrix overhead inside the
    hot loop while preserving the original interface and math.
    """
    n = len(p1)

    Q: float = delta / (1.0 - delta)
    R: float = 1.0

    a: float = 0.0
    b: float = float(beta_init)
    P00: float = 1.0
    P01: float = 0.0
    P11: float = 1.0

    alpha_out = np.empty(n, dtype=np.float64)
    beta_out = np.empty(n, dtype=np.float64)
    innov = np.empty(n, dtype=np.float64)
    innov_var = np.empty(n, dtype=np.float64)

    p1_ = np.asarray(p1, dtype=np.float64)
    p2_ = np.asarray(p2, dtype=np.float64)

    for t in range(n):
        h1: float = p2_[t]

        P00 += Q
        P11 += Q

        e: float = p1_[t] - (a + b * h1)
        S: float = P00 + 2.0 * P01 * h1 + P11 * h1 * h1 + R

        K0: float = (P00 + P01 * h1) / S
        K1: float = (P01 + P11 * h1) / S

        a += K0 * e
        b += K1 * e

        new_P00 = P00 - K0 * P00 - K0 * h1 * P01
        new_P01 = P01 - K0 * P01 - K0 * h1 * P11
        new_P11 = P11 - K1 * P01 - K1 * h1 * P11
        P00, P01, P11 = new_P00, new_P01, new_P11

        alpha_out[t] = a
        beta_out[t] = b
        innov[t] = e
        innov_var[t] = S

    return alpha_out, beta_out, innov, innov_var
