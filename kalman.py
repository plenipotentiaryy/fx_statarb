import numpy as np


def kalman_hedge(p1: np.ndarray, p2: np.ndarray,
                 delta: float = 1e-4,
                 beta_init: float = 1.0) -> tuple:
    """
    Kalman Filter for time-varying hedge ratio.
    Observation model: p1[t] = alpha[t] + beta[t] * p2[t] + noise
    State transition:  [alpha, beta] follows a random walk (Q = delta/(1-delta) * I)

    Returns: (alpha_arr, beta_arr, innovations, innovation_var)
      - innovations  : residuals e[t] = p1[t] - predicted p1[t]  (use as spread)
      - innovation_var: theoretical variance of each innovation (for normalisation)
    """
    n = len(p1)
    theta = np.array([0.0, beta_init])
    P = np.eye(2)
    Q = delta / (1.0 - delta) * np.eye(2)
    R = 1.0  # observation noise — kept at 1 so spread is in price units

    alpha_out = np.empty(n)
    beta_out  = np.empty(n)
    innov     = np.empty(n)
    innov_var = np.empty(n)

    for t in range(n):
        H = np.array([1.0, p2[t]])

        P = P + Q                        # predict covariance
        e = p1[t] - H @ theta            # innovation
        S = float(H @ P @ H) + R         # innovation variance
        K = (P @ H) / S                  # Kalman gain
        theta = theta + K * e            # update state
        P = (np.eye(2) - np.outer(K, H)) @ P  # update covariance

        alpha_out[t] = theta[0]
        beta_out[t]  = theta[1]
        innov[t]     = e
        innov_var[t] = S

    return alpha_out, beta_out, innov, innov_var
