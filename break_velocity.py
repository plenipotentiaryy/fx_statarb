import math
from dataclasses import dataclass

import numba
import numpy as np
import pandas as pd


EPS = 1e-12


@numba.njit(cache=True)
def _cusum_flags(x: np.ndarray, mu0: np.ndarray, k: float, h: float) -> np.ndarray:
    n = x.shape[0]
    flags = np.zeros(n, dtype=np.uint8)
    s_pos = 0.0
    s_neg = 0.0

    for i in range(n):
        if np.isnan(x[i]) or np.isnan(mu0[i]):
            s_pos = 0.0
            s_neg = 0.0
            continue

        inc = x[i] - mu0[i]
        s_pos = max(0.0, s_pos + inc - k)
        s_neg = min(0.0, s_neg + inc + k)

        if s_pos > h or abs(s_neg) > h:
            flags[i] = 1

    return flags


def _rolling_hurst_proxy(spread: pd.Series, window: int = 50, lag: int = 10) -> pd.Series:
    diff_1 = spread.diff(1)
    diff_lag = spread.diff(lag)
    var_1 = diff_1.rolling(window=window, min_periods=max(20, window // 2)).var()
    var_lag = diff_lag.rolling(window=window, min_periods=max(20, window // 2)).var()
    hurst = 0.5 * np.log((var_lag / var_1.replace(0, np.nan)).clip(lower=EPS)) / np.log(lag)
    return hurst.clip(lower=0.0, upper=1.5)


@dataclass
class BreakVelocityDetector:
    cusum_mean_window: int = 50
    cusum_k: float = 0.25
    cusum_h: float = 2.5
    shock_window: int = 20
    hl_window: int = 20
    hl_median_window: int = 20
    weights: tuple[float, float, float] = (1.5, 1.0, 1.0)
    break_threshold: float = 2.0
    clip_shock_ratio: tuple[float, float] = (1e-6, 1e6)
    clip_hl_ratio: tuple[float, float] = (1e-6, 1e6)

    def fit_transform(self, df: pd.DataFrame, spread_col: str) -> pd.DataFrame:
        out = df.copy()
        spread = pd.to_numeric(out[spread_col], errors="coerce").astype(float)
        warmup = max(
            self.cusum_mean_window,
            self.shock_window,
            self.hl_window + self.hl_median_window + 1,
        )

        rolling_mu = spread.rolling(
            window=self.cusum_mean_window,
            min_periods=max(5, self.cusum_mean_window // 4),
        ).mean()
        cusum_flag = _cusum_flags(
            spread.to_numpy(dtype=np.float64),
            rolling_mu.to_numpy(dtype=np.float64),
            float(self.cusum_k),
            float(self.cusum_h),
        )
        out["cusum_flag"] = cusum_flag.astype(np.int8)

        innov = spread.diff()
        innov_sq = innov.pow(2)
        innov_var = innov_sq.rolling(
            window=self.shock_window,
            min_periods=max(5, self.shock_window // 2),
        ).mean()
        shock_ratio = (innov_sq / innov_var.replace(0.0, np.nan)).clip(
            lower=self.clip_shock_ratio[0],
            upper=self.clip_shock_ratio[1],
        )
        out["shock_ratio"] = shock_ratio.replace([np.inf, -np.inf], np.nan)

        lagged = spread.shift(1)
        dx = spread.diff()
        cov = dx.rolling(
            window=self.hl_window,
            min_periods=max(5, self.hl_window // 2),
        ).cov(lagged)
        var = lagged.rolling(
            window=self.hl_window,
            min_periods=max(5, self.hl_window // 2),
        ).var()
        b = cov / var.replace(0.0, np.nan)
        theta = (-b).clip(lower=EPS)
        hl = np.log(2.0) / theta
        hl = hl.where(np.isfinite(hl), np.nan)

        hl_med = hl.shift(1).rolling(
            window=self.hl_median_window,
            min_periods=max(5, self.hl_median_window // 2),
        ).median()
        hl_ratio = (hl / hl_med.replace(0.0, np.nan)).clip(
            lower=self.clip_hl_ratio[0],
            upper=self.clip_hl_ratio[1],
        )
        out["hl_ratio"] = hl_ratio.replace([np.inf, -np.inf], np.nan)

        w1, w2, w3 = self.weights
        break_score = (
            w1 * out["cusum_flag"].astype(float)
            + w2 * np.log(out["shock_ratio"].clip(lower=self.clip_shock_ratio[0]))
            + w3 * np.log(out["hl_ratio"].clip(lower=self.clip_hl_ratio[0]))
        )
        out["break_score"] = break_score.replace([np.inf, -np.inf], np.nan)
        out.loc[out.index[:warmup], "break_score"] = np.nan
        out["is_break_active"] = out["break_score"] > self.break_threshold
        return out


def _simulate_ou_then_break(
    n_ou: int = 600,
    n_break: int = 200,
    theta: float = 0.22,
    sigma: float = 0.35,
    break_drift: float = 0.18,
    break_sigma: float = 0.8,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.zeros(n_ou + n_break, dtype=np.float64)

    for t in range(1, n_ou):
        x[t] = x[t - 1] + (-theta * x[t - 1]) + sigma * rng.normal()

    for t in range(n_ou, n_ou + n_break):
        x[t] = x[t - 1] + break_drift + break_sigma * rng.normal()

    idx = pd.date_range("2024-01-01", periods=len(x), freq="h", tz="UTC")
    return pd.DataFrame({"spread": x}, index=idx)


def _demo() -> None:
    df = _simulate_ou_then_break()
    detector = BreakVelocityDetector(
        cusum_mean_window=50,
        cusum_k=0.20,
        cusum_h=2.0,
        shock_window=20,
        hl_window=20,
        hl_median_window=20,
        weights=(1.6, 1.1, 1.1),
        break_threshold=2.2,
    )
    scored = detector.fit_transform(df, "spread")
    scored["rolling_hurst_50"] = _rolling_hurst_proxy(scored["spread"], window=50, lag=10)
    scored["hurst_break"] = scored["rolling_hurst_50"] > 0.55

    true_break_start = df.index[600]
    det_break_idx = scored.index[(scored.index >= true_break_start) & scored["is_break_active"]].min()
    hurst_break_idx = scored.index[scored["hurst_break"]].min()
    pre_break_false = int(scored.loc[scored.index < true_break_start, "is_break_active"].sum())

    print("BreakVelocityDetector demo")
    print(f"True break start:           {true_break_start}")
    print(f"Detector first trigger:     {det_break_idx}")
    print(f"Hurst(50) first > 0.55:     {hurst_break_idx}")
    print(f"Pre-break false positives:  {pre_break_false}")

    if pd.notna(det_break_idx):
        det_delay = int((det_break_idx - true_break_start) / pd.Timedelta(hours=1))
        print(f"Detector delay (bars):      {det_delay}")
    if pd.notna(hurst_break_idx):
        hurst_delay = int((hurst_break_idx - true_break_start) / pd.Timedelta(hours=1))
        print(f"Hurst delay (bars):         {hurst_delay}")
    if pd.notna(det_break_idx) and pd.notna(hurst_break_idx):
        print(f"Lead vs Hurst (bars):       {int((hurst_break_idx - det_break_idx) / pd.Timedelta(hours=1))}")

    tail = scored.loc[true_break_start:].head(12)[
        ["spread", "cusum_flag", "shock_ratio", "hl_ratio", "break_score", "is_break_active", "rolling_hurst_50"]
    ]
    print("\nFirst 12 rows after break start:")
    print(tail.round(4).to_string())


if __name__ == "__main__":
    _demo()
