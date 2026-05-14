from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass
class RegimeMemoryWeighter:
    """
    State-weighted memory for expanding-window training sets.

    Required columns:
        - half_life
        - macro_state

    Index:
        - DatetimeIndex preferred; otherwise age falls back to row distance.
    """

    age_lambda: float = 0.02
    hl_gamma: float = 1.0
    half_life_col: str = "half_life"
    state_col: str = "macro_state"
    min_weight: float = 1e-6
    normalize: bool = True

    def compute_weights(
        self,
        df: pd.DataFrame,
        now_index: pd.Timestamp | None = None,
        hl_now: float | None = None,
        state_now: float | int | str | None = None,
    ) -> np.ndarray:
        if self.half_life_col not in df.columns:
            raise KeyError(f"Missing required column: {self.half_life_col}")
        if self.state_col not in df.columns:
            raise KeyError(f"Missing required column: {self.state_col}")

        out = df.copy()
        if out.empty:
            return np.zeros(0, dtype=np.float64)

        if now_index is None:
            now_index = out.index[-1]
        if hl_now is None:
            hl_now = float(pd.to_numeric(out[self.half_life_col], errors="coerce").iloc[-1])
        if state_now is None:
            state_now = out[self.state_col].iloc[-1]

        hl_series = pd.to_numeric(out[self.half_life_col], errors="coerce").to_numpy(dtype=np.float64)
        state_series = out[self.state_col].to_numpy(copy=False)

        if isinstance(out.index, pd.DatetimeIndex):
            idx = out.index
            if idx.tz is not None and getattr(now_index, "tzinfo", None) is None:
                now_index = pd.Timestamp(now_index, tz=idx.tz)
            elif idx.tz is None and getattr(now_index, "tzinfo", None) is not None:
                now_index = pd.Timestamp(now_index).tz_localize(None)
            age_days = ((pd.Timestamp(now_index) - idx) / pd.Timedelta(days=1)).to_numpy(dtype=np.float64)
        else:
            age_days = (len(out) - 1 - np.arange(len(out), dtype=np.float64))

        age_days = np.maximum(age_days, 0.0)
        hl_now = max(float(hl_now), EPS)
        age_term = np.exp(-self.age_lambda * age_days)
        hl_term = np.exp(-self.hl_gamma * (np.abs(hl_series - hl_now) / hl_now))
        state_term = (state_series == state_now).astype(np.float64)

        weights = age_term * hl_term * state_term
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.clip(weights, 0.0, None)

        positive = weights > 0.0
        if positive.any():
            weights[positive] = np.maximum(weights[positive], self.min_weight)

        if self.normalize and weights.sum() > 0.0:
            weights = weights / weights.mean()
        return weights.astype(np.float64, copy=False)


def apply_parameter_drift_truncation(
    train_df: pd.DataFrame,
    half_life_col: str = "half_life",
    drift_threshold_q: float = 2.0,
    recent_days: int = 90,
) -> pd.DataFrame:
    """
    Hard-reset memory if current half-life regime has drifted too far away.

    Rule:
        if HL_now > q * median(HL_train), keep only the recent regime window.
    """
    if half_life_col not in train_df.columns or train_df.empty:
        return train_df

    out = train_df.copy()
    hl = pd.to_numeric(out[half_life_col], errors="coerce")
    hl_valid = hl.dropna()
    if hl_valid.empty:
        return out

    hl_now = float(hl_valid.iloc[-1])
    hl_med = float(hl_valid.median())
    if not np.isfinite(hl_now) or not np.isfinite(hl_med) or hl_med <= 0.0:
        return out
    if hl_now <= drift_threshold_q * hl_med:
        return out

    if isinstance(out.index, pd.DatetimeIndex):
        cutoff = out.index[-1] - pd.Timedelta(days=recent_days)
        truncated = out.loc[out.index >= cutoff].copy()
        return truncated if not truncated.empty else out.tail(min(len(out), recent_days)).copy()
    return out.tail(min(len(out), recent_days)).copy()
