from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np
import pandas as pd


EPS = 1e-12


def _propagate_trade_sign(delta: np.ndarray) -> np.ndarray:
    """
    Tick-rule sign with zero-delta inheritance.

    Leading zero-delta bars remain neutral until the first non-zero move appears.
    """
    sign = np.sign(delta).astype(np.int8, copy=False)
    if sign.size == 0:
        return sign

    nonzero = sign != 0
    if not nonzero.any():
        return sign

    idx = np.where(nonzero, np.arange(sign.size), 0)
    np.maximum.accumulate(idx, out=idx)
    sign = sign[idx]

    first_nz = int(np.flatnonzero(nonzero)[0])
    if first_nz > 0:
        sign[:first_nz] = 0
    return sign


@numba.njit(cache=True)
def _hawkes_intensity(
    buy_volume: np.ndarray,
    sell_volume: np.ndarray,
    mu_buy: float,
    mu_sell: float,
    alpha_bb: float,
    alpha_ss: float,
    beta_bb: float,
    beta_ss: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = buy_volume.shape[0]
    lambda_buy = np.empty(n, dtype=np.float64)
    lambda_sell = np.empty(n, dtype=np.float64)

    prev_buy = mu_buy
    prev_sell = mu_sell

    for i in range(n):
        cur_buy = mu_buy + alpha_bb * buy_volume[i] + (1.0 - beta_bb) * prev_buy
        cur_sell = mu_sell + alpha_ss * sell_volume[i] + (1.0 - beta_ss) * prev_sell
        lambda_buy[i] = cur_buy
        lambda_sell[i] = cur_sell
        prev_buy = cur_buy
        prev_sell = cur_sell

    return lambda_buy, lambda_sell


@dataclass
class ToxicityMonitor:
    """
    Proxy microstructure toxicity monitor for bar-aggregated data.

    Required columns:
        - close
        - volume

    Optional columns:
        - high
        - low

    Output columns:
        - tox_buy
        - tox_sell
    """

    ofi_window: int = 20
    microprice_window: int = 20
    microprice_ema_span: int = 5
    mu_buy: float = 0.0
    mu_sell: float = 0.0
    alpha_bb: float = 0.15
    alpha_ss: float = 0.15
    beta_bb: float = 0.25
    beta_ss: float = 0.25
    ofi_coef: float = 1.0
    microprice_coef: float = 0.5
    clip_z: float = 10.0
    volume_col: str = "volume"
    close_col: str = "close"
    high_col: str = "high"
    low_col: str = "low"

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in (self.close_col, self.volume_col) if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        out = df.copy()
        close = pd.to_numeric(out[self.close_col], errors="coerce").astype(np.float64)
        volume = pd.to_numeric(out[self.volume_col], errors="coerce").astype(np.float64)
        volume = volume.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)

        delta = close.diff().fillna(0.0).to_numpy(dtype=np.float64)
        sign = _propagate_trade_sign(delta)
        volume_arr = volume.to_numpy(dtype=np.float64, copy=False)

        buy_volume = np.where(sign > 0, volume_arr, 0.0)
        sell_volume = np.where(sign < 0, volume_arr, 0.0)
        ofi = buy_volume - sell_volume

        ofi_series = pd.Series(ofi, index=out.index, dtype=np.float64)
        min_periods = max(5, self.ofi_window // 2)
        ofi_mean = ofi_series.rolling(window=self.ofi_window, min_periods=min_periods).mean()
        ofi_std = ofi_series.rolling(window=self.ofi_window, min_periods=min_periods).std()
        ofi_z = ((ofi_series - ofi_mean) / ofi_std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        ofi_z = ofi_z.fillna(0.0).clip(lower=-self.clip_z, upper=self.clip_z)

        lambda_buy, lambda_sell = _hawkes_intensity(
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            mu_buy=float(self.mu_buy),
            mu_sell=float(self.mu_sell),
            alpha_bb=float(self.alpha_bb),
            alpha_ss=float(self.alpha_ss),
            beta_bb=float(self.beta_bb),
            beta_ss=float(self.beta_ss),
        )

        microprice_velocity = self._microprice_velocity(out, close)

        tox_buy = (
            lambda_sell
            - lambda_buy
            - float(self.ofi_coef) * ofi_z.to_numpy(dtype=np.float64, copy=False)
            + float(self.microprice_coef) * microprice_velocity.to_numpy(dtype=np.float64, copy=False)
        )
        tox_sell = (
            lambda_buy
            - lambda_sell
            + float(self.ofi_coef) * ofi_z.to_numpy(dtype=np.float64, copy=False)
            + float(self.microprice_coef) * microprice_velocity.to_numpy(dtype=np.float64, copy=False)
        )

        out["buy_volume_proxy"] = buy_volume
        out["sell_volume_proxy"] = sell_volume
        out["ofi"] = ofi
        out["ofi_z"] = ofi_z
        out["lambda_buy"] = lambda_buy
        out["lambda_sell"] = lambda_sell
        out["microprice_velocity"] = microprice_velocity
        out["tox_buy"] = np.nan_to_num(tox_buy, nan=0.0, posinf=self.clip_z, neginf=-self.clip_z)
        out["tox_sell"] = np.nan_to_num(tox_sell, nan=0.0, posinf=self.clip_z, neginf=-self.clip_z)
        return out

    def _microprice_velocity(self, df: pd.DataFrame, close: pd.Series) -> pd.Series:
        """
        Bar-data microprice proxy.

        If high/low are present, use close-location value inside the bar to proxy
        queue pressure. Otherwise fall back to close-to-close velocity.
        """
        if self.high_col in df.columns and self.low_col in df.columns:
            high = pd.to_numeric(df[self.high_col], errors="coerce").astype(np.float64)
            low = pd.to_numeric(df[self.low_col], errors="coerce").astype(np.float64)
            bar_range = (high - low).replace(0.0, np.nan)
            microprice_proxy = ((2.0 * close - high - low) / bar_range).clip(lower=-2.0, upper=2.0)
        else:
            microprice_proxy = close.diff().fillna(0.0)

        velocity = microprice_proxy.ewm(span=self.microprice_ema_span, adjust=False).mean().diff()
        scale = velocity.rolling(
            window=self.microprice_window,
            min_periods=max(5, self.microprice_window // 2),
        ).std()
        velocity_z = (velocity / scale.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        return velocity_z.fillna(0.0).clip(lower=-self.clip_z, upper=self.clip_z)
