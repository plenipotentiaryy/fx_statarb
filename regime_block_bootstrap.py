from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass
class RegimeBlockBootstrap:
    """
    Regime-aware block bootstrap for path-dependent validation.

    The sampler preserves block-level regime structure by resampling contiguous
    blocks from the same regime pool for each block slot in the original path.
    This avoids the false independence assumption of iid trade bootstrap.
    """

    block_size: int = 20
    n_bootstrap: int = 1000
    pnl_col: str = "net_pnl"
    regime_col: str = "hmm_regime"
    random_seed: int = 42

    def bootstrap_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        prepared = self._prepare_input(df)
        if prepared.empty:
            return {
                "n_obs": 0,
                "realized": {"total_pnl": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
                "confidence_intervals": {},
                "samples": pd.DataFrame(columns=["total_pnl", "sharpe", "max_drawdown"]),
            }

        base_blocks = self._make_blocks(prepared)
        samples = np.empty((self.n_bootstrap, 3), dtype=np.float64)
        rng = np.random.default_rng(self.random_seed)

        for i in range(self.n_bootstrap):
            sample = self._resample_once(base_blocks, rng)
            pnl = sample[self.pnl_col].to_numpy(dtype=np.float64)
            samples[i] = self._metric_vector(pnl)

        sample_df = pd.DataFrame(samples, columns=["total_pnl", "sharpe", "max_drawdown"])
        realized = self._metric_vector(prepared[self.pnl_col].to_numpy(dtype=np.float64))
        ci = {
            metric: (
                float(sample_df[metric].quantile(0.05)),
                float(sample_df[metric].quantile(0.95)),
            )
            for metric in sample_df.columns
        }
        return {
            "n_obs": int(len(prepared)),
            "realized": {
                "total_pnl": float(realized[0]),
                "sharpe": float(realized[1]),
                "max_drawdown": float(realized[2]),
            },
            "confidence_intervals": ci,
            "samples": sample_df,
        }

    def sample_path(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = self._prepare_input(df)
        if prepared.empty:
            return prepared
        rng = np.random.default_rng(self.random_seed)
        return self._resample_once(self._make_blocks(prepared), rng)

    def _prepare_input(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pnl_col not in df.columns:
            raise KeyError(f"Missing required column: {self.pnl_col}")
        out = df.copy()
        if self.regime_col not in out.columns:
            out[self.regime_col] = 0
        out[self.pnl_col] = pd.to_numeric(out[self.pnl_col], errors="coerce").fillna(0.0)
        out[self.regime_col] = pd.to_numeric(out[self.regime_col], errors="coerce").fillna(0).astype(int)
        return out.sort_index()

    def _make_blocks(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        n = len(df)
        for start in range(0, n, self.block_size):
            block = df.iloc[start:start + self.block_size].copy()
            if block.empty:
                continue
            regime = int(block[self.regime_col].mode(dropna=False).iloc[0])
            blocks.append({"regime": regime, "data": block})
        return blocks

    def _resample_once(
        self,
        template_blocks: list[dict[str, Any]],
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        pools: dict[int, list[pd.DataFrame]] = {}
        for block in template_blocks:
            pools.setdefault(int(block["regime"]), []).append(block["data"])

        sampled_blocks: list[pd.DataFrame] = []
        for template in template_blocks:
            regime = int(template["regime"])
            candidates = pools.get(regime, [])
            if not candidates:
                candidates = [template["data"]]
            sampled_blocks.append(candidates[int(rng.integers(0, len(candidates)))].copy())

        sample = pd.concat(sampled_blocks, axis=0, ignore_index=False)
        return sample.iloc[: sum(len(block["data"]) for block in template_blocks)].copy()

    def _metric_vector(self, pnl: np.ndarray) -> np.ndarray:
        pnl = np.nan_to_num(pnl.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        total_pnl = float(pnl.sum())
        std = float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
        sharpe = float(pnl.mean() / std * np.sqrt(len(pnl))) if std > EPS else 0.0
        curve = np.cumsum(pnl)
        max_drawdown = float((curve - np.maximum.accumulate(curve)).min()) if len(curve) > 0 else 0.0
        return np.array([total_pnl, sharpe, max_drawdown], dtype=np.float64)


def attach_regime_to_trades(
    trades_df: pd.DataFrame,
    regime_series: pd.Series,
    time_col: str = "exit_time",
    regime_col: str = "hmm_regime",
) -> pd.DataFrame:
    """Attach nearest-asof regime state to realized trades for bootstrap validation."""
    if time_col not in trades_df.columns:
        raise KeyError(f"Missing required column: {time_col}")
    if regime_series.empty:
        out = trades_df.copy()
        out[regime_col] = 0
        return out

    out = trades_df.copy()
    event_time = pd.to_datetime(out[time_col], utc=True, errors="coerce").dt.tz_localize(None)
    regime = pd.Series(regime_series).sort_index().copy()
    regime.index = pd.to_datetime(regime.index, utc=True, errors="coerce").tz_localize(None)
    out[regime_col] = [regime.asof(ts) if pd.notna(ts) else 0 for ts in event_time]
    out[regime_col] = pd.to_numeric(out[regime_col], errors="coerce").fillna(0).astype(int)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(123)
    n = 500
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    regime = np.zeros(n, dtype=int)
    regime[150:220] = 1
    regime[350:420] = 1
    pnl = rng.normal(0.2, 1.0, size=n) - regime * rng.uniform(0.4, 1.2, size=n)
    demo = pd.DataFrame({"net_pnl": pnl, "hmm_regime": regime}, index=idx)

    bootstrap = RegimeBlockBootstrap(block_size=15, n_bootstrap=500, random_seed=7)
    result = bootstrap.bootstrap_metrics(demo)

    print(f"Observations: {result['n_obs']}")
    print(f"Realized total pnl:   {result['realized']['total_pnl']:.4f}")
    print(f"Realized sharpe:      {result['realized']['sharpe']:.4f}")
    print(f"Realized max drawdown:{result['realized']['max_drawdown']:.4f}")
    for metric, (lo, hi) in result["confidence_intervals"].items():
        print(f"{metric:>14}  5-95% CI: [{lo:.4f}, {hi:.4f}]")
