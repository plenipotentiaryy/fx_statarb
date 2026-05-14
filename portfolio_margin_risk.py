from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EPS = 1e-12


def _as_weight_vector(
    weights: pd.Series | np.ndarray | list[float],
    covariance: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, list[str] | None]:
    if isinstance(covariance, pd.DataFrame):
        columns = covariance.columns.tolist()
        if isinstance(weights, pd.Series):
            aligned = weights.reindex(columns).fillna(0.0)
            return aligned.to_numpy(dtype=np.float64), columns
        arr = np.asarray(weights, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != len(columns):
            raise ValueError("weights must align to covariance columns.")
        return arr, columns

    arr = np.asarray(weights, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("weights must be a 1D vector.")
    return arr, None


def _as_covariance_matrix(covariance: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(covariance, pd.DataFrame):
        matrix = covariance.to_numpy(dtype=np.float64, copy=False)
    else:
        matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    matrix = 0.5 * (matrix + matrix.T)
    return matrix


def _project_psd(matrix: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, EPS, None)
    projected = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return 0.5 * (projected + projected.T)


@dataclass
class MarginSpiralDetector:
    vol_shock: float = 2.5
    corr_panic_level: float = 0.98
    corr_blend: float = 0.75
    confidence_z: float = 2.33
    spread_cost_multiplier: float = 3.0
    runway_buffer: float = 0.10

    def evaluate_portfolio(
        self,
        weights: pd.Series | np.ndarray | list[float],
        equity: float,
        margin_used: float,
        empirical_cov: pd.DataFrame | np.ndarray,
        spread_cost: float = 0.0,
        include_stressed_matrix: bool = False,
    ) -> dict:
        w, labels = _as_weight_vector(weights, empirical_cov)
        cov = _as_covariance_matrix(empirical_cov)
        if cov.shape[0] != w.shape[0]:
            raise ValueError("weights length must match covariance dimensions.")

        stressed_cov = self._stress_covariance(cov)

        base_var = float(w @ cov @ w)
        panic_var = float(w @ stressed_cov @ w)
        base_var = max(base_var, 0.0)
        panic_var = max(panic_var, 0.0)

        spread_penalty = max(float(spread_cost), 0.0) * self.spread_cost_multiplier
        mar_99 = self.confidence_z * np.sqrt(panic_var) + spread_penalty

        runway = float(equity) - float(margin_used)
        safe_runway = max(runway * (1.0 - self.runway_buffer), 0.0)
        imminent = mar_99 > safe_runway

        if mar_99 <= EPS:
            target_reduction_ratio = 0.0
        elif safe_runway <= 0.0:
            target_reduction_ratio = 1.0 if np.abs(w).sum() > 0.0 else 0.0
        else:
            target_reduction_ratio = max(0.0, 1.0 - (safe_runway / mar_99))
        target_reduction_ratio = float(np.clip(target_reduction_ratio, 0.0, 1.0))

        result = {
            "portfolio_variance_base": base_var,
            "portfolio_variance_panic": panic_var,
            "portfolio_vol_base": np.sqrt(base_var),
            "portfolio_vol_panic": np.sqrt(panic_var),
            "spread_widening_penalty": spread_penalty,
            "mar_99": mar_99,
            "runway": runway,
            "safe_runway": safe_runway,
            "is_margin_spiral_imminent": bool(imminent),
            "target_reduction_ratio": target_reduction_ratio,
            "gross_notional": float(np.abs(w).sum()),
            "net_notional": float(w.sum()),
        }
        if labels is not None:
            result["labels"] = labels
        if include_stressed_matrix:
            result["stressed_covariance"] = (
                pd.DataFrame(stressed_cov, index=labels, columns=labels)
                if labels is not None else stressed_cov
            )
        return result

    def _stress_covariance(self, covariance: np.ndarray) -> np.ndarray:
        vols = np.sqrt(np.clip(np.diag(covariance), EPS, None))
        inv_outer = np.outer(vols, vols)
        corr = np.divide(
            covariance,
            inv_outer,
            out=np.zeros_like(covariance, dtype=np.float64),
            where=inv_outer > EPS,
        )
        corr = np.clip(0.5 * (corr + corr.T), -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)

        sign_corr = np.sign(corr)
        panic_target = self.corr_panic_level * sign_corr
        np.fill_diagonal(panic_target, 1.0)

        corr_stress = (1.0 - self.corr_blend) * corr + self.corr_blend * panic_target
        corr_stress = np.clip(corr_stress, -1.0, 1.0)
        np.fill_diagonal(corr_stress, 1.0)
        corr_stress = _project_psd(corr_stress)

        stressed_vols = vols * np.sqrt(max(self.vol_shock, 0.0))
        stressed_cov = corr_stress * np.outer(stressed_vols, stressed_vols)
        return _project_psd(stressed_cov)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n_pairs = 50
    t_steps = 400
    n_factors = 4

    factor_returns = rng.normal(0.0, 0.012, size=(t_steps, n_factors))
    factor_loadings = rng.normal(0.0, 0.8, size=(n_pairs, n_factors))
    residual = rng.normal(0.0, 0.006, size=(t_steps, n_pairs))
    returns = factor_returns @ factor_loadings.T + residual

    columns = [f"pair_{i:02d}" for i in range(n_pairs)]
    returns_df = pd.DataFrame(returns, columns=columns)
    empirical_cov = returns_df.cov()

    notionals = pd.Series(
        rng.uniform(150_000.0, 450_000.0, size=n_pairs) * rng.choice([-1.0, 1.0], size=n_pairs),
        index=columns,
    )

    detector = MarginSpiralDetector(
        vol_shock=2.5,
        corr_panic_level=0.98,
        corr_blend=0.80,
        confidence_z=2.33,
        spread_cost_multiplier=3.0,
        runway_buffer=0.10,
    )
    result = detector.evaluate_portfolio(
        weights=notionals,
        equity=5_000_000.0,
        margin_used=4_750_000.0,
        empirical_cov=empirical_cov,
        spread_cost=75_000.0,
    )

    print(f"Gross notional:            ${result['gross_notional']:,.0f}")
    print(f"Base portfolio vol:        ${result['portfolio_vol_base']:,.2f}")
    print(f"Panic portfolio vol:       ${result['portfolio_vol_panic']:,.2f}")
    print(f"Spread widening penalty:   ${result['spread_widening_penalty']:,.2f}")
    print(f"Stressed MaR_99:           ${result['mar_99']:,.2f}")
    print(f"Runway:                    ${result['runway']:,.2f}")
    print(f"Safe runway:               ${result['safe_runway']:,.2f}")
    print(f"Margin spiral imminent:    {result['is_margin_spiral_imminent']}")
    print(f"Target reduction ratio:    {result['target_reduction_ratio']:.2%}")
