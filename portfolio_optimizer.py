from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-12


def _as_covariance(cov_matrix: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, list[str] | None]:
    if isinstance(cov_matrix, pd.DataFrame):
        labels = cov_matrix.columns.tolist()
        matrix = cov_matrix.to_numpy(dtype=np.float64, copy=False)
    else:
        labels = None
        matrix = np.asarray(cov_matrix, dtype=np.float64)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("cov_matrix must be a square matrix.")
    matrix = 0.5 * (matrix + matrix.T)
    return matrix, labels


def _as_vector(
    vector: pd.Series | pd.DataFrame | np.ndarray | list[float] | None,
    n_assets: int,
    labels: list[str] | None,
    name: str,
    fill_value: float = 0.0,
) -> np.ndarray:
    if vector is None:
        return np.full(n_assets, fill_value, dtype=np.float64)

    if isinstance(vector, pd.DataFrame):
        if vector.shape[1] != 1:
            raise ValueError(f"{name} DataFrame must have exactly one column.")
        vector = vector.iloc[:, 0]

    if isinstance(vector, pd.Series):
        if labels is not None:
            return vector.reindex(labels).fillna(fill_value).to_numpy(dtype=np.float64)
        arr = vector.to_numpy(dtype=np.float64, copy=False)
    else:
        arr = np.asarray(vector, dtype=np.float64)

    if arr.ndim != 1 or arr.shape[0] != n_assets:
        raise ValueError(f"{name} must be a 1D vector with length {n_assets}.")
    return np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)


def _project_psd(matrix: np.ndarray) -> np.ndarray:
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, EPS, None)
    projected = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return 0.5 * (projected + projected.T)


def _format_result(weights: np.ndarray, labels: list[str] | None) -> pd.Series | np.ndarray:
    if labels is None:
        return weights.astype(np.float64, copy=False)
    return pd.Series(weights.astype(np.float64, copy=False), index=labels, name="weight")


@dataclass
class RegularizedPortfolioOptimizer:
    eta: float = 1.0
    tau: float = 1e-3
    gamma: float = 1e-3
    max_gross: float = 2.0
    weight_min: float = -0.10
    weight_max: float = 0.10
    market_neutral: bool = True
    ev_epsilon: float = 1e-12
    solver_order: tuple[str, ...] = ("OSQP", "CLARABEL", "ECOS", "SCS")
    solver_options: dict[str, Any] = field(default_factory=dict)

    last_status: dict[str, Any] = field(default_factory=dict, init=False)

    def optimize_weights(
        self,
        ev_vector: pd.Series | pd.DataFrame | np.ndarray | list[float],
        cov_matrix: pd.DataFrame | np.ndarray,
        current_weights: pd.Series | pd.DataFrame | np.ndarray | list[float] | None = None,
    ) -> pd.Series | np.ndarray:
        cov, labels = _as_covariance(cov_matrix)
        n_assets = cov.shape[0]
        mu = _as_vector(ev_vector, n_assets, labels, "ev_vector")
        w_prev = _as_vector(current_weights, n_assets, labels, "current_weights")
        cov = _project_psd(cov)

        if n_assets == 0:
            raise ValueError("cov_matrix has no assets.")
        if self.max_gross < 0:
            raise ValueError("max_gross must be non-negative.")
        if self.weight_min > self.weight_max:
            raise ValueError("weight_min must be <= weight_max.")

        if np.max(np.abs(mu)) <= self.ev_epsilon:
            weights = self._project_weights_to_constraints(w_prev)
            self.last_status = {"status": "zero_ev_previous_weights", "solver": None}
            return _format_result(weights, labels)

        try:
            weights, status = self._solve_cvxpy(mu, cov, w_prev)
            self.last_status = status
            return _format_result(weights, labels)
        except Exception as exc:
            weights = self._fallback_risk_parity(mu, cov, w_prev)
            self.last_status = {
                "status": "fallback_risk_parity",
                "solver": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
            return _format_result(weights, labels)

    def _solve_cvxpy(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        w_prev: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        try:
            import cvxpy as cp
        except ImportError as exc:
            raise RuntimeError("cvxpy is required for convex optimization.") from exc

        n_assets = mu.shape[0]
        w = cp.Variable(n_assets)
        objective = cp.Minimize(
            0.5 * cp.quad_form(w, cp.psd_wrap(cov))
            - float(self.eta) * (mu @ w)
            + float(self.tau) * cp.sum_squares(w)
            + float(self.gamma) * cp.norm1(w - w_prev)
        )
        constraints = [
            cp.norm1(w) <= float(self.max_gross),
            w >= float(self.weight_min),
            w <= float(self.weight_max),
        ]
        if self.market_neutral:
            constraints.append(cp.sum(w) == 0.0)

        problem = cp.Problem(objective, constraints)
        installed = set(cp.installed_solvers())
        errors: list[str] = []
        for solver in self.solver_order:
            if solver not in installed:
                continue
            try:
                options = dict(self.solver_options.get(solver, {}))
                problem.solve(solver=solver, warm_start=True, **options)
            except cp.SolverError as exc:
                errors.append(f"{solver}: {exc}")
                continue

            if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and w.value is not None:
                weights = np.asarray(w.value, dtype=np.float64)
                weights = self._project_weights_to_constraints(weights)
                return weights, {
                    "status": problem.status,
                    "solver": solver,
                    "objective_value": float(problem.value) if problem.value is not None else np.nan,
                    "gross": float(np.abs(weights).sum()),
                }
            errors.append(f"{solver}: status={problem.status}")

        raise RuntimeError("No cvxpy solver converged. " + " | ".join(errors))

    def _project_weights_to_constraints(self, weights: np.ndarray) -> np.ndarray:
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.clip(weights, self.weight_min, self.weight_max)

        if self.market_neutral and weights.size > 0:
            weights = self._neutralize_with_bounds(weights)

        gross = float(np.abs(weights).sum())
        if gross > self.max_gross > 0.0:
            weights = weights * (self.max_gross / gross)
        elif self.max_gross == 0.0:
            weights = np.zeros_like(weights)

        if self.market_neutral and weights.size > 0:
            weights = self._neutralize_with_bounds(weights)
        return weights.astype(np.float64, copy=False)

    def _neutralize_with_bounds(self, weights: np.ndarray) -> np.ndarray:
        weights = np.clip(weights.astype(np.float64, copy=True), self.weight_min, self.weight_max)
        for _ in range(20):
            net = float(weights.sum())
            if abs(net) <= 1e-10:
                break

            if net > 0.0:
                capacity = weights - self.weight_min
                total_capacity = float(capacity.sum())
                if total_capacity <= EPS:
                    break
                adjustment = np.minimum(capacity, net * capacity / total_capacity)
                weights -= adjustment
            else:
                need = -net
                capacity = self.weight_max - weights
                total_capacity = float(capacity.sum())
                if total_capacity <= EPS:
                    break
                adjustment = np.minimum(capacity, need * capacity / total_capacity)
                weights += adjustment

            weights = np.clip(weights, self.weight_min, self.weight_max)
        return weights

    def _fallback_risk_parity(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        w_prev: np.ndarray,
    ) -> np.ndarray:
        vols = np.sqrt(np.clip(np.diag(cov), EPS, None))
        inv_vol = 1.0 / vols
        alpha_strength = np.abs(mu)

        if alpha_strength.sum() <= self.ev_epsilon:
            return self._project_weights_to_constraints(w_prev)

        raw = inv_vol * alpha_strength
        raw_sum = raw.sum()
        if raw_sum <= EPS:
            return self._project_weights_to_constraints(w_prev)

        direction = np.sign(mu)
        weights = direction * raw / raw_sum * float(self.max_gross)

        if self.market_neutral:
            long_mask = weights > 0.0
            short_mask = weights < 0.0
            if long_mask.any() and short_mask.any():
                half_gross = float(self.max_gross) / 2.0
                weights[long_mask] *= half_gross / max(weights[long_mask].sum(), EPS)
                weights[short_mask] *= half_gross / max(abs(weights[short_mask].sum()), EPS)
            else:
                return self._project_weights_to_constraints(w_prev)

        return self._project_weights_to_constraints(weights)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n_assets = 10
    labels = [f"pair_{i:02d}" for i in range(n_assets)]

    a = rng.normal(0.0, 0.02, size=(n_assets, n_assets))
    cov = a @ a.T
    cov = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    vols = rng.uniform(0.01, 0.04, size=n_assets)
    cov = cov * np.outer(vols, vols)
    cov_df = pd.DataFrame(cov, index=labels, columns=labels)

    ev = pd.Series(rng.normal(0.0, 0.02, size=n_assets), index=labels)
    ev.iloc[0] = 0.12
    ev.iloc[1] = 0.08
    w_prev = pd.Series(0.0, index=labels)
    w_prev.iloc[2] = 0.03
    w_prev.iloc[3] = -0.03

    optimizer = RegularizedPortfolioOptimizer(
        eta=1.0,
        tau=0.10,
        gamma=0.01,
        max_gross=1.0,
        weight_min=-0.15,
        weight_max=0.15,
        market_neutral=True,
    )
    weights = optimizer.optimize_weights(ev, cov_df, w_prev)

    print("Solver status:")
    print(optimizer.last_status)
    print("\nExpected values:")
    print(ev.round(4).to_string())
    print("\nOptimal weights:")
    print(weights.round(4).to_string())
    print(f"\nGross exposure: {weights.abs().sum():.4f}")
    print(f"Net exposure:   {weights.sum():.8f}")
    print(f"Max abs weight: {weights.abs().max():.4f}")
