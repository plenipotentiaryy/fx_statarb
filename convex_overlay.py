from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog


EPS = 1e-12
DEFAULT_SCENARIOS = ("gap_down", "vol_spike", "corr_collapse")


def _project_correlation_psd(corr: np.ndarray) -> np.ndarray:
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)

    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, EPS, None)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    corr = 0.5 * (corr + corr.T)

    diag = np.sqrt(np.clip(np.diag(corr), EPS, None))
    corr = corr / np.outer(diag, diag)
    corr = np.clip(0.5 * (corr + corr.T), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _series_from_input(
    values: pd.Series | dict[str, float] | np.ndarray | list[float] | None,
    labels: list[Any] | None = None,
    name: str = "values",
    fill_value: float = 0.0,
) -> pd.Series:
    if values is None:
        if labels is None:
            return pd.Series(dtype=np.float64, name=name)
        return pd.Series(fill_value, index=labels, dtype=np.float64, name=name)
    if isinstance(values, pd.Series):
        out = values.astype(np.float64, copy=False)
    elif isinstance(values, dict):
        out = pd.Series(values, dtype=np.float64, name=name)
    else:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")
        if labels is None:
            labels = [str(i) for i in range(arr.shape[0])]
        if len(labels) != arr.shape[0]:
            raise ValueError(f"{name} length does not match labels.")
        out = pd.Series(arr, index=labels, dtype=np.float64, name=name)
    if labels is not None:
        out = out.reindex(labels).fillna(fill_value)
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill_value)


def _normalize_hedge_candidates(
    hedge_candidates: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
    scenarios: tuple[str, ...],
) -> pd.DataFrame:
    if isinstance(hedge_candidates, pd.DataFrame):
        df = hedge_candidates.copy()
    elif isinstance(hedge_candidates, dict):
        if all(isinstance(v, dict) for v in hedge_candidates.values()):
            df = pd.DataFrame.from_dict(hedge_candidates, orient="index")
            if "instrument" not in df.columns:
                df["instrument"] = df.index.astype(str)
        else:
            df = pd.DataFrame(hedge_candidates)
    else:
        df = pd.DataFrame(hedge_candidates)

    if df.empty:
        return pd.DataFrame(columns=["instrument", "type", "cost", "unit_notional", "max_notional"])

    if "instrument" not in df.columns:
        df["instrument"] = [f"hedge_{i}" for i in range(len(df))]
    if "type" not in df.columns:
        df["type"] = "UNKNOWN"

    for col in ("cost", "unit_notional", "max_notional"):
        if col not in df.columns:
            raise KeyError(f"hedge_candidates missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for scenario in scenarios:
        col = f"pnl_{scenario}"
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df["cost"] = df["cost"].fillna(0.0)
    df["unit_notional"] = df["unit_notional"].fillna(0.0)
    df["max_notional"] = df["max_notional"].fillna(0.0)
    for scenario in scenarios:
        df[f"pnl_{scenario}"] = df[f"pnl_{scenario}"].fillna(0.0)

    valid = (df["cost"] > EPS) & (df["unit_notional"] > EPS) & (df["max_notional"] > EPS)
    df = df.loc[valid].copy()
    df["instrument"] = df["instrument"].astype(str)
    df["max_units"] = df["max_notional"] / df["unit_notional"]
    df["max_units"] = df["max_units"].clip(lower=0.0)
    return df.reset_index(drop=True)


@dataclass
class ConvexHedgeOptimizer:
    """
    Daily convex hedge optimizer for a short-gamma statistical-arbitrage book.

    The optimizer maps current book geometry into a baseline hedge notional:

        Z_agg = sqrt((w * z)' rho_panic (w * z))
        H_t   = k1 * max(0, Z_agg - z0)^2 + k2 * PanicProb_t

    It then solves a linear cost minimization problem over long-only option
    proxies such as SPY puts and VIX calls:

        minimize    cost' x
        subject to  book_pnl_s + payout_s' x >= -max_loss, for each scenario s
                    sum(x) >= H_t
                    0 <= x <= max_notional

    Here x is hedge notional, cost is daily premium/decay per notional, and
    payout_s is the scenario payout multiplier per notional.
    """

    k1: float = 1_000_000.0
    k2: float = 250_000.0
    z0: float = 1.0
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS
    hedge_names: tuple[str, ...] = ("SPY_PUT", "VIX_CALL")
    daily_cost: dict[str, float] = field(default_factory=lambda: {
        "SPY_PUT": 0.015,
        "VIX_CALL": 0.020,
    })
    payout_multipliers: dict[str, dict[str, float]] = field(default_factory=lambda: {
        "gap_down": {"SPY_PUT": 0.45, "VIX_CALL": 0.18},
        "vol_spike": {"SPY_PUT": 0.12, "VIX_CALL": 0.75},
        "corr_collapse": {"SPY_PUT": 0.18, "VIX_CALL": 0.35},
    })
    max_notional: dict[str, float] | None = None
    max_daily_cost: float | None = None

    def calculate_optimal_hedge(
        self,
        weights: pd.Series | dict[str, float] | np.ndarray | list[float],
        zscores: pd.Series | dict[str, float] | np.ndarray | list[float],
        panic_corr: pd.DataFrame | np.ndarray,
        panic_prob: float,
        scenario_book_pnl: pd.Series | dict[str, float],
        max_loss: float,
        current_hedges: pd.Series | dict[str, float] | None = None,
    ) -> dict[str, Any]:
        z_agg = self.calculate_aggregate_stress(weights, zscores, panic_corr)
        required_notional = self.calculate_required_notional(z_agg, panic_prob)
        book_pnl = _series_from_input(scenario_book_pnl, labels=list(self.scenarios), name="scenario_book_pnl")
        cost = self._cost_vector()
        payout = self._payout_matrix()
        bounds = self._bounds()

        if required_notional <= EPS and self._scenario_constraints_satisfied(np.zeros(len(self.hedge_names)), payout, book_pnl, max_loss):
            x = np.zeros(len(self.hedge_names), dtype=np.float64)
            status = "no_hedge_required"
        else:
            x, status = self._solve_linprog(required_notional, book_pnl, max_loss, cost, payout, bounds)
            if x is None:
                x, greedy_status = self._solve_greedy(required_notional, book_pnl, max_loss, cost, payout, bounds)
                status = f"greedy_fallback: {status}; {greedy_status}"

        return self._format_result(
            z_agg=z_agg,
            required_notional=required_notional,
            hedge_notional=x,
            current_hedges=current_hedges,
            book_pnl=book_pnl,
            max_loss=max_loss,
            cost=cost,
            payout=payout,
            solver_status=status,
        )

    def calculate_aggregate_stress(
        self,
        weights: pd.Series | dict[str, float] | np.ndarray | list[float],
        zscores: pd.Series | dict[str, float] | np.ndarray | list[float],
        panic_corr: pd.DataFrame | np.ndarray,
    ) -> float:
        labels = self._book_labels(weights, zscores, panic_corr)
        if not labels:
            return 0.0
        w = _series_from_input(weights, labels=labels, name="weights")
        z = _series_from_input(zscores, labels=labels, name="zscores")
        corr = self._aligned_corr(panic_corr, labels)
        wz = w.to_numpy(dtype=np.float64) * z.to_numpy(dtype=np.float64)
        stress_var = max(float(wz @ corr @ wz), 0.0)
        return float(np.sqrt(stress_var))

    def calculate_required_notional(self, z_agg: float, panic_prob: float) -> float:
        z_agg = max(float(np.nan_to_num(z_agg, nan=0.0)), 0.0)
        panic_prob = float(np.clip(np.nan_to_num(panic_prob, nan=0.0), 0.0, 1.0))
        excess = max(0.0, z_agg - float(self.z0))
        return float(self.k1 * excess * excess + self.k2 * panic_prob)

    def _solve_linprog(
        self,
        required_notional: float,
        book_pnl: pd.Series,
        max_loss: float,
        cost: np.ndarray,
        payout: np.ndarray,
        bounds: list[tuple[float, float | None]],
    ) -> tuple[np.ndarray | None, str]:
        a_ub = []
        b_ub = []

        for i, scenario in enumerate(self.scenarios):
            # book_pnl_s + payout_s @ x >= -max_loss
            a_ub.append(-payout[i])
            b_ub.append(float(max_loss) + float(book_pnl.get(scenario, 0.0)))

        if required_notional > EPS:
            a_ub.append(-np.ones(len(self.hedge_names), dtype=np.float64))
            b_ub.append(-float(required_notional))

        if self.max_daily_cost is not None:
            a_ub.append(cost)
            b_ub.append(float(self.max_daily_cost))

        res = linprog(
            c=cost,
            A_ub=np.asarray(a_ub, dtype=np.float64),
            b_ub=np.asarray(b_ub, dtype=np.float64),
            bounds=bounds,
            method="highs",
        )
        if not res.success or res.x is None:
            return None, f"linprog_failed: {res.message}"
        x = np.clip(np.nan_to_num(res.x, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        return x, f"linprog:{res.message}"

    def _solve_greedy(
        self,
        required_notional: float,
        book_pnl: pd.Series,
        max_loss: float,
        cost: np.ndarray,
        payout: np.ndarray,
        bounds: list[tuple[float, float | None]],
    ) -> tuple[np.ndarray, str]:
        x = np.zeros(len(self.hedge_names), dtype=np.float64)
        max_x = np.array([np.inf if b[1] is None else float(b[1]) for b in bounds], dtype=np.float64)

        for _ in range(max(10, 5 * len(self.hedge_names))):
            hedge_pnl = payout @ x
            deficits = np.array([
                max((-max_loss) - (float(book_pnl.get(s, 0.0)) + hedge_pnl[i]), 0.0)
                for i, s in enumerate(self.scenarios)
            ])
            notional_deficit = max(float(required_notional) - float(x.sum()), 0.0)
            if deficits.sum() <= EPS and notional_deficit <= EPS:
                break

            remaining = max_x - x
            available = remaining > 1e-10
            if not available.any():
                break

            benefits = np.maximum(payout, 0.0).T @ deficits
            if notional_deficit > EPS:
                benefits += notional_deficit / max(float(required_notional), EPS)
            score = np.divide(benefits, cost, out=np.zeros_like(benefits), where=cost > EPS)
            score[~available] = -np.inf
            best = int(np.argmax(score))
            if not np.isfinite(score[best]) or score[best] <= 0.0:
                break

            needed = []
            if notional_deficit > EPS:
                needed.append(notional_deficit)
            for i, deficit in enumerate(deficits):
                if deficit > EPS and payout[i, best] > EPS:
                    needed.append(deficit / payout[i, best])
            add = min(max(needed) if needed else remaining[best], remaining[best])
            if self.max_daily_cost is not None:
                remaining_cost = float(self.max_daily_cost) - float(cost @ x)
                add = min(add, max(remaining_cost / max(cost[best], EPS), 0.0))
            if add <= 1e-10:
                break
            x[best] += add

        return x, "greedy_completed"

    def _format_result(
        self,
        z_agg: float,
        required_notional: float,
        hedge_notional: np.ndarray,
        current_hedges: pd.Series | dict[str, float] | None,
        book_pnl: pd.Series,
        max_loss: float,
        cost: np.ndarray,
        payout: np.ndarray,
        solver_status: str,
    ) -> dict[str, Any]:
        target = pd.Series(hedge_notional, index=self.hedge_names, dtype=np.float64, name="target_notional")
        current = _series_from_input(current_hedges, labels=list(self.hedge_names), name="current_hedges")
        orders = target - current
        hedge_pnl = pd.Series(payout @ hedge_notional, index=self.scenarios, dtype=np.float64)
        scenario_coverage = {}
        for scenario in self.scenarios:
            book = float(book_pnl.get(scenario, 0.0))
            hedge = float(hedge_pnl.get(scenario, 0.0))
            total = book + hedge
            scenario_coverage[scenario] = {
                "book_pnl": book,
                "hedge_pnl": hedge,
                "total_pnl": total,
                "min_allowed_pnl": -float(max_loss),
                "satisfied": bool(total >= -float(max_loss) - 1e-8),
            }
        notional_ok = float(target.sum()) + 1e-8 >= float(required_notional)
        scenarios_ok = all(bool(v["satisfied"]) for v in scenario_coverage.values())
        cost_value = pd.Series(cost * hedge_notional, index=self.hedge_names, dtype=np.float64)

        allocation = pd.DataFrame({
            "target_notional": target,
            "current_notional": current,
            "order_notional": orders,
            "daily_cost": cost_value,
        })
        return {
            "z_agg": float(z_agg),
            "required_hedge_notional": float(required_notional),
            "target_hedges": target,
            "hedge_orders": allocation,
            "total_daily_cost": float(cost_value.sum()),
            "scenario_coverage": scenario_coverage,
            "constraints_satisfied": bool(notional_ok and scenarios_ok),
            "emergency_deleverage_required": not bool(notional_ok and scenarios_ok),
            "solver_status": solver_status,
        }

    def _cost_vector(self) -> np.ndarray:
        return np.array([max(float(self.daily_cost.get(h, 0.0)), EPS) for h in self.hedge_names], dtype=np.float64)

    def _payout_matrix(self) -> np.ndarray:
        matrix = np.zeros((len(self.scenarios), len(self.hedge_names)), dtype=np.float64)
        for i, scenario in enumerate(self.scenarios):
            scenario_payoffs = self.payout_multipliers.get(scenario, {})
            for j, hedge in enumerate(self.hedge_names):
                matrix[i, j] = max(float(scenario_payoffs.get(hedge, 0.0)), 0.0)
        return matrix

    def _bounds(self) -> list[tuple[float, float | None]]:
        if self.max_notional is None:
            return [(0.0, None) for _ in self.hedge_names]
        return [(0.0, max(float(self.max_notional.get(h, np.inf)), 0.0)) for h in self.hedge_names]

    def _book_labels(
        self,
        weights: pd.Series | dict[str, float] | np.ndarray | list[float],
        zscores: pd.Series | dict[str, float] | np.ndarray | list[float],
        panic_corr: pd.DataFrame | np.ndarray,
    ) -> list[Any]:
        if isinstance(weights, (pd.Series, dict)):
            labels = list(pd.Series(weights).index)
            z_idx = set(pd.Series(zscores).index) if isinstance(zscores, (pd.Series, dict)) else set(labels)
            labels = [label for label in labels if label in z_idx]
            if isinstance(panic_corr, pd.DataFrame):
                corr_idx = set(panic_corr.index).intersection(set(panic_corr.columns))
                labels = [label for label in labels if label in corr_idx]
            return labels
        arr = np.asarray(weights, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError("weights must be one-dimensional.")
        return [str(i) for i in range(arr.shape[0])]

    def _aligned_corr(self, panic_corr: pd.DataFrame | np.ndarray, labels: list[Any]) -> np.ndarray:
        if isinstance(panic_corr, pd.DataFrame):
            corr = panic_corr.reindex(index=labels, columns=labels).fillna(0.0).to_numpy(dtype=np.float64)
        else:
            corr = np.asarray(panic_corr, dtype=np.float64)
            if corr.shape != (len(labels), len(labels)):
                raise ValueError("panic_corr shape must match weights and zscores.")
        return _project_correlation_psd(corr)

    def _scenario_constraints_satisfied(
        self,
        hedge_notional: np.ndarray,
        payout: np.ndarray,
        book_pnl: pd.Series,
        max_loss: float,
    ) -> bool:
        hedge_pnl = payout @ hedge_notional
        for i, scenario in enumerate(self.scenarios):
            if float(book_pnl.get(scenario, 0.0)) + float(hedge_pnl[i]) < -float(max_loss) - 1e-8:
                return False
        return True


@dataclass
class ConvexOverlayManager:
    k1: float = 1_000_000.0
    k2: float = 250_000.0
    z0: float = 1.0
    alpha: float = 0.50
    max_premium_budget: float | None = None
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS
    alpha_scenario: str = "gap_down"
    solver_order: tuple[str, ...] = ("CLARABEL", "ECOS", "OSQP", "SCS")
    solver_options: dict[str, dict[str, Any]] = field(default_factory=dict)

    def compute_aggregate_stress(
        self,
        weights: pd.Series | dict[str, float] | np.ndarray | list[float],
        zscores: pd.Series | dict[str, float] | np.ndarray | list[float],
        panic_corr: pd.DataFrame | np.ndarray,
    ) -> float:
        labels = self._book_labels(weights, zscores, panic_corr)
        if not labels:
            return 0.0

        w = _series_from_input(weights, labels=labels, name="weights")
        z = _series_from_input(zscores, labels=labels, name="zscores")
        corr = self._aligned_corr(panic_corr, labels)

        z_weighted = (w.to_numpy(dtype=np.float64) * z.to_numpy(dtype=np.float64))
        z_agg = float(z_weighted @ corr @ z_weighted)
        return max(z_agg, 0.0)

    def compute_required_hedge_notional(self, z_agg: float, panic_prob: float) -> float:
        z_agg = max(float(np.nan_to_num(z_agg, nan=0.0)), 0.0)
        panic_prob = float(np.clip(np.nan_to_num(panic_prob, nan=0.0), 0.0, 1.0))
        stress_excess = max(0.0, z_agg - float(self.z0))
        return float(self.k1 * stress_excess * stress_excess + self.k2 * panic_prob)

    def optimize_overlay(
        self,
        weights: pd.Series | dict[str, float] | np.ndarray | list[float],
        zscores: pd.Series | dict[str, float] | np.ndarray | list[float],
        panic_corr: pd.DataFrame | np.ndarray,
        panic_prob: float,
        hedge_candidates: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
        scenario_book_pnl: pd.Series | dict[str, float],
        max_loss: float,
        current_hedges: pd.Series | dict[str, float] | None = None,
    ) -> dict[str, Any]:
        z_agg = self.compute_aggregate_stress(weights, zscores, panic_corr)
        required_notional = self.compute_required_hedge_notional(z_agg, panic_prob)
        candidates = _normalize_hedge_candidates(hedge_candidates, self.scenarios)
        book_pnl = self._scenario_series(scenario_book_pnl)
        current_units = _series_from_input(
            current_hedges,
            labels=candidates["instrument"].tolist() if not candidates.empty else [],
            name="current_hedges",
        )

        if candidates.empty:
            return self.evaluate_overlay(
                z_agg=z_agg,
                required_hedge_notional=required_notional,
                hedge_candidates=candidates,
                target_units=np.zeros(0, dtype=np.float64),
                current_units=current_units,
                scenario_book_pnl=book_pnl,
                max_loss=max_loss,
                solver_status="no_candidates",
            )

        try:
            target_units, status = self._solve_cvxpy(
                candidates=candidates,
                scenario_book_pnl=book_pnl,
                required_hedge_notional=required_notional,
                max_loss=float(max_loss),
            )
        except Exception as exc:
            target_units, status = self._solve_greedy(
                candidates=candidates,
                scenario_book_pnl=book_pnl,
                required_hedge_notional=required_notional,
                max_loss=float(max_loss),
            )
            status = f"greedy_fallback: {type(exc).__name__}: {exc}; {status}"

        return self.evaluate_overlay(
            z_agg=z_agg,
            required_hedge_notional=required_notional,
            hedge_candidates=candidates,
            target_units=target_units,
            current_units=current_units,
            scenario_book_pnl=book_pnl,
            max_loss=max_loss,
            solver_status=status,
        )

    def evaluate_overlay(
        self,
        z_agg: float,
        required_hedge_notional: float,
        hedge_candidates: pd.DataFrame,
        target_units: np.ndarray,
        current_units: pd.Series,
        scenario_book_pnl: pd.Series,
        max_loss: float,
        solver_status: str,
    ) -> dict[str, Any]:
        target_units = np.asarray(target_units, dtype=np.float64)
        if hedge_candidates.empty:
            selected = pd.DataFrame(
                columns=["instrument", "type", "target_units", "current_units", "order_units", "target_notional", "cost"]
            )
            scenario_coverage = self._coverage_dict(
                scenario_book_pnl=scenario_book_pnl,
                scenario_hedge_pnl=pd.Series(0.0, index=self.scenarios),
                max_loss=max_loss,
            )
            constraints_satisfied = self._constraints_satisfied(
                scenario_coverage, 0.0, required_hedge_notional, scenario_hedge_pnl=pd.Series(0.0, index=self.scenarios),
                scenario_book_pnl=scenario_book_pnl,
            )
            return {
                "z_agg": float(z_agg),
                "required_hedge_notional": float(required_hedge_notional),
                "selected_hedges": selected,
                "total_hedge_cost": 0.0,
                "total_hedge_notional": 0.0,
                "scenario_coverage": scenario_coverage,
                "constraints_satisfied": bool(constraints_satisfied),
                "emergency_deleverage_required": not bool(constraints_satisfied),
                "solver_status": solver_status,
            }

        instruments = hedge_candidates["instrument"].tolist()
        current_units = current_units.reindex(instruments).fillna(0.0)
        unit_notional = hedge_candidates["unit_notional"].to_numpy(dtype=np.float64)
        costs = hedge_candidates["cost"].to_numpy(dtype=np.float64)
        target_notional = target_units * unit_notional
        target_cost = target_units * costs

        scenario_hedge_pnl = {}
        for scenario in self.scenarios:
            pnl_vec = hedge_candidates[f"pnl_{scenario}"].to_numpy(dtype=np.float64)
            scenario_hedge_pnl[scenario] = float(pnl_vec @ target_units)
        scenario_hedge_pnl_s = pd.Series(scenario_hedge_pnl, dtype=np.float64)
        scenario_coverage = self._coverage_dict(scenario_book_pnl, scenario_hedge_pnl_s, max_loss)

        current_units_arr = current_units.to_numpy(dtype=np.float64)
        selected_mask = (target_units > 1e-10) | (np.abs(current_units_arr) > 1e-10)
        selected = hedge_candidates.loc[selected_mask, ["instrument", "type"]].copy()
        if selected.empty:
            selected = hedge_candidates.loc[:, ["instrument", "type"]].head(0).copy()
        selected_idx = selected.index.to_numpy(dtype=np.int64)
        selected["target_units"] = target_units[selected_idx]
        selected["current_units"] = current_units_arr[selected_idx]
        selected["order_units"] = selected["target_units"] - selected["current_units"]
        selected["target_notional"] = target_notional[selected_idx]
        selected["cost"] = target_cost[selected_idx]
        for scenario in self.scenarios:
            selected[f"pnl_{scenario}"] = (
                hedge_candidates[f"pnl_{scenario}"].to_numpy(dtype=np.float64)[selected_idx]
                * target_units[selected_idx]
            )

        total_notional = float(target_notional.sum())
        total_cost = float(target_cost.sum())
        constraints_satisfied = self._constraints_satisfied(
            scenario_coverage=scenario_coverage,
            total_notional=total_notional,
            required_notional=required_hedge_notional,
            scenario_hedge_pnl=scenario_hedge_pnl_s,
            scenario_book_pnl=scenario_book_pnl,
        )

        return {
            "z_agg": float(z_agg),
            "required_hedge_notional": float(required_hedge_notional),
            "selected_hedges": selected.reset_index(drop=True),
            "total_hedge_cost": total_cost,
            "total_hedge_notional": total_notional,
            "scenario_coverage": scenario_coverage,
            "constraints_satisfied": bool(constraints_satisfied),
            "emergency_deleverage_required": not bool(constraints_satisfied),
            "solver_status": solver_status,
        }

    def _solve_cvxpy(
        self,
        candidates: pd.DataFrame,
        scenario_book_pnl: pd.Series,
        required_hedge_notional: float,
        max_loss: float,
    ) -> tuple[np.ndarray, str]:
        try:
            import cvxpy as cp
        except ImportError as exc:
            raise RuntimeError("cvxpy is required for convex overlay optimization.") from exc

        n = len(candidates)
        q = cp.Variable(n, nonneg=True)
        costs = candidates["cost"].to_numpy(dtype=np.float64)
        unit_notional = candidates["unit_notional"].to_numpy(dtype=np.float64)
        max_units = candidates["max_units"].to_numpy(dtype=np.float64)

        constraints = [q <= max_units]
        if required_hedge_notional > EPS:
            constraints.append(unit_notional @ q >= float(required_hedge_notional))
        if self.max_premium_budget is not None:
            constraints.append(costs @ q <= float(self.max_premium_budget))

        for scenario in self.scenarios:
            pnl_vec = candidates[f"pnl_{scenario}"].to_numpy(dtype=np.float64)
            constraints.append(float(scenario_book_pnl.get(scenario, 0.0)) + pnl_vec @ q >= -float(max_loss))

        if self.alpha_scenario in self.scenarios:
            shock_loss = max(-float(scenario_book_pnl.get(self.alpha_scenario, 0.0)), 0.0)
            if shock_loss > EPS:
                pnl_vec = candidates[f"pnl_{self.alpha_scenario}"].to_numpy(dtype=np.float64)
                constraints.append(pnl_vec @ q >= float(self.alpha) * shock_loss)

        problem = cp.Problem(cp.Minimize(costs @ q), constraints)
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
            if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and q.value is not None:
                units = np.asarray(q.value, dtype=np.float64)
                units = np.clip(np.nan_to_num(units, nan=0.0, posinf=0.0, neginf=0.0), 0.0, max_units)
                return units, f"cvxpy:{solver}:{problem.status}"
            errors.append(f"{solver}: status={problem.status}")

        raise RuntimeError("No cvxpy solver converged. " + " | ".join(errors))

    def _solve_greedy(
        self,
        candidates: pd.DataFrame,
        scenario_book_pnl: pd.Series,
        required_hedge_notional: float,
        max_loss: float,
    ) -> tuple[np.ndarray, str]:
        n = len(candidates)
        units = np.zeros(n, dtype=np.float64)
        max_units = candidates["max_units"].to_numpy(dtype=np.float64)
        costs = candidates["cost"].to_numpy(dtype=np.float64)
        unit_notional = candidates["unit_notional"].to_numpy(dtype=np.float64)
        scenario_matrix = np.vstack([
            candidates[f"pnl_{scenario}"].to_numpy(dtype=np.float64) for scenario in self.scenarios
        ])

        for _ in range(max(10, n * 5)):
            hedge_pnl = scenario_matrix @ units
            total_notional = float(unit_notional @ units)
            scenario_deficits = np.array([
                max((-max_loss) - (float(scenario_book_pnl.get(scenario, 0.0)) + hedge_pnl[i]), 0.0)
                for i, scenario in enumerate(self.scenarios)
            ], dtype=np.float64)
            notional_deficit = max(float(required_hedge_notional) - total_notional, 0.0)

            alpha_deficit = 0.0
            if self.alpha_scenario in self.scenarios:
                alpha_idx = self.scenarios.index(self.alpha_scenario)
                shock_loss = max(-float(scenario_book_pnl.get(self.alpha_scenario, 0.0)), 0.0)
                alpha_deficit = max(float(self.alpha) * shock_loss - hedge_pnl[alpha_idx], 0.0)

            if scenario_deficits.sum() <= EPS and notional_deficit <= EPS and alpha_deficit <= EPS:
                break

            remaining = max_units - units
            available = remaining > 1e-10
            if not available.any():
                break

            benefits = np.zeros(n, dtype=np.float64)
            if scenario_deficits.sum() > EPS:
                positive_pnl = np.maximum(scenario_matrix, 0.0)
                benefits += scenario_deficits @ positive_pnl
            if notional_deficit > EPS:
                benefits += notional_deficit * unit_notional / max(float(required_hedge_notional), EPS)
            if alpha_deficit > EPS and self.alpha_scenario in self.scenarios:
                alpha_idx = self.scenarios.index(self.alpha_scenario)
                benefits += alpha_deficit * np.maximum(scenario_matrix[alpha_idx], 0.0)

            scores = np.divide(benefits, costs, out=np.zeros_like(benefits), where=costs > EPS)
            scores[~available] = -np.inf
            best = int(np.argmax(scores))
            if not np.isfinite(scores[best]) or scores[best] <= 0.0:
                break

            needed: list[float] = []
            if notional_deficit > EPS and unit_notional[best] > EPS:
                needed.append(notional_deficit / unit_notional[best])
            for i, deficit in enumerate(scenario_deficits):
                pnl_unit = scenario_matrix[i, best]
                if deficit > EPS and pnl_unit > EPS:
                    needed.append(deficit / pnl_unit)
            if alpha_deficit > EPS and self.alpha_scenario in self.scenarios:
                alpha_idx = self.scenarios.index(self.alpha_scenario)
                pnl_unit = scenario_matrix[alpha_idx, best]
                if pnl_unit > EPS:
                    needed.append(alpha_deficit / pnl_unit)

            add_units = max(needed) if needed else remaining[best]
            add_units = min(max(add_units, 0.0), remaining[best])
            if self.max_premium_budget is not None:
                remaining_budget = float(self.max_premium_budget) - float(costs @ units)
                add_units = min(add_units, max(remaining_budget / max(costs[best], EPS), 0.0))
            if add_units <= 1e-10:
                break
            units[best] += add_units

        return units, "greedy_completed"

    def _book_labels(
        self,
        weights: pd.Series | dict[str, float] | np.ndarray | list[float],
        zscores: pd.Series | dict[str, float] | np.ndarray | list[float],
        panic_corr: pd.DataFrame | np.ndarray,
    ) -> list[Any]:
        if isinstance(weights, pd.Series) or isinstance(weights, dict):
            w_labels = list(pd.Series(weights).index)
            z_idx = set(pd.Series(zscores).index) if isinstance(zscores, (pd.Series, dict)) else set(w_labels)
            labels = [label for label in w_labels if label in z_idx]
            if isinstance(panic_corr, pd.DataFrame):
                corr_idx = set(panic_corr.index).intersection(set(panic_corr.columns))
                labels = [label for label in labels if label in corr_idx]
            return labels

        arr = np.asarray(weights, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError("weights must be one-dimensional.")
        return [str(i) for i in range(arr.shape[0])]

    def _aligned_corr(self, panic_corr: pd.DataFrame | np.ndarray, labels: list[Any]) -> np.ndarray:
        if isinstance(panic_corr, pd.DataFrame):
            corr = panic_corr.reindex(index=labels, columns=labels).fillna(0.0).to_numpy(dtype=np.float64)
        else:
            corr = np.asarray(panic_corr, dtype=np.float64)
            if corr.shape != (len(labels), len(labels)):
                raise ValueError("panic_corr shape must match weights and zscores.")
        return _project_correlation_psd(corr)

    def _scenario_series(self, scenario_book_pnl: pd.Series | dict[str, float]) -> pd.Series:
        out = _series_from_input(scenario_book_pnl, labels=list(self.scenarios), name="scenario_book_pnl")
        return out.reindex(self.scenarios).fillna(0.0)

    def _coverage_dict(
        self,
        scenario_book_pnl: pd.Series,
        scenario_hedge_pnl: pd.Series,
        max_loss: float,
    ) -> dict[str, dict[str, float | bool]]:
        coverage: dict[str, dict[str, float | bool]] = {}
        for scenario in self.scenarios:
            book = float(scenario_book_pnl.get(scenario, 0.0))
            hedge = float(scenario_hedge_pnl.get(scenario, 0.0))
            total = book + hedge
            coverage[scenario] = {
                "book_pnl": book,
                "hedge_pnl": hedge,
                "total_pnl": total,
                "min_allowed_pnl": -float(max_loss),
                "satisfied": bool(total >= -float(max_loss) - 1e-8),
            }
        return coverage

    def _constraints_satisfied(
        self,
        scenario_coverage: dict[str, dict[str, float | bool]],
        total_notional: float,
        required_notional: float,
        scenario_hedge_pnl: pd.Series,
        scenario_book_pnl: pd.Series,
    ) -> bool:
        scenarios_ok = all(bool(v["satisfied"]) for v in scenario_coverage.values())
        notional_ok = total_notional + 1e-8 >= float(required_notional)
        alpha_ok = True
        if self.alpha_scenario in self.scenarios:
            shock_loss = max(-float(scenario_book_pnl.get(self.alpha_scenario, 0.0)), 0.0)
            if shock_loss > EPS:
                alpha_ok = float(scenario_hedge_pnl.get(self.alpha_scenario, 0.0)) + 1e-8 >= float(self.alpha) * shock_loss
        budget_ok = True
        return bool(scenarios_ok and notional_ok and alpha_ok and budget_ok)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n_pairs = 20
    pairs = [f"pair_{i:02d}" for i in range(n_pairs)]

    weights = pd.Series(rng.normal(0.0, 0.08, size=n_pairs), index=pairs)
    weights = weights / max(weights.abs().sum(), EPS) * 2.0
    zscores = pd.Series(rng.normal(0.0, 1.2, size=n_pairs), index=pairs)
    zscores.iloc[:6] = np.array([4.2, -3.9, 3.4, -4.5, 3.1, -3.6])

    rho = np.full((n_pairs, n_pairs), 0.82)
    np.fill_diagonal(rho, 1.0)
    panic_corr = pd.DataFrame(rho, index=pairs, columns=pairs)

    scenario_book_pnl = pd.Series({
        "gap_down": -1_250_000.0,
        "vol_spike": -850_000.0,
        "corr_collapse": -950_000.0,
    })

    optimizer = ConvexHedgeOptimizer(
        k1=1_250_000.0,
        k2=400_000.0,
        z0=0.35,
        daily_cost={"SPY_PUT": 0.015, "VIX_CALL": 0.020},
        payout_multipliers={
            "gap_down": {"SPY_PUT": 0.50, "VIX_CALL": 0.18},
            "vol_spike": {"SPY_PUT": 0.12, "VIX_CALL": 0.80},
            "corr_collapse": {"SPY_PUT": 0.20, "VIX_CALL": 0.38},
        },
        max_notional={"SPY_PUT": 4_000_000.0, "VIX_CALL": 3_000_000.0},
        max_daily_cost=140_000.0,
    )
    result = optimizer.calculate_optimal_hedge(
        weights=weights,
        zscores=zscores,
        panic_corr=panic_corr,
        panic_prob=0.85,
        scenario_book_pnl=scenario_book_pnl,
        max_loss=350_000.0,
        current_hedges={"SPY_PUT": 250_000.0, "VIX_CALL": 0.0},
    )

    print(f"Z_agg:                   {result['z_agg']:.6f}")
    print(f"Required hedge notional: ${result['required_hedge_notional']:,.2f}")
    print(f"Total daily cost:        ${result['total_daily_cost']:,.2f}")
    print(f"Solver status:           {result['solver_status']}")
    print("\nTarget hedge notionals and orders:")
    print(result["hedge_orders"].round(2).to_string())
    print("\nScenario coverage:")
    for scenario, coverage in result["scenario_coverage"].items():
        print(
            f"  {scenario:<15} book={coverage['book_pnl']:>+10,.0f} "
            f"hedge={coverage['hedge_pnl']:>+10,.0f} "
            f"total={coverage['total_pnl']:>+10,.0f} "
            f"ok={coverage['satisfied']}"
        )
    print(f"\nConstraints satisfied:       {result['constraints_satisfied']}")
    print(f"Emergency deleveraging:      {result['emergency_deleverage_required']}")
