from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    signal_index: int
    fill_index: int | None
    fill_time: pd.Timestamp | None
    side: int
    requested_price: float
    fill_price: float
    effective_spread: float
    slippage: float
    fill_probability: float
    reason: str


@dataclass
class ExecutionSimulator:
    """
    Execution friction model for strict validation.

    The simulator precomputes dynamic spread and limit-fill probability vectors
    for the full bar series, then exposes simulate_order() as the API a
    backtester can call when a signal wants an actual fill.
    """

    base_spread: float = 0.0001
    spread_gamma: float = 1.5
    panic_multiplier: float = 3.0
    entry_delay: int = 1
    slippage_variance: float = 0.0
    fill_kappa: float = 8.0
    passive: bool = True
    use_vwap: bool = True
    random_seed: int = 42
    toxicity_threshold: float | None = None
    price_col: str = "close"
    vwap_col: str = "vwap"
    open_col: str = "open"
    vol_ratio_col: str = "vol_ratio"
    hmm_col: str = "hmm_regime"
    tox_buy_col: str = "tox_buy"
    tox_sell_col: str = "tox_sell"

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)
        self._cache_key: tuple[int, int] | None = None
        self._precomputed: pd.DataFrame | None = None

    def precompute(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        index = prices_df.index
        vol_ratio = self._series_or_default(prices_df, self.vol_ratio_col, 1.0)
        panic_state = self._series_or_default(prices_df, self.hmm_col, 0.0)
        panic_flag = pd.to_numeric(panic_state, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) > 0

        vol_arr = pd.to_numeric(vol_ratio, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0)
        spread = (
            float(self.base_spread)
            * (1.0 + float(self.spread_gamma) * np.maximum(vol_arr.to_numpy(dtype=np.float64) - 1.0, 0.0))
            * np.where(panic_flag, float(self.panic_multiplier), 1.0)
        )
        spread = np.nan_to_num(spread, nan=self.base_spread, posinf=self.base_spread * self.panic_multiplier, neginf=self.base_spread)

        mid = self._execution_reference_price(prices_df)
        distance = 0.5 * spread
        denom = np.maximum(np.abs(mid.to_numpy(dtype=np.float64)), EPS)
        normalized_distance = distance / denom
        fill_probability = np.exp(-float(self.fill_kappa) * normalized_distance)
        fill_probability = np.clip(fill_probability, 0.0, 1.0)

        out = pd.DataFrame(
            {
                "effective_spread": spread,
                "fill_probability": fill_probability,
                "execution_mid": mid.to_numpy(dtype=np.float64),
            },
            index=index,
        )
        self._precomputed = out
        self._cache_key = (id(prices_df), len(prices_df))
        return out

    def simulate_order(
        self,
        prices_df: pd.DataFrame,
        signal_index: int,
        side: int,
        requested_price: float | None = None,
        order_type: str = "limit",
        force_fill: bool = False,
    ) -> ExecutionResult:
        """
        Return actual fill status and price for one order.

        side: +1 buy / long, -1 sell / short.
        Stops and panic exits should pass order_type="market" or force_fill=True.
        """
        if side not in (-1, 1):
            raise ValueError("side must be +1 for buy or -1 for sell.")
        if signal_index < 0 or signal_index >= len(prices_df):
            raise IndexError("signal_index out of bounds.")

        fill_index = int(signal_index + max(int(self.entry_delay), 0))
        if fill_index >= len(prices_df):
            return ExecutionResult(
                status="MISSED",
                signal_index=signal_index,
                fill_index=None,
                fill_time=None,
                side=side,
                requested_price=float("nan") if requested_price is None else float(requested_price),
                fill_price=float("nan"),
                effective_spread=0.0,
                slippage=0.0,
                fill_probability=0.0,
                reason="NO_BAR_AFTER_LATENCY",
            )

        pre = self._ensure_precomputed(prices_df)
        mid = float(pre["execution_mid"].iloc[fill_index])
        effective_spread = float(pre["effective_spread"].iloc[fill_index])
        fill_probability = float(pre["fill_probability"].iloc[fill_index])
        requested = mid if requested_price is None else float(requested_price)

        if order_type.lower() == "limit" and self._should_cancel_toxic(prices_df, fill_index, side):
            return ExecutionResult(
                status="CANCELED",
                signal_index=signal_index,
                fill_index=fill_index,
                fill_time=prices_df.index[fill_index],
                side=side,
                requested_price=requested,
                fill_price=float("nan"),
                effective_spread=effective_spread,
                slippage=0.0,
                fill_probability=0.0,
                reason="TOXIC_FLOW_CANCEL",
            )

        if order_type.lower() == "limit" and self.passive and not force_fill:
            distance_to_mid = abs(requested - mid)
            denom = max(abs(mid), EPS)
            fill_probability = float(np.exp(-float(self.fill_kappa) * distance_to_mid / denom))
            fill_probability = float(np.clip(fill_probability, 0.0, 1.0))
            if self.rng.random() > fill_probability:
                return ExecutionResult(
                    status="MISSED",
                    signal_index=signal_index,
                    fill_index=fill_index,
                    fill_time=prices_df.index[fill_index],
                    side=side,
                    requested_price=requested,
                    fill_price=float("nan"),
                    effective_spread=effective_spread,
                    slippage=0.0,
                    fill_probability=fill_probability,
                    reason="LIMIT_NOT_FILLED",
                )

        spread_penalty = 0.5 * effective_spread if order_type.lower() == "limit" else effective_spread
        slippage = self._adverse_slippage()
        fill_price = mid + side * (spread_penalty + slippage)
        return ExecutionResult(
            status="FILLED",
            signal_index=signal_index,
            fill_index=fill_index,
            fill_time=prices_df.index[fill_index],
            side=side,
            requested_price=requested,
            fill_price=float(fill_price),
            effective_spread=effective_spread,
            slippage=float(slippage),
            fill_probability=fill_probability,
            reason="FILLED",
        )

    def stress_exit_price(self, prices_df: pd.DataFrame, bar_index: int, side_to_close: int) -> float:
        """
        Market exit price with maximum spread penalty.

        side_to_close is +1 for buy-to-cover, -1 for sell-to-close.
        """
        pre = self._ensure_precomputed(prices_df)
        idx = min(max(int(bar_index), 0), len(prices_df) - 1)
        mid = float(pre["execution_mid"].iloc[idx])
        spread = float(pre["effective_spread"].iloc[idx])
        return float(mid + int(side_to_close) * (spread + self._adverse_slippage()))

    def _adverse_slippage(self) -> float:
        if self.slippage_variance <= 0.0:
            return 0.0
        return float(abs(self.rng.normal(0.0, np.sqrt(float(self.slippage_variance)))))

    def _ensure_precomputed(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        key = (id(prices_df), len(prices_df))
        if self._precomputed is None or self._cache_key != key:
            return self.precompute(prices_df)
        return self._precomputed

    def _series_or_default(self, df: pd.DataFrame, col: str, default: float) -> pd.Series:
        if col in df.columns:
            return df[col]
        return pd.Series(default, index=df.index, dtype=np.float64)

    def _should_cancel_toxic(self, prices_df: pd.DataFrame, fill_index: int, side: int) -> bool:
        if self.toxicity_threshold is None:
            return False
        col = self.tox_buy_col if side == 1 else self.tox_sell_col
        if col not in prices_df.columns:
            return False
        tox = pd.to_numeric(prices_df[col], errors="coerce").iloc[fill_index]
        if pd.isna(tox):
            return False
        return float(tox) > float(self.toxicity_threshold)

    def _execution_reference_price(self, prices_df: pd.DataFrame) -> pd.Series:
        if self.use_vwap and self.vwap_col in prices_df.columns:
            ref = prices_df[self.vwap_col]
        elif self.open_col in prices_df.columns:
            ref = prices_df[self.open_col]
        elif self.price_col in prices_df.columns:
            ref = prices_df[self.price_col]
        else:
            raise KeyError(f"prices_df must contain one of {self.vwap_col}, {self.open_col}, {self.price_col}.")
        return pd.to_numeric(ref, errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().bfill()


@dataclass
class CounterfactualTester:
    base_simulator: ExecutionSimulator
    entry_z: float = 1.5
    exit_z: float = 0.25
    stop_z: float = 3.0
    max_hold_bars: int = 30
    weight_noise: float = 0.10
    hedge_ratio_noise: float = 0.10
    random_seed: int = 7

    def run_counterfactual_suite(self, signals_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        scenarios = {
            "Baseline": self.base_simulator,
            "Delayed Execution": replace(self.base_simulator, entry_delay=2, random_seed=self.base_simulator.random_seed + 11),
            "Delayed Execution 3 Bars": replace(self.base_simulator, entry_delay=3, random_seed=self.base_simulator.random_seed + 17),
            "Stressed Microstructure": replace(
                self.base_simulator,
                base_spread=self.base_simulator.base_spread * 2.0,
                slippage_variance=max(self.base_simulator.slippage_variance, self.base_simulator.base_spread ** 2),
                random_seed=self.base_simulator.random_seed + 23,
            ),
            "Widened Stops": replace(self.base_simulator, random_seed=self.base_simulator.random_seed + 29),
            "Perturbed Hedge Ratio": replace(self.base_simulator, random_seed=self.base_simulator.random_seed + 31),
            "Perturbed Weights": replace(self.base_simulator, random_seed=self.base_simulator.random_seed + 37),
            "Sector Pair Shuffle": replace(self.base_simulator, random_seed=self.base_simulator.random_seed + 41),
        }

        rows = []
        for name, simulator in scenarios.items():
            trades = self._run_vectorized_backtest(
                signals_df,
                prices_df,
                simulator,
                perturbed_weights=name == "Perturbed Weights",
                perturbed_hedge_ratio=name == "Perturbed Hedge Ratio",
                widened_stops=name == "Widened Stops",
                shuffled_pairs=name == "Sector Pair Shuffle",
            )
            rows.append(self._summary_row(name, trades))
        return pd.DataFrame(rows).set_index("scenario")

    def run_robustness_suite(self, signals_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        return self.run_counterfactual_suite(signals_df, prices_df)

    def _run_vectorized_backtest(
        self,
        signals_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        simulator: ExecutionSimulator,
        perturbed_weights: bool = False,
        perturbed_hedge_ratio: bool = False,
        widened_stops: bool = False,
        shuffled_pairs: bool = False,
    ) -> pd.DataFrame:
        if "zscore" not in signals_df.columns:
            raise KeyError("signals_df must contain zscore.")
        if "close" not in prices_df.columns:
            raise KeyError("prices_df must contain close.")

        aligned = signals_df.join(prices_df, how="inner", rsuffix="_px")
        if aligned.empty:
            return pd.DataFrame(columns=["entry_time", "exit_time", "side", "net_pnl", "status"])

        prices = prices_df.reindex(aligned.index).copy()
        simulator.precompute(prices)
        working = aligned.copy()
        if shuffled_pairs:
            working = self._shuffle_within_cluster(working)

        z = pd.to_numeric(working["zscore"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        close = pd.to_numeric(prices["close"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)
        weights = (
            pd.to_numeric(working["weight"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
            if "weight" in working.columns else np.ones(len(working), dtype=np.float64)
        )
        if perturbed_weights:
            rng = np.random.default_rng(self.random_seed)
            weights = weights * rng.uniform(1.0 - self.weight_noise, 1.0 + self.weight_noise, size=len(weights))
        hedge_ratio = (
            pd.to_numeric(working["hedge_ratio"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
            if "hedge_ratio" in working.columns else np.ones(len(working), dtype=np.float64)
        )
        if perturbed_hedge_ratio:
            rng = np.random.default_rng(self.random_seed + 101)
            hedge_ratio = hedge_ratio * rng.uniform(
                1.0 - self.hedge_ratio_noise,
                1.0 + self.hedge_ratio_noise,
                size=len(hedge_ratio),
            )

        stop_z = abs(self.stop_z) * (1.5 if widened_stops else 1.0)

        trades: list[dict[str, Any]] = []
        i = 0
        while i < len(working) - 1:
            side = 0
            if z[i] <= -abs(self.entry_z):
                side = 1
            elif z[i] >= abs(self.entry_z):
                side = -1
            if side == 0:
                i += 1
                continue

            requested = float(close[i])
            fill = simulator.simulate_order(prices, i, side=side, requested_price=requested, order_type="limit")
            if fill.status != "FILLED" or fill.fill_index is None:
                trades.append({
                    "entry_time": aligned.index[i],
                    "exit_time": aligned.index[i],
                    "side": side,
                    "net_pnl": 0.0,
                    "status": fill.status,
                    "reason": fill.reason,
                })
                i += 1
                continue

            exit_idx = min(fill.fill_index + self.max_hold_bars, len(aligned) - 1)
            for j in range(fill.fill_index + 1, exit_idx + 1):
                hit_exit = (side == 1 and z[j] >= -abs(self.exit_z)) or (side == -1 and z[j] <= abs(self.exit_z))
                hit_stop = (side == 1 and z[j] <= -stop_z) or (side == -1 and z[j] >= stop_z)
                if hit_exit or hit_stop:
                    exit_idx = j
                    break

            exit_side = -side
            exit_price = simulator.stress_exit_price(prices, exit_idx, side_to_close=exit_side)
            gross = side * (exit_price - fill.fill_price)
            hedge_error = abs(hedge_ratio[min(exit_idx, len(hedge_ratio) - 1)] - hedge_ratio[i])
            gross -= hedge_error * abs(close[exit_idx] - close[fill.fill_index])
            # Two paid crossings: entry modeled in fill price, exit modeled by stress_exit_price.
            net_pnl = gross * float(weights[i])
            trades.append({
                "entry_time": fill.fill_time,
                "exit_time": working.index[exit_idx],
                "side": side,
                "entry_price": fill.fill_price,
                "exit_price": exit_price,
                "effective_spread": fill.effective_spread,
                "slippage": fill.slippage,
                "fill_probability": fill.fill_probability,
                "weight": float(weights[i]),
                "hedge_ratio": float(hedge_ratio[i]),
                "net_pnl": float(net_pnl),
                "status": "FILLED",
                "reason": "EXIT_SIGNAL" if exit_idx < fill.fill_index + self.max_hold_bars else "TIME_STOP",
            })
            i = max(exit_idx + 1, i + 1)

        return pd.DataFrame(trades)

    def _summary_row(self, scenario: str, trades: pd.DataFrame) -> dict[str, Any]:
        if trades.empty:
            return {
                "scenario": scenario,
                "trades": 0,
                "fills": 0,
                "missed": 0,
                "canceled": 0,
                "fill_rate": 0.0,
                "net_pnl": 0.0,
                "avg_trade": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "hit_rate": 0.0,
                "turnover": 0.0,
            }
        filled = trades[trades["status"] == "FILLED"]
        fills = len(filled)
        missed = int((trades["status"] == "MISSED").sum())
        canceled = int((trades["status"] == "CANCELED").sum()) if "status" in trades.columns else 0
        net_pnl = float(trades["net_pnl"].sum())
        pnl = filled["net_pnl"].to_numpy(dtype=np.float64) if fills > 0 else np.array([], dtype=np.float64)
        sharpe = float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(len(pnl))) if len(pnl) > 1 and pnl.std(ddof=1) > EPS else 0.0
        curve = np.cumsum(pnl) if len(pnl) > 0 else np.array([0.0], dtype=np.float64)
        running_max = np.maximum.accumulate(curve)
        max_drawdown = float((curve - running_max).min()) if len(curve) > 0 else 0.0
        hit_rate = float((pnl > 0).mean()) if len(pnl) > 0 else 0.0
        turnover = float(filled["weight"].abs().sum()) if "weight" in filled.columns else float(fills)
        return {
            "scenario": scenario,
            "trades": len(trades),
            "fills": fills,
            "missed": missed,
            "canceled": canceled,
            "fill_rate": fills / max(len(trades), 1),
            "net_pnl": net_pnl,
            "avg_trade": net_pnl / max(fills, 1),
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "hit_rate": hit_rate,
            "turnover": turnover,
        }

    def _shuffle_within_cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        if "sector" not in df.columns and "cluster" not in df.columns:
            return df
        group_col = "sector" if "sector" in df.columns else "cluster"
        out = df.copy()
        rng = np.random.default_rng(self.random_seed + 211)
        for _, idx in out.groupby(group_col).groups.items():
            idx_list = list(idx)
            if len(idx_list) < 2:
                continue
            shuffled = rng.permutation(idx_list)
            out.loc[idx_list, "zscore"] = out.loc[shuffled, "zscore"].to_numpy()
            if "weight" in out.columns:
                out.loc[idx_list, "weight"] = out.loc[shuffled, "weight"].to_numpy()
        return out


def run_robustness_suite(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    simulator: ExecutionSimulator | None = None,
) -> pd.DataFrame:
    simulator = simulator or ExecutionSimulator()
    tester = CounterfactualTester(base_simulator=simulator)
    return tester.run_robustness_suite(signals_df, prices_df)


def run_counterfactual_suite(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    simulator: ExecutionSimulator | None = None,
) -> pd.DataFrame:
    simulator = simulator or ExecutionSimulator()
    tester = CounterfactualTester(base_simulator=simulator)
    return tester.run_counterfactual_suite(signals_df, prices_df)


if __name__ == "__main__":
    rng = np.random.default_rng(123)
    n = 1_000
    idx = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    ret = rng.normal(0.0, 0.0006, size=n)
    close = 100.0 + np.cumsum(ret)
    open_px = close + rng.normal(0.0, 0.0002, size=n)
    vwap = 0.5 * (open_px + close)
    vol_ratio = pd.Series(1.0 + np.abs(rng.normal(0.0, 0.35, size=n)), index=idx).clip(0.5, 4.0)
    hmm = pd.Series(0, index=idx)
    hmm.iloc[700:780] = 1

    z = pd.Series(rng.normal(0.0, 0.8, size=n), index=idx)
    shock_points = np.arange(60, n, 95)
    z.iloc[shock_points] = rng.choice([-2.4, 2.4], size=len(shock_points))
    for p in shock_points:
        end = min(p + 18, n)
        z.iloc[p:end] = np.linspace(z.iloc[p], 0.0, end - p)

    prices = pd.DataFrame(
        {
            "open": open_px,
            "close": close,
            "vwap": vwap,
            "vol_ratio": vol_ratio,
            "hmm_regime": hmm,
        },
        index=idx,
    )
    signals = pd.DataFrame(
        {
            "zscore": z,
            "weight": 1.0,
        },
        index=idx,
    )

    base = ExecutionSimulator(
        base_spread=0.00008,
        spread_gamma=1.5,
        panic_multiplier=3.0,
        entry_delay=1,
        slippage_variance=0.0,
        fill_kappa=10.0,
        passive=True,
        random_seed=99,
    )
    comparison = run_robustness_suite(signals, prices, simulator=base)
    print(comparison.round(6).to_string())
