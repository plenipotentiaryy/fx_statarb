from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from execution_stress import CounterfactualTester, ExecutionSimulator
from regime_block_bootstrap import RegimeBlockBootstrap, attach_regime_to_trades


@dataclass
class ValidationStack:
    """
    Unified strict-validation runner.

    This coordinates:
    - temporal leakage checks
    - execution/counterfactual robustness tests
    - regime-aware block bootstrap confidence intervals
    """

    execution_tester: CounterfactualTester | None = None
    bootstrapper: RegimeBlockBootstrap | None = None

    def __post_init__(self) -> None:
        if self.execution_tester is None:
            self.execution_tester = CounterfactualTester(base_simulator=ExecutionSimulator())
        if self.bootstrapper is None:
            self.bootstrapper = RegimeBlockBootstrap()

    def run(
        self,
        signals_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        trades_df: pd.DataFrame | None = None,
        regime_series: pd.Series | None = None,
        train_index: pd.Index | None = None,
        valid_index: pd.Index | None = None,
        test_index: pd.Index | None = None,
        lookahead_bars: int = 0,
    ) -> dict[str, Any]:
        results: dict[str, Any] = {
            "leakage_audit": audit_temporal_leakage(
                train_index=train_index,
                valid_index=valid_index,
                test_index=test_index,
                lookahead_bars=lookahead_bars,
            ),
            "counterfactual": self.execution_tester.run_counterfactual_suite(signals_df, prices_df),
        }

        if trades_df is not None:
            boot_input = trades_df.copy()
            if regime_series is not None:
                boot_input = attach_regime_to_trades(boot_input, regime_series)
            results["bootstrap"] = self.bootstrapper.bootstrap_metrics(boot_input)
        return results


def audit_temporal_leakage(
    train_index: pd.Index | None,
    valid_index: pd.Index | None,
    test_index: pd.Index | None,
    lookahead_bars: int = 0,
) -> dict[str, Any]:
    """
    Audit the most important temporal leakage channels.

    The check is intentionally strict: any overlap across train/valid/test or
    any label lookahead spilling into the subsequent split is reported.
    """
    issues: list[str] = []
    checks: dict[str, bool] = {}

    def _normalize(idx: pd.Index | None) -> pd.Index:
        if idx is None:
            return pd.Index([])
        return pd.Index(pd.to_datetime(idx))

    train = _normalize(train_index)
    valid = _normalize(valid_index)
    test = _normalize(test_index)

    checks["train_valid_overlap"] = len(train.intersection(valid)) == 0
    checks["train_test_overlap"] = len(train.intersection(test)) == 0
    checks["valid_test_overlap"] = len(valid.intersection(test)) == 0

    if not checks["train_valid_overlap"]:
        issues.append("Inner train overlaps inner validation.")
    if not checks["train_test_overlap"]:
        issues.append("Outer train overlaps OOS test.")
    if not checks["valid_test_overlap"]:
        issues.append("Inner validation overlaps OOS test.")

    checks["label_lookahead_safe_train_valid"] = True
    checks["label_lookahead_safe_valid_test"] = True
    if lookahead_bars > 0 and len(train) > 0 and len(valid) > 0:
        spill = min(len(train), int(lookahead_bars))
        checks["label_lookahead_safe_train_valid"] = train[-spill:].max() < valid.min()
        if not checks["label_lookahead_safe_train_valid"]:
            issues.append("Lookahead labels from inner train spill into inner validation.")
    if lookahead_bars > 0 and len(valid) > 0 and len(test) > 0:
        spill = min(len(valid), int(lookahead_bars))
        checks["label_lookahead_safe_valid_test"] = valid[-spill:].max() < test.min()
        if not checks["label_lookahead_safe_valid_test"]:
            issues.append("Lookahead labels from inner validation spill into OOS test.")

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "issues": issues,
    }


if __name__ == "__main__":
    idx = pd.date_range("2025-01-01", periods=400, freq="h", tz="UTC")
    signals = pd.DataFrame({"zscore": 1.8 * np.sin(np.linspace(0, 18, len(idx))), "weight": 1.0}, index=idx)
    prices = pd.DataFrame(
        {
            "open": 100 + np.cumsum(np.random.default_rng(7).normal(0, 0.05, len(idx))),
            "close": 100 + np.cumsum(np.random.default_rng(8).normal(0, 0.05, len(idx))),
            "vwap": 100 + np.cumsum(np.random.default_rng(9).normal(0, 0.05, len(idx))),
            "vol_ratio": 1.0,
            "hmm_regime": 0,
        },
        index=idx,
    )
    trades = pd.DataFrame(
        {
            "exit_time": idx[::12][:20],
            "net_pnl": np.random.default_rng(10).normal(0.1, 1.0, 20),
        }
    )

    runner = ValidationStack(
        execution_tester=CounterfactualTester(
            base_simulator=ExecutionSimulator(),
            entry_z=0.9,
            exit_z=0.2,
        )
    )
    result = runner.run(
        signals_df=signals,
        prices_df=prices,
        trades_df=trades,
        regime_series=prices["hmm_regime"].resample("D").last(),
        train_index=idx[:220],
        valid_index=idx[220:320],
        test_index=idx[320:],
        lookahead_bars=10,
    )
    print(result["counterfactual"].round(4).to_string())
    print(result["leakage_audit"])
    if "bootstrap" in result:
        print(result["bootstrap"]["realized"])
