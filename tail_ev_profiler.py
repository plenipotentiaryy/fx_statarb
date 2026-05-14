from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import genpareto
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from regime_memory import apply_parameter_drift_truncation


@dataclass(frozen=True)
class _TailParams:
    xi: float
    beta: float
    threshold: float
    exceedance_rate: float
    var_alpha: float
    fit_count: int


class TailAdjustedEVProfiler:
    """
    Tail-adjusted Expected Value profiler using POT/EVT + logistic reversion model.

    Required columns in fit/predict input:
        - zscore
        - velocity
        - vol_ratio
        - hurst

    Required in fit only:
        - revert_label  (1 if TP/reversion hit before stop, else 0)

    Optional:
        - expected_gain : precomputed gain in normalized spread units
        - tail_loss     : adverse shock magnitude to fit tails on; defaults to abs(zscore)

    Notes:
        - EV is expressed in normalized spread units.
        - GPD tail fitting is done on absolute shocks above threshold `u`.
        - During predict, the logistic model is frozen after train; only the tail model
          may be updated chunkwise using observed OOS shocks if `tail_refit_freq` is set.
    """

    feature_cols = ("zscore", "velocity", "vol_ratio", "hurst")

    def __init__(
        self,
        tail_threshold: float = 2.5,
        confidence_level: float = 0.95,
        rr_threshold: float = 0.5,
        exit_z: float = 0.0,
        label_col: str = "revert_label",
        gain_col: str | None = None,
        tail_col: str | None = None,
        min_tail_exceedances: int = 50,
        logistic_c: float = 1.0,
        logistic_max_iter: int = 2000,
        tail_refit_freq: str | None = None,
        clip_feature_abs: float = 12.0,
        random_state: int = 42,
        half_life_col: str = "half_life",
        state_col: str = "macro_state",
        hl_drift_threshold_q: float = 2.0,
        drift_recent_days: int = 90,
    ) -> None:
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if tail_threshold <= 0.0:
            raise ValueError("tail_threshold must be > 0")
        if rr_threshold <= 0.0:
            raise ValueError("rr_threshold must be > 0")
        self.tail_threshold = float(tail_threshold)
        self.confidence_level = float(confidence_level)
        self.rr_threshold = float(rr_threshold)
        self.exit_z = float(exit_z)
        self.label_col = label_col
        self.gain_col = gain_col
        self.tail_col = tail_col
        self.min_tail_exceedances = int(min_tail_exceedances)
        self.tail_refit_freq = tail_refit_freq
        self.clip_feature_abs = float(clip_feature_abs)
        self.half_life_col = half_life_col
        self.state_col = state_col
        self.hl_drift_threshold_q = float(hl_drift_threshold_q)
        self.drift_recent_days = int(drift_recent_days)

        self._logit = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=logistic_c,
                        max_iter=logistic_max_iter,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        self._tail_params: _TailParams | None = None
        self._tail_history: np.ndarray | None = None
        self._fitted = False

    def fit(self, train_df: pd.DataFrame, sample_weights: np.ndarray | None = None) -> "TailAdjustedEVProfiler":
        df = self._prepare_frame(train_df, require_label=True)
        if df.empty:
            raise ValueError("No valid rows remain after filtering training data.")

        X = df.loc[:, self.feature_cols].to_numpy(dtype=np.float64)
        y = df[self.label_col].astype(np.int8).to_numpy()
        if np.unique(y).size < 2:
            raise ValueError("Training labels must contain both classes for logistic fit.")

        fit_params = {}
        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights, dtype=np.float64)
            if sample_weights.ndim != 1:
                raise ValueError("sample_weights must be a 1D array.")
            if sample_weights.shape[0] == len(train_df):
                sample_weights = pd.Series(sample_weights, index=train_df.index).reindex(df.index).to_numpy(dtype=np.float64)
            elif sample_weights.shape[0] != len(df):
                raise ValueError("sample_weights must align to either raw train_df or filtered training rows.")
            sample_weights = np.nan_to_num(sample_weights, nan=0.0, posinf=0.0, neginf=0.0)
            sample_weights = np.clip(sample_weights, 0.0, None)
            if sample_weights.sum() <= 0.0:
                raise ValueError("sample_weights must contain positive mass.")
            fit_params["clf__sample_weight"] = sample_weights

        self._logit.fit(X, y, **fit_params)

        tail_fit_df = apply_parameter_drift_truncation(
            df,
            half_life_col=self.half_life_col,
            drift_threshold_q=self.hl_drift_threshold_q,
            recent_days=self.drift_recent_days,
        )
        tail_series = self._tail_series(tail_fit_df)
        self._tail_history = tail_series.to_numpy(dtype=np.float64, copy=True)
        self._tail_params = self._fit_tail_params(self._tail_history)
        self._fitted = True
        return self

    def predict_ev(self, oos_df: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        df = self._prepare_frame(oos_df, require_label=False)
        if df.empty:
            return df.copy()

        if self.tail_refit_freq is None:
            scored = self._score_block(df, self._tail_params)
            return scored

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("tail_refit_freq requires a DatetimeIndex on oos_df.")

        blocks: list[pd.DataFrame] = []
        history = self._tail_history.copy()
        params = self._tail_params

        period_index = df.index.tz_localize(None).to_period(self.tail_refit_freq)
        for _, block in df.groupby(period_index, sort=True):
            blocks.append(self._score_block(block, params))
            history = np.concatenate(
                [history, self._tail_series(block).to_numpy(dtype=np.float64, copy=False)]
            )
            params = self._fit_tail_params(history)

        out = pd.concat(blocks).sort_index()
        return out

    def _score_block(self, df: pd.DataFrame, tail_params: _TailParams) -> pd.DataFrame:
        X = df.loc[:, self.feature_cols].to_numpy(dtype=np.float64)
        p_revert = self._logit.predict_proba(X)[:, 1]

        shock_abs = self._tail_series(df).to_numpy(dtype=np.float64, copy=False)
        expected_gain = self._expected_gain(df, shock_abs)
        es_95 = self._expected_shortfall_from_state(shock_abs, tail_params)
        ev = p_revert * expected_gain - (1.0 - p_revert) * es_95
        ratio = np.divide(
            ev,
            es_95,
            out=np.full_like(ev, -np.inf, dtype=np.float64),
            where=es_95 > 0.0,
        )
        approved = (ev > 0.0) & (ratio > self.rr_threshold)

        out = df.copy()
        out["p_revert"] = p_revert
        out["expected_gain"] = expected_gain
        out["tail_var"] = tail_params.var_alpha
        out["tail_es_95"] = es_95
        out["tail_ev"] = ev
        out["tail_rr_ratio"] = ratio
        out["tail_signal_ok"] = approved
        out["tail_xi"] = tail_params.xi
        out["tail_beta"] = tail_params.beta
        out["tail_threshold"] = tail_params.threshold
        out["tail_exceedance_rate"] = tail_params.exceedance_rate
        return out

    def _prepare_frame(self, df: pd.DataFrame, require_label: bool) -> pd.DataFrame:
        required = list(self.feature_cols)
        if require_label:
            required.append(self.label_col)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        out = df.copy()
        for col in self.feature_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].clip(-self.clip_feature_abs, self.clip_feature_abs)
        if require_label:
            out[self.label_col] = pd.to_numeric(out[self.label_col], errors="coerce")
        if self.gain_col and self.gain_col in out.columns:
            out[self.gain_col] = pd.to_numeric(out[self.gain_col], errors="coerce")
        if self.tail_col and self.tail_col in out.columns:
            out[self.tail_col] = pd.to_numeric(out[self.tail_col], errors="coerce")
        if self.half_life_col in out.columns:
            out[self.half_life_col] = pd.to_numeric(out[self.half_life_col], errors="coerce")

        subset = list(self.feature_cols)
        if require_label:
            subset.append(self.label_col)
        out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=subset)
        return out

    def _tail_series(self, df: pd.DataFrame) -> pd.Series:
        if self.tail_col and self.tail_col in df.columns:
            series = pd.to_numeric(df[self.tail_col], errors="coerce").abs()
        else:
            series = pd.to_numeric(df["zscore"], errors="coerce").abs()
        return series.replace([np.inf, -np.inf], np.nan).dropna()

    def _expected_gain(self, df: pd.DataFrame, shock_abs: np.ndarray) -> np.ndarray:
        if self.gain_col and self.gain_col in df.columns:
            gain = pd.to_numeric(df[self.gain_col], errors="coerce").to_numpy(dtype=np.float64)
            return np.nan_to_num(gain, nan=0.0, posinf=0.0, neginf=0.0)
        return np.maximum(shock_abs - abs(self.exit_z), 0.0)

    def _fit_tail_params(self, shocks_abs: np.ndarray) -> _TailParams:
        shocks_abs = np.asarray(shocks_abs, dtype=np.float64)
        shocks_abs = shocks_abs[np.isfinite(shocks_abs)]
        if shocks_abs.size == 0:
            raise ValueError("Cannot fit tail model on empty shock history.")

        exceed_mask = shocks_abs > self.tail_threshold
        exceed = shocks_abs[exceed_mask] - self.tail_threshold
        exceedance_rate = float(exceed_mask.mean())

        if exceed.size == 0:
            beta = max(float(np.std(shocks_abs)), 1e-6)
            xi = 0.0
            var_alpha = max(float(np.quantile(shocks_abs, self.confidence_level)), self.tail_threshold)
            return _TailParams(
                xi=xi,
                beta=beta,
                threshold=self.tail_threshold,
                exceedance_rate=max(exceedance_rate, 1.0 / shocks_abs.size),
                var_alpha=var_alpha,
                fit_count=int(shocks_abs.size),
            )

        if exceed.size < self.min_tail_exceedances:
            xi = 0.0
            beta = max(float(exceed.mean()), 1e-6)
        else:
            xi_hat, _, beta_hat = genpareto.fit(exceed, floc=0.0)
            if not np.isfinite(xi_hat):
                xi_hat = 0.0
            if not np.isfinite(beta_hat) or beta_hat <= 0.0:
                beta_hat = float(exceed.mean())
            xi = float(np.clip(xi_hat, -0.5, 0.95))
            beta = max(float(beta_hat), 1e-6)

        var_alpha = self._tail_var_from_params(
            xi=xi,
            beta=beta,
            threshold=self.tail_threshold,
            exceedance_rate=max(exceedance_rate, 1.0 / shocks_abs.size),
            confidence_level=self.confidence_level,
            empirical_shocks=shocks_abs,
        )

        return _TailParams(
            xi=xi,
            beta=beta,
            threshold=self.tail_threshold,
            exceedance_rate=max(exceedance_rate, 1.0 / shocks_abs.size),
            var_alpha=var_alpha,
            fit_count=int(shocks_abs.size),
        )

    @staticmethod
    def _tail_var_from_params(
        xi: float,
        beta: float,
        threshold: float,
        exceedance_rate: float,
        confidence_level: float,
        empirical_shocks: np.ndarray,
    ) -> float:
        tail_mass = max(exceedance_rate, 1e-8)
        alpha = float(confidence_level)

        if 1.0 - alpha >= tail_mass:
            return max(float(np.quantile(empirical_shocks, alpha)), threshold)

        scale = (1.0 - alpha) / tail_mass
        scale = max(scale, 1e-12)
        if abs(xi) < 1e-8:
            return threshold - beta * np.log(scale)
        return threshold + (beta / xi) * (scale ** (-xi) - 1.0)

    @staticmethod
    def _vectorized_mean_excess(levels: np.ndarray, params: _TailParams) -> np.ndarray:
        levels = np.asarray(levels, dtype=np.float64)
        levels = np.maximum(levels, params.threshold)
        if abs(params.xi) < 1e-8:
            return np.full_like(levels, params.beta)
        scale_x = params.beta + params.xi * (levels - params.threshold)
        scale_x = np.maximum(scale_x, 1e-8)
        if params.xi >= 1.0:
            return np.full_like(levels, np.inf)
        return scale_x / max(1.0 - params.xi, 1e-8)

    def _expected_shortfall_from_state(
        self,
        shock_abs: np.ndarray,
        params: _TailParams,
    ) -> np.ndarray:
        current_level = np.maximum.reduce(
            [
                np.asarray(shock_abs, dtype=np.float64),
                np.full_like(shock_abs, params.threshold, dtype=np.float64),
                np.full_like(shock_abs, params.var_alpha, dtype=np.float64),
            ]
        )
        mean_excess = self._vectorized_mean_excess(current_level, params)
        es_total = current_level + mean_excess
        es_loss = np.maximum(es_total - np.asarray(shock_abs, dtype=np.float64), 0.0)
        return np.nan_to_num(es_loss, nan=np.inf, posinf=np.inf)

    def _require_fitted(self) -> None:
        if not self._fitted or self._tail_params is None or self._tail_history is None:
            raise RuntimeError("TailAdjustedEVProfiler must be fitted before predict_ev().")


def _build_synthetic_case(n: int = 6000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = np.zeros(n, dtype=np.float64)
    vol = np.ones(n, dtype=np.float64)

    jump_flags = rng.random(n) < 0.01
    shocks = rng.standard_t(df=4, size=n) * 0.28
    jumps = jump_flags * rng.normal(0.0, 3.2, size=n)

    for i in range(1, n):
        if abs(z[i - 1]) > 3.0:
            phi = 0.80
        else:
            phi = 0.92
        vol[i] = 0.95 * vol[i - 1] + 0.05 * (1.0 + abs(shocks[i]))
        z[i] = phi * z[i - 1] + shocks[i] + jumps[i]

    velocity = np.diff(z, prepend=z[0])
    rolling_std = pd.Series(z).rolling(60, min_periods=20).std().bfill().to_numpy()
    vol_ratio = rolling_std / max(float(np.nanmedian(rolling_std)), 1e-6)
    hurst = np.clip(0.42 + 0.02 * np.abs(z) + 0.03 * (vol_ratio - 1.0), 0.35, 0.70)

    horizon = 30
    tp_z = 0.3
    stop_z = 4.0
    labels = np.zeros(n, dtype=np.int8)
    gains = np.zeros(n, dtype=np.float64)

    for i in range(n - horizon - 1):
        zi = z[i]
        if abs(zi) < 2.0:
            continue
        future = z[i + 1 : i + horizon + 1]
        if zi > 0:
            tp_hits = np.flatnonzero(future <= tp_z)
            sl_hits = np.flatnonzero(future >= stop_z)
            gains[i] = max(abs(zi) - tp_z, 0.0)
        else:
            tp_hits = np.flatnonzero(future >= -tp_z)
            sl_hits = np.flatnonzero(future <= -stop_z)
            gains[i] = max(abs(zi) - tp_z, 0.0)
        first_tp = tp_hits[0] if tp_hits.size else np.inf
        first_sl = sl_hits[0] if sl_hits.size else np.inf
        labels[i] = int(first_tp < first_sl)

    idx = pd.date_range("2020-01-01", periods=n, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "zscore": z,
            "velocity": velocity,
            "vol_ratio": vol_ratio,
            "hurst": hurst,
            "revert_label": labels,
            "expected_gain": gains,
            "tail_loss": np.abs(z),
        },
        index=idx,
    )
    return df.dropna()


def _demo() -> None:
    df = _build_synthetic_case()
    split = int(len(df) * 0.7)
    train = df.iloc[:split]
    test = df.iloc[split:]

    profiler = TailAdjustedEVProfiler(
        tail_threshold=2.5,
        confidence_level=0.95,
        rr_threshold=0.5,
        gain_col="expected_gain",
        tail_col="tail_loss",
        tail_refit_freq="W",
    )
    profiler.fit(train)
    scored = profiler.predict_ev(test)

    approved = scored["tail_signal_ok"]
    print("Synthetic EVT demo")
    print(f"train rows: {len(train):,}  |  test rows: {len(test):,}")
    print(f"approved signals: {int(approved.sum()):,} / {len(scored):,}")
    print(
        "mean tail EV on approved: "
        f"{scored.loc[approved, 'tail_ev'].mean():+.4f}"
        if approved.any()
        else "mean tail EV on approved: n/a"
    )
    print(
        "mean ES95 on approved: "
        f"{scored.loc[approved, 'tail_es_95'].mean():.4f}"
        if approved.any()
        else "mean ES95 on approved: n/a"
    )
    cols = [
        "zscore",
        "velocity",
        "vol_ratio",
        "hurst",
        "p_revert",
        "tail_es_95",
        "tail_ev",
        "tail_rr_ratio",
        "tail_signal_ok",
    ]
    print(scored.loc[:, cols].head(10).to_string())


if __name__ == "__main__":
    _demo()
