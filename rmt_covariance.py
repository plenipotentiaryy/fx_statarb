from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-12


def _prepare_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame.")
    if returns_df.empty:
        raise ValueError("returns_df is empty.")

    clean = returns_df.apply(pd.to_numeric, errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if clean.shape[0] < 2 or clean.shape[1] < 2:
        raise ValueError("returns_df must contain at least 2 assets and 2 complete observations.")
    return clean


def _corr_and_vols(returns_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], int, int]:
    clean = _prepare_returns(returns_df)
    values = clean.to_numpy(dtype=np.float64, copy=False)
    values = values - values.mean(axis=0, keepdims=True)

    vols = values.std(axis=0, ddof=1)
    safe_vols = np.where(vols > EPS, vols, 1.0)
    standardized = values / safe_vols
    corr = (standardized.T @ standardized) / max(values.shape[0] - 1, 1)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return corr, vols, clean.columns.tolist(), values.shape[0], values.shape[1]


def _project_to_correlation(matrix: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 1.0)

    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, EPS, None)
    matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
    matrix = 0.5 * (matrix + matrix.T)

    diag = np.sqrt(np.clip(np.diag(matrix), EPS, None))
    matrix = matrix / np.outer(diag, diag)
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def clean_covariance_rmt(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean an empirical covariance matrix via Marchenko-Pastur eigenvalue clipping.

    Parameters
    ----------
    returns_df
        DataFrame of asset returns with shape (T, N).

    Returns
    -------
    pd.DataFrame
        RMT-cleaned covariance matrix aligned to the surviving numeric columns.
    """
    corr, vols, columns, n_obs, n_assets = _corr_and_vols(returns_df)

    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    q = n_assets / n_obs
    lambda_plus = (1.0 + np.sqrt(q)) ** 2

    noise_mask = eigvals <= lambda_plus
    cleaned_eigvals = eigvals.copy()
    if noise_mask.any():
        cleaned_eigvals[noise_mask] = eigvals[noise_mask].mean()

    corr_clean = eigvecs @ np.diag(cleaned_eigvals) @ eigvecs.T
    corr_clean = _project_to_correlation(corr_clean)

    safe_vols = np.where(vols > EPS, vols, EPS)
    cov_clean = corr_clean * np.outer(safe_vols, safe_vols)
    cov_clean = 0.5 * (cov_clean + cov_clean.T)

    cov_eigvals, cov_eigvecs = np.linalg.eigh(cov_clean)
    cov_eigvals = np.clip(cov_eigvals, EPS, None)
    cov_clean = cov_eigvecs @ np.diag(cov_eigvals) @ cov_eigvecs.T
    cov_clean = 0.5 * (cov_clean + cov_clean.T)

    return pd.DataFrame(cov_clean, index=columns, columns=columns)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    t_steps = 300
    n_assets = 100
    n_factors = 3

    factor_returns = rng.normal(0.0, 0.02, size=(t_steps, n_factors))
    factor_loadings = rng.normal(0.0, 1.0, size=(n_assets, n_factors))
    idiosyncratic = rng.normal(0.0, 0.01, size=(t_steps, n_assets))
    returns = factor_returns @ factor_loadings.T + idiosyncratic

    columns = [f"asset_{i:03d}" for i in range(n_assets)]
    returns_df = pd.DataFrame(returns, columns=columns)

    corr_emp, vols_emp, cols, n_obs, n_dim = _corr_and_vols(returns_df)
    cov_clean_df = clean_covariance_rmt(returns_df)
    corr_clean = cov_clean_df.to_numpy(dtype=np.float64) / np.outer(vols_emp, vols_emp)
    corr_clean = _project_to_correlation(corr_clean)

    eig_emp = np.sort(np.linalg.eigvalsh(corr_emp))[::-1]
    eig_clean = np.sort(np.linalg.eigvalsh(corr_clean))[::-1]
    lambda_plus = (1.0 + np.sqrt(n_dim / n_obs)) ** 2

    print(f"T={n_obs}, N={n_dim}, q={n_dim / n_obs:.4f}, lambda_plus={lambda_plus:.4f}")
    print("Top 10 empirical corr eigenvalues:")
    print(np.round(eig_emp[:10], 6))
    print("Top 10 cleaned corr eigenvalues:")
    print(np.round(eig_clean[:10], 6))
    print(f"Trace empirical corr: {np.trace(corr_emp):.10f}")
    print(f"Trace cleaned corr:   {np.trace(corr_clean):.10f}")
    print(f"Trace preserved:      {np.isclose(np.trace(corr_emp), np.trace(corr_clean), atol=1e-8)}")
