"""Fold-wise association screening (simple GWAS) without label leakage."""
from __future__ import annotations

import numpy as np
from scipy import stats


def gwas_neglog10_p(
    X: np.ndarray,
    y: np.ndarray,
    covariates: np.ndarray | None = None,
) -> np.ndarray:
    """Per-SNP association -log10(p) using OLS with optional covariates.

    X: (n_samples, n_snps), already imputed.
    y: (n_samples,)
    covariates: (n_samples, n_cov) e.g. PCs; intercept always included.
    """
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    if covariates is None:
        Z = np.ones((n, 1), dtype=float)
    else:
        Z = np.column_stack([np.ones(n), np.asarray(covariates, dtype=float)])

    # Residualize y on covariates
    beta_y, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
    y_res = y - Z @ beta_y
    y_ss = float(np.dot(y_res, y_res))
    if y_ss < 1e-12:
        return np.zeros(p, dtype=float)

    # Residualize each SNP on covariates via projection
    # For speed: if covariates are few, compute Q from QR
    q, _ = np.linalg.qr(Z, mode="reduced")
    X_res = X - q @ (q.T @ X)
    # correlation / t-test equivalent
    x_ss = np.einsum("ij,ij->j", X_res, X_res)
    xy = X_res.T @ y_res
    with np.errstate(invalid="ignore", divide="ignore"):
        r = xy / np.sqrt(x_ss * y_ss)
        r = np.clip(r, -0.999999, 0.999999)
        df = n - Z.shape[1] - 1
        t = r * np.sqrt(df / (1.0 - r * r))
        pvals = 2.0 * stats.t.sf(np.abs(t), df)
        neglog = -np.log10(np.clip(pvals, 1e-300, 1.0))
    neglog[~np.isfinite(neglog)] = 0.0
    neglog[x_ss < 1e-12] = 0.0
    return neglog.astype(float)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(int(k), len(scores))
    if k <= 0:
        return np.array([], dtype=int)
    # argpartition for top-k
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]
