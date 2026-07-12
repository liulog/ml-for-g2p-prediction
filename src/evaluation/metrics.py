"""Regression and breeding-selection metrics for G2P evaluation."""
from __future__ import annotations

import numpy as np
from scipy import stats


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 3:
        return {
            "n": float(len(y_true)),
            "pearson_r": np.nan,
            "spearman_rho": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
        }
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        pearson_r = np.nan
        spearman_rho = np.nan
    else:
        pearson_r = float(stats.pearsonr(y_true, y_pred)[0])
        spearman_rho = float(stats.spearmanr(y_true, y_pred)[0])
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {
        "n": float(len(y_true)),
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def topk_overlap(y_true: np.ndarray, y_pred: np.ndarray, frac: float = 0.10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    n = len(y_true)
    if n == 0:
        return np.nan
    k = max(1, int(round(n * frac)))
    true_top = set(np.argsort(y_true)[-k:])
    pred_top = set(np.argsort(y_pred)[-k:])
    return float(len(true_top & pred_top) / k)
