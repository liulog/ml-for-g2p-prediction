"""Explicit genomic prediction models including RR-BLUP."""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold

from src.evaluation.metrics import regression_metrics


class MeanModel:
    def __init__(self) -> None:
        self.mu_ = 0.0

    def fit(self, y: np.ndarray) -> "MeanModel":
        self.mu_ = float(np.mean(y))
        return self

    def predict(self, n: int) -> np.ndarray:
        return np.full(n, self.mu_, dtype=float)


class GBLUP:
    """GBLUP: (K_tt + λ I) α = y_c; ŷ = K_pt α + μ, λ=(1-h2)/h2."""

    def __init__(self, h2: float = 0.5) -> None:
        self.h2 = float(h2)
        self.mu_ = 0.0
        self.alpha_: np.ndarray | None = None
        self.train_idx_: np.ndarray | None = None
        self.K_: np.ndarray | None = None

    def fit(self, K: np.ndarray, y_full: np.ndarray, train_idx: np.ndarray) -> "GBLUP":
        self.K_ = K
        self.train_idx_ = np.asarray(train_idx, dtype=int)
        y_tr = np.asarray(y_full, dtype=float)[self.train_idx_]
        self.mu_ = float(np.mean(y_tr))
        y_c = y_tr - self.mu_
        h2 = min(max(self.h2, 1e-4), 1.0 - 1e-4)
        lam = (1.0 - h2) / h2
        K_tt = K[np.ix_(self.train_idx_, self.train_idx_)]
        self.alpha_ = np.linalg.solve(K_tt + lam * np.eye(len(self.train_idx_)), y_c)
        return self

    def predict(self, pred_idx: np.ndarray) -> np.ndarray:
        assert self.K_ is not None and self.alpha_ is not None and self.train_idx_ is not None
        K_pt = self.K_[np.ix_(np.asarray(pred_idx, dtype=int), self.train_idx_)]
        return K_pt @ self.alpha_ + self.mu_


class RRBLUP:
    """Ridge regression on marker dosages (RR-BLUP)."""

    def __init__(self, alpha: float = 100.0) -> None:
        self.alpha = float(alpha)
        self.model = Ridge(alpha=self.alpha, fit_intercept=True)
        self.mu_: np.ndarray | None = None
        self.sd_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RRBLUP":
        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self.sd_[self.sd_ < 1e-8] = 1.0
        self.model.fit((X - self.mu_) / self.sd_, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.mu_ is not None and self.sd_ is not None
        return self.model.predict((X - self.mu_) / self.sd_)


def choose_gblup_h2(
    K: np.ndarray,
    y_full: np.ndarray,
    train_idx: np.ndarray,
    h2_grid: list[float] | None = None,
    seed: int = 2026,
) -> float:
    if h2_grid is None:
        h2_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
    train_idx = np.asarray(train_idx, dtype=int)
    best_h2, best_score = 0.5, -np.inf
    n_splits = min(5, len(train_idx))
    if n_splits < 2:
        return best_h2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for h2 in h2_grid:
        preds = np.full(len(train_idx), np.nan)
        y_local = y_full[train_idx]
        for tr, va in kf.split(train_idx):
            tr_ids = train_idx[tr]
            va_ids = train_idx[va]
            model = GBLUP(h2=h2).fit(K, y_full, tr_ids)
            preds[va] = model.predict(va_ids)
        score = regression_metrics(y_local, preds)["pearson_r"]
        if np.isfinite(score) and score > best_score:
            best_score, best_h2 = score, h2
    return best_h2


def fit_elastic_net(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    random_state: int = 2026,
) -> ElasticNet:
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=2000,
        tol=1e-3,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def fit_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 2026,
) -> LGBMRegressor:
    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.3,
        random_state=random_state,
        verbosity=-1,
        force_col_wise=True,
    )
    model.fit(X_train, y_train)
    return model
