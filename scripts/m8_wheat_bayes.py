#!/usr/bin/env python3
"""Plan remainder: wheat Bayesian Ridge / ARD (Bayes-style) vs GBLUP/RR-BLUP/LGBM.

Uses LD-pruned dosages + kinship GroupKFold on pilot traits.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import ARDRegression, BayesianRidge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.wheat import kinship_groups, load_wheat_arrays  # noqa: E402
from src.evaluation.metrics import regression_metrics, topk_overlap  # noqa: E402
from src.models.baselines import GBLUP, MeanModel, RRBLUP, fit_lightgbm  # noqa: E402


def load_cfg():
    with open(ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def eval_pack(y_true, y_pred):
    m = regression_metrics(y_true, y_pred)
    m["top10_overlap"] = topk_overlap(y_true, y_pred, 0.10)
    return m


def standardize(tr, te):
    mu, sd = tr.mean(0), tr.std(0)
    sd[sd < 1e-8] = 1.0
    return (tr - mu) / sd, (te - mu) / sd


def main() -> int:
    cfg = load_cfg()
    seed = cfg["project"]["seed"]
    data = load_wheat_arrays(ROOT / "data" / "interim" / "wheat")
    X, K, pheno = data["X_ld"], data["K"], data["pheno"]
    # BayesianRidge on top-2k var SNPs; ARD on top-400 (ARD is O(n*p^2) heavy)
    var = X.var(axis=0)
    keep_br = np.argsort(var)[-min(2000, X.shape[1]) :]
    keep_ard = np.argsort(var)[-min(400, X.shape[1]) :]
    Xbr = X[:, keep_br]
    Xard = X[:, keep_ard]
    groups = kinship_groups(K, cfg["cv"]["n_splits"])
    gkf = GroupKFold(n_splits=cfg["cv"]["n_splits"])

    rows = []
    for trait in cfg["wheat"]["pilot_traits"]:
        y = pd.to_numeric(pheno[trait], errors="coerce").to_numpy(float)
        y = np.where(np.isfinite(y), y, np.nanmean(y))
        print(f"[{trait}]", flush=True)
        for fold, (tr, te) in enumerate(gkf.split(np.arange(len(y)), groups=groups)):
            Xtr_br, Xte_br = standardize(Xbr[tr], Xbr[te])
            Xtr_ard, Xte_ard = standardize(Xard[tr], Xard[te])
            preds = {
                "mean": MeanModel().fit(y[tr]).predict(len(te)),
                "gblup": GBLUP(0.5).fit(K, y, tr).predict(te),
                "rrblup": RRBLUP().fit(X[tr], y[tr]).predict(X[te]),
                "bayesian_ridge": BayesianRidge().fit(Xtr_br, y[tr]).predict(Xte_br),
                "ard": ARDRegression(max_iter=100, tol=1e-2).fit(Xtr_ard, y[tr]).predict(Xte_ard),
                "lightgbm": fit_lightgbm(X[tr], y[tr], random_state=seed).predict(X[te]),
            }
            for name, p in preds.items():
                rows.append({"trait": trait, "fold": fold, "model": name, "scheme": "kinship_group", **eval_pack(y[te], p)})
            print(f"  fold{fold} done", flush=True)

    metrics = pd.DataFrame(rows)
    out = ROOT / "results" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "wheat_m8_bayes_metrics_by_fold.csv", index=False)
    summary = (
        metrics.groupby(["trait", "model"], as_index=False)
        .agg(pearson_r_mean=("pearson_r", "mean"), pearson_r_std=("pearson_r", "std"), rmse_mean=("rmse", "mean"), n_folds=("rmse", "count"))
        .sort_values(["trait", "pearson_r_mean"], ascending=[True, False])
    )
    summary.to_csv(out / "wheat_m8_bayes_metrics_summary.csv", index=False)
    print("M8 Bayes OK")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
