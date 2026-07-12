#!/usr/bin/env python3
"""M5b: maize Yield ablation on raw Hybrid×Env means (so E-only is identifiable).

Complements m5 (env-centered target) with y_raw_mean under the same CV schemes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import regression_metrics, topk_overlap  # noqa: E402
from src.models.baselines import MeanModel  # noqa: E402


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


def predict(model_name, Xtr, ytr, Xte, seed):
    if model_name == "mean":
        return MeanModel().fit(ytr).predict(len(Xte))
    if model_name == "ridge":
        a, b = standardize(Xtr, Xte)
        return Ridge(alpha=10.0).fit(a, ytr).predict(b)
    model = LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8,
        colsample_bytree=0.8, random_state=seed, verbosity=-1, force_col_wise=True,
    )
    return model.fit(Xtr, ytr).predict(Xte)


def main() -> int:
    cfg = load_cfg()
    seed = cfg["project"]["seed"]
    interim = ROOT / cfg["paths"]["interim"] / "maize"
    metrics_dir = ROOT / cfg["paths"]["results"] / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    gxe = pd.read_parquet(interim / "maize_routeB_yield_gxe.parquet")
    g_pc = pd.read_csv(interim / "maize_genotype_pca.csv")
    e_feat = pd.read_parquet(interim / "maize_env_features.parquet").reset_index()
    if "environment_id" not in e_feat.columns:
        e_feat = e_feat.rename(columns={e_feat.columns[0]: "environment_id"})
    e_pc = pd.read_csv(interim / "maize_env_pca.csv")
    if "environment_id" not in e_pc.columns:
        e_pc = e_pc.rename(columns={e_pc.columns[0]: "environment_id"})

    d = (
        gxe.merge(g_pc, on="genotype_id", how="left")
        .merge(e_feat, on="environment_id", how="left")
        .merge(e_pc, on="environment_id", how="left")
        .dropna(subset=["y_raw_mean"])
    )
    g_cols = [c for c in d.columns if c.startswith("G_PC")]
    e_pc_cols = [c for c in d.columns if c.startswith("E_PC")]
    key_e = [c for c in d.columns if c.startswith("full__") and any(k in c for k in ["GDD__sum", "PRECTOT__sum", "T2M__mean", "VPD__mean", "hot_days", "RAD__sum"])]
    e_cols = e_pc_cols + key_e
    y = d["y_raw_mean"].to_numpy(float)
    G = d[g_cols].to_numpy(float)
    E = d[e_cols].to_numpy(float)
    GE = np.einsum("ij,ik->ijk", G[:, :5], E[:, :5]).reshape(len(d), -1)
    mats = {"G": G, "E": E, "G+E": np.hstack([G, E]), "G+E+GxE": np.hstack([G, E, GE])}

    rows = []
    for scheme, groups in {
        "leave_genotype": d["genotype_id"].to_numpy(),
        "leave_environment": d["environment_id"].to_numpy(),
        "leave_year": d["Year"].to_numpy(),
    }.items():
        splits = min(5, pd.Series(groups).nunique())
        gkf = GroupKFold(n_splits=splits)
        print(f"{scheme} splits={splits}", flush=True)
        for fold, (tr, te) in enumerate(gkf.split(np.arange(len(d)), groups=groups)):
            ytr, yte = y[tr], y[te]
            p = MeanModel().fit(ytr).predict(len(te))
            rows.append({"scheme": scheme, "fold": fold, "model": "mean", "features": "none", **eval_pack(yte, p)})
            for feat, X in mats.items():
                for model in ["ridge", "lightgbm"]:
                    p = predict(model, X[tr], ytr, X[te], seed)
                    rows.append({"scheme": scheme, "fold": fold, "model": model, "features": feat, **eval_pack(yte, p)})

    metrics = pd.DataFrame(rows)
    summary = (
        metrics.groupby(["scheme", "model", "features"], as_index=False)
        .agg(pearson_r_mean=("pearson_r", "mean"), pearson_r_std=("pearson_r", "std"), rmse_mean=("rmse", "mean"), n_folds=("rmse", "count"))
        .sort_values(["scheme", "pearson_r_mean"], ascending=[True, False])
    )
    metrics.to_csv(metrics_dir / "maize_m5b_raw_yield_metrics_by_fold.csv", index=False)
    summary.to_csv(metrics_dir / "maize_m5b_raw_yield_metrics_summary.csv", index=False)

    gate = {}
    for scheme in summary.scheme.unique():
        sub = summary[summary.scheme == scheme]
        best = {f: sub[sub.features == f].sort_values("pearson_r_mean", ascending=False).iloc[0] for f in ["G", "E", "G+E", "G+E+GxE"] if (sub.features == f).any()}
        gate[scheme] = {
            k: {"model": v["model"], "r": float(v["pearson_r_mean"]), "rmse": float(v["rmse_mean"])} for k, v in best.items()
        }
        if "G" in best and "E" in best and "G+E" in best:
            gate[scheme]["ge_beats_g_and_e"] = bool(
                best["G+E"]["pearson_r_mean"] >= max(best["G"]["pearson_r_mean"], best["E"]["pearson_r_mean"]) - 1e-6
                or best["G+E+GxE"]["pearson_r_mean"] >= max(best["G"]["pearson_r_mean"], best["E"]["pearson_r_mean"]) - 1e-6
            )
    with open(metrics_dir / "maize_m5b_gate.json", "w") as f:
        json.dump(gate, f, indent=2)
    print("M5b OK")
    print(summary.to_string(index=False))
    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
