#!/usr/bin/env python3
"""M5: maize Yield G/E/G×E baselines under leave-genotype / leave-environment CV.

Uses Route B Hybrid×Env env-centered yield + genotype PCs + environment features.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import regression_metrics, topk_overlap  # noqa: E402
from src.models.baselines import MeanModel  # noqa: E402


def load_cfg() -> dict:
    with open(ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def eval_pack(y_true, y_pred):
    m = regression_metrics(y_true, y_pred)
    m["top10_overlap"] = topk_overlap(y_true, y_pred, 0.10)
    return m


def standardize(tr, te):
    mu = tr.mean(axis=0)
    sd = tr.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (tr - mu) / sd, (te - mu) / sd


def fit_predict(model_name, Xtr, ytr, Xte, seed):
    if model_name == "mean":
        return MeanModel().fit(ytr).predict(len(Xte))
    if model_name == "ridge":
        Xtr_z, Xte_z = standardize(Xtr, Xte)
        return Ridge(alpha=10.0).fit(Xtr_z, ytr).predict(Xte_z)
    if model_name == "elastic_net":
        Xtr_z, Xte_z = standardize(Xtr, Xte)
        return ElasticNet(alpha=0.2, l1_ratio=0.5, max_iter=4000, random_state=seed).fit(Xtr_z, ytr).predict(Xte_z)
    if model_name == "lightgbm":
        model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbosity=-1,
            force_col_wise=True,
        )
        return model.fit(Xtr, ytr).predict(Xte)
    raise ValueError(model_name)


def main() -> int:
    cfg = load_cfg()
    seed = cfg["project"]["seed"]
    np.random.seed(seed)
    interim = ROOT / cfg["paths"]["interim"] / "maize"
    metrics_dir = ROOT / cfg["paths"]["results"] / "metrics"
    pred_dir = ROOT / cfg["paths"]["results"] / "predictions"
    splits_dir = ROOT / cfg["paths"]["results"] / "splits"
    for d in (metrics_dir, pred_dir, splits_dir):
        d.mkdir(parents=True, exist_ok=True)

    gxe = pd.read_parquet(interim / "maize_routeB_yield_gxe.parquet")
    g_pc = pd.read_csv(interim / "maize_genotype_pca.csv")
    e_feat = pd.read_parquet(interim / "maize_env_features.parquet").reset_index()
    if "environment_id" not in e_feat.columns:
        e_feat = e_feat.rename(columns={e_feat.columns[0]: "environment_id"})
    e_pc = pd.read_csv(interim / "maize_env_pca.csv")
    if "environment_id" not in e_pc.columns:
        e_pc = e_pc.rename(columns={e_pc.columns[0]: "environment_id"})

    df = (
        gxe.merge(g_pc, on="genotype_id", how="left")
        .merge(e_feat, on="environment_id", how="left")
        .merge(e_pc, on="environment_id", how="left")
    )

    g_cols = [c for c in df.columns if c.startswith("G_PC")]
    e_pc_cols = [c for c in df.columns if c.startswith("E_PC")]
    # use compact env: E PCs + a few key full-window stats if present
    key_e = [c for c in df.columns if c.startswith("full__") and any(k in c for k in ["GDD__sum", "PRECTOT__sum", "T2M__mean", "VPD__mean", "hot_days", "RAD__sum"])]
    e_cols = e_pc_cols + key_e
    # drop rows with missing features/target
    use_cols = g_cols + e_cols
    d = df.dropna(subset=["y_env_centered"] + g_cols[:1] + e_pc_cols[:1]).copy()
    y = d["y_env_centered"].to_numpy(dtype=float)
    G = d[g_cols].to_numpy(dtype=float)
    E = d[e_cols].to_numpy(dtype=float)
    # GxE interactions: top 5 G PCs × top 5 E PCs
    g5, e5 = G[:, :5], E[:, :5]
    GE = np.einsum("ij,ik->ijk", g5, e5).reshape(len(d), -1)
    X = {
        "G": G,
        "E": E,
        "G+E": np.hstack([G, E]),
        "G+E+GxE": np.hstack([G, E, GE]),
    }

    schemes = {
        "leave_genotype": d["genotype_id"].to_numpy(),
        "leave_environment": d["environment_id"].to_numpy(),
        "leave_year": d["Year"].to_numpy(),
    }

    metric_rows = []
    pred_rows = []
    n_splits = 5

    for scheme, groups in schemes.items():
        # year may have fewer unique -> adjust splits
        n_unique = pd.Series(groups).nunique()
        splits = min(n_splits, n_unique)
        if splits < 2:
            continue
        gkf = GroupKFold(n_splits=splits)
        print(f"Scheme {scheme}: {splits} folds, n={len(d)}", flush=True)
        for fold, (tr, te) in enumerate(gkf.split(np.arange(len(d)), groups=groups)):
            ytr, yte = y[tr], y[te]
            # mean baseline
            pred = MeanModel().fit(ytr).predict(len(te))
            metric_rows.append({"scheme": scheme, "fold": fold, "model": "mean", "features": "none", **eval_pack(yte, pred)})
            pred_rows.append(_pred(d, te, yte, pred, "mean", "none", scheme, fold))

            for feat_name, mat in X.items():
                Xtr, Xte = mat[tr], mat[te]
                # choose model: ridge for G-only compactness, lgbm for E and fused
                models = ["ridge", "lightgbm"] if feat_name in {"E", "G+E", "G+E+GxE"} else ["ridge", "lightgbm"]
                for model_name in models:
                    print(f"  [{scheme} fold{fold}] {model_name}|{feat_name}", flush=True)
                    p = fit_predict(model_name, Xtr, ytr, Xte, seed)
                    metric_rows.append(
                        {"scheme": scheme, "fold": fold, "model": model_name, "features": feat_name, **eval_pack(yte, p)}
                    )
                    pred_rows.append(_pred(d, te, yte, p, model_name, feat_name, scheme, fold))

    metrics = pd.DataFrame(metric_rows)
    preds = pd.concat(pred_rows, ignore_index=True)
    metrics.to_csv(metrics_dir / "maize_m5_yield_metrics_by_fold.csv", index=False)
    preds.to_csv(pred_dir / "maize_m5_yield_predictions.csv", index=False)

    summary = (
        metrics.groupby(["scheme", "model", "features"], as_index=False)
        .agg(
            pearson_r_mean=("pearson_r", "mean"),
            pearson_r_std=("pearson_r", "std"),
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            top10_overlap_mean=("top10_overlap", "mean"),
            n_folds=("rmse", "count"),
        )
        .sort_values(["scheme", "pearson_r_mean"], ascending=[True, False])
    )
    summary.to_csv(metrics_dir / "maize_m5_yield_metrics_summary.csv", index=False)

    gate = {"schemes": {}, "pass": True}
    for scheme in summary.scheme.unique():
        sub = summary[summary.scheme == scheme]
        mean_rmse = float(sub.loc[sub.model == "mean", "rmse_mean"].iloc[0])
        cand = sub[sub.model != "mean"].sort_values(["pearson_r_mean", "rmse_mean"], ascending=[False, True]).iloc[0]
        # G+E should beat G-only and E-only on leave_environment ideally
        g_only = sub[(sub.features == "G")].sort_values("pearson_r_mean", ascending=False).head(1)
        e_only = sub[(sub.features == "E")].sort_values("pearson_r_mean", ascending=False).head(1)
        ge = sub[(sub.features.isin(["G+E", "G+E+GxE"]))].sort_values("pearson_r_mean", ascending=False).head(1)
        beats_mean = bool(cand["rmse_mean"] < mean_rmse)
        ge_beats = True
        if scheme == "leave_environment" and len(g_only) and len(e_only) and len(ge):
            ge_beats = bool(
                ge.iloc[0]["pearson_r_mean"] >= max(g_only.iloc[0]["pearson_r_mean"], e_only.iloc[0]["pearson_r_mean"]) - 0.02
            )
        gate["schemes"][scheme] = {
            "best_model": cand["model"],
            "best_features": cand["features"],
            "best_r": None if pd.isna(cand["pearson_r_mean"]) else float(cand["pearson_r_mean"]),
            "best_rmse": float(cand["rmse_mean"]),
            "mean_rmse": mean_rmse,
            "beats_mean": beats_mean,
            "ge_competitive_on_new_env": ge_beats,
        }
        if not beats_mean:
            gate["pass"] = False
    with open(metrics_dir / "maize_m5_gate.json", "w") as f:
        json.dump(gate, f, indent=2)

    # save row ids used
    d[["genotype_id", "environment_id", "Year", "Field_Location"]].to_csv(
        splits_dir / "maize_m5_modeling_rows.csv", index=False
    )

    print("M5 OK")
    print(summary.to_string(index=False))
    print(json.dumps(gate, indent=2))
    return 0


def _pred(d, te, y_true, y_pred, model, features, scheme, fold):
    sub = d.iloc[te]
    return pd.DataFrame(
        {
            "genotype_id": sub["genotype_id"].to_numpy(),
            "environment_id": sub["environment_id"].to_numpy(),
            "Year": sub["Year"].to_numpy(),
            "y_true": y_true,
            "y_pred": y_pred,
            "model": model,
            "features": features,
            "scheme": scheme,
            "fold": fold,
        }
    )


if __name__ == "__main__":
    sys.exit(main())
