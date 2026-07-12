#!/usr/bin/env python3
"""Gap-fill: maize Route A — plot-level Yield prediction with design covariates + G/E."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
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


def main() -> int:
    cfg = load_cfg()
    seed = cfg["project"]["seed"]
    interim = ROOT / "data" / "interim" / "maize"
    metrics_dir = ROOT / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    plot = pd.read_parquet(interim / "maize_routeA_plot.parquet")
    g_pc = pd.read_csv(interim / "maize_genotype_pca.csv")
    e_pc = pd.read_csv(interim / "maize_env_pca.csv")
    if "environment_id" not in e_pc.columns:
        e_pc = e_pc.rename(columns={e_pc.columns[0]: "environment_id"})

    d = plot.merge(g_pc, on="genotype_id").merge(e_pc, on="environment_id")
    ycol = cfg["maize"]["primary_trait"]
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
    d = d.dropna(subset=[ycol]).copy()

    # design covariates
    for c in ["Replicate", "Block", "Range", "Pass", "Plot"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    design = [c for c in ["Replicate", "Block", "Range", "Pass"] if c in d.columns]
    g_cols = [c for c in d.columns if c.startswith("G_PC")]
    e_cols = [c for c in d.columns if c.startswith("E_PC")]

    y = d[ycol].to_numpy(float)
    mats = {
        "design": d[design].fillna(0).to_numpy(float) if design else np.zeros((len(d), 1)),
        "G+design": np.hstack([d[g_cols].to_numpy(float), d[design].fillna(0).to_numpy(float)]) if design else d[g_cols].to_numpy(float),
        "E+design": np.hstack([d[e_cols].to_numpy(float), d[design].fillna(0).to_numpy(float)]) if design else d[e_cols].to_numpy(float),
        "G+E+design": np.hstack(
            [d[g_cols].to_numpy(float), d[e_cols].to_numpy(float), d[design].fillna(0).to_numpy(float)]
        ),
    }

    rows = []
    # group by Hybrid|Env to avoid leakage of plot replicates
    groups_ge = (d["genotype_id"].astype(str) + "|" + d["environment_id"].astype(str)).to_numpy()
    for scheme, groups in {
        "leave_gxe": groups_ge,
        "leave_environment": d["environment_id"].to_numpy(),
        "leave_genotype": d["genotype_id"].to_numpy(),
    }.items():
        splits = min(5, pd.Series(groups).nunique())
        gkf = GroupKFold(n_splits=splits)
        print(f"RouteA {scheme} n={len(d)} splits={splits}", flush=True)
        for fold, (tr, te) in enumerate(gkf.split(np.arange(len(d)), groups=groups)):
            ytr, yte = y[tr], y[te]
            p = MeanModel().fit(ytr).predict(len(te))
            rows.append({"scheme": scheme, "fold": fold, "model": "mean", "features": "none", **eval_pack(yte, p)})
            for feat, X in mats.items():
                model = LGBMRegressor(
                    n_estimators=250,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=seed,
                    verbosity=-1,
                    force_col_wise=True,
                )
                p = model.fit(X[tr], ytr).predict(X[te])
                rows.append({"scheme": scheme, "fold": fold, "model": "lightgbm", "features": feat, **eval_pack(yte, p)})

    metrics = pd.DataFrame(rows)
    summary = (
        metrics.groupby(["scheme", "model", "features"], as_index=False)
        .agg(pearson_r_mean=("pearson_r", "mean"), pearson_r_std=("pearson_r", "std"), rmse_mean=("rmse", "mean"), n_folds=("rmse", "count"))
        .sort_values(["scheme", "pearson_r_mean"], ascending=[True, False])
    )
    metrics.to_csv(metrics_dir / "maize_routeA_metrics_by_fold.csv", index=False)
    summary.to_csv(metrics_dir / "maize_routeA_metrics_summary.csv", index=False)
    with open(metrics_dir / "maize_routeA_gate.json", "w") as f:
        json.dump({"n_plots": int(len(d)), "design_cols": design, "best": summary.groupby("scheme").head(1).to_dict(orient="records")}, f, indent=2)
    print("RouteA OK")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
